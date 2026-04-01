// shaders/fft.metal  (v2 — shared-memory Cooley-Tukey)
//
// Key optimisation: the entire FFT runs inside threadgroup (shared) memory.
// The old design issued log2(N) separate kernel dispatches, each reading and
// writing global memory.  This design does all log2(N) stages with one
// global-read and one global-write, touching ~16 KB of threadgroup SRAM
// (8 bytes/sample × 2048 samples = 16 KB — fits Apple Silicon's 32 KB limit).
//
// Single-pass path  — N ≤ 2048, one threadgroup per FFT, N/2 threads.
// Multi-pass path   — N > 2048, one dispatch per stage (global memory).
// Both paths fuse Hann windowing on load (no separate CPU pass).
//
// Params buffer layout  (constant uint[6]):
//   [0] stage        — current stage for multi-pass (ignored in single-pass)
//   [1] fft_size     — N
//   [2] batch_size   — number of FFTs in the buffer
//   [3] use_window   — 1 → apply Hann window on load, 0 → raw
//   [4] log2_N       — precomputed log2(N)
//   [5] half1        — N/2 + 1

#include <metal_stdlib>
using namespace metal;

// ── Complex helpers ─────────────────────────────────────────────────────────

inline float2 cmul(float2 a, float2 b) {
    return float2(fma(a.x, b.x, -a.y * b.y),
                  fma(a.x, b.y,  a.y * b.x));
}

// Unit twiddle W_N^k = e^{-2πi·k/N}
inline float2 twiddle_unit(uint k, uint N) {
    float theta = -2.0f * M_PI_F * float(k) / float(N);
    float s, c;
    s = sincos(theta, c);
    return float2(c, s);
}

// ── Bit-reversal (log2(N) passed in params) ──────────────────────────────────

inline uint bit_rev(uint x, uint log2N) {
    uint r = 0;
    for (uint i = 0; i < log2N; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 1: fft_shared
//   Single-pass FFT for N ≤ 2048 (N/2 threads per group, one group per FFT).
//   Fuses: bit-reversal + all butterfly stages + windowing + magnitude.
//   Global memory is touched exactly once in, once out — ~log2(N)× speedup
//   over the old per-stage global dispatch.
// ─────────────────────────────────────────────────────────────────────────────
kernel void fft_shared(
    device   const float*  real_in  [[ buffer(0) ]],
    device   const float*  window   [[ buffer(1) ]],
    device         float2* cplx_out [[ buffer(2) ]],
    device         float*  mag_out  [[ buffer(3) ]],
    constant       uint*   params   [[ buffer(4) ]],
    threadgroup    float2* shmem    [[ threadgroup(0) ]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  gid [[threadgroup_position_in_grid]])
{
    const uint N       = params[1];
    const uint batch   = params[2];
    const uint use_win = params[3];
    const uint log2N   = params[4];
    const uint half1   = params[5];

    if (gid >= batch) return;

    const uint base_in  = gid * N;
    const uint half_N   = N / 2;

    // ── Load with bit-reversal + optional windowing ───────────────────────
    // Each thread loads two elements (positions tid and tid + N/2).
    {
        uint a_src = bit_rev(tid,          log2N);
        uint b_src = bit_rev(tid + half_N, log2N);
        float sa = real_in[base_in + a_src];
        float sb = real_in[base_in + b_src];
        if (use_win) {
            sa *= window[a_src];
            sb *= window[b_src];
        }
        shmem[tid]          = float2(sa, 0.0f);
        shmem[tid + half_N] = float2(sb, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── All butterfly stages (in threadgroup SRAM) ────────────────────────
    for (uint s = 1; s <= log2N; ++s) {
        const uint half_span = 1u << (s - 1);
        const uint span      = half_span << 1;
        const uint group     = tid / half_span;
        const uint leg       = tid % half_span;
        const uint i         = group * span + leg;
        const uint j         = i + half_span;

        float2 W  = twiddle_unit(leg, span);
        float2 a  = shmem[i];
        float2 tb = cmul(W, shmem[j]);
        shmem[i]  = a + tb;
        shmem[j]  = a - tb;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write complex output + inline magnitude ───────────────────────────
    const uint base_out = gid * N;

    uint i = tid;
    uint j = tid + half_N;
    cplx_out[base_out + i] = shmem[i];
    cplx_out[base_out + j] = shmem[j];

    if (i < half1) {
        float2 c = shmem[i];
        mag_out[gid * half1 + i] = sqrt(c.x*c.x + c.y*c.y);
    }
    // Nyquist bin: index N/2, covered by tid == half_N-1 incrementing to half1
    if (tid == 0) {
        float2 ny = shmem[half_N];
        mag_out[gid * half1 + half_N] = sqrt(ny.x*ny.x + ny.y*ny.y);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2: fft_pass_global
//   One butterfly stage over global memory — used for N > 2048 (multi-pass).
//   Fuses optional windowing into stage 0 (after bit_reverse_global).
// ─────────────────────────────────────────────────────────────────────────────
kernel void fft_pass_global(
    device         float2* data   [[ buffer(0) ]],
    device   const float*  window [[ buffer(1) ]],
    constant       uint*   params [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    const uint stage     = params[0];
    const uint N         = params[1];
    const uint batch     = params[2];
    const uint use_win   = params[3];

    const uint fft_idx   = gid.y;
    const uint butterfly = gid.x;
    const uint half_N    = N >> 1;

    if (fft_idx >= batch || butterfly >= half_N) return;

    const uint base      = fft_idx * N;
    const uint half_span = 1u << stage;
    const uint span      = half_span << 1;
    const uint group     = butterfly / half_span;
    const uint leg       = butterfly % half_span;
    const uint i         = base + group * span + leg;
    const uint j         = i + half_span;

    float2 a = data[i];
    float2 b = data[j];

    if (stage == 0 && use_win) {
        a.x *= window[i - base];
        b.x *= window[j - base];
    }

    float2 W  = twiddle_unit(leg, span);
    float2 tb = cmul(W, b);
    data[i]   = a + tb;
    data[j]   = a - tb;
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 3: bit_reverse_global  (multi-pass path only)
// ─────────────────────────────────────────────────────────────────────────────
kernel void bit_reverse_global(
    device   float2* data   [[ buffer(0) ]],
    constant uint*   params [[ buffer(1) ]],
    uint2            gid    [[ thread_position_in_grid ]])
{
    const uint N      = params[1];
    const uint batch  = params[2];
    const uint log2N  = params[4];

    const uint fft_idx = gid.y;
    const uint idx     = gid.x;
    if (fft_idx >= batch || idx >= N) return;

    uint rev = bit_rev(idx, log2N);
    if (rev > idx) {
        uint base        = fft_idx * N;
        float2 t         = data[base + idx];
        data[base + idx] = data[base + rev];
        data[base + rev] = t;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 4: magnitude_only  (multi-pass path only)
// ─────────────────────────────────────────────────────────────────────────────
kernel void magnitude_only(
    device const float2* cplx_in [[ buffer(0) ]],
    device       float*  mag_out [[ buffer(1) ]],
    constant     uint*   params  [[ buffer(2) ]],
    uint2                gid     [[ thread_position_in_grid ]])
{
    const uint N      = params[1];
    const uint batch  = params[2];
    const uint half1  = params[5];

    const uint fft_idx = gid.y;
    const uint bin     = gid.x;
    if (fft_idx >= batch || bin >= half1) return;

    float2 c = cplx_in[fft_idx * N + bin];
    mag_out[fft_idx * half1 + bin] = sqrt(c.x*c.x + c.y*c.y);
}
