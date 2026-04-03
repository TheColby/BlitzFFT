// shaders/blitz_fft.cu
//
// BlitzFFT native CUDA kernel — no cuFFT dependency.
//
// Algorithm: iterative Cooley-Tukey DIT radix-2 in shared memory.
//   • One CUDA block handles one FFT.
//   • blockDim.x == N/2 threads.
//   • Entire FFT lives in shared memory (16 KB per block for N=2048).
//   • A global-memory multi-pass path handles N > 2048.
//   • The real-to-complex post-processing (unpack) is a separate kernel.
//   • A Hann window may be fused into the load step.
//
// Kernels exported (extern "C" for PTX mangling):
//   blitz_fft_shared       — single-pass, N ≤ SHMEM_MAX_N
//   blitz_fft_pass         — one butterfly stage, N > SHMEM_MAX_N
//   blitz_bit_rev          — bit-reversal permutation (multi-pass path)
//   blitz_magnitude        — compute magnitude from complex output
//   blitz_hann_window      — fused Hann windowing (element-wise, before pass 0)

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Maximum N that fits in a single shared-memory pass.
// 2048 complex float2 = 2048 × 8 bytes = 16 KB ≤ typical 48 KB shared mem.
#define SHMEM_MAX_N 2048

// ── Helpers ───────────────────────────────────────────────────────────────────

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(
        fmaf(a.x, b.x, -a.y * b.y),
        fmaf(a.x, b.y,  a.y * b.x)
    );
}

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// Twiddle factor W_N^k = exp(-2πi·k/N).
__device__ __forceinline__ float2 twiddle(uint32_t k, uint32_t N) {
    float theta = -2.0f * (float)M_PI * (float)k / (float)N;
    float s, c;
    sincosf(theta, &s, &c);
    return make_float2(c, s);
}

// Iterative bit-reversal (log2N bits).
__device__ __forceinline__ uint32_t bit_rev(uint32_t x, uint32_t log2N) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2N; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// ── Kernel 1: blitz_fft_shared ────────────────────────────────────────────────
//
// Single-pass shared-memory FFT for N ≤ SHMEM_MAX_N.
//
// Grid:  (batch, 1, 1) thread-groups
// Block: (N/2, 1, 1) threads
// Smem:  N × sizeof(float2) bytes  (allocated by caller)
//
// Params layout (device buffer, 6 × uint32):
//   [0] unused (stage, for compat with multi-pass path)
//   [1] N (FFT size)
//   [2] batch (number of FFTs)
//   [3] use_window (1 = apply Hann on load)
//   [4] log2_N
//   [5] half1 = N/2+1

extern "C" __global__ void blitz_fft_shared(
    const float* __restrict__ real_in,    // [batch × N] real input
    const float* __restrict__ window,     // [N] Hann coefficients (ignored if use_window=0)
    float2*      __restrict__ cplx_out,   // [batch × N] complex output
    float*       __restrict__ mag_out,    // [batch × half1] magnitude output
    const uint32_t*            params     // [6]
) {
    extern __shared__ float2 shmem[];

    const uint32_t N        = params[1];
    const uint32_t batch    = params[2];
    const uint32_t use_win  = params[3];
    const uint32_t log2N    = params[4];
    const uint32_t half1    = params[5];
    const uint32_t half_N   = N >> 1;

    const uint32_t fft_idx  = blockIdx.x;
    const uint32_t tid      = threadIdx.x;  // 0 .. N/2-1

    if (fft_idx >= batch || tid >= half_N) return;

    const uint32_t base_in = fft_idx * N;

    // ── Load with bit-reversal + optional Hann window ─────────────────────
    {
        uint32_t a_src = bit_rev(tid,        log2N);
        uint32_t b_src = bit_rev(tid + half_N, log2N);
        float sa = real_in[base_in + a_src];
        float sb = real_in[base_in + b_src];
        if (use_win) {
            sa *= window[a_src];
            sb *= window[b_src];
        }
        shmem[tid]        = make_float2(sa, 0.0f);
        shmem[tid + half_N] = make_float2(sb, 0.0f);
    }
    __syncthreads();

    // ── All log2(N) butterfly stages in SRAM ─────────────────────────────
    for (uint32_t s = 1; s <= log2N; ++s) {
        const uint32_t half_span = 1u << (s - 1);
        const uint32_t span      = half_span << 1;
        const uint32_t group     = tid / half_span;
        const uint32_t leg       = tid % half_span;
        const uint32_t i         = group * span + leg;
        const uint32_t j         = i + half_span;

        float2 W  = twiddle(leg, span);
        float2 a  = shmem[i];
        float2 tb = cmul(W, shmem[j]);
        shmem[i]  = cadd(a, tb);
        shmem[j]  = csub(a, tb);
        __syncthreads();
    }

    // ── Write complex output + inline magnitude ───────────────────────────
    const uint32_t base_out = fft_idx * N;

    uint32_t i = tid;
    uint32_t j = tid + half_N;
    cplx_out[base_out + i] = shmem[i];
    cplx_out[base_out + j] = shmem[j];

    if (i < half1) {
        float2 c = shmem[i];
        mag_out[fft_idx * half1 + i] = sqrtf(c.x * c.x + c.y * c.y);
    }
    // Nyquist bin: tid == 0 writes half_N into mag_out.
    if (tid == 0) {
        float2 ny = shmem[half_N];
        mag_out[fft_idx * half1 + half_N] = sqrtf(ny.x * ny.x + ny.y * ny.y);
    }
}

// ── Kernel 2: blitz_bit_rev ───────────────────────────────────────────────────
//
// Bit-reversal permutation into complex output buffer (multi-pass path).
// Grid: (ceil(N/blockDim.x), batch, 1), Block: (min(512,N), 1, 1)

extern "C" __global__ void blitz_bit_rev(
    float2*          __restrict__ cplx_out, // [batch × N] — written in-place
    const float*     __restrict__ real_in,  // [batch × N] real input
    const float*     __restrict__ window,
    const uint32_t*              params
) {
    const uint32_t N       = params[1];
    const uint32_t batch   = params[2];
    const uint32_t use_win = params[3];
    const uint32_t log2N   = params[4];

    const uint32_t fft_idx = blockIdx.y;
    const uint32_t idx     = blockIdx.x * blockDim.x + threadIdx.x;

    if (fft_idx >= batch || idx >= N) return;

    uint32_t rev = bit_rev(idx, log2N);
    const uint32_t base = fft_idx * N;
    float val = real_in[base + idx];
    if (use_win) val *= window[idx];
    cplx_out[base + rev] = make_float2(val, 0.0f);
}

// ── Kernel 3: blitz_fft_pass ─────────────────────────────────────────────────
//
// One butterfly stage over global memory (multi-pass path, N > SHMEM_MAX_N).
// Grid: (ceil(N/2/blockDim.x), batch, 1), Block: (512, 1, 1)
// Params[0] = current stage index s (0-indexed).

extern "C" __global__ void blitz_fft_pass(
    float2*       __restrict__ data,   // [batch × N] complex buffer
    const uint32_t*             params
) {
    const uint32_t stage    = params[0];
    const uint32_t N        = params[1];
    const uint32_t batch    = params[2];

    const uint32_t fft_idx   = blockIdx.y;
    const uint32_t butterfly = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half_N    = N >> 1;

    if (fft_idx >= batch || butterfly >= half_N) return;

    const uint32_t base      = fft_idx * N;
    const uint32_t half_span = 1u << stage;
    const uint32_t span      = half_span << 1;
    const uint32_t group     = butterfly / half_span;
    const uint32_t leg       = butterfly % half_span;
    const uint32_t i         = base + group * span + leg;
    const uint32_t j         = i + half_span;

    float2 W  = twiddle(leg, span);
    float2 a  = data[i];
    float2 tb = cmul(W, data[j]);
    data[i]   = cadd(a, tb);
    data[j]   = csub(a, tb);
}

// ── Kernel 4: blitz_magnitude ─────────────────────────────────────────────────
//
// Compute magnitude from complex buffer; only positive-frequency half.
// Grid: (ceil(half1/blockDim.x), batch, 1), Block: (512, 1, 1)

extern "C" __global__ void blitz_magnitude(
    const float2* __restrict__ cplx_in,  // [batch × N]
    float*        __restrict__ mag_out,  // [batch × half1]
    const uint32_t*             params
) {
    const uint32_t N       = params[1];
    const uint32_t batch   = params[2];
    const uint32_t half1   = params[5];

    const uint32_t fft_idx = blockIdx.y;
    const uint32_t bin     = blockIdx.x * blockDim.x + threadIdx.x;

    if (fft_idx >= batch || bin >= half1) return;

    float2 c = cplx_in[fft_idx * N + bin];
    mag_out[fft_idx * half1 + bin] = sqrtf(c.x * c.x + c.y * c.y);
}
