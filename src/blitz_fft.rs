// src/blitz_fft.rs
//
// BlitzFFT — native Rust FFT engine.  No external FFT libraries are used.
//
// Algorithms
// ──────────
//   • Power-of-two N  → iterative Cooley-Tukey DIT radix-2
//   • Arbitrary N     → Bluestein chirp-z transform (inner FFT is power-of-two)
//   • Real-input N    → half-size complex trick (pow2) or direct Bluestein (arb)
//   • Precision       → f32 (BlitzFftPlan) and f64 (BlitzFftPlan64)
//
// SIMD acceleration
// ─────────────────
//   • aarch64 NEON   → 2 complex butterflies per instruction group
//   • x86_64 AVX2+FMA→ 4 complex butterflies per instruction group
//   • Scalar fallback → always compiled, used when SIMD unavailable or half < SIMD_MIN
//
// Plan caching
// ────────────
//   Plans are cached globally; repeated calls for the same N reuse the plan.

#![allow(clippy::excessive_precision)]

use std::{
    collections::HashMap,
    f32::consts::PI as PI32,
    f64::consts::PI as PI64,
    sync::{Arc, Mutex},
};

use num_complex::{Complex32, Complex64};
use once_cell::sync::Lazy;

// ─── f32 Plan ─────────────────────────────────────────────────────────────────

/// Pre-planned FFT for a specific size N (f32 precision).
///
/// Supports power-of-two N via Cooley-Tukey and arbitrary N via Bluestein.
pub struct BlitzFftPlan {
    /// Full signal length N.
    pub n: usize,
    /// Inner complex FFT size for the real-input trick (N/2, only used when N is pow2).
    m: usize,
    /// Precomputed bit-reversal indices for the inner M-point FFT (only pow2).
    bit_rev: Vec<u32>,
    /// Twiddle factors for the inner M-point FFT: W_M^k = exp(-2πi·k/M), k=0..M/2-1.
    twiddles: Vec<Complex32>,
    /// Unpack twiddles for the real-to-complex post-processing: W_N^k, k=0..M.
    unpack: Vec<Complex32>,
    /// Whether N (or M for the real trick) is a power of two.
    pow2: bool,
}

impl BlitzFftPlan {
    fn new(n: usize) -> Self {
        assert!(n >= 2, "FFT size must be at least 2");
        let pow2 = n.is_power_of_two();
        let m = if pow2 { n / 2 } else { 0 };

        let bit_rev = if pow2 {
            let log2m = m.trailing_zeros();
            (0..m as u32)
                .map(|i| i.reverse_bits() >> (32 - log2m))
                .collect()
        } else {
            vec![]
        };

        let twiddles = if pow2 {
            // W_M^k = exp(-2πi·k/M) for k = 0..M/2
            (0..m / 2)
                .map(|k| {
                    let theta = -2.0 * PI32 * k as f32 / m as f32;
                    Complex32::new(theta.cos(), theta.sin())
                })
                .collect()
        } else {
            vec![]
        };

        let unpack = if pow2 {
            // W_N^k = exp(-2πi·k/N) for k = 0..=M
            (0..=m)
                .map(|k| {
                    let theta = -2.0 * PI32 * k as f32 / n as f32;
                    Complex32::new(theta.cos(), theta.sin())
                })
                .collect()
        } else {
            vec![]
        };

        Self { n, m, bit_rev, twiddles, unpack, pow2 }
    }

    // ── Scalar butterfly stage ──────────────────────────────────────────────

    #[inline(never)]
    fn butterfly_stage_scalar(
        twiddles: &[Complex32],
        buf: &mut [Complex32],
        step: usize,
        m: usize,
    ) {
        let half = step >> 1;
        let stride = m / step; // twiddle table stride for this stage
        let mut start = 0usize;
        while start < m {
            for k in 0..half {
                let w = twiddles[k * stride];
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw_re = w.re * b.re - w.im * b.im;
                let bw_im = w.re * b.im + w.im * b.re;
                buf[start + k] = Complex32::new(a.re + bw_re, a.im + bw_im);
                buf[start + k + half] = Complex32::new(a.re - bw_re, a.im - bw_im);
            }
            start += step;
        }
    }

    // ── NEON butterfly stage (aarch64) ──────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn butterfly_stage_neon(
        twiddles: &[Complex32],
        buf: &mut [Complex32],
        step: usize,
        m: usize,
    ) {
        use std::arch::aarch64::*;

        let half = step >> 1;
        let stride = m / step;

        // sign mask: [-0.0f, 0.0f, -0.0f, 0.0f] — flips sign of real parts
        let sign_mask: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
        let vsign = vld1q_f32(sign_mask.as_ptr());

        let mut start = 0usize;
        while start < m {
            let mut k = 0usize;

            // Process 2 butterflies at a time when half >= 2.
            while k + 1 < half {
                let w0 = twiddles[k * stride];
                let w1 = twiddles[(k + 1) * stride];
                let ai = start + k;
                let bi = start + k + half;

                // Load [a0.re, a0.im, a1.re, a1.im]
                let a_vec = vld1q_f32(buf.as_ptr().add(ai) as *const f32);
                // Load [b0.re, b0.im, b1.re, b1.im]
                let b_vec = vld1q_f32(buf.as_ptr().add(bi) as *const f32);
                // Pack twiddles: [w0.re, w0.im, w1.re, w1.im]
                let w0_f = vld1_f32(&w0 as *const Complex32 as *const f32);
                let w1_f = vld1_f32(&w1 as *const Complex32 as *const f32);
                let w_vec = vcombine_f32(w0_f, w1_f);

                // Duplicate real and imaginary parts of w:
                // w_re = [w0.re, w0.re, w1.re, w1.re]
                // w_im = [w0.im, w0.im, w1.im, w1.im]
                let w_re = vtrn1q_f32(w_vec, w_vec);
                let w_im = vtrn2q_f32(w_vec, w_vec);

                // b_swap = [b0.im, b0.re, b1.im, b1.re]  (swap re/im within each pair)
                let b_swap = vrev64q_f32(b_vec);

                // t1 = [w0.re*b0.re, w0.re*b0.im, w1.re*b1.re, w1.re*b1.im]
                let t1 = vmulq_f32(w_re, b_vec);
                // t2 = [w0.im*b0.im, w0.im*b0.re, w1.im*b1.im, w1.im*b1.re]
                let t2 = vmulq_f32(w_im, b_swap);

                // Flip re parts of t2: t2_adj = [-w.im*b.im, w.im*b.re, ...]
                let t2_adj = vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(t2),
                    vreinterpretq_u32_f32(vsign),
                ));

                // twisted = t1 + t2_adj = [w.re*b.re - w.im*b.im, w.re*b.im + w.im*b.re, ...]
                let twisted = vaddq_f32(t1, t2_adj);

                let out_a = vaddq_f32(a_vec, twisted);
                let out_b = vsubq_f32(a_vec, twisted);

                vst1q_f32(buf.as_mut_ptr().add(ai) as *mut f32, out_a);
                vst1q_f32(buf.as_mut_ptr().add(bi) as *mut f32, out_b);

                k += 2;
            }

            // Scalar tail for odd half or remainder.
            while k < half {
                let w = twiddles[k * stride];
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw_re = w.re * b.re - w.im * b.im;
                let bw_im = w.re * b.im + w.im * b.re;
                buf[start + k] = Complex32::new(a.re + bw_re, a.im + bw_im);
                buf[start + k + half] = Complex32::new(a.re - bw_re, a.im - bw_im);
                k += 1;
            }

            start += step;
        }
    }

    // ── AVX2 + FMA butterfly stage (x86_64) ─────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn butterfly_stage_avx2(
        twiddles: &[Complex32],
        buf: &mut [Complex32],
        step: usize,
        m: usize,
    ) {
        use std::arch::x86_64::*;

        let half = step >> 1;
        let stride = m / step;

        let mut start = 0usize;
        while start < m {
            let mut k = 0usize;

            // Process 4 butterflies at a time.
            while k + 3 < half {
                let w0 = twiddles[k * stride];
                let w1 = twiddles[(k + 1) * stride];
                let w2 = twiddles[(k + 2) * stride];
                let w3 = twiddles[(k + 3) * stride];

                let ai = start + k;
                let bi = start + k + half;

                // Load [a0.re, a0.im, a1.re, a1.im, a2.re, a2.im, a3.re, a3.im]
                let a_vec = _mm256_loadu_ps(buf.as_ptr().add(ai) as *const f32);
                // Load b similarly
                let b_vec = _mm256_loadu_ps(buf.as_ptr().add(bi) as *const f32);

                // Gather twiddles into [w0.re,w0.im,w1.re,w1.im,w2.re,w2.im,w3.re,w3.im]
                let w_arr: [f32; 8] = [w0.re, w0.im, w1.re, w1.im, w2.re, w2.im, w3.re, w3.im];
                let w_vec = _mm256_loadu_ps(w_arr.as_ptr());

                // _mm256_moveldup_ps duplicates even-indexed floats:
                // [w0.re, w0.re, w1.re, w1.re, w2.re, w2.re, w3.re, w3.re]
                let w_re = _mm256_moveldup_ps(w_vec);
                // _mm256_movehdup_ps duplicates odd-indexed floats:
                // [w0.im, w0.im, w1.im, w1.im, w2.im, w2.im, w3.im, w3.im]
                let w_im = _mm256_movehdup_ps(w_vec);

                // b_swap = [b0.im, b0.re, b1.im, b1.re, ...] (swap re/im pairs)
                // imm8=0xB1 = 10_11_00_01: swaps adjacent pairs within 128-bit lanes
                let b_swap = _mm256_permute_ps(b_vec, 0xB1);

                // t1 = w_re * b = [w.re*b.re, w.re*b.im, ...]
                let t1 = _mm256_mul_ps(w_re, b_vec);
                // t2 = w_im * b_swap = [w.im*b.im, w.im*b.re, ...]
                let t2 = _mm256_mul_ps(w_im, b_swap);
                // twisted[even] = t1 - t2 = w.re*b.re - w.im*b.im  (real part)
                // twisted[odd]  = t1 + t2 = w.re*b.im + w.im*b.re  (imag part)
                let twisted = _mm256_addsub_ps(t1, t2);

                let out_a = _mm256_add_ps(a_vec, twisted);
                let out_b = _mm256_sub_ps(a_vec, twisted);

                _mm256_storeu_ps(buf.as_mut_ptr().add(ai) as *mut f32, out_a);
                _mm256_storeu_ps(buf.as_mut_ptr().add(bi) as *mut f32, out_b);

                k += 4;
            }

            // Scalar tail.
            while k < half {
                let w = twiddles[k * stride];
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw_re = w.re * b.re - w.im * b.im;
                let bw_im = w.re * b.im + w.im * b.re;
                buf[start + k] = Complex32::new(a.re + bw_re, a.im + bw_im);
                buf[start + k + half] = Complex32::new(a.re - bw_re, a.im - bw_im);
                k += 1;
            }

            start += step;
        }
    }

    // ── In-place complex DIT FFT for power-of-two length M ─────────────────

    /// In-place Cooley-Tukey DIT FFT.  `buf` must have length `self.m` (= N/2).
    pub fn fft_pow2_inplace(&self, buf: &mut [Complex32]) {
        debug_assert!(self.pow2);
        debug_assert_eq!(buf.len(), self.m);

        // Bit-reversal permutation.
        for i in 0..self.m {
            let r = self.bit_rev[i] as usize;
            if r > i {
                buf.swap(i, r);
            }
        }

        // Butterfly stages: step = 2, 4, 8, …, m.
        let m = self.m;
        let twiddles = &self.twiddles;

        let mut step = 2usize;
        while step <= m {
            #[cfg(target_arch = "aarch64")]
            {
                if step >= 4 {
                    // SAFETY: always available on aarch64.
                    unsafe { Self::butterfly_stage_neon(twiddles, buf, step, m) };
                } else {
                    Self::butterfly_stage_scalar(twiddles, buf, step, m);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if step >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe { Self::butterfly_stage_avx2(twiddles, buf, step, m) };
                } else {
                    Self::butterfly_stage_scalar(twiddles, buf, step, m);
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            Self::butterfly_stage_scalar(twiddles, buf, step, m);

            step <<= 1;
        }
    }

    // ── Real-to-complex FFT for power-of-two N ─────────────────────────────

    /// Real-to-complex FFT using the half-size complex trick.
    ///
    /// `input`   — N real samples
    /// `scratch` — work buffer of length N/2
    /// `output`  — N/2+1 complex bins (DC … Nyquist)
    pub fn fft_real_pow2(
        &self,
        input: &[f32],
        scratch: &mut [Complex32],
        output: &mut [Complex32],
    ) {
        debug_assert!(self.pow2);
        let m = self.m;
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(scratch.len(), m);
        debug_assert_eq!(output.len(), m + 1);

        // Pack: z[k] = x[2k] + i·x[2k+1]
        for k in 0..m {
            scratch[k] = Complex32::new(input[2 * k], input[2 * k + 1]);
        }

        // M-point complex FFT.
        self.fft_pow2_inplace(scratch);

        // Unpack to N/2+1 bins.
        let z0 = scratch[0];
        output[0] = Complex32::new(z0.re + z0.im, 0.0);
        output[m] = Complex32::new(z0.re - z0.im, 0.0);

        // For k = 1 .. M/2 we compute both X[k] and X[M-k] from Z[k] and Z[M-k].
        // We iterate in pairs to avoid reading overwritten values.
        for k in 1..=(m / 2) {
            let zk = scratch[k];
            let zmk = scratch[m - k];
            let zmk_conj = Complex32::new(zmk.re, -zmk.im);

            let even = Complex32::new(
                (zk.re + zmk_conj.re) * 0.5,
                (zk.im + zmk_conj.im) * 0.5,
            );
            let diff = Complex32::new(
                (zk.re - zmk_conj.re) * 0.5,
                (zk.im - zmk_conj.im) * 0.5,
            );
            // diff / (i) = (diff.im, -diff.re), so -(i·diff) / 2 = (diff.im, -diff.re) * 0.5
            // We already have the 0.5 factor in `diff`.
            let neg_i_diff = Complex32::new(diff.im, -diff.re);

            let w = self.unpack[k];
            let xk = Complex32::new(
                even.re + w.re * neg_i_diff.re - w.im * neg_i_diff.im,
                even.im + w.re * neg_i_diff.im + w.im * neg_i_diff.re,
            );
            output[k] = xk;

            // X[M-k] (only if distinct from X[k])
            if m - k != k {
                let zk2 = zmk;
                let zmk2_conj = Complex32::new(zk.re, -zk.im);
                let even2 = Complex32::new(
                    (zk2.re + zmk2_conj.re) * 0.5,
                    (zk2.im + zmk2_conj.im) * 0.5,
                );
                let diff2 = Complex32::new(
                    (zk2.re - zmk2_conj.re) * 0.5,
                    (zk2.im - zmk2_conj.im) * 0.5,
                );
                let neg_i_diff2 = Complex32::new(diff2.im, -diff2.re);
                let w2 = self.unpack[m - k];
                output[m - k] = Complex32::new(
                    even2.re + w2.re * neg_i_diff2.re - w2.im * neg_i_diff2.im,
                    even2.im + w2.re * neg_i_diff2.im + w2.im * neg_i_diff2.re,
                );
            }
        }
    }

    // ── Bluestein chirp-z for arbitrary-length complex FFT ──────────────────

    /// Complex FFT of arbitrary length via Bluestein's algorithm.
    ///
    /// Computes `buf = DFT(buf)` for any N (not just power-of-two).
    /// Uses an internal power-of-two FFT of size ≥ 2N-1.
    fn fft_bluestein_inplace(buf: &mut [Complex32]) {
        let n = buf.len();
        if n <= 1 {
            return;
        }
        if n.is_power_of_two() {
            // Use the fast path.
            fft_pow2_scratch(buf);
            return;
        }

        // Choose M = next power of two ≥ 2N-1 for the convolution.
        let conv_len = (2 * n - 1).next_power_of_two();

        // Precompute chirp: W_N^(k²/2) = exp(-πi·k²/N)
        let mut chirp: Vec<Complex32> = Vec::with_capacity(n);
        for k in 0..n {
            let theta = -PI32 * (k * k % (2 * n)) as f32 / n as f32;
            chirp.push(Complex32::new(theta.cos(), theta.sin()));
        }

        // y[k] = x[k] * chirp[k]
        let mut y = vec![Complex32::new(0.0, 0.0); conv_len];
        for k in 0..n {
            y[k] = buf[k] * chirp[k];
        }

        // h[k] = conj(chirp[k]) for k = 0..N-1, zero-padded, wrap-around for negative indices.
        let mut h = vec![Complex32::new(0.0, 0.0); conv_len];
        for k in 0..n {
            let c = Complex32::new(chirp[k].re, -chirp[k].im); // conj(chirp[k])
            h[k] = c;
            if k > 0 {
                h[conv_len - k] = c;
            }
        }

        // Convolve y and h via FFT convolution.
        fft_pow2_scratch(&mut y);
        fft_pow2_scratch(&mut h);

        for i in 0..conv_len {
            let yr = y[i].re;
            let yi = y[i].im;
            let hr = h[i].re;
            let hi = h[i].im;
            y[i] = Complex32::new(yr * hr - yi * hi, yr * hi + yi * hr);
        }

        // IFFT via conj + FFT + conj + scale.
        for v in y.iter_mut() {
            *v = Complex32::new(v.re, -v.im);
        }
        fft_pow2_scratch(&mut y);
        let scale = 1.0 / conv_len as f32;
        for v in y.iter_mut() {
            *v = Complex32::new(v.re * scale, -v.im * scale);
        }

        // Output: X[k] = chirp[k] * y[k]
        for k in 0..n {
            buf[k] = chirp[k] * y[k];
        }
    }

    // ── Public real-to-complex FFT (arbitrary N) ────────────────────────────

    /// Real-to-complex forward FFT for arbitrary N (power-of-two or not).
    ///
    /// `input`  — N real samples
    /// `output` — N/2+1 complex bins
    /// `scratch` — work buffer of length N/2 (used only when N is a power of two)
    pub fn fft_real(
        &self,
        input: &[f32],
        scratch: &mut [Complex32],
        output: &mut [Complex32],
    ) {
        if self.pow2 {
            self.fft_real_pow2(input, scratch, output);
        } else {
            // For arbitrary N: Bluestein over the full complex signal.
            // `scratch` is unused here; the Bluestein path allocates internally.
            let _ = scratch;
            let n = self.n;
            let half1 = n / 2 + 1;
            debug_assert_eq!(output.len(), half1);

            let mut buf: Vec<Complex32> = input
                .iter()
                .map(|&r| Complex32::new(r, 0.0))
                .collect();
            Self::fft_bluestein_inplace(&mut buf);

            output[..half1].copy_from_slice(&buf[..half1]);
        }
    }
}

// ── Stand-alone power-of-two in-place FFT (no plan struct) ────────────────────
// Used internally by Bluestein as the inner FFT.

fn fft_pow2_scratch(buf: &mut [Complex32]) {
    let m = buf.len();
    debug_assert!(m.is_power_of_two());
    if m <= 1 {
        return;
    }

    let log2m = m.trailing_zeros();

    // Bit-reversal.
    for i in 0..m as u32 {
        let r = i.reverse_bits() >> (32 - log2m);
        if r > i {
            buf.swap(i as usize, r as usize);
        }
    }

    // Butterfly stages (scalar — Bluestein inner sizes may not be audio-sized).
    let mut step = 2usize;
    while step <= m {
        let half = step >> 1;
        let angle = -PI32 * 2.0 / step as f32;
        let w_step = Complex32::new(angle.cos(), angle.sin());

        let mut start = 0;
        while start < m {
            let mut w = Complex32::new(1.0, 0.0);
            for k in 0..half {
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw = Complex32::new(
                    w.re * b.re - w.im * b.im,
                    w.re * b.im + w.im * b.re,
                );
                buf[start + k] = Complex32::new(a.re + bw.re, a.im + bw.im);
                buf[start + k + half] = Complex32::new(a.re - bw.re, a.im - bw.im);
                w = Complex32::new(
                    w.re * w_step.re - w.im * w_step.im,
                    w.re * w_step.im + w.im * w_step.re,
                );
            }
            start += step;
        }
        step <<= 1;
    }
}

// ─── f64 Plan ─────────────────────────────────────────────────────────────────

/// Pre-planned FFT for a specific size N (f64 precision).
pub struct BlitzFftPlan64 {
    pub n: usize,
    m: usize,
    bit_rev: Vec<u32>,
    twiddles: Vec<Complex64>,
    unpack: Vec<Complex64>,
}

impl BlitzFftPlan64 {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "f64 plan requires power-of-two N");
        assert!(n >= 2);
        let m = n / 2;
        let log2m = m.trailing_zeros();

        let bit_rev = (0..m as u32)
            .map(|i| i.reverse_bits() >> (32 - log2m))
            .collect();

        let twiddles = (0..m / 2)
            .map(|k| {
                let theta = -2.0 * PI64 * k as f64 / m as f64;
                Complex64::new(theta.cos(), theta.sin())
            })
            .collect();

        let unpack = (0..=m)
            .map(|k| {
                let theta = -2.0 * PI64 * k as f64 / n as f64;
                Complex64::new(theta.cos(), theta.sin())
            })
            .collect();

        Self { n, m, bit_rev, twiddles, unpack }
    }

    fn butterfly_stage_scalar(
        twiddles: &[Complex64],
        buf: &mut [Complex64],
        step: usize,
        m: usize,
    ) {
        let half = step >> 1;
        let stride = m / step;
        let mut start = 0usize;
        while start < m {
            for k in 0..half {
                let w = twiddles[k * stride];
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw_re = w.re * b.re - w.im * b.im;
                let bw_im = w.re * b.im + w.im * b.re;
                buf[start + k] = Complex64::new(a.re + bw_re, a.im + bw_im);
                buf[start + k + half] = Complex64::new(a.re - bw_re, a.im - bw_im);
            }
            start += step;
        }
    }

    pub fn fft_pow2_inplace(&self, buf: &mut [Complex64]) {
        debug_assert_eq!(buf.len(), self.m);

        for i in 0..self.m {
            let r = self.bit_rev[i] as usize;
            if r > i {
                buf.swap(i, r);
            }
        }

        let m = self.m;
        let twiddles = &self.twiddles;
        let mut step = 2usize;
        while step <= m {
            Self::butterfly_stage_scalar(twiddles, buf, step, m);
            step <<= 1;
        }
    }

    /// Real-to-complex FFT for power-of-two N (f64).
    pub fn fft_real_pow2(
        &self,
        input: &[f64],
        scratch: &mut [Complex64],
        output: &mut [Complex64],
    ) {
        let m = self.m;
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(scratch.len(), m);
        debug_assert_eq!(output.len(), m + 1);

        for k in 0..m {
            scratch[k] = Complex64::new(input[2 * k], input[2 * k + 1]);
        }
        self.fft_pow2_inplace(scratch);

        let z0 = scratch[0];
        output[0] = Complex64::new(z0.re + z0.im, 0.0);
        output[m] = Complex64::new(z0.re - z0.im, 0.0);

        for k in 1..=(m / 2) {
            let zk = scratch[k];
            let zmk = scratch[m - k];
            let zmk_conj = Complex64::new(zmk.re, -zmk.im);
            let even = Complex64::new(
                (zk.re + zmk_conj.re) * 0.5,
                (zk.im + zmk_conj.im) * 0.5,
            );
            let diff = Complex64::new(
                (zk.re - zmk_conj.re) * 0.5,
                (zk.im - zmk_conj.im) * 0.5,
            );
            let neg_i_diff = Complex64::new(diff.im, -diff.re);
            let w = self.unpack[k];
            let xk = Complex64::new(
                even.re + w.re * neg_i_diff.re - w.im * neg_i_diff.im,
                even.im + w.re * neg_i_diff.im + w.im * neg_i_diff.re,
            );
            output[k] = xk;

            if m - k != k {
                let zk2 = zmk;
                let zmk2_conj = Complex64::new(zk.re, -zk.im);
                let even2 = Complex64::new(
                    (zk2.re + zmk2_conj.re) * 0.5,
                    (zk2.im + zmk2_conj.im) * 0.5,
                );
                let diff2 = Complex64::new(
                    (zk2.re - zmk2_conj.re) * 0.5,
                    (zk2.im - zmk2_conj.im) * 0.5,
                );
                let neg_i_diff2 = Complex64::new(diff2.im, -diff2.re);
                let w2 = self.unpack[m - k];
                output[m - k] = Complex64::new(
                    even2.re + w2.re * neg_i_diff2.re - w2.im * neg_i_diff2.im,
                    even2.im + w2.re * neg_i_diff2.im + w2.im * neg_i_diff2.re,
                );
            }
        }
    }
}

// ─── Standalone f64 power-of-two FFT (for Bluestein inner convolution) ────────

fn fft_pow2_scratch_f64(buf: &mut [Complex64]) {
    let m = buf.len();
    debug_assert!(m.is_power_of_two());
    if m <= 1 {
        return;
    }
    let log2m = m.trailing_zeros();
    for i in 0..m as u32 {
        let r = i.reverse_bits() >> (32 - log2m);
        if r > i {
            buf.swap(i as usize, r as usize);
        }
    }
    let mut step = 2usize;
    while step <= m {
        let half = step >> 1;
        let angle = -PI64 * 2.0 / step as f64;
        let w_step = Complex64::new(angle.cos(), angle.sin());
        let mut start = 0;
        while start < m {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw = Complex64::new(
                    w.re * b.re - w.im * b.im,
                    w.re * b.im + w.im * b.re,
                );
                buf[start + k] = Complex64::new(a.re + bw.re, a.im + bw.im);
                buf[start + k + half] = Complex64::new(a.re - bw.re, a.im - bw.im);
                w = Complex64::new(
                    w.re * w_step.re - w.im * w_step.im,
                    w.re * w_step.im + w.im * w_step.re,
                );
            }
            start += step;
        }
        step <<= 1;
    }
}

// ─── Public helpers for arbitrary-length real FFT (used by whole_fft.rs) ──────

/// Forward real-to-complex FFT for any N (f32).
/// Output length = N/2+1.  Handles both power-of-two (fast path) and other N.
pub fn fft_real_arbitrary_f32(input: &[f32]) -> Vec<Complex32> {
    let n = input.len();
    let half1 = n / 2 + 1;
    let plan = get_plan(n);
    let mut scratch = vec![Complex32::new(0.0, 0.0); n / 2];
    let mut output = vec![Complex32::new(0.0, 0.0); half1];
    plan.fft_real(input, &mut scratch, &mut output);
    output
}

/// Forward real-to-complex FFT for any N (f64).
/// Output length = N/2+1.  Handles both power-of-two and other N.
pub fn fft_real_arbitrary_f64(input: &[f64]) -> Vec<Complex64> {
    let n = input.len();
    let half1 = n / 2 + 1;

    if n.is_power_of_two() {
        let plan = get_plan_64(n);
        let mut scratch = vec![Complex64::new(0.0, 0.0); n / 2];
        let mut output = vec![Complex64::new(0.0, 0.0); half1];
        plan.fft_real_pow2(input, &mut scratch, &mut output);
        return output;
    }

    // Bluestein for arbitrary N (f64).
    let conv_len = (2 * n - 1).next_power_of_two();
    let mut chirp: Vec<Complex64> = (0..n)
        .map(|k| {
            let theta = -PI64 * (k * k % (2 * n)) as f64 / n as f64;
            Complex64::new(theta.cos(), theta.sin())
        })
        .collect();

    let mut y = vec![Complex64::new(0.0, 0.0); conv_len];
    for k in 0..n {
        y[k] = Complex64::new(input[k], 0.0) * chirp[k];
    }

    let mut h = vec![Complex64::new(0.0, 0.0); conv_len];
    for k in 0..n {
        let c = Complex64::new(chirp[k].re, -chirp[k].im);
        h[k] = c;
        if k > 0 {
            h[conv_len - k] = c;
        }
    }

    fft_pow2_scratch_f64(&mut y);
    fft_pow2_scratch_f64(&mut h);

    for i in 0..conv_len {
        let yr = y[i].re; let yi = y[i].im;
        let hr = h[i].re; let hi = h[i].im;
        y[i] = Complex64::new(yr * hr - yi * hi, yr * hi + yi * hr);
    }

    // IFFT via conj + FFT + conj + scale.
    for v in y.iter_mut() { *v = Complex64::new(v.re, -v.im); }
    fft_pow2_scratch_f64(&mut y);
    let scale = 1.0 / conv_len as f64;
    for v in y.iter_mut() { *v = Complex64::new(v.re * scale, -v.im * scale); }

    // Output X[k] = chirp[k] * y[k], keep only 0..N/2+1
    let mut output = vec![Complex64::new(0.0, 0.0); half1];
    for k in 0..half1 {
        let yk = y[k];
        let ck = chirp[k];
        output[k] = Complex64::new(
            ck.re * yk.re - ck.im * yk.im,
            ck.re * yk.im + ck.im * yk.re,
        );
    }
    output
}

// ─── Plan caches ──────────────────────────────────────────────────────────────

static PLAN_CACHE: Lazy<Mutex<HashMap<usize, Arc<BlitzFftPlan>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static PLAN_CACHE_64: Lazy<Mutex<HashMap<usize, Arc<BlitzFftPlan64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Return a shared plan for size `n` (f32), creating it on first use.
pub fn get_plan(n: usize) -> Arc<BlitzFftPlan> {
    let mut cache = PLAN_CACHE.lock().unwrap();
    if let Some(p) = cache.get(&n) {
        return Arc::clone(p);
    }
    let p = Arc::new(BlitzFftPlan::new(n));
    cache.insert(n, Arc::clone(&p));
    p
}

/// Return a shared plan for size `n` (f64, must be power-of-two).
pub fn get_plan_64(n: usize) -> Arc<BlitzFftPlan64> {
    let mut cache = PLAN_CACHE_64.lock().unwrap();
    if let Some(p) = cache.get(&n) {
        return Arc::clone(p);
    }
    let p = Arc::new(BlitzFftPlan64::new(n));
    cache.insert(n, Arc::clone(&p));
    p
}
