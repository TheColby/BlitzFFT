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

struct BlitzBluesteinPlan32 {
    conv_len: usize,
    chirp: Vec<Complex32>,
    kernel_fft: Vec<Complex32>,
    inner: Box<BlitzFftPlan>,
}

impl BlitzBluesteinPlan32 {
    fn new(n: usize) -> Self {
        let conv_len = (2 * n - 1).next_power_of_two();
        let inner = Box::new(BlitzFftPlan::new(2 * conv_len));
        let chirp: Vec<Complex32> = (0..n)
            .map(|k| {
                let theta = -PI32 * (k * k % (2 * n)) as f32 / n as f32;
                Complex32::new(theta.cos(), theta.sin())
            })
            .collect();

        let mut kernel_fft = vec![Complex32::new(0.0, 0.0); conv_len];
        for k in 0..n {
            let c = Complex32::new(chirp[k].re, -chirp[k].im);
            kernel_fft[k] = c;
            if k > 0 {
                kernel_fft[conv_len - k] = c;
            }
        }
        inner.fft_pow2_inplace(&mut kernel_fft);

        Self {
            conv_len,
            chirp,
            kernel_fft,
            inner,
        }
    }
}

struct BlitzBluesteinPlan64 {
    conv_len: usize,
    chirp: Vec<Complex64>,
    kernel_fft: Vec<Complex64>,
    inner: Box<BlitzFftPlan64>,
}

impl BlitzBluesteinPlan64 {
    fn new(n: usize) -> Self {
        let conv_len = (2 * n - 1).next_power_of_two();
        let inner = Box::new(BlitzFftPlan64::new(2 * conv_len));
        let chirp: Vec<Complex64> = (0..n)
            .map(|k| {
                let theta = -PI64 * (k * k % (2 * n)) as f64 / n as f64;
                Complex64::new(theta.cos(), theta.sin())
            })
            .collect();

        let mut kernel_fft = vec![Complex64::new(0.0, 0.0); conv_len];
        for k in 0..n {
            let c = Complex64::new(chirp[k].re, -chirp[k].im);
            kernel_fft[k] = c;
            if k > 0 {
                kernel_fft[conv_len - k] = c;
            }
        }
        inner.fft_pow2_inplace(&mut kernel_fft);

        Self {
            conv_len,
            chirp,
            kernel_fft,
            inner,
        }
    }
}

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
    /// Bluestein setup reused for arbitrary-length transforms.
    bluestein: Option<BlitzBluesteinPlan32>,
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

        let bluestein = if pow2 {
            None
        } else {
            Some(BlitzBluesteinPlan32::new(n))
        };

        Self {
            n,
            m,
            bit_rev,
            twiddles,
            unpack,
            pow2,
            bluestein,
        }
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
                if step >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
                {
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

            let even = Complex32::new((zk.re + zmk_conj.re) * 0.5, (zk.im + zmk_conj.im) * 0.5);
            let diff = Complex32::new((zk.re - zmk_conj.re) * 0.5, (zk.im - zmk_conj.im) * 0.5);
            // diff / (i) = (diff.im, -diff.re), so -(i·diff) / 2 = (diff.im, -diff.re) * 0.5
            // We already have the 0.5 factor in `diff`.
            let neg_i_diff = Complex32::new(diff.im, -diff.re);

            let w = self.unpack[k];
            let twist = Complex32::new(
                w.re * neg_i_diff.re - w.im * neg_i_diff.im,
                w.re * neg_i_diff.im + w.im * neg_i_diff.re,
            );
            output[k] = Complex32::new(even.re + twist.re, even.im + twist.im);

            // X[M-k] (only if distinct from X[k])
            if m - k != k {
                output[m - k] = Complex32::new(even.re - twist.re, twist.im - even.im);
            }
        }
    }

    // ── Bluestein chirp-z for arbitrary-length complex FFT ──────────────────

    /// Complex FFT of arbitrary length via Bluestein's algorithm.
    ///
    /// Computes `buf = DFT(buf)` for any N (not just power-of-two).
    /// Uses an internal power-of-two FFT of size ≥ 2N-1.
    fn fft_bluestein_real(&self, input: &[f32], output: &mut [Complex32], work: &mut [Complex32]) {
        let bluestein = self
            .bluestein
            .as_ref()
            .expect("Bluestein plan is only present for arbitrary lengths");
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n / 2 + 1);
        debug_assert_eq!(work.len(), bluestein.conv_len);

        work.fill(Complex32::new(0.0, 0.0));
        for (slot, (&sample, &chirp)) in work
            .iter_mut()
            .zip(input.iter().zip(bluestein.chirp.iter()))
        {
            *slot = Complex32::new(sample * chirp.re, sample * chirp.im);
        }

        bluestein.inner.fft_pow2_inplace(work);

        for (slot, kernel) in work.iter_mut().zip(bluestein.kernel_fft.iter()) {
            let wr = slot.re;
            let wi = slot.im;
            let kr = kernel.re;
            let ki = kernel.im;
            *slot = Complex32::new(wr * kr - wi * ki, wr * ki + wi * kr);
        }

        for value in work.iter_mut() {
            *value = Complex32::new(value.re, -value.im);
        }
        bluestein.inner.fft_pow2_inplace(work);
        let scale = 1.0 / bluestein.conv_len as f32;
        for value in work.iter_mut() {
            *value = Complex32::new(value.re * scale, -value.im * scale);
        }

        for (bin, out) in output.iter_mut().enumerate() {
            let wk = work[bin];
            let ck = bluestein.chirp[bin];
            *out = Complex32::new(ck.re * wk.re - ck.im * wk.im, ck.re * wk.im + ck.im * wk.re);
        }
    }

    // ── Public real-to-complex FFT (arbitrary N) ────────────────────────────

    /// Real-to-complex forward FFT for arbitrary N (power-of-two or not).
    ///
    /// `input`  — N real samples
    /// `output` — N/2+1 complex bins
    /// `scratch` — work buffer of length N/2 (used only when N is a power of two)
    pub fn fft_real(&self, input: &[f32], scratch: &mut [Complex32], output: &mut [Complex32]) {
        if self.pow2 {
            self.fft_real_pow2(input, scratch, output);
        } else {
            let _ = scratch;
            let mut work = vec![Complex32::new(0.0, 0.0); self.bluestein_work_len()];
            self.fft_real_with_work(input, output, &mut work);
        }
    }

    pub fn bluestein_work_len(&self) -> usize {
        self.bluestein.as_ref().map_or(0, |plan| plan.conv_len)
    }

    pub fn fft_real_with_work(
        &self,
        input: &[f32],
        output: &mut [Complex32],
        work: &mut [Complex32],
    ) {
        if self.pow2 {
            debug_assert_eq!(work.len(), self.m);
            self.fft_real_pow2(input, work, output);
        } else {
            self.fft_bluestein_real(input, output, work);
        }
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

fn fft_real_bluestein_f64_with_plan(
    input: &[f64],
    output: &mut [Complex64],
    work: &mut [Complex64],
    plan: &BlitzBluesteinPlan64,
) {
    debug_assert_eq!(input.len(), plan.chirp.len());
    debug_assert_eq!(output.len(), input.len() / 2 + 1);
    debug_assert_eq!(work.len(), plan.conv_len);

    work.fill(Complex64::new(0.0, 0.0));
    for (slot, (&sample, &chirp)) in work.iter_mut().zip(input.iter().zip(plan.chirp.iter())) {
        *slot = Complex64::new(sample * chirp.re, sample * chirp.im);
    }

    plan.inner.fft_pow2_inplace(work);

    for (slot, kernel) in work.iter_mut().zip(plan.kernel_fft.iter()) {
        let wr = slot.re;
        let wi = slot.im;
        let kr = kernel.re;
        let ki = kernel.im;
        *slot = Complex64::new(wr * kr - wi * ki, wr * ki + wi * kr);
    }

    for value in work.iter_mut() {
        *value = Complex64::new(value.re, -value.im);
    }
    plan.inner.fft_pow2_inplace(work);
    let scale = 1.0 / plan.conv_len as f64;
    for value in work.iter_mut() {
        *value = Complex64::new(value.re * scale, -value.im * scale);
    }

    for (bin, out) in output.iter_mut().enumerate() {
        let wk = work[bin];
        let ck = plan.chirp[bin];
        *out = Complex64::new(ck.re * wk.re - ck.im * wk.im, ck.re * wk.im + ck.im * wk.re);
    }
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

        Self {
            n,
            m,
            bit_rev,
            twiddles,
            unpack,
        }
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

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn butterfly_stage_neon(
        twiddles: &[Complex64],
        buf: &mut [Complex64],
        step: usize,
        m: usize,
    ) {
        use std::arch::aarch64::*;

        let half = step >> 1;
        let stride = m / step;
        let sign_mask: [f64; 2] = [-0.0, 0.0];
        let vsign = vld1q_f64(sign_mask.as_ptr());

        let mut start = 0usize;
        while start < m {
            let mut k = 0usize;
            while k < half {
                let w = twiddles[k * stride];
                let ai = start + k;
                let bi = start + k + half;

                let a_vec = vld1q_f64(buf.as_ptr().add(ai) as *const f64);
                let b_vec = vld1q_f64(buf.as_ptr().add(bi) as *const f64);
                let w_re = vdupq_n_f64(w.re);
                let w_im = vdupq_n_f64(w.im);
                let b_swap = vextq_f64(b_vec, b_vec, 1);
                let t1 = vmulq_f64(w_re, b_vec);
                let t2 = vmulq_f64(w_im, b_swap);
                let t2_adj = vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(t2),
                    vreinterpretq_u64_f64(vsign),
                ));
                let twisted = vaddq_f64(t1, t2_adj);
                let out_a = vaddq_f64(a_vec, twisted);
                let out_b = vsubq_f64(a_vec, twisted);

                vst1q_f64(buf.as_mut_ptr().add(ai) as *mut f64, out_a);
                vst1q_f64(buf.as_mut_ptr().add(bi) as *mut f64, out_b);

                k += 1;
            }

            start += step;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn butterfly_stage_avx2(
        twiddles: &[Complex64],
        buf: &mut [Complex64],
        step: usize,
        m: usize,
    ) {
        use std::arch::x86_64::*;

        let half = step >> 1;
        let stride = m / step;
        let sign_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);

        let mut start = 0usize;
        while start < m {
            let mut k = 0usize;

            while k + 1 < half {
                let w0 = twiddles[k * stride];
                let w1 = twiddles[(k + 1) * stride];
                let ai = start + k;
                let bi = start + k + half;

                let a_vec = _mm256_loadu_pd(buf.as_ptr().add(ai) as *const f64);
                let b_vec = _mm256_loadu_pd(buf.as_ptr().add(bi) as *const f64);
                let w_re = _mm256_setr_pd(w0.re, w0.re, w1.re, w1.re);
                let w_im = _mm256_setr_pd(w0.im, w0.im, w1.im, w1.im);
                let b_swap = _mm256_permute_pd(b_vec, 0b0101);
                let t1 = _mm256_mul_pd(w_re, b_vec);
                let t2 = _mm256_mul_pd(w_im, b_swap);
                let t2_adj = _mm256_xor_pd(t2, sign_mask);
                let twisted = _mm256_add_pd(t1, t2_adj);
                let out_a = _mm256_add_pd(a_vec, twisted);
                let out_b = _mm256_sub_pd(a_vec, twisted);

                _mm256_storeu_pd(buf.as_mut_ptr().add(ai) as *mut f64, out_a);
                _mm256_storeu_pd(buf.as_mut_ptr().add(bi) as *mut f64, out_b);

                k += 2;
            }

            while k < half {
                let w = twiddles[k * stride];
                let a = buf[start + k];
                let b = buf[start + k + half];
                let bw_re = w.re * b.re - w.im * b.im;
                let bw_im = w.re * b.im + w.im * b.re;
                buf[start + k] = Complex64::new(a.re + bw_re, a.im + bw_im);
                buf[start + k + half] = Complex64::new(a.re - bw_re, a.im - bw_im);
                k += 1;
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
            #[cfg(target_arch = "aarch64")]
            {
                if step >= 4 {
                    unsafe { Self::butterfly_stage_neon(twiddles, buf, step, m) };
                } else {
                    Self::butterfly_stage_scalar(twiddles, buf, step, m);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if step >= 4 && is_x86_feature_detected!("avx2") {
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
            let even = Complex64::new((zk.re + zmk_conj.re) * 0.5, (zk.im + zmk_conj.im) * 0.5);
            let diff = Complex64::new((zk.re - zmk_conj.re) * 0.5, (zk.im - zmk_conj.im) * 0.5);
            let neg_i_diff = Complex64::new(diff.im, -diff.re);
            let w = self.unpack[k];
            let twist = Complex64::new(
                w.re * neg_i_diff.re - w.im * neg_i_diff.im,
                w.re * neg_i_diff.im + w.im * neg_i_diff.re,
            );
            output[k] = Complex64::new(even.re + twist.re, even.im + twist.im);

            if m - k != k {
                output[m - k] = Complex64::new(even.re - twist.re, twist.im - even.im);
            }
        }
    }
}

// ─── Public helpers for arbitrary-length real FFT (used by whole_fft.rs) ──────

/// Forward real-to-complex FFT for any N (f32).
/// Output length = N/2+1.  Handles both power-of-two (fast path) and other N.
#[allow(dead_code)]
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
#[allow(dead_code)]
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

    let mut output = vec![Complex64::new(0.0, 0.0); half1];
    let mut work = vec![Complex64::new(0.0, 0.0); bluestein_work_len_64(n)];
    fft_real_arbitrary_f64_with_work(input, &mut output, &mut work);
    output
}

// ─── Plan caches ──────────────────────────────────────────────────────────────

static PLAN_CACHE: Lazy<Mutex<HashMap<usize, Arc<BlitzFftPlan>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static PLAN_CACHE_64: Lazy<Mutex<HashMap<usize, Arc<BlitzFftPlan64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static BLUESTEIN_PLAN_CACHE_64: Lazy<Mutex<HashMap<usize, Arc<BlitzBluesteinPlan64>>>> =
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

fn get_bluestein_plan_64(n: usize) -> Arc<BlitzBluesteinPlan64> {
    let mut cache = BLUESTEIN_PLAN_CACHE_64.lock().unwrap();
    if let Some(p) = cache.get(&n) {
        return Arc::clone(p);
    }
    let p = Arc::new(BlitzBluesteinPlan64::new(n));
    cache.insert(n, Arc::clone(&p));
    p
}

pub fn bluestein_work_len_64(n: usize) -> usize {
    get_bluestein_plan_64(n).conv_len
}

pub fn fft_real_arbitrary_f64_with_work(
    input: &[f64],
    output: &mut [Complex64],
    work: &mut [Complex64],
) {
    let n = input.len();
    if n.is_power_of_two() {
        let plan = get_plan_64(n);
        debug_assert_eq!(work.len(), n / 2);
        plan.fft_real_pow2(input, work, output);
        return;
    }

    let bluestein = get_bluestein_plan_64(n);
    fft_real_bluestein_f64_with_plan(input, output, work, &bluestein);
}

#[cfg(test)]
mod tests {
    use super::{
        bluestein_work_len_64, fft_real_arbitrary_f32, fft_real_arbitrary_f64,
        fft_real_arbitrary_f64_with_work, get_plan, get_plan_64,
    };
    use num_complex::{Complex32, Complex64};
    use std::sync::Arc;

    fn naive_rfft_f32(input: &[f32]) -> Vec<Complex32> {
        let n = input.len();
        (0..=n / 2)
            .map(|k| {
                let mut sum = Complex32::new(0.0, 0.0);
                for (n_idx, &sample) in input.iter().enumerate() {
                    let theta =
                        -2.0 * std::f32::consts::PI * (k as f32) * (n_idx as f32) / (n as f32);
                    let twiddle = Complex32::new(theta.cos(), theta.sin());
                    sum += twiddle * sample;
                }
                sum
            })
            .collect()
    }

    fn naive_rfft_f64(input: &[f64]) -> Vec<Complex64> {
        let n = input.len();
        (0..=n / 2)
            .map(|k| {
                let mut sum = Complex64::new(0.0, 0.0);
                for (n_idx, &sample) in input.iter().enumerate() {
                    let theta =
                        -2.0 * std::f64::consts::PI * (k as f64) * (n_idx as f64) / (n as f64);
                    let twiddle = Complex64::new(theta.cos(), theta.sin());
                    sum += twiddle * sample;
                }
                sum
            })
            .collect()
    }

    fn assert_bins_close_f32(actual: &[Complex32], expected: &[Complex32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual.re - expected.re).abs() <= tolerance,
                "real bin {index} differed: {} vs {}",
                actual.re,
                expected.re
            );
            assert!(
                (actual.im - expected.im).abs() <= tolerance,
                "imag bin {index} differed: {} vs {}",
                actual.im,
                expected.im
            );
        }
    }

    fn assert_bins_close_f64(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual.re - expected.re).abs() <= tolerance,
                "real bin {index} differed: {} vs {}",
                actual.re,
                expected.re
            );
            assert!(
                (actual.im - expected.im).abs() <= tolerance,
                "imag bin {index} differed: {} vs {}",
                actual.im,
                expected.im
            );
        }
    }

    #[test]
    fn pow2_real_fft_matches_naive_reference() {
        let input = [0.25, -1.0, 0.5, 0.75, -0.125, 0.0, 1.5, -0.25];
        let plan = get_plan(input.len());
        let mut scratch = vec![Complex32::new(0.0, 0.0); input.len() / 2];
        let mut output = vec![Complex32::new(0.0, 0.0); input.len() / 2 + 1];

        plan.fft_real(&input, &mut scratch, &mut output);

        let expected = naive_rfft_f32(&input);
        assert_bins_close_f32(&output, &expected, 1e-4);
    }

    #[test]
    fn arbitrary_length_real_fft_matches_naive_reference() {
        let input = [0.5, -0.25, 1.0, 0.125, -0.75, 0.3, 0.2, -0.4, 0.9];
        let plan = get_plan(input.len());
        let mut scratch = vec![Complex32::new(0.0, 0.0); input.len() / 2];
        let mut output = vec![Complex32::new(0.0, 0.0); input.len() / 2 + 1];

        plan.fft_real(&input, &mut scratch, &mut output);

        let expected = naive_rfft_f32(&input);
        assert_bins_close_f32(&output, &expected, 1e-3);
        assert_bins_close_f32(&fft_real_arbitrary_f32(&input), &expected, 1e-3);
    }

    #[test]
    fn bluestein_reused_work_matches_naive_for_prime_and_composite_lengths() {
        for n in [15, 31, 63, 127, 257] {
            let input: Vec<f32> = (0..n)
                .map(|i| ((i * i + 7 * i) as f32 * 0.137).sin())
                .collect();
            let plan = get_plan(n);
            let mut work = vec![Complex32::new(0.0, 0.0); plan.bluestein_work_len()];
            let mut output = vec![Complex32::new(0.0, 0.0); n / 2 + 1];
            for _ in 0..2 {
                plan.fft_real_with_work(&input, &mut output, &mut work);
                assert_bins_close_f32(&output, &naive_rfft_f32(&input), 2e-3);
            }
            let input64: Vec<f64> = input.iter().copied().map(f64::from).collect();
            let mut work64 = vec![Complex64::new(0.0, 0.0); bluestein_work_len_64(n)];
            let mut output64 = vec![Complex64::new(0.0, 0.0); n / 2 + 1];
            for _ in 0..2 {
                fft_real_arbitrary_f64_with_work(&input64, &mut output64, &mut work64);
                assert_bins_close_f64(&output64, &naive_rfft_f64(&input64), 1e-8);
            }
        }
    }

    #[test]
    fn f64_paths_match_naive_reference() {
        let input = [0.125, 0.5, -1.5, 0.25, 0.75, -0.125, 0.0, 1.25];
        let plan = get_plan_64(input.len());
        let mut scratch = vec![Complex64::new(0.0, 0.0); input.len() / 2];
        let mut output = vec![Complex64::new(0.0, 0.0); input.len() / 2 + 1];

        plan.fft_real_pow2(&input, &mut scratch, &mut output);

        let expected = naive_rfft_f64(&input);
        assert_bins_close_f64(&output, &expected, 1e-9);

        let arbitrary_input = [0.125, -0.75, 0.5, 1.0, -0.25, 0.875, -0.5];
        let expected_arbitrary = naive_rfft_f64(&arbitrary_input);
        let actual_arbitrary = fft_real_arbitrary_f64(&arbitrary_input);
        assert_bins_close_f64(&actual_arbitrary, &expected_arbitrary, 1e-9);
    }

    #[test]
    fn plan_cache_reuses_allocated_plans() {
        let first = get_plan(32);
        let second = get_plan(32);
        let first64 = get_plan_64(64);
        let second64 = get_plan_64(64);

        assert!(Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first64, &second64));
    }
}
