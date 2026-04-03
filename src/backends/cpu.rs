// src/backends/cpu.rs
//
// CPU framed-FFT backend — uses the native BlitzFFT engine (no external FFT libs).
//
// f32 path  : BlitzFftPlan (precomputed twiddles + SIMD), Rayon parallel batches.
// f64 path  : BlitzFftPlan64 (precomputed twiddles, scalar f64).
// quad path : hand-rolled radix-2 over Quad (binary-128 precision).

use std::cell::RefCell;
use std::sync::Arc;

use anyhow::Result;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

use super::{FftBackend, FftFrame};
use crate::blitz_fft::{get_plan, get_plan_64, BlitzFftPlan, BlitzFftPlan64};
use crate::quad::Quad;

// ─── f32 thread-local work buffers ───────────────────────────────────────────

struct WorkBuf32 {
    fft_size: usize,
    scratch: Vec<Complex32>, // length N/2
    output: Vec<Complex32>,  // length N/2+1
}

thread_local! {
    static WORK32: RefCell<Option<WorkBuf32>> = const { RefCell::new(None) };
}

fn with_work32<R>(plan: &Arc<BlitzFftPlan>, f: impl FnOnce(&mut WorkBuf32) -> R) -> R {
    WORK32.with(|cell| {
        let mut guard = cell.borrow_mut();
        let needs_reset = guard.as_ref().map_or(true, |w| w.fft_size != plan.n);
        if needs_reset {
            *guard = Some(WorkBuf32 {
                fft_size: plan.n,
                scratch: vec![Complex32::new(0.0, 0.0); plan.n / 2],
                output: vec![Complex32::new(0.0, 0.0); plan.n / 2 + 1],
            });
        }
        f(guard.as_mut().unwrap())
    })
}

// ─── f64 thread-local work buffers ───────────────────────────────────────────

struct WorkBuf64 {
    fft_size: usize,
    scratch: Vec<Complex64>,
    output: Vec<Complex64>,
}

thread_local! {
    static WORK64: RefCell<Option<WorkBuf64>> = const { RefCell::new(None) };
}

fn with_work64<R>(plan: &Arc<BlitzFftPlan64>, f: impl FnOnce(&mut WorkBuf64) -> R) -> R {
    WORK64.with(|cell| {
        let mut guard = cell.borrow_mut();
        let needs_reset = guard.as_ref().map_or(true, |w| w.fft_size != plan.n);
        if needs_reset {
            *guard = Some(WorkBuf64 {
                fft_size: plan.n,
                scratch: vec![Complex64::new(0.0, 0.0); plan.n / 2],
                output: vec![Complex64::new(0.0, 0.0); plan.n / 2 + 1],
            });
        }
        f(guard.as_mut().unwrap())
    })
}

// ─── Public compute functions ─────────────────────────────────────────────────

/// Compute a batch of f32 frames in parallel using the native BlitzFFT engine.
pub fn compute_batch_f32_native(
    frames: &[&[f32]],
    fft_size: usize,
) -> Result<Vec<FftFrame>> {
    let plan = get_plan(fft_size);

    frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            with_work32(&plan, |work| {
                // Zero-pad if frame is shorter than fft_size.
                let len = frame.len().min(fft_size);
                let padded: Vec<f32> = if len == fft_size {
                    frame.to_vec()
                } else {
                    let mut v = vec![0.0f32; fft_size];
                    v[..len].copy_from_slice(&frame[..len]);
                    v
                };

                plan.fft_real(&padded, &mut work.scratch, &mut work.output);

                let magnitude = work
                    .output
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                    .collect();

                Ok(FftFrame { frame_index: i, magnitude })
            })
        })
        .collect()
}

/// Compute a batch of f64 frames using the native BlitzFFT f64 engine.
pub fn compute_batch_f64(frames: &[Vec<f64>], fft_size: usize) -> Result<Vec<FftFrame>> {
    let plan = get_plan_64(fft_size);

    frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            with_work64(&plan, |work| {
                let len = frame.len().min(fft_size);
                let padded: Vec<f64> = if len == fft_size {
                    frame.clone()
                } else {
                    let mut v = vec![0.0f64; fft_size];
                    v[..len].copy_from_slice(&frame[..len]);
                    v
                };

                plan.fft_real_pow2(&padded, &mut work.scratch, &mut work.output);

                let magnitude = work
                    .output
                    .iter()
                    .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
                    .collect();

                Ok(FftFrame { frame_index: i, magnitude })
            })
        })
        .collect()
}

// ─── quad (binary-128) path ───────────────────────────────────────────────────
// Retained as-is: already a native implementation with no external FFT library.

#[derive(Clone, Copy)]
struct ComplexQuad {
    re: Quad,
    im: Quad,
}

impl ComplexQuad {
    const fn new(re: Quad, im: Quad) -> Self {
        Self { re, im }
    }
}

fn bit_reverse(index: usize, bits: u32) -> usize {
    index.reverse_bits() >> (usize::BITS - bits)
}

fn fft_real_qd_inner(frame: &[Quad], fft_size: usize) -> Vec<Quad> {
    let bits = fft_size.trailing_zeros();
    let mut buffer = vec![ComplexQuad::new(Quad::ZERO, Quad::ZERO); fft_size];

    for (index, &sample) in frame.iter().take(fft_size).enumerate() {
        let reversed = bit_reverse(index, bits);
        buffer[reversed] = ComplexQuad::new(sample, Quad::ZERO);
    }

    let mut step = 2usize;
    while step <= fft_size {
        let half_step = step / 2;
        let angle = -(Quad::TWO_PI / Quad::from(step as f64));
        let twiddle_step = ComplexQuad::new(angle.cos(), angle.sin());

        for start in (0..fft_size).step_by(step) {
            let mut twiddle = ComplexQuad::new(Quad::ONE, Quad::ZERO);
            for offset in 0..half_step {
                let even = buffer[start + offset];
                let odd = buffer[start + offset + half_step];
                let product = ComplexQuad::new(
                    twiddle.re * odd.re - twiddle.im * odd.im,
                    twiddle.re * odd.im + twiddle.im * odd.re,
                );
                buffer[start + offset] =
                    ComplexQuad::new(even.re + product.re, even.im + product.im);
                buffer[start + offset + half_step] =
                    ComplexQuad::new(even.re - product.re, even.im - product.im);
                twiddle = ComplexQuad::new(
                    twiddle.re * twiddle_step.re - twiddle.im * twiddle_step.im,
                    twiddle.re * twiddle_step.im + twiddle.im * twiddle_step.re,
                );
            }
        }
        step *= 2;
    }

    buffer[..(fft_size / 2 + 1)]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}

pub fn compute_batch_qd(frames: &[Vec<Quad>], fft_size: usize) -> Result<Vec<FftFrame>> {
    frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            let magnitude = fft_real_qd_inner(frame, fft_size)
                .into_iter()
                .map(|v| v.to_f64() as f32)
                .collect();
            Ok(FftFrame { frame_index: i, magnitude })
        })
        .collect()
}

// ─── FftBackend impl ──────────────────────────────────────────────────────────

pub struct CpuFftBackend;

impl CpuFftBackend {
    pub fn new() -> Self {
        Self
    }
}

impl FftBackend for CpuFftBackend {
    fn name(&self) -> &str {
        "BlitzFFT native (CPU — precomputed twiddles + SIMD, Rayon parallel)"
    }

    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
        compute_batch_f32_native(frames, fft_size)
    }
}
