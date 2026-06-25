// src/backends/cpu.rs
//
// CPU framed-FFT backend — uses the native BlitzFFT engine (no external FFT libs).
//
// f32 path  : BlitzFftPlan (precomputed twiddles + SIMD), Rayon parallel batches.
// f64 path  : BlitzFftPlan64 (precomputed twiddles, scalar f64).
// quad path : hand-rolled radix-2 over Quad (true binary128 only when the
//             optional `binary128` feature is enabled).

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
    input: Vec<f32>,
    input_len: usize,
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
                input: vec![0.0f32; plan.n],
                input_len: 0,
                scratch: vec![Complex32::new(0.0, 0.0); plan.n / 2],
                output: vec![Complex32::new(0.0, 0.0); plan.n / 2 + 1],
            });
        }
        f(guard.as_mut().unwrap())
    })
}

struct PairWorkBuf32 {
    fft_size: usize,
    complex: Vec<Complex32>,
}

thread_local! {
    static PAIR_WORK32: RefCell<Option<PairWorkBuf32>> = const { RefCell::new(None) };
}

fn with_pair_work32<R>(fft_size: usize, f: impl FnOnce(&mut PairWorkBuf32) -> R) -> R {
    PAIR_WORK32.with(|cell| {
        let mut guard = cell.borrow_mut();
        let needs_reset = guard.as_ref().map_or(true, |w| w.fft_size != fft_size);
        if needs_reset {
            *guard = Some(PairWorkBuf32 {
                fft_size,
                complex: vec![Complex32::new(0.0, 0.0); fft_size],
            });
        }
        f(guard.as_mut().unwrap())
    })
}

// ─── f64 thread-local work buffers ───────────────────────────────────────────

struct WorkBuf64 {
    fft_size: usize,
    input: Vec<f64>,
    input_len: usize,
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
                input: vec![0.0f64; plan.n],
                input_len: 0,
                scratch: vec![Complex64::new(0.0, 0.0); plan.n / 2],
                output: vec![Complex64::new(0.0, 0.0); plan.n / 2 + 1],
            });
        }
        f(guard.as_mut().unwrap())
    })
}

// ─── Public compute functions ─────────────────────────────────────────────────

fn compute_single_f32_frame(
    plan: &Arc<BlitzFftPlan>,
    frame_index: usize,
    frame: &[f32],
    fft_size: usize,
) -> Result<FftFrame> {
    with_work32(plan, |work| {
        let len = frame.len().min(fft_size);
        let WorkBuf32 {
            input,
            input_len,
            scratch,
            output,
            ..
        } = work;
        let input = if len == fft_size {
            &frame[..fft_size]
        } else {
            input[..len].copy_from_slice(&frame[..len]);
            if *input_len > len {
                input[len..*input_len].fill(0.0);
            }
            *input_len = len;
            &input[..]
        };

        plan.fft_real(input, scratch, output);

        let magnitude = output
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        Ok(FftFrame {
            frame_index,
            magnitude,
        })
    })
}

fn compute_paired_f32_frames(
    pair_plan: &Arc<BlitzFftPlan>,
    first_index: usize,
    first: &[f32],
    second: &[f32],
    fft_size: usize,
) -> Result<[FftFrame; 2]> {
    with_pair_work32(fft_size, |work| {
        for (index, slot) in work.complex.iter_mut().enumerate() {
            let re = first.get(index).copied().unwrap_or(0.0);
            let im = second.get(index).copied().unwrap_or(0.0);
            *slot = Complex32::new(re, im);
        }

        pair_plan.fft_pow2_inplace(&mut work.complex);

        let half1 = fft_size / 2 + 1;
        let mut first_magnitude = Vec::with_capacity(half1);
        let mut second_magnitude = Vec::with_capacity(half1);

        for bin in 0..half1 {
            let mirror = if bin == 0 { 0 } else { fft_size - bin };
            let forward = work.complex[bin];
            let mirrored = work.complex[mirror].conj();

            let first_bin = Complex32::new(
                (forward.re + mirrored.re) * 0.5,
                (forward.im + mirrored.im) * 0.5,
            );
            let diff = Complex32::new(forward.re - mirrored.re, forward.im - mirrored.im);
            let second_bin = Complex32::new(diff.im * 0.5, -diff.re * 0.5);

            first_magnitude
                .push((first_bin.re * first_bin.re + first_bin.im * first_bin.im).sqrt());
            second_magnitude
                .push((second_bin.re * second_bin.re + second_bin.im * second_bin.im).sqrt());
        }

        Ok([
            FftFrame {
                frame_index: first_index,
                magnitude: first_magnitude,
            },
            FftFrame {
                frame_index: first_index + 1,
                magnitude: second_magnitude,
            },
        ])
    })
}

/// Compute a batch of f32 frames in parallel using the native BlitzFFT engine.
pub fn compute_batch_f32_native(frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
    let plan = get_plan(fft_size);
    let pair_plan = get_plan(fft_size * 2);

    let grouped = frames
        .par_chunks(2)
        .enumerate()
        .map(|(pair_index, chunk)| {
            let first_index = pair_index * 2;
            if chunk.len() == 2 {
                let [first, second] = compute_paired_f32_frames(
                    &pair_plan,
                    first_index,
                    chunk[0],
                    chunk[1],
                    fft_size,
                )?;
                Ok(vec![first, second])
            } else {
                Ok(vec![compute_single_f32_frame(
                    &plan,
                    first_index,
                    chunk[0],
                    fft_size,
                )?])
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(grouped.into_iter().flatten().collect())
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
                let WorkBuf64 {
                    input,
                    input_len,
                    scratch,
                    output,
                    ..
                } = work;
                let input = if len == fft_size {
                    &frame[..fft_size]
                } else {
                    input[..len].copy_from_slice(&frame[..len]);
                    if *input_len > len {
                        input[len..*input_len].fill(0.0);
                    }
                    *input_len = len;
                    &input[..]
                };

                plan.fft_real_pow2(input, scratch, output);

                let magnitude = output
                    .iter()
                    .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
                    .collect();

                Ok(FftFrame {
                    frame_index: i,
                    magnitude,
                })
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
            Ok(FftFrame {
                frame_index: i,
                magnitude,
            })
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
        "BlitzFFT native (CPU — paired real-frame batching + SIMD, Rayon parallel)"
    }

    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
        compute_batch_f32_native(frames, fft_size)
    }
}

#[cfg(test)]
mod tests {
    use super::compute_batch_f32_native;
    use crate::blitz_fft::get_plan;
    use num_complex::Complex32;

    fn reference_magnitude(frame: &[f32]) -> Vec<f32> {
        let plan = get_plan(frame.len());
        let mut scratch = vec![Complex32::new(0.0, 0.0); frame.len() / 2];
        let mut output = vec![Complex32::new(0.0, 0.0); frame.len() / 2 + 1];
        plan.fft_real(frame, &mut scratch, &mut output);
        output
            .iter()
            .map(|bin| (bin.re * bin.re + bin.im * bin.im).sqrt())
            .collect()
    }

    #[test]
    fn paired_real_frame_batch_matches_single_frame_reference() {
        let frames = [
            vec![0.25, -1.0, 0.5, 0.75, -0.125, 0.0, 1.5, -0.25],
            vec![1.0, 0.5, -0.25, 0.125, -0.75, 0.3, 0.2, -0.4],
            vec![0.0, 0.2, 0.4, -0.6, 0.8, -1.0, 0.1, -0.3],
        ];
        let refs: Vec<&[f32]> = frames.iter().map(Vec::as_slice).collect();

        let actual = compute_batch_f32_native(&refs, 8).expect("paired batch should compute");

        assert_eq!(actual.len(), frames.len());
        for (frame_index, frame) in frames.iter().enumerate() {
            assert_eq!(actual[frame_index].frame_index, frame_index);
            let expected = reference_magnitude(frame);
            assert_eq!(actual[frame_index].magnitude.len(), expected.len());
            for (bin, (&actual, &expected)) in actual[frame_index]
                .magnitude
                .iter()
                .zip(expected.iter())
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "frame {frame_index} bin {bin}: {actual} != {expected}"
                );
            }
        }
    }
}
