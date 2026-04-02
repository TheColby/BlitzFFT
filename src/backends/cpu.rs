use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use realfft::{RealFftPlanner, RealToComplex};

use super::{FftBackend, FftFrame};
use crate::quad::Quad;

type RealPlan = Arc<dyn RealToComplex<f32>>;
type RealPlan64 = Arc<dyn RealToComplex<f64>>;

static PLAN_CACHE: once_cell::sync::Lazy<Mutex<HashMap<usize, RealPlan>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));
static PLAN_CACHE_64: once_cell::sync::Lazy<Mutex<HashMap<usize, RealPlan64>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

fn get_plan(fft_size: usize) -> RealPlan {
    let mut cache = PLAN_CACHE.lock().unwrap();
    if let Some(plan) = cache.get(&fft_size) {
        return Arc::clone(plan);
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(fft_size);
    cache.insert(fft_size, Arc::clone(&plan));
    plan
}

fn get_plan_f64(fft_size: usize) -> RealPlan64 {
    let mut cache = PLAN_CACHE_64.lock().unwrap();
    if let Some(plan) = cache.get(&fft_size) {
        return Arc::clone(plan);
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(fft_size);
    cache.insert(fft_size, Arc::clone(&plan));
    plan
}

struct WorkBuffers {
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

struct WorkBuffers64 {
    input: Vec<f64>,
    spectrum: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

thread_local! {
    static WORK: RefCell<Option<WorkBuffers>> = const { RefCell::new(None) };
    static WORK_64: RefCell<Option<WorkBuffers64>> = const { RefCell::new(None) };
}

fn with_work_buffers<R>(
    plan: &RealPlan,
    fft_size: usize,
    f: impl FnOnce(&mut WorkBuffers) -> R,
) -> R {
    WORK.with(|cell| {
        let mut guard = cell.borrow_mut();
        let needs_resize = guard.as_ref().map_or(true, |work| {
            work.input.len() != fft_size || work.spectrum.len() != plan.complex_len()
        });

        if needs_resize {
            *guard = Some(WorkBuffers {
                input: vec![0.0; fft_size],
                spectrum: plan.make_output_vec(),
                scratch: plan.make_scratch_vec(),
            });
        }

        f(guard.as_mut().unwrap())
    })
}

fn with_work_buffers_f64<R>(
    plan: &RealPlan64,
    fft_size: usize,
    f: impl FnOnce(&mut WorkBuffers64) -> R,
) -> R {
    WORK_64.with(|cell| {
        let mut guard = cell.borrow_mut();
        let needs_resize = guard.as_ref().map_or(true, |work| {
            work.input.len() != fft_size || work.spectrum.len() != plan.complex_len()
        });

        if needs_resize {
            *guard = Some(WorkBuffers64 {
                input: vec![0.0; fft_size],
                spectrum: plan.make_output_vec(),
                scratch: plan.make_scratch_vec(),
            });
        }

        f(guard.as_mut().unwrap())
    })
}

pub struct CpuFftBackend;

impl CpuFftBackend {
    pub fn new() -> Self {
        Self
    }
}

pub fn compute_batch_f64(frames: &[Vec<f64>], fft_size: usize) -> Result<Vec<FftFrame>> {
    let plan = get_plan_f64(fft_size);

    frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            with_work_buffers_f64(&plan, fft_size, |work| {
                let len = frame.len().min(fft_size);
                work.input[..len].copy_from_slice(&frame[..len]);
                work.input[len..].fill(0.0);

                plan.process_with_scratch(&mut work.input, &mut work.spectrum, &mut work.scratch)?;

                let magnitude = work
                    .spectrum
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

fn quad_from_f64(value: f64) -> Quad {
    Quad::from(value)
}

fn quad_to_f64(value: Quad) -> f64 {
    value.to_f64()
}

fn bit_reverse(index: usize, bits: u32) -> usize {
    index.reverse_bits() >> (usize::BITS - bits)
}

fn fft_real_qd(frame: &[Quad], fft_size: usize) -> Vec<Quad> {
    let bits = fft_size.trailing_zeros();
    let mut buffer = vec![ComplexQuad::new(Quad::ZERO, Quad::ZERO); fft_size];

    for (index, &sample) in frame.iter().take(fft_size).enumerate() {
        let reversed = bit_reverse(index, bits);
        buffer[reversed] = ComplexQuad::new(sample, Quad::ZERO);
    }

    let mut step = 2usize;
    while step <= fft_size {
        let half_step = step / 2;
        let angle = -(Quad::TWO_PI / quad_from_f64(step as f64));
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
            let magnitude = fft_real_qd(frame, fft_size)
                .into_iter()
                .map(|value| quad_to_f64(value) as f32)
                .collect();

            Ok(FftFrame {
                frame_index: i,
                magnitude,
            })
        })
        .collect()
}

impl FftBackend for CpuFftBackend {
    fn name(&self) -> &str {
        "RealFFT (CPU - plan-cached, Rayon parallel)"
    }

    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
        let plan = get_plan(fft_size);

        frames
            .par_iter()
            .enumerate()
            .map(|(i, frame)| {
                with_work_buffers(&plan, fft_size, |work| {
                    let len = frame.len().min(fft_size);
                    work.input[..len].copy_from_slice(&frame[..len]);
                    work.input[len..].fill(0.0);

                    plan.process_with_scratch(
                        &mut work.input,
                        &mut work.spectrum,
                        &mut work.scratch,
                    )?;

                    let magnitude = work
                        .spectrum
                        .iter()
                        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                        .collect();

                    Ok(FftFrame {
                        frame_index: i,
                        magnitude,
                    })
                })
            })
            .collect()
    }
}
