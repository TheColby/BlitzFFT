use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use num_complex::Complex32;
use rayon::prelude::*;
use realfft::{RealFftPlanner, RealToComplex};

use super::{FftBackend, FftFrame};

type RealPlan = Arc<dyn RealToComplex<f32>>;

static PLAN_CACHE: once_cell::sync::Lazy<Mutex<HashMap<usize, RealPlan>>> =
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

struct WorkBuffers {
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

thread_local! {
    static WORK: RefCell<Option<WorkBuffers>> = const { RefCell::new(None) };
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

pub struct CpuFftBackend;

impl CpuFftBackend {
    pub fn new() -> Self {
        Self
    }
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
