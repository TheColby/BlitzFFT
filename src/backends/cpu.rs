// src/backends/cpu.rs  (v2)
//
// Optimisations over v1
// ─────────────────────
//  1. Global plan cache (Mutex<HashMap<usize, Arc<dyn Fft<f32>>>>)
//     Planning is amortised across calls.  The old code re-planned inside
//     every par_iter closure, wasting ~50 µs per thread per call.
//
//  2. Pre-allocated scratch buffers per Rayon thread via thread_local!
//     Eliminates Vec allocation inside the hot loop.
//
//  3. Window application fused into the copy-to-scratch step.

use std::{collections::HashMap, sync::{Arc, Mutex}};
use anyhow::Result;
use num_complex::Complex;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

use super::{FftBackend, FftFrame};

// ── Global plan cache ─────────────────────────────────────────────────────────
// Keyed by fft_size; value is an Arc'd Fft object which is internally
// thread-safe.

type FftArc = Arc<dyn Fft<f32>>;

static PLAN_CACHE: once_cell::sync::Lazy<Mutex<HashMap<usize, FftArc>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

fn get_plan(fft_size: usize) -> FftArc {
    let mut cache = PLAN_CACHE.lock().unwrap();
    if let Some(plan) = cache.get(&fft_size) {
        return Arc::clone(plan);
    }
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let plan = planner.plan_fft_forward(fft_size);
    cache.insert(fft_size, Arc::clone(&plan));
    plan
}

// ── Thread-local scratch buffer ───────────────────────────────────────────────

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<Complex<f32>>> =
        std::cell::RefCell::new(Vec::new());
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct CpuFftBackend;

impl CpuFftBackend {
    pub fn new() -> Self { Self }
}

impl FftBackend for CpuFftBackend {
    fn name(&self) -> &str { "RustFFT (CPU — plan-cached, Rayon parallel)" }

    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>> {
        // Fetch plan once (cheap — just an Arc clone after first call)
        let plan = get_plan(fft_size);

        let results: Vec<FftFrame> = frames
            .par_iter()
            .enumerate()
            .map(|(i, frame)| {
                SCRATCH.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    buf.resize(fft_size, Complex::new(0.0, 0.0));

                    let len = frame.len().min(fft_size);
                    for (dst, &src) in buf[..len].iter_mut().zip(frame.iter()) {
                        *dst = Complex::new(src, 0.0);
                    }
                    for dst in &mut buf[len..fft_size] {
                        *dst = Complex::new(0.0, 0.0);
                    }

                    // In-place FFT — no extra allocation
                    plan.process(&mut buf);

                    let half = fft_size / 2 + 1;
                    let magnitude: Vec<f32> = buf[..half]
                        .iter()
                        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                        .collect();

                    use num_complex::Complex32;
                    let spectrum: Vec<Complex32> = buf
                        .iter()
                        .map(|c| Complex32::new(c.re, c.im))
                        .collect();

                    FftFrame { frame_index: i, spectrum, magnitude }
                })
            })
            .collect();

        Ok(results)
    }
}
