// src/backends/mod.rs
use anyhow::Result;
use num_complex::Complex32;

/// One FFT hop: index, complex spectrum, magnitude spectrum.
#[derive(Debug)]
pub struct FftFrame {
    pub frame_index : usize,
    pub spectrum    : Vec<Complex32>,
    pub magnitude   : Vec<f32>,     // only positive-frequency bins (N/2+1)
}

/// Common interface every backend must satisfy.
pub trait FftBackend: Send + Sync {
    /// Human-readable name shown in CLI output.
    fn name(&self) -> &str;

    /// Compute FFTs on a batch of real frames.
    ///
    /// * `frames`   — slice of frames, each of exactly `fft_size` f32 samples
    /// * `fft_size` — must be a power of two
    ///
    /// Returns one `FftFrame` per input frame, in order.
    fn compute_batch(&self, frames: &[&[f32]], fft_size: usize) -> Result<Vec<FftFrame>>;
}

// ── Sub-modules (conditionally compiled) ────────────────────────────────────
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

// ── Backend selection ────────────────────────────────────────────────────────
use std::sync::Arc;

/// Probe available hardware and return the fastest backend.
/// Priority: CUDA > Metal > CPU.
pub fn select_backend(force: Option<&str>) -> Arc<dyn FftBackend> {
    if let Some(name) = force {
        return create_named(name);
    }

    #[cfg(feature = "cuda")]
    if let Some(b) = cuda::CudaFftBackend::try_init() {
        return Arc::new(b);
    }

    #[cfg(feature = "metal")]
    if let Some(b) = metal::MetalFftBackend::try_init() {
        return Arc::new(b);
    }

    Arc::new(cpu::CpuFftBackend::new())
}

fn create_named(name: &str) -> Arc<dyn FftBackend> {
    match name.to_lowercase().as_str() {
        #[cfg(feature = "cuda")]
        "cuda" => {
            Arc::new(
                cuda::CudaFftBackend::try_init()
                    .expect("CUDA backend requested but initialisation failed"),
            )
        }
        #[cfg(feature = "metal")]
        "metal" => {
            Arc::new(
                metal::MetalFftBackend::try_init()
                    .expect("Metal backend requested but no Metal device found"),
            )
        }
        "cpu" => Arc::new(cpu::CpuFftBackend::new()),
        other => {
            eprintln!("Unknown backend '{}', falling back to CPU", other);
            Arc::new(cpu::CpuFftBackend::new())
        }
    }
}
