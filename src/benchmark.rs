// src/benchmark.rs
//
// Timing harness.
// Runs both the selected GPU backend and the CPU baseline, prints a
// side-by-side comparison table, and reports the speedup ratio.

use anyhow::Result;
use colored::Colorize;
use std::{sync::Arc, time::Instant};

use crate::backends::{cpu::CpuFftBackend, FftBackend, FftFrame};

pub struct BenchResult {
    pub backend_name: String,
    pub frames: usize,
    pub fft_size: usize,
    pub elapsed_ms: f64,
    pub throughput_mfft: f64, // millions of FFTs / second
}

/// Run `backend` and `cpu` on the same data; return (gpu_result, cpu_result).
pub fn run(
    backend: &Arc<dyn FftBackend>,
    frames: &[Vec<f32>],
    fft_size: usize,
    repeats: usize,
) -> Result<(BenchResult, BenchResult)> {
    let frame_refs: Vec<&[f32]> = frames.iter().map(|v| v.as_slice()).collect();
    let n = frames.len();

    // ── Warm-up ────────────────────────────────────────────────────────────
    let _ = backend.compute_batch(&frame_refs, fft_size)?;

    // ── GPU / selected backend ─────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..repeats {
        let _: Vec<FftFrame> = backend.compute_batch(&frame_refs, fft_size)?;
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / repeats as f64;

    // ── CPU baseline ───────────────────────────────────────────────────────
    let cpu = CpuFftBackend::new();
    let _ = cpu.compute_batch(&frame_refs, fft_size)?; // warm-up

    let t1 = Instant::now();
    for _ in 0..repeats {
        let _: Vec<FftFrame> = cpu.compute_batch(&frame_refs, fft_size)?;
    }
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / repeats as f64;

    let mfft = |ms: f64| n as f64 / (ms / 1000.0) / 1_000_000.0;

    Ok((
        BenchResult {
            backend_name: backend.name().to_owned(),
            frames: n,
            fft_size,
            elapsed_ms: gpu_ms,
            throughput_mfft: mfft(gpu_ms),
        },
        BenchResult {
            backend_name: cpu.name().to_owned(),
            frames: n,
            fft_size,
            elapsed_ms: cpu_ms,
            throughput_mfft: mfft(cpu_ms),
        },
    ))
}

pub fn print_table(gpu: &BenchResult, cpu: &BenchResult) {
    let speedup = cpu.elapsed_ms / gpu.elapsed_ms;
    let win = if speedup >= 1.0 { gpu } else { cpu };
    let lose = if speedup >= 1.0 { cpu } else { gpu };

    println!();
    println!(
        "{}",
        "═══════════════════════════════════════════════════════".cyan()
    );
    println!("{}", "  audiofft  —  Benchmark Results".cyan().bold());
    println!(
        "{}",
        "═══════════════════════════════════════════════════════".cyan()
    );
    println!("  FFT size  : {} samples", gpu.fft_size);
    println!("  Batch     : {} frames", gpu.frames);
    println!();
    println!(
        "  {:<28} {:>10}  {:>14}",
        "Backend", "Time (ms)", "Throughput"
    );
    println!("  {}", "─".repeat(56));
    println!(
        "  {:<28} {:>10.3}  {:>10.3} Mfft/s",
        gpu.backend_name, gpu.elapsed_ms, gpu.throughput_mfft
    );
    println!(
        "  {:<28} {:>10.3}  {:>10.3} Mfft/s",
        cpu.backend_name, cpu.elapsed_ms, cpu.throughput_mfft
    );
    println!("  {}", "─".repeat(56));

    if speedup >= 1.0 {
        println!(
            "  {} is {:.2}× faster than {}",
            win.backend_name.green().bold(),
            speedup,
            lose.backend_name
        );
    } else {
        println!(
            "  {} is {:.2}× faster than {}",
            win.backend_name.yellow().bold(),
            1.0 / speedup,
            lose.backend_name
        );
    }
    println!(
        "{}",
        "═══════════════════════════════════════════════════════".cyan()
    );
    println!();
}
