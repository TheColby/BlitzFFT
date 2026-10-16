// src/benchmark.rs
//
// Timing harness.
// Runs both the selected GPU backend and the CPU baseline, prints a
// side-by-side comparison table, and reports the speedup ratio.

use anyhow::{ensure, Result};
use colored::Colorize;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

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
    ensure!(repeats > 0, "benchmark repeats must be positive");
    ensure!(!frames.is_empty(), "benchmark requires at least one frame");
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

    let mfft = |ms: f64| {
        if ms > 0.0 {
            n as f64 / (ms * 1000.0)
        } else {
            0.0
        }
    };

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

/// Write one v0.2 framed-benchmark report without changing the human-readable table.
pub fn write_json_report(
    path: &Path,
    selected: &BenchResult,
    cpu: &BenchResult,
    repeats: usize,
) -> Result<()> {
    ensure!(repeats > 0, "benchmark repeats must be positive");
    ensure!(
        selected.frames == cpu.frames && selected.fft_size == cpu.fft_size,
        "benchmark results must describe the same workload"
    );
    ensure!(
        selected.frames > 0 && selected.fft_size > 0,
        "benchmark workload must be nonempty"
    );
    for row in [selected, cpu] {
        ensure!(
            row.elapsed_ms.is_finite() && row.elapsed_ms >= 0.0,
            "invalid elapsed time"
        );
        ensure!(
            row.throughput_mfft.is_finite() && row.throughput_mfft >= 0.0,
            "invalid throughput"
        );
    }
    let mut writer = BufWriter::new(File::create(path)?);
    write!(writer, "{{\n  \"schema_version\": \"0.2\",\n  \"kind\": \"framed_fft_benchmark\",\n  \"fft_size\": {},\n  \"frames\": {},\n  \"repeats\": {},\n  \"timing\": {{\"unit\": \"milliseconds\", \"statistic\": \"arithmetic_mean\", \"scope\": \"compute_batch calls; one untimed warm-up per backend; selected backend first\"}},\n  \"results\": [\n", selected.fft_size, selected.frames, repeats)?;
    for (index, row) in [selected, cpu].iter().enumerate() {
        write!(
            writer,
            "    {{\"role\": \"{}\", \"backend\": ",
            if index == 0 { "selected" } else { "baseline" }
        )?;
        write_json_string(&mut writer, &row.backend_name)?;
        write!(
            writer,
            ", \"elapsed_ms\": {}, \"throughput_mfft_per_second\": {}}}{}\n",
            row.elapsed_ms,
            row.throughput_mfft,
            if index == 0 { "," } else { "" }
        )?;
    }
    writer.write_all(b"  ]\n}\n")?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn write_json_string(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
    writer.write_all(b"\"")?;
    for ch in value.chars() {
        match ch {
            '"' => writer.write_all(b"\\\"")?,
            '\\' => writer.write_all(b"\\\\")?,
            '\n' => writer.write_all(b"\\n")?,
            '\r' => writer.write_all(b"\\r")?,
            '\t' => writer.write_all(b"\\t")?,
            c if c < '\u{20}' => write!(writer, "\\u{:04x}", c as u32)?,
            c => write!(writer, "{c}")?,
        }
    }
    writer.write_all(b"\"")
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
    println!("{}", "  blitzfft  —  Benchmark Results".cyan().bold());
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
        "  {:<28} {:>10.6}  {:>10.6} Mfft/s",
        gpu.backend_name, gpu.elapsed_ms, gpu.throughput_mfft
    );
    println!(
        "  {:<28} {:>10.6}  {:>10.6} Mfft/s",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_string_escapes_control_characters() {
        let mut output = Vec::new();
        write_json_string(&mut output, "CPU \"fast\"\\\n\u{1}").unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "\"CPU \\\"fast\\\"\\\\\\n\\u0001\""
        );
    }

    #[test]
    fn framed_report_rejects_mismatched_workloads_before_creating_file() {
        let path =
            std::env::temp_dir().join(format!("blitzfft-framed-{}.json", std::process::id()));
        let a = BenchResult {
            backend_name: "GPU".into(),
            frames: 2,
            fft_size: 8,
            elapsed_ms: 1.0,
            throughput_mfft: 0.002,
        };
        let b = BenchResult {
            backend_name: "CPU".into(),
            frames: 3,
            fft_size: 8,
            elapsed_ms: 2.0,
            throughput_mfft: 0.001,
        };
        assert!(write_json_report(&path, &a, &b, 5).is_err());
        assert!(!path.exists());
    }
}
