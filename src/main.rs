// src/main.rs
//
// audiofft — GPU-accelerated FFT CLI
//
// Priority: CUDA (cuFFT) → Metal → CPU (RustFFT)
//
// Usage examples
// ──────────────
//   audiofft input.wav                        # auto-select backend, print summary
//   audiofft input.wav --backend metal        # force Metal
//   audiofft input.wav --fft-size 4096 --hop 1024 --output spec.csv --format csv
//   audiofft input.wav --benchmark            # compare GPU vs CPU
//   audiofft --generate-sine 440 48000 2 | audiofft /dev/stdin --format text

mod audio;
mod backends;
mod benchmark;
mod output;

use std::{
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use audio::{frame_signal, hann_window, load_wav};
use backends::select_backend;
use output::{write_frames, OutputFormat, print_summary};

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendChoice {
    Auto,
    Cuda,
    Metal,
    Cpu,
}

#[derive(Parser, Debug)]
#[command(
    name    = "audiofft",
    version = "0.1.0",
    author  = "Colby Leider <colby@leider.org>",
    about   = "GPU-accelerated FFT for audio — CUDA (cuFFT) and Metal backends"
)]
struct Args {
    /// Input WAV file (16/24/32-bit PCM or f32)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// FFT frame size (must be a power of two)
    #[arg(short = 'n', long, default_value = "2048")]
    fft_size: usize,

    /// Hop size in samples (default = fft_size/2)
    #[arg(long)]
    hop: Option<usize>,

    /// Number of frames to process per GPU batch (default = all frames)
    #[arg(long, default_value = "0")]
    batch_size: usize,

    /// Force a specific backend
    #[arg(short, long, value_enum, default_value = "auto")]
    backend: BackendChoice,

    /// Output file (optional; stdout if omitted for text/csv)
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Output format
    #[arg(short = 'f', long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Only emit the N loudest bins per frame (0 = all bins)
    #[arg(long, default_value = "0")]
    top_bins: usize,

    /// Run benchmark comparing selected backend against CPU baseline
    #[arg(long)]
    benchmark: bool,

    /// Number of benchmark repeats (increases timing accuracy)
    #[arg(long, default_value = "5")]
    bench_repeats: usize,

    /// Synthesise a pure sine tone and pipe it as the input (no WAV file needed)
    /// Format: <frequency_hz>,<sample_rate>,<duration_secs>  e.g. 440,48000,2
    #[arg(long, value_name = "Hz,SR,Secs")]
    generate_sine: Option<String>,

    /// Print per-frame peak-frequency summary to stdout
    #[arg(long)]
    summary: bool,
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn is_power_of_two(n: usize) -> bool { n > 0 && (n & (n - 1)) == 0 }

/// Generate a mono sine tone as Vec<f32>.
fn synthesise_sine(freq: f32, sample_rate: u32, duration_secs: f32) -> (Vec<f32>, u32) {
    use std::f32::consts::TAU;
    let n = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| (TAU * freq * i as f32 / sample_rate as f32).sin())
        .collect();
    (samples, sample_rate)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // ── Validate fft_size ──────────────────────────────────────────────────
    if !is_power_of_two(args.fft_size) {
        return Err(anyhow!("--fft-size {} is not a power of two", args.fft_size));
    }

    // ── Load or synthesise audio ───────────────────────────────────────────
    let (mono, sample_rate) = if let Some(ref spec) = args.generate_sine {
        let parts: Vec<&str> = spec.split(',').collect();
        if parts.len() != 3 {
            return Err(anyhow!("--generate-sine format: freq_hz,sample_rate,duration_secs"));
        }
        let freq: f32  = parts[0].trim().parse()?;
        let sr:   u32  = parts[1].trim().parse()?;
        let dur:  f32  = parts[2].trim().parse()?;
        eprintln!("  Synthesising {:.1} Hz sine, {} Hz, {:.2}s", freq, sr, dur);
        let (s, r) = synthesise_sine(freq, sr, dur);
        (s, r)
    } else {
        let input = args.input.as_ref()
            .ok_or_else(|| anyhow!("Provide an INPUT file or use --generate-sine"))?;
        let (samples, info) = load_wav(input)?;
        eprintln!(
            "  Loaded {:?}  {:.3}s  {} Hz  {} ch",
            input.file_name().unwrap_or_default(),
            info.duration_secs,
            info.sample_rate,
            info.channels,
        );
        (samples, info.sample_rate)
    };

    // ── Build backend ──────────────────────────────────────────────────────
    let force = match args.backend {
        BackendChoice::Auto  => None,
        BackendChoice::Cuda  => Some("cuda"),
        BackendChoice::Metal => Some("metal"),
        BackendChoice::Cpu   => Some("cpu"),
    };
    let backend: Arc<dyn backends::FftBackend> = select_backend(force);
    eprintln!("  Backend : {}", backend.name().green().bold());

    // ── Frame the signal ───────────────────────────────────────────────────
    // GPU backends (CUDA v2, Metal v2) apply the Hann window on-device.
    // For the CPU backend we apply it here on the host.
    let hop      = args.hop.unwrap_or(args.fft_size / 2);
    let is_gpu   = !matches!(force, Some("cpu")) && backend.name().contains("GPU")
                   || backend.name().contains("CUDA");
    let window   = hann_window(args.fft_size);
    let flat_win = vec![1.0f32; args.fft_size]; // identity — GPU will window
    let win_ref  = if is_gpu { &flat_win } else { &window };
    let frames   = frame_signal(&mono, args.fft_size, hop, win_ref);
    eprintln!(
        "  Frames  : {}  (size={}, hop={})",
        frames.len(), args.fft_size, hop
    );

    // ── Benchmark mode ────────────────────────────────────────────────────
    if args.benchmark {
        eprintln!();
        eprintln!("  Running benchmark ({} repeats) …", args.bench_repeats);
        let (gpu_res, cpu_res) = benchmark::run(
            &backend, &frames, args.fft_size, args.bench_repeats,
        )?;
        benchmark::print_table(&gpu_res, &cpu_res);
        return Ok(());
    }

    // ── Compute FFTs ───────────────────────────────────────────────────────
    let batch_size = if args.batch_size == 0 { frames.len() } else { args.batch_size };

    let pb = ProgressBar::new(frames.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "  {bar:40.cyan/blue} {pos}/{len} frames  [{elapsed_precise}]"
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );

    let mut all_results = Vec::with_capacity(frames.len());
    let t_start = Instant::now();

    for chunk in frames.chunks(batch_size) {
        let refs: Vec<&[f32]> = chunk.iter().map(|v| v.as_slice()).collect();
        let mut batch_results = backend.compute_batch(&refs, args.fft_size)?;
        pb.inc(chunk.len() as u64);
        all_results.append(&mut batch_results);
    }

    pb.finish_and_clear();

    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  Done    : {:.2} ms total  ({:.1} μs/frame)",
        elapsed_ms,
        elapsed_ms * 1000.0 / all_results.len() as f64,
    );

    // ── Optional summary ───────────────────────────────────────────────────
    if args.summary {
        println!();
        println!("{}", "  Frame   Peak Freq     Magnitude".dimmed());
        println!("{}", "  ──────────────────────────────".dimmed());
        print_summary(&all_results, args.fft_size, sample_rate);
        println!();
    }

    // ── Write output ───────────────────────────────────────────────────────
    write_frames(
        &all_results,
        args.fft_size,
        sample_rate,
        args.format,
        args.output.as_deref(),
        args.top_bins,
    )?;

    Ok(())
}
