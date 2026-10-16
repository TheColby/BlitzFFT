use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex32, Complex64};
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::benchmark::write_json_string;
use crate::blitz_fft::{
    bluestein_work_len_64, fft_real_arbitrary_f64_with_work, get_plan, get_plan_64,
};

#[cfg(any(have_fftw, have_kissfft))]
use anyhow::Context;

#[cfg(have_kissfft)]
use anyhow::bail;

#[cfg(any(have_fftw, have_kissfft, have_pocketfft))]
use std::{ffi::c_void, os::raw::c_int};

#[cfg(have_fftw)]
use std::os::raw::c_uint;

#[cfg(any(have_fftw, have_kissfft))]
use std::ptr::NonNull;

#[derive(Debug, Clone)]
pub struct WholeFftBenchResult {
    pub algorithm: &'static str,
    pub setup_secs: f64,
    pub exec_secs: f64,
    pub peak_bin: usize,
    pub peak_freq_hz: f64,
    pub peak_mag: f32,
}

const PEAK_FREQ_DECIMALS: usize = 12;

pub fn run_whole_signal_benchmark(
    signal: &[f32],
    sample_rate: u32,
    repeats: usize,
) -> Result<Vec<WholeFftBenchResult>> {
    ensure!(repeats > 0, "benchmark repeats must be positive");
    let mut results = Vec::with_capacity(6);

    results.push(bench_blitzfft_real(signal, sample_rate, repeats)?);
    results.push(bench_realfft(signal, sample_rate, repeats)?);
    results.push(bench_rustfft(signal, sample_rate, repeats)?);
    #[cfg(have_fftw)]
    results.push(bench_fftw(signal, sample_rate, repeats)?);
    #[cfg(have_kissfft)]
    results.push(bench_kissfft(signal, sample_rate, repeats)?);
    #[cfg(have_pocketfft)]
    results.push(bench_pocketfft(signal, sample_rate, repeats)?);

    Ok(results)
}

pub fn run_whole_signal_benchmark_f64(
    signal: &[f64],
    sample_rate: u32,
    repeats: usize,
) -> Result<Vec<WholeFftBenchResult>> {
    ensure!(repeats > 0, "benchmark repeats must be positive");
    let mut results = Vec::with_capacity(3);

    results.push(bench_blitzfft_real_f64(signal, sample_rate, repeats)?);
    results.push(bench_realfft_f64(signal, sample_rate, repeats)?);
    results.push(bench_rustfft_f64(signal, sample_rate, repeats)?);

    Ok(results)
}

pub fn print_whole_signal_table(results: &[WholeFftBenchResult], len: usize, sample_rate: u32) {
    println!();
    println!("Whole-file FFT benchmark");
    println!("  samples     : {}", len);
    println!("  sample rate : {} Hz", sample_rate);
    println!(
        "  duration    : {:.2} minutes",
        len as f64 / sample_rate as f64 / 60.0
    );
    println!();
    println!(
        "  {:<26} {:>12} {:>12} {:>12} {:>21} {:>14}",
        "Algorithm", "Setup (s)", "Exec (s)", "Peak bin", "Peak est. (Hz)", "Peak mag"
    );
    println!("  {}", "-".repeat(104));
    for result in results {
        println!(
            "  {:<26} {:>12.9} {:>12.9} {:>12} {:>21.*} {:>14.6}",
            result.algorithm,
            result.setup_secs,
            result.exec_secs,
            result.peak_bin,
            PEAK_FREQ_DECIMALS,
            result.peak_freq_hz,
            result.peak_mag,
        );
    }
    println!();
}

/// Write a dependency-free, machine-readable v0.2 whole-signal benchmark report.
pub fn write_json_report(
    path: &Path,
    results: &[WholeFftBenchResult],
    len: usize,
    sample_rate: u32,
    precision: u32,
    repeats: usize,
) -> Result<()> {
    ensure!(
        len > 0 && sample_rate > 0 && repeats > 0,
        "invalid benchmark workload"
    );
    ensure!(
        precision == 32 || precision == 64,
        "whole-file benchmark precision must be 32 or 64"
    );
    ensure!(!results.is_empty(), "benchmark results must not be empty");
    for row in results {
        ensure!(
            row.setup_secs.is_finite() && row.setup_secs >= 0.0,
            "invalid setup time"
        );
        ensure!(
            row.exec_secs.is_finite() && row.exec_secs >= 0.0,
            "invalid execution time"
        );
        ensure!(
            row.peak_bin <= len / 2 && row.peak_freq_hz.is_finite() && row.peak_mag.is_finite(),
            "invalid peak result"
        );
    }

    let mut writer = BufWriter::new(File::create(path)?);
    write!(writer, "{{\n  \"schema_version\": \"0.2\",\n  \"kind\": \"whole_file_fft_benchmark\",\n  \"samples\": {len},\n  \"sample_rate_hz\": {sample_rate},\n  \"precision_bits\": {precision},\n  \"repeats\": {repeats},\n  \"timing\": {{\"unit\": \"seconds\", \"execution_statistic\": \"arithmetic_mean\", \"scope\": \"setup once per algorithm; FFT calls timed individually; input restoration excluded; fixed algorithm order\"}},\n  \"results\": [\n")?;
    for (index, row) in results.iter().enumerate() {
        write!(writer, "    {{\"algorithm\": ")?;
        write_json_string(&mut writer, row.algorithm)?;
        write!(writer, ", \"setup_seconds\": {}, \"execution_seconds\": {}, \"peak_bin\": {}, \"peak_hz\": {}, \"peak_magnitude\": {}}}{}\n",
            row.setup_secs, row.exec_secs, row.peak_bin, row.peak_freq_hz, row.peak_mag,
            if index + 1 == results.len() { "" } else { "," })?;
    }
    writer.write_all(b"  ]\n}\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod json_tests {
    use super::*;

    #[test]
    fn report_contains_full_precision_numbers_and_schema() {
        let path = std::env::temp_dir().join(format!("blitzfft-whole-{}.json", std::process::id()));
        let rows = [WholeFftBenchResult {
            algorithm: "FFT \"test\"",
            setup_secs: 0.000000000123,
            exec_secs: 0.000000000456,
            peak_bin: 1,
            peak_freq_hz: 123.456789012345,
            peak_mag: 0.5,
        }];
        write_json_report(&path, &rows, 8, 48_000, 64, 3).unwrap();
        let report = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(report.contains("\"schema_version\": \"0.2\""));
        assert!(report.contains("\"precision_bits\": 64"));
        assert!(report.contains("\"algorithm\": \"FFT \\\"test\\\"\""));
        assert!(report.contains("0.000000000123"));
        assert!(report.contains("123.456789012345"));
    }
}

fn bench_blitzfft_real(
    signal: &[f32],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();
    let half1 = len / 2 + 1;

    // Setup: build (or fetch cached) plan + allocate scratch / output buffers.
    let setup_start = Instant::now();
    let plan = get_plan(len);
    let mut scratch = if len.is_power_of_two() {
        vec![num_complex::Complex32::new(0.0, 0.0); len / 2]
    } else {
        vec![]
    };
    let mut work = if len.is_power_of_two() {
        vec![]
    } else {
        vec![num_complex::Complex32::new(0.0, 0.0); plan.bluestein_work_len()]
    };
    let mut output = vec![num_complex::Complex32::new(0.0, 0.0); half1];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut exec_total_secs = 0.0;
    for _ in 0..repeats {
        let exec_start = Instant::now();
        if len.is_power_of_two() {
            plan.fft_real(signal, &mut scratch, &mut output);
        } else {
            plan.fft_real_with_work(signal, &mut output, &mut work);
        }
        exec_total_secs += exec_start.elapsed().as_secs_f64();
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) = peak_from_complex32(&output, len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "BlitzFFT native",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

fn bench_blitzfft_real_f64(
    signal: &[f64],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();
    let half1 = len / 2 + 1;
    let is_pow2 = len.is_power_of_two();

    // Setup: build/fetch plan + allocate buffers.
    let setup_start = Instant::now();
    let pow2_plan = is_pow2.then(|| get_plan_64(len));
    let mut work = if is_pow2 {
        vec![num_complex::Complex64::new(0.0, 0.0); len / 2]
    } else {
        vec![num_complex::Complex64::new(0.0, 0.0); bluestein_work_len_64(len)]
    };
    let mut output = vec![num_complex::Complex64::new(0.0, 0.0); half1];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut exec_total_secs = 0.0;
    for _ in 0..repeats {
        let exec_start = Instant::now();
        if let Some(plan) = pow2_plan.as_ref() {
            plan.fft_real_pow2(signal, &mut work, &mut output);
        } else {
            fft_real_arbitrary_f64_with_work(signal, &mut output, &mut work);
        }
        exec_total_secs += exec_start.elapsed().as_secs_f64();
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) = peak_from_complex64(&output, len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "BlitzFFT native (f64)",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

fn bench_realfft(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(len);
    let mut input = plan.make_input_vec();
    input.copy_from_slice(signal);
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut exec_total_secs = 0.0;
    for repeat in 0..repeats {
        if repeat > 0 {
            input.copy_from_slice(signal);
        }
        let exec_start = Instant::now();
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|err| anyhow!("RealFFT failed: {err}"))?;
        exec_total_secs += exec_start.elapsed().as_secs_f64();
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) = peak_from_complex32(&output, len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "RealFFT",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

fn bench_realfft_f64(
    signal: &[f64],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(len);
    let mut input = plan.make_input_vec();
    input.copy_from_slice(signal);
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut exec_total_secs = 0.0;
    for repeat in 0..repeats {
        if repeat > 0 {
            input.copy_from_slice(signal);
        }
        let exec_start = Instant::now();
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|err| anyhow!("RealFFT f64 failed: {err}"))?;
        exec_total_secs += exec_start.elapsed().as_secs_f64();
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) = peak_from_complex64(&output, len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "RealFFT (f64)",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

fn bench_rustfft(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = FftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(len);
    let input_template: Vec<Complex32> = signal
        .iter()
        .map(|&sample| Complex32::new(sample, 0.0))
        .collect();
    let mut buffer = input_template.clone();
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut exec_total_secs = 0.0;
    for repeat in 0..repeats {
        if repeat > 0 {
            buffer.copy_from_slice(&input_template);
        }
        let exec_start = Instant::now();
        plan.process(&mut buffer);
        exec_total_secs += exec_start.elapsed().as_secs_f64();
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let half_len = len / 2 + 1;
    let (peak_bin, peak_mag, peak_freq_hz) =
        peak_from_complex32(&buffer[..half_len], len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "RustFFT complex",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

fn bench_rustfft_f64(
    signal: &[f64],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(len);
    let input_template: Vec<Complex64> = signal
        .iter()
        .map(|&sample| Complex64::new(sample, 0.0))
        .collect();
    let mut buffer = input_template.clone();
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut exec_total_secs = 0.0;
    for repeat in 0..repeats {
        if repeat > 0 {
            buffer.copy_from_slice(&input_template);
        }
        let exec_start = Instant::now();
        plan.process(&mut buffer);
        exec_total_secs += exec_start.elapsed().as_secs_f64();
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let half_len = len / 2 + 1;
    let (peak_bin, peak_mag, peak_freq_hz) =
        peak_from_complex64(&buffer[..half_len], len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "RustFFT complex (f64)",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

#[cfg(have_fftw)]
fn bench_fftw(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let fftw = FftwContext::new(len)?;
    let setup_secs = fftw.setup_secs;

    let mut exec_total_secs = 0.0;
    for _ in 0..repeats {
        unsafe {
            let input = fftw.input_slice_mut();
            input.copy_from_slice(signal);
            let exec_start = Instant::now();
            fftwf_execute(fftw.plan);
            exec_total_secs += exec_start.elapsed().as_secs_f64();
        }
    }
    let exec_secs = exec_total_secs / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) =
        unsafe { peak_from_fftw(fftw.output_slice(), len, sample_rate) };

    Ok(WholeFftBenchResult {
        algorithm: "FFTW3f",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

#[cfg(have_kissfft)]
fn bench_kissfft(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();
    if len % 2 != 0 {
        bail!(
            "KissFFT real FFT requires an even sample count, got {}",
            len
        );
    }

    let setup_start = Instant::now();
    let cfg =
        unsafe { kiss_fftr_alloc(len as c_int, 0, std::ptr::null_mut(), std::ptr::null_mut()) };
    let cfg: NonNull<c_void> = NonNull::new(cfg).context("kiss_fftr_alloc returned null")?;
    let mut output = vec![KissFftCpx { r: 0.0, i: 0.0 }; len / 2 + 1];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        unsafe {
            kiss_fftr(cfg.as_ptr(), signal.as_ptr(), output.as_mut_ptr());
        }
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) = peak_from_kiss(&output, len, sample_rate);

    unsafe {
        free(cfg.as_ptr());
    }

    Ok(WholeFftBenchResult {
        algorithm: "KissFFT",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

#[cfg(have_pocketfft)]
fn bench_pocketfft(
    signal: &[f32],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut output = vec![PocketFftComplex { re: 0.0, im: 0.0 }; len / 2 + 1];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        let status =
            unsafe { pocketfft_r2c_f32(len, signal.as_ptr(), output.as_mut_ptr().cast::<f32>()) };
        if status != 0 {
            bail!("PocketFFT bridge returned error status {}", status);
        }
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag, peak_freq_hz) = peak_from_pocket(&output, len, sample_rate);

    Ok(WholeFftBenchResult {
        algorithm: "PocketFFT",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz,
        peak_mag,
    })
}

fn bin_to_hz_f64(bin: f64, fft_size: usize, sample_rate: u32) -> f64 {
    bin * sample_rate as f64 / fft_size as f64
}

fn peak_from_complex32(
    values: &[Complex32],
    fft_size: usize,
    sample_rate: u32,
) -> (usize, f32, f64) {
    peak_from_slice(values.len(), fft_size, sample_rate, |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt()
    })
}

fn peak_from_complex64(
    values: &[Complex64],
    fft_size: usize,
    sample_rate: u32,
) -> (usize, f32, f64) {
    peak_from_slice(values.len(), fft_size, sample_rate, |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt() as f32
    })
}

#[cfg(have_fftw)]
unsafe fn peak_from_fftw(
    values: &[FftwComplex],
    fft_size: usize,
    sample_rate: u32,
) -> (usize, f32, f64) {
    peak_from_slice(values.len(), fft_size, sample_rate, |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt()
    })
}

#[cfg(have_kissfft)]
fn peak_from_kiss(values: &[KissFftCpx], fft_size: usize, sample_rate: u32) -> (usize, f32, f64) {
    peak_from_slice(values.len(), fft_size, sample_rate, |i| {
        let c = values[i];
        (c.r * c.r + c.i * c.i).sqrt()
    })
}

#[cfg(have_pocketfft)]
fn peak_from_pocket(
    values: &[PocketFftComplex],
    fft_size: usize,
    sample_rate: u32,
) -> (usize, f32, f64) {
    peak_from_slice(values.len(), fft_size, sample_rate, |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt()
    })
}

fn peak_from_slice(
    len: usize,
    fft_size: usize,
    sample_rate: u32,
    magnitude_at: impl Fn(usize) -> f32,
) -> (usize, f32, f64) {
    let mut best_bin = 0usize;
    let mut best_mag = f32::MIN;
    for i in 0..len {
        let mag = magnitude_at(i);
        if mag > best_mag {
            best_bin = i;
            best_mag = mag;
        }
    }

    let peak_freq_hz = interpolated_peak_hz(best_bin, len, fft_size, sample_rate, &magnitude_at);
    (best_bin, best_mag, peak_freq_hz)
}

fn interpolated_peak_hz(
    peak_bin: usize,
    spectrum_len: usize,
    fft_size: usize,
    sample_rate: u32,
    magnitude_at: &impl Fn(usize) -> f32,
) -> f64 {
    let base_bin = peak_bin as f64;
    if peak_bin == 0 || peak_bin + 1 >= spectrum_len {
        return bin_to_hz_f64(base_bin, fft_size, sample_rate);
    }

    let alpha = magnitude_at(peak_bin - 1) as f64;
    let beta = magnitude_at(peak_bin) as f64;
    let gamma = magnitude_at(peak_bin + 1) as f64;
    let denominator = alpha - 2.0 * beta + gamma;
    if denominator.abs() < f64::EPSILON {
        return bin_to_hz_f64(base_bin, fft_size, sample_rate);
    }

    let delta = (0.5 * (alpha - gamma) / denominator).clamp(-0.5, 0.5);
    bin_to_hz_f64(base_bin + delta, fft_size, sample_rate)
}

#[cfg(have_fftw)]
struct FftwContext {
    plan: FftwfPlan,
    input: *mut f32,
    output: *mut FftwComplex,
    len: usize,
    half_len: usize,
    setup_secs: f64,
}

#[cfg(have_fftw)]
impl FftwContext {
    fn new(len: usize) -> Result<Self> {
        let setup_start = Instant::now();
        let half_len = len / 2 + 1;

        let input = unsafe { fftwf_malloc(len * std::mem::size_of::<f32>()) }.cast::<f32>();
        let output = unsafe { fftwf_malloc(half_len * std::mem::size_of::<FftwComplex>()) }
            .cast::<FftwComplex>();

        let input = NonNull::new(input).context("fftwf_malloc input returned null")?;
        let output = NonNull::new(output).context("fftwf_malloc output returned null")?;

        let plan = unsafe {
            fftwf_plan_dft_r2c_1d(len as c_int, input.as_ptr(), output.as_ptr(), FFTW_ESTIMATE)
        };
        let plan = NonNull::new(plan).context("fftwf_plan_dft_r2c_1d returned null")?;

        Ok(Self {
            plan: plan.as_ptr(),
            input: input.as_ptr(),
            output: output.as_ptr(),
            len,
            half_len,
            setup_secs: setup_start.elapsed().as_secs_f64(),
        })
    }

    unsafe fn input_slice_mut(&self) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.input, self.len)
    }

    unsafe fn output_slice(&self) -> &[FftwComplex] {
        std::slice::from_raw_parts(self.output, self.half_len)
    }
}

#[cfg(have_fftw)]
impl Drop for FftwContext {
    fn drop(&mut self) {
        unsafe {
            fftwf_destroy_plan(self.plan);
            fftwf_free(self.input.cast());
            fftwf_free(self.output.cast());
        }
    }
}

#[cfg(have_fftw)]
#[repr(C)]
#[derive(Clone, Copy)]
struct FftwComplex {
    re: f32,
    im: f32,
}

#[cfg(have_kissfft)]
#[repr(C)]
#[derive(Clone, Copy)]
struct KissFftCpx {
    r: f32,
    i: f32,
}

#[cfg(have_pocketfft)]
#[repr(C)]
#[derive(Clone, Copy)]
struct PocketFftComplex {
    re: f32,
    im: f32,
}

#[cfg(have_fftw)]
type FftwfPlan = *mut c_void;

#[cfg(have_fftw)]
const FFTW_ESTIMATE: c_uint = 1 << 6;

#[cfg(any(have_fftw, have_kissfft, have_pocketfft))]
unsafe extern "C" {
    #[cfg(have_fftw)]
    fn fftwf_malloc(n: usize) -> *mut c_void;
    #[cfg(have_fftw)]
    fn fftwf_free(p: *mut c_void);
    #[cfg(have_fftw)]
    fn fftwf_plan_dft_r2c_1d(
        n: c_int,
        input: *mut f32,
        output: *mut FftwComplex,
        flags: c_uint,
    ) -> FftwfPlan;
    #[cfg(have_fftw)]
    fn fftwf_execute(plan: FftwfPlan);
    #[cfg(have_fftw)]
    fn fftwf_destroy_plan(plan: FftwfPlan);

    #[cfg(have_kissfft)]
    fn kiss_fftr_alloc(
        nfft: c_int,
        inverse_fft: c_int,
        mem: *mut c_void,
        lenmem: *mut usize,
    ) -> *mut c_void;
    #[cfg(have_kissfft)]
    fn kiss_fftr(cfg: *mut c_void, timedata: *const f32, freqdata: *mut KissFftCpx);
    #[cfg(have_kissfft)]
    fn free(ptr: *mut c_void);

    #[cfg(have_pocketfft)]
    fn pocketfft_r2c_f32(len: usize, input: *const f32, output_interleaved: *mut f32) -> c_int;
}
