use std::{
    ffi::c_void,
    os::raw::{c_int, c_uint},
    ptr::NonNull,
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use num_complex::{Complex32, Complex64};
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::audio::bin_to_hz;

#[derive(Debug, Clone)]
pub struct WholeFftBenchResult {
    pub algorithm: &'static str,
    pub setup_secs: f64,
    pub exec_secs: f64,
    pub peak_bin: usize,
    pub peak_freq_hz: f32,
    pub peak_mag: f32,
}

pub fn run_whole_signal_benchmark(
    signal: &[f32],
    sample_rate: u32,
    repeats: usize,
) -> Result<Vec<WholeFftBenchResult>> {
    let repeats = repeats.max(1);
    let mut results = Vec::with_capacity(6);

    results.push(bench_blitzfft_real(signal, sample_rate, repeats)?);
    results.push(bench_realfft(signal, sample_rate, repeats)?);
    results.push(bench_rustfft(signal, sample_rate, repeats)?);
    results.push(bench_fftw(signal, sample_rate, repeats)?);
    results.push(bench_kissfft(signal, sample_rate, repeats)?);
    results.push(bench_pocketfft(signal, sample_rate, repeats)?);

    Ok(results)
}

pub fn run_whole_signal_benchmark_f64(
    signal: &[f64],
    sample_rate: u32,
    repeats: usize,
) -> Result<Vec<WholeFftBenchResult>> {
    let repeats = repeats.max(1);
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
        "  {:<26} {:>12} {:>12} {:>12} {:>14} {:>14}",
        "Algorithm", "Setup (s)", "Exec (s)", "Peak bin", "Peak freq (Hz)", "Peak mag"
    );
    println!("  {}", "-".repeat(100));
    for result in results {
        println!(
            "  {:<26} {:>12.3} {:>12.3} {:>12} {:>14.6} {:>14.3}",
            result.algorithm,
            result.setup_secs,
            result.exec_secs,
            result.peak_bin,
            result.peak_freq_hz,
            result.peak_mag,
        );
    }
    println!();
}

fn bench_blitzfft_real(
    signal: &[f32],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(len);
    let mut input = vec![0.0f32; len];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        input.copy_from_slice(signal);
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|err| anyhow!("BlitzFFT exact-real path failed: {err}"))?;
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag) = peak_from_complex32(&output);

    Ok(WholeFftBenchResult {
        algorithm: "BlitzFFT exact-real",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

fn bench_blitzfft_real_f64(
    signal: &[f64],
    sample_rate: u32,
    repeats: usize,
) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(len);
    let mut input = vec![0.0f64; len];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        input.copy_from_slice(signal);
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|err| anyhow!("BlitzFFT exact-real f64 path failed: {err}"))?;
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag) = peak_from_complex64(&output);

    Ok(WholeFftBenchResult {
        algorithm: "BlitzFFT exact-real (f64)",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

fn bench_realfft(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(len);
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut last_output = plan.make_output_vec();
    let exec_start = Instant::now();
    for _ in 0..repeats {
        let mut input = signal.to_vec();
        last_output.fill(Complex32::new(0.0, 0.0));
        plan.process(&mut input, &mut last_output)
            .map_err(|err| anyhow!("RealFFT failed: {err}"))?;
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag) = peak_from_complex32(&last_output);

    Ok(WholeFftBenchResult {
        algorithm: "RealFFT",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
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
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let mut last_output = plan.make_output_vec();
    let exec_start = Instant::now();
    for _ in 0..repeats {
        let mut input = signal.to_vec();
        last_output.fill(Complex64::new(0.0, 0.0));
        plan.process(&mut input, &mut last_output)
            .map_err(|err| anyhow!("RealFFT f64 failed: {err}"))?;
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag) = peak_from_complex64(&last_output);

    Ok(WholeFftBenchResult {
        algorithm: "RealFFT (f64)",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

fn bench_rustfft(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let setup_start = Instant::now();
    let mut planner = FftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(len);
    let mut buffer = vec![Complex32::new(0.0, 0.0); len];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        for (dst, &src) in buffer.iter_mut().zip(signal.iter()) {
            *dst = Complex32::new(src, 0.0);
        }
        plan.process(&mut buffer);
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let half_len = len / 2 + 1;
    let (peak_bin, peak_mag) = peak_from_complex32(&buffer[..half_len]);

    Ok(WholeFftBenchResult {
        algorithm: "RustFFT complex",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
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
    let mut buffer = vec![Complex64::new(0.0, 0.0); len];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        for (dst, &src) in buffer.iter_mut().zip(signal.iter()) {
            *dst = Complex64::new(src, 0.0);
        }
        plan.process(&mut buffer);
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let half_len = len / 2 + 1;
    let (peak_bin, peak_mag) = peak_from_complex64(&buffer[..half_len]);

    Ok(WholeFftBenchResult {
        algorithm: "RustFFT complex (f64)",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

fn bench_fftw(signal: &[f32], sample_rate: u32, repeats: usize) -> Result<WholeFftBenchResult> {
    let len = signal.len();

    let fftw = FftwContext::new(len)?;
    let setup_secs = fftw.setup_secs;

    let exec_start = Instant::now();
    for _ in 0..repeats {
        unsafe {
            let input = fftw.input_slice_mut();
            input.copy_from_slice(signal);
            fftwf_execute(fftw.plan);
        }
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag) = unsafe { peak_from_fftw(fftw.output_slice()) };

    Ok(WholeFftBenchResult {
        algorithm: "FFTW3f",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

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
    let mut input = vec![0.0f32; len];
    let mut output = vec![KissFftCpx { r: 0.0, i: 0.0 }; len / 2 + 1];
    let setup_secs = setup_start.elapsed().as_secs_f64();

    let exec_start = Instant::now();
    for _ in 0..repeats {
        input.copy_from_slice(signal);
        unsafe {
            kiss_fftr(cfg.as_ptr(), input.as_ptr(), output.as_mut_ptr());
        }
    }
    let exec_secs = exec_start.elapsed().as_secs_f64() / repeats as f64;
    let (peak_bin, peak_mag) = peak_from_kiss(&output);

    unsafe {
        free(cfg.as_ptr());
    }

    Ok(WholeFftBenchResult {
        algorithm: "KissFFT",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

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
    let (peak_bin, peak_mag) = peak_from_pocket(&output);

    Ok(WholeFftBenchResult {
        algorithm: "PocketFFT",
        setup_secs,
        exec_secs,
        peak_bin,
        peak_freq_hz: bin_to_hz(peak_bin, len, sample_rate),
        peak_mag,
    })
}

fn peak_from_complex32(values: &[Complex32]) -> (usize, f32) {
    peak_from_slice(values.len(), |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt()
    })
}

fn peak_from_complex64(values: &[Complex64]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let mag = (value.re * value.re + value.im * value.im).sqrt() as f32;
            (i, mag)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0))
}

unsafe fn peak_from_fftw(values: &[FftwComplex]) -> (usize, f32) {
    peak_from_slice(values.len(), |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt()
    })
}

fn peak_from_kiss(values: &[KissFftCpx]) -> (usize, f32) {
    peak_from_slice(values.len(), |i| {
        let c = values[i];
        (c.r * c.r + c.i * c.i).sqrt()
    })
}

fn peak_from_pocket(values: &[PocketFftComplex]) -> (usize, f32) {
    peak_from_slice(values.len(), |i| {
        let c = values[i];
        (c.re * c.re + c.im * c.im).sqrt()
    })
}

fn peak_from_slice(len: usize, magnitude_at: impl Fn(usize) -> f32) -> (usize, f32) {
    let mut best_bin = 0usize;
    let mut best_mag = f32::MIN;
    for i in 0..len {
        let mag = magnitude_at(i);
        if mag > best_mag {
            best_bin = i;
            best_mag = mag;
        }
    }
    (best_bin, best_mag)
}

struct FftwContext {
    plan: FftwfPlan,
    input: *mut f32,
    output: *mut FftwComplex,
    len: usize,
    half_len: usize,
    setup_secs: f64,
}

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

impl Drop for FftwContext {
    fn drop(&mut self) {
        unsafe {
            fftwf_destroy_plan(self.plan);
            fftwf_free(self.input.cast());
            fftwf_free(self.output.cast());
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FftwComplex {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KissFftCpx {
    r: f32,
    i: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PocketFftComplex {
    re: f32,
    im: f32,
}

type FftwfPlan = *mut c_void;

const FFTW_ESTIMATE: c_uint = 1 << 6;

unsafe extern "C" {
    fn fftwf_malloc(n: usize) -> *mut c_void;
    fn fftwf_free(p: *mut c_void);
    fn fftwf_plan_dft_r2c_1d(
        n: c_int,
        input: *mut f32,
        output: *mut FftwComplex,
        flags: c_uint,
    ) -> FftwfPlan;
    fn fftwf_execute(plan: FftwfPlan);
    fn fftwf_destroy_plan(plan: FftwfPlan);

    fn kiss_fftr_alloc(
        nfft: c_int,
        inverse_fft: c_int,
        mem: *mut c_void,
        lenmem: *mut usize,
    ) -> *mut c_void;
    fn kiss_fftr(cfg: *mut c_void, timedata: *const f32, freqdata: *mut KissFftCpx);
    fn free(ptr: *mut c_void);

    fn pocketfft_r2c_f32(len: usize, input: *const f32, output_interleaved: *mut f32) -> c_int;
}
