// src/audio.rs
//
// WAV loading and framing utilities.
//
// Supports: 16-bit PCM, 24-bit PCM, 32-bit float.
// Multi-channel input can be downmixed or a single channel can be selected.
// Frames are extracted with a configurable hop size (default = fft_size/2).

use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use hound::{SampleFormat, WavReader};
use rayon::prelude::*;
use rustfft::num_traits::Float;
use std::{fmt, path::Path, str::FromStr};

use crate::quad::Quad;

/// Raw audio metadata
#[derive(Debug, Clone)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub num_samples: usize, // per channel
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelSelection {
    Average,
    Left,
    Right,
    Index(usize),
}

impl fmt::Display for ChannelSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Average => write!(f, "avg"),
            Self::Left => write!(f, "left"),
            Self::Right => write!(f, "right"),
            Self::Index(index) => write!(f, "{index}"),
        }
    }
}

impl FromStr for ChannelSelection {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "avg" | "average" | "mono" | "mix" => Ok(Self::Average),
            "left" | "l" => Ok(Self::Left),
            "right" | "r" => Ok(Self::Right),
            _ => normalized
                .parse::<usize>()
                .map(Self::Index)
                .map_err(|_| format!("invalid channel selection '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WindowFunction {
    Rect,
    Hann,
    Hamming,
    Blackman,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProcessingPrecision {
    #[value(name = "32")]
    Bits32,
    #[value(name = "64")]
    Bits64,
    #[value(name = "128")]
    Bits128,
}

impl fmt::Display for ProcessingPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Bits32 => "32",
            Self::Bits64 => "64",
            Self::Bits128 => "128",
        };
        write!(f, "{label}")
    }
}

impl fmt::Display for WindowFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Rect => "rect",
            Self::Hann => "hann",
            Self::Hamming => "hamming",
            Self::Blackman => "blackman",
        };
        write!(f, "{label}")
    }
}

/// Load a WAV file, select/downmix channels, normalise to [-1, 1].
pub fn load_wav(path: &Path, channel_selection: ChannelSelection) -> Result<(Vec<f32>, AudioInfo)> {
    let mut reader = WavReader::open(path).with_context(|| format!("Cannot open {:?}", path))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;

    let mono: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            collect_channel_samples(reader.samples::<f32>(), channels, channel_selection, |s| s)?
        }
        (SampleFormat::Int, 16) => collect_channel_samples(
            reader.samples::<i16>(),
            channels,
            channel_selection,
            |s| s as f32 / 32768.0,
        )?,
        (SampleFormat::Int, 24) => collect_channel_samples(
            reader.samples::<i32>(),
            channels,
            channel_selection,
            |s| s as f32 / 8_388_608.0,
        )?,
        (SampleFormat::Int, 32) => collect_channel_samples(
            reader.samples::<i32>(),
            channels,
            channel_selection,
            |s| s as f32 / 2_147_483_648.0,
        )?,
        (fmt, bits) => return Err(anyhow!("Unsupported WAV format: {:?} {}-bit", fmt, bits)),
    };

    let num_samples = mono.len();
    let sample_rate = spec.sample_rate;
    let duration_secs = num_samples as f64 / sample_rate as f64;

    let info = AudioInfo {
        sample_rate,
        channels: spec.channels,
        num_samples,
        duration_secs,
    };

    Ok((mono, info))
}

fn collect_channel_samples<T, I, F>(
    iter: I,
    channels: usize,
    selection: ChannelSelection,
    map_sample: F,
) -> Result<Vec<f32>>
where
    I: IntoIterator<Item = std::result::Result<T, hound::Error>>,
    F: Fn(T) -> f32,
{
    let selected_channel = match selection {
        ChannelSelection::Average => None,
        ChannelSelection::Left => Some(0),
        ChannelSelection::Right => Some(1),
        ChannelSelection::Index(index) => Some(index),
    };
    if let Some(index) = selected_channel {
        if index >= channels {
            return Err(anyhow!(
                "Requested channel {} but WAV only has {} channel{}",
                index,
                channels,
                if channels == 1 { "" } else { "s" }
            ));
        }
    }

    let mut mono = Vec::new();
    let mut sum = 0.0f32;
    let mut selected_value = 0.0f32;
    let mut channel_index = 0usize;

    for sample in iter {
        let sample = map_sample(sample.context("Invalid WAV sample")?);
        if let Some(index) = selected_channel {
            if channel_index == index {
                selected_value = sample;
            }
        } else {
            sum += sample;
        }
        channel_index += 1;

        if channel_index == channels {
            mono.push(if selected_channel.is_some() {
                selected_value
            } else {
                sum / channels as f32
            });
            sum = 0.0;
            selected_value = 0.0;
            channel_index = 0;
        }
    }

    Ok(mono)
}

pub fn window_coeffs(window: WindowFunction, size: usize) -> Vec<f32> {
    window_coeffs_generic(window, size)
}

pub fn window_coeffs_f64(window: WindowFunction, size: usize) -> Vec<f64> {
    window_coeffs_generic(window, size)
}

pub fn window_coeffs_qd(window: WindowFunction, size: usize) -> Vec<Quad> {
    if size <= 1 {
        return vec![Quad::ONE; size];
    }

    let denom = Quad::from((size - 1) as f64);
    let phase_step = Quad::TWO_PI / denom;
    match window {
        WindowFunction::Rect => vec![Quad::ONE; size],
        WindowFunction::Hann => (0..size)
            .map(|n| {
                let phase = phase_step * Quad::from(n as f64);
                Quad::from(0.5) * (Quad::ONE - phase.cos())
            })
            .collect(),
        WindowFunction::Hamming => (0..size)
            .map(|n| {
                let phase = phase_step * Quad::from(n as f64);
                Quad::from(0.54) - Quad::from(0.46) * phase.cos()
            })
            .collect(),
        WindowFunction::Blackman => (0..size)
            .map(|n| {
                let phase = phase_step * Quad::from(n as f64);
                Quad::from(0.42) - Quad::from(0.5) * phase.cos()
                    + Quad::from(0.08) * (phase + phase).cos()
            })
            .collect(),
    }
}

fn window_coeffs_generic<T: Float>(window: WindowFunction, size: usize) -> Vec<T> {
    use std::f64::consts::PI;

    if size <= 1 {
        return vec![T::one(); size];
    }

    let cast = |v: f64| T::from(v).expect("window coefficient fits target float type");
    match window {
        WindowFunction::Rect => vec![T::one(); size],
        WindowFunction::Hann => (0..size)
            .map(|n| cast(0.5 * (1.0 - (2.0 * PI * n as f64 / (size - 1) as f64).cos())))
            .collect(),
        WindowFunction::Hamming => (0..size)
            .map(|n| cast(0.54 - 0.46 * (2.0 * PI * n as f64 / (size - 1) as f64).cos()))
            .collect(),
        WindowFunction::Blackman => (0..size)
            .map(|n| {
                let phase = 2.0 * PI * n as f64 / (size - 1) as f64;
                cast(0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos())
            })
            .collect(),
    }
}

pub fn apply_window_in_place(signal: &mut [f32], window: WindowFunction) {
    apply_window_in_place_generic(signal, window);
}

pub fn apply_window_in_place_f64(signal: &mut [f64], window: WindowFunction) {
    apply_window_in_place_generic(signal, window);
}

pub fn apply_window_in_place_qd(signal: &mut [Quad], window: WindowFunction) {
    if signal.len() <= 1 || window == WindowFunction::Rect {
        return;
    }

    let coeffs = window_coeffs_qd(window, signal.len());
    for (sample, coeff) in signal.iter_mut().zip(coeffs.into_iter()) {
        *sample *= coeff;
    }
}

fn apply_window_in_place_generic<T>(signal: &mut [T], window: WindowFunction)
where
    T: Float + Send + Sync,
{
    if signal.len() <= 1 {
        return;
    }

    if window == WindowFunction::Rect {
        return;
    }

    let coeffs = window_coeffs_generic(window, signal.len());
    signal
        .par_iter_mut()
        .zip(coeffs.into_par_iter())
        .for_each(|(sample, coeff)| *sample = *sample * coeff);
}

pub fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("Cannot create {:?}", path))?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Slice mono audio into overlapping frames.
///
/// * `signal`   — mono f32 samples
/// * `fft_size` — frame length (padded with zeros if audio is shorter)
/// * `hop`      — advance per frame (default fft_size/2 for 50% overlap)
pub fn frame_signal<'a>(
    signal: &'a [f32],
    fft_size: usize,
    hop: usize,
    window: &[f32],
) -> Vec<Vec<f32>> {
    frame_signal_generic(signal, fft_size, hop, window)
}

pub fn frame_signal_f64<'a>(
    signal: &'a [f64],
    fft_size: usize,
    hop: usize,
    window: &[f64],
) -> Vec<Vec<f64>> {
    frame_signal_generic(signal, fft_size, hop, window)
}

pub fn frame_signal_qd(signal: &[Quad], fft_size: usize, hop: usize, window: &[Quad]) -> Vec<Vec<Quad>> {
    if signal.is_empty() || hop == 0 {
        return vec![];
    }

    let frame_count = if signal.len() <= fft_size {
        1
    } else {
        1 + (signal.len() - 1) / hop
    };
    let mut frames = Vec::with_capacity(frame_count);
    let mut offset = 0usize;

    while offset < signal.len() {
        let end = (offset + fft_size).min(signal.len());
        let src = &signal[offset..end];

        let mut frame = vec![Quad::ZERO; fft_size];
        for (i, &sample) in src.iter().enumerate() {
            frame[i] = sample * window[i];
        }
        frames.push(frame);
        offset += hop;
    }

    frames
}

fn frame_signal_generic<T>(signal: &[T], fft_size: usize, hop: usize, window: &[T]) -> Vec<Vec<T>>
where
    T: Float + Copy,
{
    if signal.is_empty() || hop == 0 {
        return vec![];
    }

    let frame_count = if signal.len() <= fft_size {
        1
    } else {
        1 + (signal.len() - 1) / hop
    };
    let mut frames = Vec::with_capacity(frame_count);
    let mut offset = 0usize;

    while offset < signal.len() {
        let end = (offset + fft_size).min(signal.len());
        let src = &signal[offset..end];

        let mut frame = vec![T::zero(); fft_size];
        for (i, &s) in src.iter().enumerate() {
            frame[i] = s * window[i];
        }
        frames.push(frame);
        offset += hop;
    }

    frames
}

/// Convert bin index to frequency in Hz.
#[inline]
pub fn bin_to_hz(bin: usize, fft_size: usize, sample_rate: u32) -> f32 {
    bin as f32 * sample_rate as f32 / fft_size as f32
}

/// Find the peak-magnitude bin in a spectrum.
pub fn peak_bin(magnitude: &[f32]) -> (usize, f32) {
    magnitude
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}
