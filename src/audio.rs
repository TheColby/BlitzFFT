// src/audio.rs
//
// WAV loading and framing utilities.
//
// Supports: 16-bit PCM, 24-bit PCM, 32-bit float.
// Stereo → mono by averaging channels.
// Frames are extracted with a configurable hop size (default = fft_size/2).

use anyhow::{anyhow, Context, Result};
use hound::{WavReader, SampleFormat};
use std::path::Path;

/// Raw audio metadata
#[derive(Debug, Clone)]
pub struct AudioInfo {
    pub sample_rate  : u32,
    pub channels     : u16,
    pub num_samples  : usize,   // per channel
    pub duration_secs: f64,
}

/// Load a WAV file, downmix to mono, normalise to [-1, 1].
pub fn load_wav(path: &Path) -> Result<(Vec<f32>, AudioInfo)> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Cannot open {:?}", path))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples_f32: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        }
        (SampleFormat::Int, 16) => {
            reader.samples::<i16>()
                  .map(|s| s.unwrap() as f32 / 32768.0)
                  .collect()
        }
        (SampleFormat::Int, 24) => {
            reader.samples::<i32>()
                  .map(|s| s.unwrap() as f32 / 8_388_608.0)
                  .collect()
        }
        (SampleFormat::Int, 32) => {
            reader.samples::<i32>()
                  .map(|s| s.unwrap() as f32 / 2_147_483_648.0)
                  .collect()
        }
        (fmt, bits) => {
            return Err(anyhow!("Unsupported WAV format: {:?} {}‑bit", fmt, bits));
        }
    };

    // Downmix to mono
    let mono: Vec<f32> = if channels == 1 {
        samples_f32
    } else {
        samples_f32
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
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

/// Von Hann window coefficients for FFT frame apodisation.
pub fn hann_window(size: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    (0..size)
        .map(|n| (0.5 * (1.0 - (2.0 * PI * n as f64 / (size - 1) as f64).cos())) as f32)
        .collect()
}

/// Slice mono audio into overlapping frames.
///
/// * `signal`   — mono f32 samples
/// * `fft_size` — frame length (padded with zeros if audio is shorter)
/// * `hop`      — advance per frame (default fft_size/2 for 50% overlap)
pub fn frame_signal<'a>(
    signal   : &'a [f32],
    fft_size : usize,
    hop      : usize,
    window   : &[f32],
) -> Vec<Vec<f32>> {
    if signal.is_empty() { return vec![]; }

    let mut frames = Vec::new();
    let mut offset = 0usize;

    while offset < signal.len() {
        let end  = (offset + fft_size).min(signal.len());
        let src  = &signal[offset..end];

        let mut frame = vec![0.0f32; fft_size];
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
