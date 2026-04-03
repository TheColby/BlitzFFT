// src/output.rs
//
// Spectrum output formats:
//   text  — human-readable table of (frame, bin, Hz, magnitude)
//   csv   — comma-separated, suitable for import into MATLAB / Python / Excel
//   json  — structured per-frame bins for scripting and downstream tooling
//   bin   — compact little-endian f32 binary (frame × bins)

use anyhow::{anyhow, Result};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    audio::{bin_to_hz, peak_bin},
    backends::FftFrame,
};

/// How to write output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Csv,
    Json,
    Bin,
    None,
}

pub fn write_frames(
    frames: &[FftFrame],
    fft_size: usize,
    sample_rate: u32,
    format: OutputFormat,
    out_path: Option<&Path>,
    top_bins: usize, // only emit the N loudest bins per frame (0 = all)
    min_hz: Option<f32>,
    max_hz: Option<f32>,
) -> Result<()> {
    match format {
        OutputFormat::None => return Ok(()),
        OutputFormat::Text => {
            let mut w: Box<dyn Write> = match out_path {
                Some(p) => Box::new(BufWriter::new(File::create(p)?)),
                None => Box::new(BufWriter::new(std::io::stdout())),
            };
            write_text(
                &mut *w,
                frames,
                fft_size,
                sample_rate,
                top_bins,
                min_hz,
                max_hz,
            )
        }
        OutputFormat::Csv => {
            let mut w: Box<dyn Write> = match out_path {
                Some(p) => Box::new(BufWriter::new(File::create(p)?)),
                None => Box::new(BufWriter::new(std::io::stdout())),
            };
            write_csv(
                &mut *w,
                frames,
                fft_size,
                sample_rate,
                top_bins,
                min_hz,
                max_hz,
            )
        }
        OutputFormat::Json => {
            let mut w: Box<dyn Write> = match out_path {
                Some(p) => Box::new(BufWriter::new(File::create(p)?)),
                None => Box::new(BufWriter::new(std::io::stdout())),
            };
            write_json(
                &mut *w,
                frames,
                fft_size,
                sample_rate,
                top_bins,
                min_hz,
                max_hz,
            )
        }
        OutputFormat::Bin => {
            if top_bins != 0 || min_hz.is_some() || max_hz.is_some() {
                return Err(anyhow!(
                    "--format bin does not support --top-bins, --min-hz, or --max-hz"
                ));
            }
            let p = out_path.expect("--output required for binary format");
            write_bin(p, frames)
        }
    }
}

fn write_text(
    w: &mut dyn Write,
    frames: &[FftFrame],
    fft_size: usize,
    sample_rate: u32,
    top_bins: usize,
    min_hz: Option<f32>,
    max_hz: Option<f32>,
) -> Result<()> {
    writeln!(w, "# audiofft — magnitude spectrum")?;
    writeln!(w, "# fft_size={} sample_rate={}", fft_size, sample_rate)?;
    writeln!(w, "# frame | bin | freq_hz | magnitude")?;

    for f in frames {
        let bins = emit_bins(
            &f.magnitude,
            fft_size,
            sample_rate,
            top_bins,
            min_hz,
            max_hz,
        );
        for (bin, mag) in bins {
            let hz = bin_to_hz(bin, fft_size, sample_rate);
            writeln!(w, "{:6} {:6} {:10.2} {:12.6}", f.frame_index, bin, hz, mag)?;
        }
    }
    Ok(())
}

fn write_csv(
    w: &mut dyn Write,
    frames: &[FftFrame],
    fft_size: usize,
    sample_rate: u32,
    top_bins: usize,
    min_hz: Option<f32>,
    max_hz: Option<f32>,
) -> Result<()> {
    writeln!(w, "frame,bin,freq_hz,magnitude")?;
    for f in frames {
        let bins = emit_bins(
            &f.magnitude,
            fft_size,
            sample_rate,
            top_bins,
            min_hz,
            max_hz,
        );
        for (bin, mag) in bins {
            let hz = bin_to_hz(bin, fft_size, sample_rate);
            writeln!(w, "{},{},{:.4},{:.8}", f.frame_index, bin, hz, mag)?;
        }
    }
    Ok(())
}

fn write_json(
    w: &mut dyn Write,
    frames: &[FftFrame],
    fft_size: usize,
    sample_rate: u32,
    top_bins: usize,
    min_hz: Option<f32>,
    max_hz: Option<f32>,
) -> Result<()> {
    writeln!(w, "[")?;
    for (frame_index, frame) in frames.iter().enumerate() {
        let bins = emit_bins(
            &frame.magnitude,
            fft_size,
            sample_rate,
            top_bins,
            min_hz,
            max_hz,
        );
        writeln!(w, "  {{")?;
        writeln!(w, "    \"frame\": {},", frame.frame_index)?;
        writeln!(w, "    \"bins\": [")?;
        for (bin_index, (bin, mag)) in bins.iter().enumerate() {
            let hz = bin_to_hz(*bin, fft_size, sample_rate);
            writeln!(
                w,
                "      {{\"bin\": {}, \"freq_hz\": {:.4}, \"magnitude\": {:.8}}}{}",
                bin,
                hz,
                mag,
                if bin_index + 1 == bins.len() { "" } else { "," }
            )?;
        }
        writeln!(w, "    ]")?;
        writeln!(
            w,
            "  }}{}",
            if frame_index + 1 == frames.len() {
                ""
            } else {
                ","
            }
        )?;
    }
    writeln!(w, "]")?;
    Ok(())
}

fn write_bin(path: &Path, frames: &[FftFrame]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for f in frames {
        for &m in &f.magnitude {
            w.write_all(&m.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Return (bin_index, magnitude) pairs; sorted by magnitude descending if
/// top_bins > 0, otherwise all bins in order.
fn emit_bins(
    magnitude: &[f32],
    fft_size: usize,
    sample_rate: u32,
    top_bins: usize,
    min_hz: Option<f32>,
    max_hz: Option<f32>,
) -> Vec<(usize, f32)> {
    let min_hz = min_hz.unwrap_or(0.0);
    let max_hz = max_hz.unwrap_or(f32::INFINITY);
    let all: Vec<(usize, f32)> = magnitude
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| {
            let hz = bin_to_hz(i, fft_size, sample_rate);
            (hz >= min_hz && hz <= max_hz).then_some((i, m))
        })
        .collect();

    if top_bins == 0 || top_bins >= all.len() {
        all
    } else {
        let mut sorted = all.clone();
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(top_bins);
        sorted.sort_unstable_by_key(|&(i, _)| i); // restore frequency order
        sorted
    }
}

/// Print a quick one-line summary per frame to stdout.
pub fn print_summary(
    frames: &[FftFrame],
    fft_size: usize,
    sample_rate: u32,
    min_hz: Option<f32>,
    max_hz: Option<f32>,
) {
    for f in frames {
        let bins = emit_bins(&f.magnitude, fft_size, sample_rate, 0, min_hz, max_hz);
        if bins.is_empty() {
            println!(
                "  frame {:>5}  peak      n/a  mag        n/a",
                f.frame_index
            );
        } else {
            let magnitudes: Vec<f32> = bins.iter().map(|(_, mag)| *mag).collect();
            let (peak_offset, mag) = peak_bin(&magnitudes);
            let bin = bins[peak_offset].0;
            let hz = bin_to_hz(bin, fft_size, sample_rate);
            println!(
                "  frame {:>5}  peak {:>7.1} Hz  mag {:>10.4}",
                f.frame_index, hz, mag
            );
        }
    }
}
