# BlitzFFT v0.2.0

This release makes FFT results easier to validate and benchmark comparisons easier to reproduce. It does not claim a general speedup over v0.1.0 or over other FFT libraries.

## Highlights

- `--benchmark-json PATH` writes full-precision, machine-readable results for framed and whole-file benchmarks. The output is separate from spectral `--format json`.
- `scripts/measure_whole_fft.py` now validates CLI JSON reports and records raw trials plus per-algorithm medians, rather than scraping rounded console tables.
- Expanded randomized tests compare native FFT results with `RealFFT` and `RustFFT` across power-of-two and arbitrary lengths, plan reuse, and paired-frame batches.
- Fixed stale-sample handling when a short frame follows summary or top-bin processing on a reused CPU work buffer.
- README benchmark methodology now describes timing scope, input precision, fixed algorithm order, and the limits of the measured comparisons.

The native f64 and arbitrary-length paths still have performance headroom. A candidate optimization that did not show a reliable improvement was reverted rather than included in this release.

## Install

```bash
python3 -m pip install --upgrade blitzfft
blitzfft --version
```

The PyPI package installs a CLI, not an importable Python module. The default wheels use the native Rust CPU build; optional GPU, foreign-library, and experimental binary128 modes require a separate Cargo build.
