# 60-Minute Whole-File FFT Benchmark

Run date: `2026-04-01`

Input:

- File: `data/sine_440hz_60min_48khz_f32_hann.wav`
- Format: mono, 32-bit float WAV
- Sample rate: `48,000 Hz`
- Duration: `3,600 s`
- Samples: `172,800,000`
- Window: full-signal Hann window applied before benchmarking

Command:

```bash
target/release/audiofft \
  --generate-sine 440,48000,3600 \
  --apply-full-hann \
  --write-generated-wav data/sine_440hz_60min_48khz_f32_hann.wav \
  --whole-file-benchmark \
  --bench-repeats 1 \
  -f none
```

Results:

| Algorithm | Setup (s) | Exec (s) | Peak bin | Peak freq (Hz) |
|---|---:|---:|---:|---:|
| PocketFFT | 0.036 | 1.214 | 1,584,000 | 440.000000 |
| RealFFT | 0.591 | 2.311 | 1,584,000 | 440.000000 |
| BlitzFFT exact-real | 0.910 | 2.508 | 1,584,000 | 440.000000 |
| FFTW3f | 0.206 | 4.096 | 1,584,000 | 440.000000 |
| RustFFT complex | 1.015 | 4.379 | 1,584,000 | 440.000000 |
| KissFFT | 0.758 | 5.240 | 1,584,000 | 440.000000 |
