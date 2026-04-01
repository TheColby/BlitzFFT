# BlitzFFT

`BlitzFFT` is a Rust CLI for audio FFT work with two modes:

- Framed/STFT-style analysis with auto-selected `CUDA -> Metal -> CPU`.
- Exact whole-file FFT benchmarking on a single long waveform, including FFTW, KissFFT, PocketFFT, RustFFT, and the repo's optimized real-input path.

## What changed

- The CPU backend now uses `RealFFT` instead of packing real audio into a full complex `RustFFT` buffer for every frame.
- Framed results no longer materialize an unused full complex spectrum for every frame.
- There is now an exact whole-file benchmark path for non-power-of-two signals.
- The repo includes a generated one-hour 32-bit float sine-wave test asset with a full-file Hann window:
  `data/sine_440hz_60min_48khz_f32_hann.wav`

## Build

```bash
# CPU + exact whole-file benchmarks
cargo build --release

# Apple GPU backend
cargo build --release --features metal

# NVIDIA GPU backend
cargo build --release --features cuda

# Everything enabled
cargo build --release --features "cuda metal"
```

Notes:

- `FFTW3f` is linked from the local system install. On this machine it was discovered through `pkg-config`.
- `KissFFT` and `PocketFFT` are vendored in `vendor/`.
- The Metal shader is compiled by `build.rs` when the `metal` feature is enabled.

## CLI

The project name is `BlitzFFT`. The current binary name remains `audiofft`.

```text
Usage: audiofft [OPTIONS] [INPUT]

Arguments:
  [INPUT]  Input WAV file (16/24/32-bit PCM or f32)

Options:
  -n, --fft-size <FFT_SIZE>            FFT frame size (must be a power of two) [default: 2048]
      --hop <HOP>                      Hop size in samples (default = fft_size/2)
      --batch-size <BATCH_SIZE>        Number of frames to process per GPU batch (default = all frames) [default: 0]
  -b, --backend <BACKEND>              Force a specific backend [default: auto]
  -o, --output <PATH>                  Output file (optional; stdout if omitted for text/csv)
  -f, --format <FORMAT>                Output format [default: text]
      --top-bins <TOP_BINS>            Only emit the N loudest bins per frame (0 = all bins) [default: 0]
      --benchmark                      Run framed benchmark comparing selected backend against CPU baseline
      --bench-repeats <BENCH_REPEATS>  Number of benchmark repeats [default: 5]
      --whole-file-benchmark           Run one exact FFT over the entire signal
      --generate-sine <Hz,SR,Secs>     Synthesise a sine wave in memory
      --write-generated-wav <PATH>     Persist the generated/loaded mono signal as 32-bit float WAV
      --apply-full-hann                Apply a Hann window across the entire loaded/generated signal
      --summary                        Print per-frame peak-frequency summary to stdout
```

## Examples

```bash
# Standard framed analysis
cargo run --release -- input.wav --backend cpu --summary -f none

# Whole-file exact FFT benchmark on an existing WAV
cargo run --release -- input.wav --apply-full-hann --whole-file-benchmark --bench-repeats 1 -f none

# Reproduce the hour-long benchmark asset and timings from this repo
target/release/audiofft \
  --generate-sine 440,48000,3600 \
  --apply-full-hann \
  --write-generated-wav data/sine_440hz_60min_48khz_f32_hann.wav \
  --whole-file-benchmark \
  --bench-repeats 1 \
  -f none
```

## Measured 60-Minute Whole-File FFT

Measured on `2026-04-01` in this repo after generating a mono 48 kHz 32-bit float WAV with:

- Frequency: `440 Hz`
- Duration: `3600 s`
- Samples: `172,800,000`
- Frequency resolution: `sample_rate / N = 48000 / 172800000 = 1 / 3600 Hz = 0.0002777778 Hz`
- Window: full-signal Hann window applied before the FFT
- Command: the reproduction command shown above

The generated asset on disk is about `659 MiB`.

The table below reports one exact forward FFT over the full signal. `Setup` includes planner/config creation. `Exec` is the measured transform run, excluding WAV load but including any algorithm-specific input marshaling from the real sample buffer.

| Algorithm | Setup (s) | Exec (s) | Peak bin | Peak freq (Hz) |
|---|---:|---:|---:|---:|
| PocketFFT | 0.036 | 1.214 | 1,584,000 | 440.000000 |
| RealFFT | 0.591 | 2.311 | 1,584,000 | 440.000000 |
| BlitzFFT exact-real | 0.910 | 2.508 | 1,584,000 | 440.000000 |
| FFTW3f | 0.206 | 4.096 | 1,584,000 | 440.000000 |
| RustFFT complex | 1.015 | 4.379 | 1,584,000 | 440.000000 |
| KissFFT | 0.758 | 5.240 | 1,584,000 | 440.000000 |

## Layout

```text
BlitzFFT/
├── data/
│   └── sine_440hz_60min_48khz_f32_hann.wav
├── src/
│   ├── main.rs
│   ├── audio.rs
│   ├── benchmark.rs
│   ├── output.rs
│   ├── whole_fft.rs
│   ├── native/
│   │   └── pocketfft_bridge.cc
│   └── backends/
│       ├── mod.rs
│       ├── cpu.rs
│       ├── cuda.rs
│       └── metal.rs
├── vendor/
│   ├── kissfft/
│   └── pocketfft/
└── shaders/
    └── fft.metal
```

## License

MIT
