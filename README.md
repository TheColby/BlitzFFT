# audiofft

GPU-accelerated FFT CLI for audio files.  
**CUDA (cuFFT) > Metal (Apple GPU) > CPU (RustFFT)** — auto-selected at runtime.

---

## Why faster than FFTW?

| Backend | Strategy | When it wins |
|---|---|---|
| **cuFFT** | Massively-parallel 1D R2C batched transforms on NVIDIA GPU | N ≥ 2048, batch ≥ 64 |
| **Metal** | Stockham radix-2 FFT shader on Apple GPU (unified memory, zero-copy) | N ≥ 2048 on M-series |
| **RustFFT** | Planner-cached SIMD Cooley-Tukey (FFTW-competitive) | Fallback on any CPU |

For large audio files (hundreds of overlapping frames), GPU batching amortises transfer overhead and achieves **5–50× throughput** over single-threaded FFTW.

---

## Prerequisites

| Feature | Requirement |
|---|---|
| `cuda`  | CUDA 11+ runtime (`libcudart.so`, `libcufft.so`) — **no compile-time CUDA SDK needed** |
| `metal` | macOS 12+ with Xcode command-line tools (`xcrun`) |
| CPU     | Nothing extra — Rust stable toolchain only |

---

## Build

```bash
# CPU fallback only (works everywhere)
cargo build --release

# NVIDIA GPU support
cargo build --release --features cuda

# Apple GPU support (macOS)
cargo build --release --features metal

# Both (heterogeneous build)
cargo build --release --features "cuda metal"
```

---

## Usage

```
USAGE:
    audiofft [OPTIONS] [INPUT]

ARGS:
    <INPUT>    WAV file (16/24/32-bit PCM or f32)

OPTIONS:
    -n, --fft-size <N>          FFT frame size, must be power of two [default: 2048]
        --hop <SAMPLES>         Hop size in samples [default: fft_size/2]
        --batch-size <N>        Frames per GPU batch (0 = all) [default: 0]
    -b, --backend <BACKEND>     Force backend: auto|cuda|metal|cpu [default: auto]
    -o, --output <PATH>         Output file (stdout if omitted for text/csv)
    -f, --format <FORMAT>       text|csv|bin|none [default: text]
        --top-bins <N>          Only emit N loudest bins per frame (0=all) [default: 0]
        --benchmark             Compare GPU vs CPU and print speedup table
        --bench-repeats <N>     Timing repetitions [default: 5]
        --generate-sine <SPEC>  Synthesise input: "440,48000,2" = 440 Hz, 48 kHz, 2 s
        --summary               Print peak frequency per frame to stdout
    -h, --help
    -V, --version
```

### Examples

```bash
# Auto-select best GPU, write CSV spectrum
audiofft recording.wav -f csv -o spectrum.csv

# Benchmark Metal vs CPU on 4096-pt FFT
audiofft recording.wav -n 4096 --benchmark --backend metal

# Synthesise 440 Hz tone, force CUDA, print peak-freq summary
audiofft --generate-sine 440,48000,2 --backend cuda --summary

# Top-10 bins per frame as text to stdout
audiofft recording.wav --top-bins 10

# Large batch: 8192-pt FFT, 25% overlap
audiofft recording.wav -n 8192 --hop 2048 -f bin -o spectrum.bin
```

---

## Architecture

```
audiofft/
├── src/
│   ├── main.rs           CLI (clap), dispatch, progress bar
│   ├── audio.rs          WAV loading, Hann windowing, framing
│   ├── benchmark.rs      Timing harness, speedup table
│   ├── output.rs         text / CSV / binary writers
│   └── backends/
│       ├── mod.rs        FftBackend trait + runtime selection
│       ├── cpu.rs        RustFFT (Rayon parallel, one planner/thread)
│       ├── cuda.rs       cuFFT via runtime dlopen (no compile-time SDK)
│       └── metal.rs      Stockham FFT + magnitude on Apple GPU
└── shaders/
    └── fft.metal         Bit-reversal + radix-2 butterfly + magnitude kernels
```

### CUDA backend details
- Dynamically loads `libcudart` and `libcufft` at startup — binary runs on
  any machine; CUDA unavailability is a soft failure.
- Uses **batched cufftPlanMany R2C** — one plan covers all frames in the batch.
- Pinned host memory (`cudaHostAlloc`) for maximum PCIe throughput.

### Metal backend details
- `build.rs` compiles `fft.metal` → `fft.metallib` via `xcrun metal`.
- **Unified memory** on Apple Silicon: `MTLStorageModeShared` buffers are
  zero-copy — no explicit host↔device transfers.
- Three-kernel pipeline: `bit_reverse` → N × `fft_pass` → `magnitude`.
- Fully batched: one command buffer covers all frames.

---

## Output formats

| Format | Description |
|---|---|
| `text` | `frame bin freq_hz magnitude` — human-readable, stdout |
| `csv`  | Header + rows, importable by Python/MATLAB/Excel |
| `bin`  | Raw little-endian `f32` array, `[frames × (N/2+1)]` magnitudes |
| `none` | Suppress output (useful with `--benchmark` or `--summary`) |

---

## License

MIT
