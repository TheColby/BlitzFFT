# BlitzFFT Roadmap

This roadmap is meant to keep the project honest and useful.
It favors build reliability, correctness, and benchmark trust over flashy claims.

## Principles

- The default build should stay native Rust and easy to install.
- Optional backends and comparisons should be clearly opt-in.
- Numerical claims should be backed by tests or explicitly labeled as modeled.
- Benchmark output should optimize for trust, not spectacle.
- Experimental precision modes should be treated as research features until proven otherwise.

## Near Term

### 1. Ship a stable native-Rust baseline

Goal: make `cargo build` and `cargo run` feel boring and dependable.

Work:

- Keep the default build free of required FFTW, KISS FFT, and PocketFFT dependencies.
- Keep `cuda`, `metal`, `fftw`, `kissfft`, `pocketfft`, and `binary128` behind explicit Cargo features.
- Finish the remaining naming cleanup so all user-facing output consistently says `blitzfft`.
- Add a short capability summary command such as `--list-backends` or `--version --verbose`.

Success looks like:

- a new user can build and run the CPU path on stable Rust without hunting for external FFT libraries
- optional comparison backends do not block the default install path

### 2. Strengthen native FFT correctness coverage

Goal: make the native engine trustworthy across more sizes and inputs.

Work:

- Expand the unit tests in `src/blitz_fft.rs` beyond a few small reference vectors.
- Add randomized comparisons against naive DFTs for very small sizes.
- Add cross-checks against `rustfft` and `realfft` across power-of-two and awkward lengths.
- Add regression tests for plan reuse, real-input packing/unpacking, and Bluestein edge cases.
- Add explicit error tolerances for `f32` and `f64` modes.

Success looks like:

- the native engine has a clear, documented numerical envelope
- correctness regressions are caught before benchmark work starts

### 3. Make benchmark claims easier to trust

Goal: keep benchmark output informative without implying more than the code can support.

Work:

- Keep the default whole-file benchmark table limited to the native Rust comparison set.
- Treat FFTW, KISS FFT, and PocketFFT as optional comparison layers, not as part of the core product.
- Separate measured tables from simulated tables more clearly in the docs.
- Keep peak-frequency estimates labeled as estimates, with conservative printed precision.
- Add a reproducible benchmark script and record machine details alongside published numbers.

Success looks like:

- readers can immediately tell what is measured, what is simulated, and what features were enabled
- benchmark tables remain useful without looking like marketing

## Mid Term

### 4. Make the CLI feel complete

Goal: turn the repo from a promising benchmark harness into a reliable DSP tool.

Work:

- Add capability-reporting flags such as `--list-backends`, `--list-precisions`, and `--list-formats`.
- Improve error messages for feature-gated modes like `--precision 128`.
- Add a machine-readable benchmark export mode for CI and scripting.
- Add small example commands and expected outputs for the main workflows.
- Consider adding a compact spectrogram export path or a stable JSON schema for analysis output.

Success looks like:

- common tasks are discoverable from `--help`
- scripting and regression tracking do not require scraping human-formatted tables

### 5. Reduce architectural risk in the native engine

Goal: make the code easier to maintain and audit.

Work:

- Split `src/blitz_fft.rs` into smaller modules by concern: plan building, power-of-two kernels, arbitrary-length kernels, SIMD, and cache management.
- Document the exact algorithm choices and fast paths in code, not just in the README.
- Add comments only where they reduce real cognitive load, especially around packing/unpacking and Bluestein math.
- Keep public APIs small and explicit.

Success looks like:

- future optimization work does not require holding the whole file in your head at once
- correctness reviews become much easier

## Long Term

### 6. Decide what `128-bit` means for this project

Goal: make the precision story deliberate instead of ambiguous.

There are two viable directions:

1. Keep `binary128` as an experimental nightly-only research path.
2. Invest in a validated high-precision subsystem with serious tests, docs, and known limits.

Questions to answer:

- Is the main value scientific frequency estimation, or is it broad audio tooling?
- Does the project want stable-Rust portability more than true `binary128`?
- Should high precision remain CPU-only?

Success looks like:

- the docs and CLI make the tradeoff obvious
- users know whether `128-bit` is a demo, a lab tool, or a supported mode

### 7. Decide how much GPU scope the repo should own

Goal: avoid half-supporting too many acceleration stories.

Questions to answer:

- Is CPU-native BlitzFFT the flagship, with GPU backends as optional accelerators?
- Or should CUDA and Metal become first-class supported products with their own validation and benchmark discipline?
- Should foreign-library comparisons stay in this crate, or move to a benchmark-only companion crate?

Success looks like:

- the repo has a clear center of gravity
- maintenance effort matches the actual value each backend provides

## Recommended Release Shape

### `v0.2`

Focus:

- stable native-Rust default build
- stronger native FFT tests
- cleaner benchmark honesty
- more consistent naming and help output

### `v0.3`

Focus:

- better CLI ergonomics
- reproducible benchmark tooling
- modularized native FFT implementation

### `v0.4`

Focus:

- deliberate decision on `binary128`
- deliberate decision on long-term GPU scope

## Not Yet

These are intentionally not top priority right now:

- chasing "fastest FFT" marketing claims
- adding more optional comparison libraries before the native engine is better validated
- expanding experimental precision modes before the stable path is rock solid
- publishing more simulated benchmark tables without better reproducibility
