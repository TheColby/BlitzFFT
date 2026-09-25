# BlitzFFT Roadmap

This roadmap is meant to keep the repo ambitious without getting slippery.
The project gets better when it becomes faster, easier to trust, and easier to build at the same time.

## What "better" means here

For BlitzFFT, "better" is not just one thing.

- Faster native Rust CPU kernels
- More trustworthy correctness and benchmark claims
- Cleaner defaults and a simpler install story
- A CLI that feels like a real tool instead of only a benchmark harness
- Clearer boundaries around experimental features such as `binary128`

## Current diagnosis

The repo already has a good center:

- a native Rust CPU path
- framed analysis plus exact whole-file benchmarking
- optional foreign-library comparisons
- explicit precision and backend choices

The main gaps are now more specific:

- the native `f64` whole-file path is still meaningfully slower than `RealFFT`
- the arbitrary-length Bluestein path still leaves speed on the table
- `binary128` and GPU spectrum output still store full per-frame magnitudes for top-bin requests
- benchmark presentation is much better than before, but reproducibility can still tighten
- the repo narrative is strong, but the README can do a better job of explaining priorities and tradeoffs

Recent progress: Homebrew packaging, `f64` SIMD butterflies, paired real-frame batching, a summary-only `f32` CPU path, a bounded top-bin CPU path, a SIMD-backed Bluestein convolution, and a reproducible local benchmark runner are implemented.

## Near-term priorities

### 1. Make the native CPU engine materially faster

Goal: improve real throughput where BlitzFFT is trying to compete, not just benchmark harness cosmetics.

Work:

- Benchmark and tune the existing `f64` SIMD butterflies across supported CPUs.
- Measure and further tune the new SIMD-backed Bluestein inner FFT across awkward lengths.
- Keep reducing allocation and copy overhead in the native real-input paths.
- Extend bounded spectrum output to other precisions and GPU backends where useful.
- Measure framed and whole-file performance separately so kernel wins are visible.

Success looks like:

- `BlitzFFT native (f64)` closes more of the gap to `RealFFT (f64)`
- framed CPU throughput rises without widening correctness risk
- awkward non-power-of-two lengths stop looking disproportionately expensive

### 2. Tighten correctness around every speedup

Goal: make each optimization safe to believe.

Work:

- Expand `src/blitz_fft.rs` tests beyond the current small reference vectors.
- Add randomized small-size comparisons against naive DFTs.
- Add more cross-checks against `rustfft` and `realfft`.
- Add targeted regression tests for unpack math, plan reuse, and Bluestein edge cases.
- Document expected numerical tolerances for `f32` and `f64`.

Success looks like:

- speed work lands with immediate numerical checks
- regressions show up in tests before they show up in benchmark tables

### 3. Keep the default build boring and dependable

Goal: make stable-Rust CPU usage the easiest path.

Work:

- Keep foreign-library comparisons fully optional.
- Keep GPU backends additive instead of required.
- Make capability reporting obvious from the CLI.
- Keep CI validating the default native path first.

Success looks like:

- new users can build and run the CPU tool without external FFT dependencies
- optional backends never make the default path feel fragile

## Mid-term priorities

### 4. Make the CLI smarter about work it does not need to do

Goal: stop paying for data movement and formatting that the user did not ask for.

Work:

- Extend the `f32` CPU summary-only fast path to other precision modes where useful.
- Profile the new top-bin path for small and large selection limits.
- Add machine-readable benchmark export for CI and scripted comparisons.
- Keep output schemas stable enough for automation.

Success looks like:

- `--summary` and filtered-output workflows get faster
- scripting no longer depends on scraping human-readable tables

### 5. Make benchmark claims easier to reproduce

Goal: make it obvious what is measured, what is simulated, and how to rerun it.

Work:

- Publish representative local measured artifacts alongside the reproducible runner.
- Keep measured and simulated tables visually separate in the docs.
- Record feature flags and precision mode alongside published results.
- Add a short benchmark policy section explaining what counts as a fair comparison.

Success looks like:

- readers can rerun the published workflow on their own machine
- benchmark trust improves even when BlitzFFT is not yet the fastest entry

### 6. Reduce complexity in the native engine layout

Goal: make future optimization work easier to reason about.

Work:

- Split `src/blitz_fft.rs` by concern: plans, pow2 kernels, Bluestein, SIMD, and helpers.
- Keep code comments focused on the non-obvious math and packing details.
- Make hot-path ownership and scratch-buffer expectations explicit in the APIs.

Success looks like:

- performance work becomes easier to isolate and review
- correctness review no longer requires holding one giant file in your head

## Long-term decisions

### 7. Decide how serious `binary128` should be

The repo should eventually choose one of two honest stories:

1. `binary128` stays a research-grade, nightly-only experiment.
2. `binary128` becomes a validated subsystem with real tests, limits, and documented use cases.

The project gets worse if it tries to sound production-grade without choosing.

### 8. Decide how much GPU scope BlitzFFT really wants

There are two healthy shapes here:

1. CPU-native BlitzFFT is the flagship, with CUDA and Metal as optional accelerators.
2. GPU backends become first-class supported surfaces with their own validation and benchmark discipline.

The repo gets muddier if it tries to imply both without resourcing both.

## Suggested release shape

### `v0.2`

Focus:

- faster native CPU kernels
- stronger correctness coverage for the native engine
- clearer benchmark reproducibility
- README and CLI polish

### `v0.3`

Focus:

- additional-precision and GPU spectrum fast paths
- benchmark artifact publication and scripting polish
- modularized native FFT implementation

### `v0.4`

Focus:

- explicit `binary128` positioning
- explicit long-term GPU positioning

## Not the priority right now

These are tempting, but they are not the highest-leverage next moves:

- adding more comparison libraries before the native engine improves
- publishing more simulated mega-benchmark tables without better reproducibility
- marketing "fastest FFT" claims before the native `f64` path closes more of the real gap
- broadening experimental precision support before the stable paths are tighter
