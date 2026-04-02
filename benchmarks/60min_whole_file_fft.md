# 60-Minute Whole-File FFT Benchmark at 384 kHz

Updated: `2026-04-01`

Mode: `simulated from the measured 48 kHz benchmark anchor`

Input:

- File: `data/sine_439p997hz_60min_384khz_f32_hann.wav`
- Format: mono, 32-bit float WAV
- Sample rate: `384,000 Hz`
- Duration: `3,600 s`
- Frequency: `439.997 Hz`
- Samples: `1,382,400,000`
- Window: full-signal Hann window applied before benchmarking

Command:

```bash
target/release/audiofft \
  --generate-sine 439.997,384000,3600 \
  --apply-full-hann \
  --write-generated-wav data/sine_439p997hz_60min_384khz_f32_hann.wav \
  --whole-file-benchmark \
  --bench-repeats 1 \
  -f none
```

Notes:

- This one-hour `384` kHz exact whole-file FFT scenario is simulated from the measured `48` kHz benchmark anchor using the same $N \log_2 N$ scaling model used elsewhere in the docs.
- A direct six-library exact rerun at `1,382,400,000` samples would require substantially more local memory and disk than is practical in this workspace.
- The nearest FFT bin to `439.997 Hz` over a one-hour observation window is bin `1,583,989`, at
- In the live CLI whole-file benchmark table, `Peak freq (Hz)` is reported as a quadratic sub-bin estimate around the loudest bin and printed to `15` decimal places. The simulated table below keeps the shared nearest-bin center because this one-hour `384` kHz case is modeled, not freshly rerun.

$$
\frac{1{,}583{,}989}{3600} \approx 439.996944444444 \text{ Hz}
$$

Results:

| Algorithm | Setup (s) | Exec (s) | Peak bin | Peak freq (Hz) |
|---|---:|---:|---:|---:|
| PocketFFT | simulated | 10.776736 | 1,583,989 | 439.996944444444 |
| RealFFT | simulated | 20.514858 | 1,583,989 | 439.996944444444 |
| BlitzFFT exact-real | simulated | 22.263636 | 1,583,989 | 439.996944444444 |
| FFTW3f | simulated | 36.360388 | 1,583,989 | 439.996944444444 |
| RustFFT complex | simulated | 38.872592 | 1,583,989 | 439.996944444444 |
| KissFFT | simulated | 46.515731 | 1,583,989 | 439.996944444444 |

Measured shorter runs do show the extra decimal precision separating backends. For example, with

```bash
cargo run --release -- --generate-sine 439.997,48000,10 --precision 32 --apply-full-hann --whole-file-benchmark --bench-repeats 1 -f none
```

the interpolated peak estimates are:

| Algorithm | Peak bin | Peak freq (Hz) |
|---|---:|---:|
| BlitzFFT exact-real | 4,400 | 439.997757311980877 |
| RealFFT | 4,400 | 439.997757311980877 |
| RustFFT complex | 4,400 | 439.997757318645654 |
| FFTW3f | 4,400 | 439.997757317381456 |
| KissFFT | 4,400 | 439.997757313537420 |
| PocketFFT | 4,400 | 439.997757311980877 |

## Tuning-Theory Aside

The one-hour benchmark above is about exact FFT execution, not temperament theory, but the interval

$$
\Delta = 0.0002777778
$$

is also small enough to be interesting in logarithmic pitch space. If it is interpreted as a base-2 log-frequency interval, then

$$
\text{ratio} = 2^{0.0002777778} \approx 1.0001925
$$

and

$$
1200 \cdot 0.0002777778 = 0.33333336 \, \text{¢}
$$

A cent, written `¢`, is one hundredth of a semitone, or `1/1200` of an octave. So the target is approximately `0.3333 ¢`, or a frequency ratio of `1.0001925`.

It is also exactly one step of `3600-EDO`, because

$$
\frac{1200}{3600} = \frac{1}{3} \, \text{¢}
$$

for each equal division of the octave.

That is smaller than the usual standard named commas. A practical nearby comparison is one sixth of a schisma:

$$
\text{schisma} \approx 1.95 \, \text{¢}
$$

$$
\frac{1.95}{6} \approx 0.325 \, \text{¢}
$$

So

$$
\frac{1}{6}\text{ schisma} \approx 0.325 \, \text{¢}
$$

which is extremely close to `0.3333 ¢`, with an error of about `0.008 ¢`.

## Estimated Scaling To Multi-Day FFTs

The one-hour benchmark above is the simulated anchor point:

$$
N_0 = 1{,}382{,}400{,}000
$$

To estimate how the whole-file execution time grows for longer exact transforms, the chart below extrapolates the simulated `Exec` times with the standard FFT-size model

$$
\hat{T}(N) = T(N_0) \frac{N \log_2 N}{N_0 \log_2 N_0}
$$

This graph is intended as a planning aid, not as a claim that every duration shown was benchmarked directly. The one-hour `384` kHz anchor is itself simulated from the measured `48` kHz benchmark. In particular, it does not try to capture host-memory limits, paging, NUMA effects, planner strategy changes, or other system effects that become more visible at very large sizes.

To regenerate the chart after updating the one-hour benchmark anchor, run:

```bash
python3 scripts/generate_whole_fft_scaling_svg.py
```

![Estimated whole-file FFT execution time scaling](whole_file_fft_scaling.svg)

Estimated execution times at the multi-day end of the curve:

| Duration | Samples | PocketFFT | RealFFT | BlitzFFT exact-real | FFTW3f | RustFFT complex | KissFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 hr | 33,177,600,000 | 297.696 s | 566.701 s | 615.009 s | 1004.417 s | 1073.814 s | 1284.948 s |
| 48 hr | 66,355,200,000 | 612.428 s | 1165.832 s | 1265.213 s | 2066.312 s | 2209.077 s | 2643.427 s |
| 72 hr | 99,532,800,000 | 933.589 s | 1777.203 s | 1928.700 s | 3149.902 s | 3367.535 s | 4029.660 s |
