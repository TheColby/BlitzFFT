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

## Tuning-Theory Aside

The measured one-hour benchmark above is about exact FFT execution, not temperament theory, but the interval

$$
\Delta = 0.0002777778
$$

is also small enough to be interesting in logarithmic pitch space. If it is interpreted as a base-2 log-frequency interval, then

$$
\text{ratio} = 2^{0.0002777778} \approx 1.0001925
$$

and

$$
1200 \cdot 0.0002777778 = 0.33333336 \text{ cents}
$$

So the target is approximately `0.3333` cents, or a frequency ratio of `1.0001925`.

It is also exactly one step of `3600-EDO`, because

$$
\frac{1200}{3600} = \frac{1}{3} \text{ cent}
$$

for each equal division of the octave.

That is smaller than the usual standard named commas. A practical nearby comparison is one sixth of a schisma:

$$
\text{schisma} \approx 1.95 \text{ cents}
$$

$$
\frac{1.95}{6} \approx 0.325 \text{ cents}
$$

So

$$
\frac{1}{6}\text{ schisma} \approx 0.325 \text{ cents}
$$

which is extremely close to `0.3333` cents, with an error of about `0.008` cents.

## Estimated Scaling To Multi-Day FFTs

The one-hour benchmark above is the measured anchor point:

$$
N_0 = 172{,}800{,}000
$$

To estimate how the whole-file execution time grows for longer exact transforms, the chart below extrapolates the measured `Exec` times with the standard FFT-size model

$$
\hat{T}(N) = T(N_0) \frac{N \log_2 N}{N_0 \log_2 N_0}
$$

This graph is intended as a planning aid, not as a claim that every duration shown was benchmarked directly. In particular, it does not try to capture host-memory limits, paging, NUMA effects, planner strategy changes, or other system effects that become more visible at very large sizes.

To regenerate the chart after updating the one-hour benchmark anchor, run:

```bash
python3 scripts/generate_whole_fft_scaling_svg.py
```

![Estimated whole-file FFT execution time scaling](whole_file_fft_scaling.svg)

Estimated execution times at the multi-day end of the curve:

| Duration | Samples | PocketFFT | RealFFT | BlitzFFT exact-real | FFTW3f | RustFFT complex | KissFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 hr | 4,147,200,000 | 34.018 s | 64.757 s | 70.277 s | 114.775 s | 122.705 s | 146.831 s |
| 48 hr | 8,294,400,000 | 70.165 s | 133.568 s | 144.954 s | 236.735 s | 253.091 s | 302.854 s |
| 72 hr | 12,441,600,000 | 107.116 s | 203.909 s | 221.291 s | 361.406 s | 386.377 s | 462.346 s |
