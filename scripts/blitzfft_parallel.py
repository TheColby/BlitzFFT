#!/usr/bin/env python3
"""Parallel Python reference CLI for framed FFT analysis.

This is a dependency-light reference implementation of the core framed-analysis
workflow. It uses a pure-Python radix-2 FFT and `ProcessPoolExecutor` to spread
frame work across CPU processes.
"""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
import os
import struct
import sys
import wave
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    num_samples: int
    duration_secs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="blitzfft-parallel.py",
        description="Pure-Python parallel framed FFT reference implementation",
    )
    parser.add_argument("input", nargs="?", help="Input WAV file")
    parser.add_argument(
        "--generate-sine",
        metavar="Hz,SR,Secs",
        help="Synthesize a sine wave instead of reading a WAV",
    )
    parser.add_argument("-n", "--fft-size", type=int, default=2048, help="FFT size (power of two)")
    parser.add_argument("--hop", type=int, help="Hop size in samples (default: fft_size/2)")
    parser.add_argument("--window", choices=("rect", "hann", "hamming", "blackman"), default="hann")
    parser.add_argument("--channel", default="avg", help="avg, left, right, or zero-based index")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0 = cpu count)")
    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
        help="Parallel executor type (default: auto)",
    )
    parser.add_argument("--summary", action="store_true", help="Print frame peak summary")
    parser.add_argument("--format", choices=("text", "csv", "json", "none"), default="text")
    parser.add_argument("--output", help="Write output to a file instead of stdout")
    parser.add_argument("--top-bins", type=int, default=0, help="Only emit the N loudest bins per frame")
    parser.add_argument("--min-hz", type=float, help="Only emit bins at or above this frequency")
    parser.add_argument("--max-hz", type=float, help="Only emit bins at or below this frequency")
    return parser.parse_args()


def parse_channel_selection(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"avg", "average", "mono", "mix"}:
        return None
    if normalized in {"left", "l"}:
        return 0
    if normalized in {"right", "r"}:
        return 1
    return int(normalized)


def load_wav_mono(path: Path, channel: str) -> tuple[list[float], AudioInfo]:
    selected_channel = parse_channel_selection(channel)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    if selected_channel is not None and selected_channel >= channels:
        raise ValueError(f"requested channel {selected_channel} but WAV only has {channels} channel(s)")

    mono: list[float] = []
    frame_size = channels * sample_width
    for frame_index in range(frame_count):
        offset = frame_index * frame_size
        values = []
        for channel_index in range(channels):
            start = offset + channel_index * sample_width
            sample_bytes = frames[start : start + sample_width]
            values.append(decode_sample(sample_bytes, sample_width))
        if selected_channel is None:
            mono.append(sum(values) / len(values))
        else:
            mono.append(values[selected_channel])

    return mono, AudioInfo(
        sample_rate=sample_rate,
        channels=channels,
        num_samples=len(mono),
        duration_secs=len(mono) / sample_rate,
    )


def decode_sample(sample_bytes: bytes, sample_width: int) -> float:
    if sample_width == 2:
        return struct.unpack("<h", sample_bytes)[0] / 32768.0
    if sample_width == 3:
        raw = int.from_bytes(sample_bytes, "little", signed=False)
        if raw & 0x800000:
            raw -= 0x1000000
        return raw / 8388608.0
    if sample_width == 4:
        try:
            value = struct.unpack("<f", sample_bytes)[0]
            if math.isfinite(value) and -1.5 <= value <= 1.5:
                return float(value)
        except struct.error:
            pass
        return struct.unpack("<i", sample_bytes)[0] / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")


def synthesize_sine(spec: str) -> tuple[list[float], AudioInfo]:
    freq_text, sample_rate_text, duration_text = [part.strip() for part in spec.split(",")]
    frequency = float(freq_text)
    sample_rate = int(sample_rate_text)
    duration_secs = float(duration_text)
    sample_count = int(sample_rate * duration_secs)
    samples = [
        math.sin(math.tau * frequency * index / sample_rate)
        for index in range(sample_count)
    ]
    return samples, AudioInfo(
        sample_rate=sample_rate,
        channels=1,
        num_samples=sample_count,
        duration_secs=duration_secs,
    )


def window_coeffs(kind: str, size: int) -> list[float]:
    if size <= 1:
        return [1.0] * size
    if kind == "rect":
        return [1.0] * size
    coeffs = []
    for n in range(size):
        phase = 2.0 * math.pi * n / (size - 1)
        if kind == "hann":
            coeff = 0.5 * (1.0 - math.cos(phase))
        elif kind == "hamming":
            coeff = 0.54 - 0.46 * math.cos(phase)
        elif kind == "blackman":
            coeff = 0.42 - 0.5 * math.cos(phase) + 0.08 * math.cos(2.0 * phase)
        else:
            raise ValueError(f"unsupported window: {kind}")
        coeffs.append(coeff)
    return coeffs


def frame_signal(signal: list[float], fft_size: int, hop: int, window: list[float]) -> list[tuple[int, list[float]]]:
    if not signal or hop <= 0:
        return []
    frames: list[tuple[int, list[float]]] = []
    offset = 0
    frame_index = 0
    while offset < len(signal):
        end = min(offset + fft_size, len(signal))
        frame = [0.0] * fft_size
        source = signal[offset:end]
        for idx, sample in enumerate(source):
            frame[idx] = sample * window[idx]
        frames.append((frame_index, frame))
        frame_index += 1
        offset += hop
    return frames


def fft_complex(values: list[complex]) -> list[complex]:
    size = len(values)
    if size == 0 or size & (size - 1):
        raise ValueError("FFT size must be a power of two")
    bits = size.bit_length() - 1
    output = [0j] * size
    for index, value in enumerate(values):
        reversed_index = int(f"{index:0{bits}b}"[::-1], 2)
        output[reversed_index] = value
    step = 2
    while step <= size:
        half = step // 2
        twiddle_step = cmath.exp(-2j * math.pi / step)
        for start in range(0, size, step):
            twiddle = 1 + 0j
            for k in range(half):
                even = output[start + k]
                odd = output[start + k + half] * twiddle
                output[start + k] = even + odd
                output[start + k + half] = even - odd
                twiddle *= twiddle_step
        step *= 2
    return output


def process_frame(task: tuple[int, list[float]]) -> tuple[int, list[float]]:
    frame_index, frame = task
    spectrum = fft_complex([complex(sample, 0.0) for sample in frame])
    half = len(frame) // 2 + 1
    magnitudes = [abs(value) for value in spectrum[:half]]
    return frame_index, magnitudes


def bin_to_hz(bin_index: int, fft_size: int, sample_rate: int) -> float:
    return bin_index * sample_rate / fft_size


def filter_bins(
    magnitudes: list[float],
    fft_size: int,
    sample_rate: int,
    top_bins: int,
    min_hz: float | None,
    max_hz: float | None,
) -> list[tuple[int, float]]:
    pairs = [
        (bin_index, magnitude)
        for bin_index, magnitude in enumerate(magnitudes)
        if (min_hz is None or bin_to_hz(bin_index, fft_size, sample_rate) >= min_hz)
        and (max_hz is None or bin_to_hz(bin_index, fft_size, sample_rate) <= max_hz)
    ]
    if top_bins > 0:
        pairs.sort(key=lambda item: item[1], reverse=True)
        pairs = pairs[:top_bins]
        pairs.sort(key=lambda item: item[0])
    return pairs


def print_summary(
    results: list[tuple[int, list[float]]],
    fft_size: int,
    sample_rate: int,
    min_hz: float | None,
    max_hz: float | None,
) -> None:
    print()
    print("  Frame   Peak Freq     Magnitude")
    print("  ──────────────────────────────")
    for frame_index, magnitudes in results:
        candidates = filter_bins(magnitudes, fft_size, sample_rate, 1, min_hz, max_hz)
        if not candidates:
            continue
        bin_index, magnitude = max(candidates, key=lambda item: item[1])
        print(
            f"  frame {frame_index:5d}  peak {bin_to_hz(bin_index, fft_size, sample_rate):7.1f} Hz"
            f"  mag {magnitude:10.4f}"
        )
    print()


def write_text(
    stream,
    results: list[tuple[int, list[float]]],
    fft_size: int,
    sample_rate: int,
    top_bins: int,
    min_hz: float | None,
    max_hz: float | None,
) -> None:
    stream.write("# blitzfft-parallel.py — magnitude spectrum\n")
    stream.write(f"# fft_size={fft_size} sample_rate={sample_rate}\n")
    stream.write("# frame | bin | freq_hz | magnitude\n")
    for frame_index, magnitudes in results:
        for bin_index, magnitude in filter_bins(magnitudes, fft_size, sample_rate, top_bins, min_hz, max_hz):
            stream.write(
                f"{frame_index:6d} {bin_index:6d} {bin_to_hz(bin_index, fft_size, sample_rate):10.2f}"
                f" {magnitude:12.6f}\n"
            )


def write_csv_output(
    stream,
    results: list[tuple[int, list[float]]],
    fft_size: int,
    sample_rate: int,
    top_bins: int,
    min_hz: float | None,
    max_hz: float | None,
) -> None:
    writer = csv.writer(stream)
    writer.writerow(["frame", "bin", "freq_hz", "magnitude"])
    for frame_index, magnitudes in results:
        for bin_index, magnitude in filter_bins(magnitudes, fft_size, sample_rate, top_bins, min_hz, max_hz):
            writer.writerow([frame_index, bin_index, f"{bin_to_hz(bin_index, fft_size, sample_rate):.4f}", f"{magnitude:.8f}"])


def write_json_output(
    stream,
    results: list[tuple[int, list[float]]],
    fft_size: int,
    sample_rate: int,
    top_bins: int,
    min_hz: float | None,
    max_hz: float | None,
) -> None:
    payload = []
    for frame_index, magnitudes in results:
        bins = [
            {
                "bin": bin_index,
                "freq_hz": round(bin_to_hz(bin_index, fft_size, sample_rate), 4),
                "magnitude": round(magnitude, 8),
            }
            for bin_index, magnitude in filter_bins(magnitudes, fft_size, sample_rate, top_bins, min_hz, max_hz)
        ]
        payload.append({"frame": frame_index, "bins": bins})
    json.dump(payload, stream, indent=2)
    stream.write("\n")


def run_parallel(
    tasks: Iterable[tuple[int, list[float]]],
    workers: int,
    executor_kind: str,
) -> tuple[list[tuple[int, list[float]]], str]:
    if workers == 1:
        return [process_frame(task) for task in tasks], "serial"

    task_list = list(tasks)
    if executor_kind in {"auto", "process"}:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(process_frame, task_list, chunksize=8))
            results.sort(key=lambda item: item[0])
            return results, "process"
        except (PermissionError, OSError):
            if executor_kind == "process":
                raise

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_frame, task_list))
    results.sort(key=lambda item: item[0])
    return results, "thread"


def open_output(path: str | None):
    if path is None:
        return sys.stdout
    return open(path, "w", encoding="utf-8", newline="")


def main() -> int:
    args = parse_args()
    if args.fft_size <= 0 or args.fft_size & (args.fft_size - 1):
        raise SystemExit("--fft-size must be a positive power of two")
    hop = args.hop if args.hop is not None else args.fft_size // 2
    if hop <= 0:
        raise SystemExit("--hop must be greater than zero")

    if args.generate_sine:
        signal, info = synthesize_sine(args.generate_sine)
    elif args.input:
        signal, info = load_wav_mono(Path(args.input), args.channel)
    else:
        raise SystemExit("provide an INPUT WAV or use --generate-sine")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 1))
    window = window_coeffs(args.window, args.fft_size)
    frames = frame_signal(signal, args.fft_size, hop, window)
    results, executor_used = run_parallel(frames, workers, args.executor)

    print(
        f"Processed {len(frames)} frame(s) at {info.sample_rate} Hz using {workers} worker(s)"
        f" with the {executor_used} executor.",
        file=sys.stderr,
    )

    if args.summary:
        print_summary(results, args.fft_size, info.sample_rate, args.min_hz, args.max_hz)

    if args.format == "none":
        return 0

    with open_output(args.output) as stream:
        if args.format == "text":
            write_text(stream, results, args.fft_size, info.sample_rate, args.top_bins, args.min_hz, args.max_hz)
        elif args.format == "csv":
            write_csv_output(stream, results, args.fft_size, info.sample_rate, args.top_bins, args.min_hz, args.max_hz)
        elif args.format == "json":
            write_json_output(stream, results, args.fft_size, info.sample_rate, args.top_bins, args.min_hz, args.max_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
