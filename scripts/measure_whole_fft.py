#!/usr/bin/env python3
"""Run reproducible, measured whole-file FFT comparisons on generated signals."""

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(
    r"^\s{2}(BlitzFFT native(?: \(f64\))?|RealFFT(?: \(f64\))?|"
    r"RustFFT complex(?: \(f64\))?|FFTW3f|KissFFT|PocketFFT)\s+"
    r"([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def optional_command(*args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1009, 1024, 4093, 4096, 65521, 65536])
    parser.add_argument("--precision", choices=["32", "64"], default="64")
    parser.add_argument("--repeats", type=int, default=5, help="FFT calls averaged by each CLI run")
    parser.add_argument("--trials", type=int, default=3, help="independent CLI runs per size")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--frequency", type=float, default=439.997)
    parser.add_argument("--features", default="", help="comma-separated optional Cargo features")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1 or args.trials < 1 or args.sample_rate < 1:
        parser.error("sizes must be >= 2; repeats, trials, and sample rate must be positive")

    features = [feature.strip() for feature in args.features.split(",") if feature.strip()]
    build = ["cargo", "build", "--release"]
    if features:
        build.extend(["--features", ",".join(features)])
    subprocess.run(build, cwd=ROOT, check=True)

    binary = ROOT / "target" / "release" / "blitzfft"
    report = {
        "kind": "measured_whole_file_fft",
        "method": "fixed algorithm order; setup once per CLI trial; execution averaged over bench_repeats",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": (optional_command("sysctl", "-n", "machdep.cpu.brand_string")
                      or optional_command("sysctl", "-n", "hw.model")
                      or platform.processor()),
        "cpu_count": os.cpu_count(),
        "memory_bytes": optional_command("sysctl", "-n", "hw.memsize"),
        "rustc": command("rustc", "--version"),
        "cargo": command("cargo", "--version"),
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "build_command": build,
        "precision_bits": int(args.precision),
        "features": features,
        "sample_rate_hz": args.sample_rate,
        "frequency_hz": args.frequency,
        "bench_repeats": args.repeats,
        "trials": args.trials,
        "runs": [],
    }

    for size in args.sizes:
        # The generator truncates an f32 duration to a sample count. Stay safely
        # inside the interval that truncates to exactly `size` samples.
        duration = (size + 0.25) / args.sample_rate
        run_command = [str(binary), "--generate-sine", f"{args.frequency},{args.sample_rate},{duration:.12f}",
                       "--precision", args.precision, "--whole-file-benchmark",
                       "--bench-repeats", str(args.repeats), "-f", "none"]
        for trial in range(args.trials):
            completed = subprocess.run(run_command, cwd=ROOT, check=True, text=True, capture_output=True)
            count = re.search(r"^  samples\s+: (\d+)\s*$", completed.stdout, re.MULTILINE)
            if not count or int(count.group(1)) != size:
                raise RuntimeError(f"requested {size} samples, got {count.group(1) if count else 'unknown'}")
            algorithms = {}
            for line in completed.stdout.splitlines():
                match = ROW.match(line)
                if match:
                    name, setup, execution, peak_bin, peak_hz, peak_mag = match.groups()
                    algorithms[name] = {
                        "setup_seconds": float(setup),
                        "execution_seconds": float(execution),
                        "peak_bin": int(peak_bin),
                        "peak_hz": float(peak_hz),
                        "peak_magnitude": float(peak_mag),
                    }
            expected = {"BlitzFFT native", "RealFFT", "RustFFT complex"}
            if args.precision == "64":
                expected = {f"{name} (f64)" for name in expected}
            if not expected.issubset(algorithms):
                raise RuntimeError(f"missing benchmark rows for size {size}: {sorted(expected - algorithms.keys())}")
            report["runs"].append({"samples": size, "trial": trial + 1, "command": run_command,
                                   "algorithms": algorithms})
            print(f"{size} samples, trial {trial + 1}/{args.trials}: " +
                  ", ".join(f"{name}={row['execution_seconds']:.6f}s" for name, row in algorithms.items()), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
