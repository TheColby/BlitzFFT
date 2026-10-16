#!/usr/bin/env python3
"""Run reproducible, measured whole-file FFT comparisons on generated signals."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def optional_command(*args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_report(path, size, sample_rate, precision, repeats):
    data = json.loads(path.read_text())
    expected = {
        "schema_version": "0.2", "kind": "whole_file_fft_benchmark",
        "samples": size, "sample_rate_hz": sample_rate,
        "precision_bits": int(precision), "repeats": repeats,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"{path}: expected {key}={value!r}, got {data.get(key)!r}")
    names = [row["algorithm"] for row in data["results"]]
    required = {"BlitzFFT native", "RealFFT", "RustFFT complex"}
    if precision == "64":
        required = {f"{name} (f64)" for name in required}
    if not required.issubset(names) or len(names) != len(set(names)):
        raise RuntimeError(f"{path}: missing or duplicate benchmark algorithms: {names}")
    for row in data["results"]:
        if row["setup_seconds"] < 0 or row["execution_seconds"] <= 0:
            raise RuntimeError(f"{path}: invalid timing for {row['algorithm']}")
    return data


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
    if args.precision == "32" and {"kissfft", "foreign-fft"}.intersection(features) and any(size % 2 for size in args.sizes):
        parser.error("KissFFT real FFT requires even sizes; pass an even --sizes list when enabling kissfft or foreign-fft")
    build = ["cargo", "build", "--release"]
    if features:
        build.extend(["--features", ",".join(features)])
    subprocess.run(build, cwd=ROOT, check=True)

    binary = ROOT / "target" / "release" / "blitzfft"
    report = {
        "schema_version": "0.2",
        "kind": "measured_whole_file_fft",
        "method": "separate process per trial; fixed algorithm order; setup once per algorithm; execution arithmetic mean over bench_repeats; median across trials",
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
        "medians": [],
    }

    with tempfile.TemporaryDirectory(prefix="blitzfft-bench-") as temp_dir:
        for size in args.sizes:
            # main.rs parses duration as f32 and truncates sample_rate * duration.
            duration = (size + 0.25) / args.sample_rate
            base_command = [str(binary), "--generate-sine", f"{args.frequency},{args.sample_rate},{duration:.12f}",
                            "--precision", args.precision, "--whole-file-benchmark",
                            "--bench-repeats", str(args.repeats), "-f", "none"]
            trials = []
            for trial in range(1, args.trials + 1):
                json_path = Path(temp_dir) / f"size-{size}-trial-{trial}.json"
                run_command = base_command + ["--benchmark-json", str(json_path)]
                subprocess.run(run_command, cwd=ROOT, check=True, text=True, capture_output=True)
                data = read_report(json_path, size, args.sample_rate, args.precision, args.repeats)
                algorithms = {row["algorithm"]: {key: value for key, value in row.items() if key != "algorithm"}
                              for row in data["results"]}
                trials.append(algorithms)
                report["runs"].append({"samples": size, "trial": trial, "command": run_command,
                                       "algorithms": algorithms})
                print(f"{size} samples, trial {trial}/{args.trials}: " +
                      ", ".join(f"{name}={row['execution_seconds']:.6f}s" for name, row in algorithms.items()), flush=True)
            report["medians"].append({
                "samples": size,
                "algorithms": {name: {
                    "setup_seconds": statistics.median(run[name]["setup_seconds"] for run in trials),
                    "execution_seconds": statistics.median(run[name]["execution_seconds"] for run in trials),
                } for name in trials[0]},
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
