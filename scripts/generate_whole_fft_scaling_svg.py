#!/usr/bin/env python3
"""Generate the whole-file FFT scaling SVG and table rows.

This script extrapolates execution time from the measured 1-hour benchmark
using the standard FFT-size growth model:

    T_hat(N) = T(N0) * (N log2 N) / (N0 log2 N0)

It writes the SVG chart used in the docs and prints the multi-day table rows
for convenient copy/paste into Markdown.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent.parent
SVG_PATH = ROOT / "benchmarks" / "whole_file_fft_scaling.svg"

SAMPLE_RATE = 48_000
ANCHOR_SAMPLES = 172_800_000
ANCHOR_DURATION_LABEL = "1 hr"

ALGORITHMS = {
    "PocketFFT": (1.214, "#1f77b4"),
    "RealFFT": (2.311, "#2ca02c"),
    "BlitzFFT exact-real": (2.508, "#d62728"),
    "FFTW3f": (4.096, "#9467bd"),
    "RustFFT complex": (4.379, "#ff7f0e"),
    "KissFFT": (5.240, "#8c564b"),
}

DURATIONS = [
    ("1 min", 60),
    ("10 min", 600),
    ("1 hr", 3600),
    ("6 hr", 21_600),
    ("12 hr", 43_200),
    ("24 hr", 86_400),
    ("48 hr", 172_800),
    ("72 hr", 259_200),
]

MULTI_DAY_ROWS = [
    ("24 hr", 86_400),
    ("48 hr", 172_800),
    ("72 hr", 259_200),
]


def estimate_exec_secs(anchor_exec_secs: float, samples: int) -> float:
    anchor_weight = ANCHOR_SAMPLES * math.log2(ANCHOR_SAMPLES)
    weight = samples * math.log2(samples)
    return anchor_exec_secs * weight / anchor_weight


def format_samples(samples: int) -> str:
    if samples >= 1_000_000_000:
        return f"{samples / 1_000_000_000:.3f}B"
    return f"{samples / 1_000_000:.2f}M"


def render_svg(series: dict[str, tuple[list[float], str]]) -> str:
    width, height = 1120, 760
    margin_left, margin_right, margin_top, margin_bottom = 110, 40, 60, 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_min, y_max = 0.01, 1000.0
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)

    def x_pos(index: int) -> float:
        return margin_left + index * plot_width / (len(DURATIONS) - 1)

    def y_pos(value: float) -> float:
        lv = math.log10(value)
        return margin_top + (log_max - lv) / (log_max - log_min) * plot_height

    parts: list[str] = []
    add = parts.append

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
    )
    add("<title id=\"title\">Estimated whole-file FFT execution time vs signal length</title>")
    add(
        "<desc id=\"desc\">Log-scale chart of estimated execution time for BlitzFFT "
        "exact-real, RealFFT, PocketFFT, FFTW3f, RustFFT complex, and KissFFT from "
        "one minute to seventy-two hours at 48 kHz, extrapolated from the measured "
        "one-hour benchmark using N log2 N scaling.</desc>"
    )
    add('<rect width="100%" height="100%" fill="#ffffff"/>')
    add(
        '<text x="110" y="34" font-family="Helvetica, Arial, sans-serif" '
        'font-size="28" font-weight="700" fill="#111827">'
        "Estimated Exact FFT Execution Time vs Signal Length</text>"
    )
    add(
        '<text x="110" y="58" font-family="Helvetica, Arial, sans-serif" '
        'font-size="16" fill="#4b5563">'
        "48 kHz mono whole-file FFTs; curves extrapolated from the measured "
        "1-hour benchmark with T_est(N) = T(N0) * N log2 N / (N0 log2 N0)</text>"
    )

    for tick in [0.01, 0.1, 1, 10, 100, 1000]:
        y = y_pos(tick)
        add(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        add(
            f'<text x="{margin_left - 14}" y="{y + 5:.2f}" text-anchor="end" '
            'font-family="Helvetica, Arial, sans-serif" font-size="14" '
            f'fill="#374151">{escape(f"{tick:g} s")}</text>'
        )

    add(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    add(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" '
        'stroke="#111827" stroke-width="1.5"/>'
    )

    for index, (label, seconds) in enumerate(DURATIONS):
        x = x_pos(index)
        samples = SAMPLE_RATE * seconds
        add(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom + 8}" stroke="#111827" stroke-width="1"/>'
        )
        add(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 28}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="14" '
            f'fill="#374151">{escape(label)}</text>'
        )
        add(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 48}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" '
            f'fill="#6b7280">N={escape(format_samples(samples))}</text>'
        )

    axis_y = margin_top + plot_height / 2
    add(
        f'<text x="32" y="{axis_y:.2f}" transform="rotate(-90 32 {axis_y:.2f})" '
        'text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="16" fill="#111827">Estimated execution time '
        "(seconds, log scale)</text>"
    )
    add(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 28}" '
        'text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="16" fill="#111827">Exact whole-file FFT size at 48 kHz</text>'
    )

    for name, (values, color) in series.items():
        points = " ".join(
            f"{x_pos(index):.2f},{y_pos(value):.2f}"
            for index, value in enumerate(values)
        )
        add(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'points="{points}"/>'
        )
        for index, value in enumerate(values):
            x = x_pos(index)
            y = y_pos(value)
            add(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" '
                'stroke="#ffffff" stroke-width="1.5"/>'
            )

    legend_x = 735
    legend_y = 92
    add(
        f'<rect x="{legend_x}" y="{legend_y}" width="320" height="168" rx="10" '
        'fill="#ffffff" stroke="#d1d5db"/>'
    )
    add(
        f'<text x="{legend_x + 16}" y="{legend_y + 24}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="16" '
        'font-weight="700" fill="#111827">Algorithms</text>'
    )
    for idx, (name, (_values, color)) in enumerate(series.items()):
        y = legend_y + 48 + idx * 20
        add(
            f'<line x1="{legend_x + 16}" y1="{y}" x2="{legend_x + 42}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        add(
            f'<circle cx="{legend_x + 29}" cy="{y}" r="4" fill="{color}" '
            'stroke="#fff" stroke-width="1"/>'
        )
        add(
            f'<text x="{legend_x + 52}" y="{y + 5}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="14" '
            f'fill="#111827">{escape(name)}</text>'
        )

    callout_x = 110
    callout_y = 570
    callout_lines = [
        f"Measured anchor point: {ANCHOR_DURATION_LABEL}, 172.8M samples",
        "Measured execution times come from benchmarks/60min_whole_file_fft.md",
        "Values away from that point are modeled estimates, not directly measured runs",
    ]
    add(
        f'<rect x="{callout_x}" y="{callout_y}" width="980" height="116" rx="10" '
        'fill="#f9fafb" stroke="#d1d5db"/>'
    )
    for idx, line in enumerate(callout_lines):
        add(
            f'<text x="{callout_x + 18}" y="{callout_y + 28 + idx * 28}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="15" '
            f'fill="#374151">{escape(line)}</text>'
        )

    add("</svg>")
    return "\n".join(parts)


def main() -> None:
    series: dict[str, tuple[list[float], str]] = {}
    for name, (anchor_exec_secs, color) in ALGORITHMS.items():
        values = [
            estimate_exec_secs(anchor_exec_secs, SAMPLE_RATE * seconds)
            for _label, seconds in DURATIONS
        ]
        series[name] = (values, color)

    SVG_PATH.write_text(render_svg(series), encoding="utf-8")
    print(f"Wrote {SVG_PATH.relative_to(ROOT)}")
    print()
    print("| Duration | Samples | PocketFFT | RealFFT | BlitzFFT exact-real | FFTW3f | RustFFT complex | KissFFT |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, seconds in MULTI_DAY_ROWS:
        samples = SAMPLE_RATE * seconds
        values = [
            estimate_exec_secs(ALGORITHMS[name][0], samples)
            for name in ALGORITHMS
        ]
        cells = " | ".join(f"{value:.3f} s" for value in values)
        print(f"| {label} | {samples:,} | {cells} |")


if __name__ == "__main__":
    main()
