#!/usr/bin/env python3
"""Generate the whole-file FFT scaling SVG and table rows.

This script extrapolates execution time from a simulated 1-hour 384 kHz anchor
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

SAMPLE_RATE = 384_000
ANCHOR_SAMPLES = 1_382_400_000
ANCHOR_DURATION_LABEL = "1 hr"

ALGORITHMS = {
    "PocketFFT": (10.776736, "#1f77b4"),
    "RealFFT": (20.514858, "#2ca02c"),
    "BlitzFFT exact-real": (22.263636, "#d62728"),
    "FFTW3f": (36.360388, "#9467bd"),
    "RustFFT complex": (38.872592, "#ff7f0e"),
    "KissFFT": (46.515731, "#8c564b"),
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
    width, height = 1320, 940
    margin_left, margin_right, margin_top, margin_bottom = 110, 280, 88, 250
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_axis_y = height - margin_bottom
    y_min, y_max = 0.1, 10000.0
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)
    anchor_index = next(
        index for index, (label, _seconds) in enumerate(DURATIONS) if label == ANCHOR_DURATION_LABEL
    )
    line_styles = {
        "PocketFFT": "",
        "RealFFT": "9 7",
        "BlitzFFT exact-real": "",
        "FFTW3f": "14 8",
        "RustFFT complex": "4 6",
        "KissFFT": "18 8 4 8",
    }

    def x_pos(index: int) -> float:
        return margin_left + index * plot_width / (len(DURATIONS) - 1)

    def y_pos(value: float) -> float:
        lv = math.log10(value)
        return margin_top + (log_max - lv) / (log_max - log_min) * plot_height

    def relax_label_positions(items: list[tuple[float, str, str, float]]) -> list[tuple[float, str, str, float]]:
        min_gap = 24.0
        low = margin_top + 14
        high = margin_top + plot_height - 14
        relaxed = sorted(items, key=lambda item: item[0])
        for idx in range(1, len(relaxed)):
            prev_y = relaxed[idx - 1][0]
            y, name, color, value = relaxed[idx]
            if y - prev_y < min_gap:
                relaxed[idx] = (prev_y + min_gap, name, color, value)
        for idx in range(len(relaxed) - 2, -1, -1):
            next_y = relaxed[idx + 1][0]
            y, name, color, value = relaxed[idx]
            if next_y - y < min_gap:
                relaxed[idx] = (next_y - min_gap, name, color, value)
        return [
            (min(max(y, low), high), name, color, value)
            for y, name, color, value in relaxed
        ]

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
        "one minute to seventy-two hours at 384 kHz, extrapolated from a simulated "
        "one-hour anchor using N log2 N scaling.</desc>"
    )
    add('<rect width="100%" height="100%" fill="#ffffff"/>')
    add(
        '<text x="110" y="34" font-family="Helvetica, Arial, sans-serif" '
        'font-size="30" font-weight="700" fill="#111827">'
        "Estimated Exact FFT Execution Time vs Signal Length</text>"
    )
    add(
        '<text x="110" y="60" font-family="Helvetica, Arial, sans-serif" '
        'font-size="17" fill="#4b5563">'
        "384 kHz mono whole-file FFTs. The 1-hour anchor is simulated from the measured 48 kHz run; the rest use N log2 N scaling.</text>"
    )
    add(
        f'<rect x="{margin_left}" y="{x_axis_y - 2}" width="{plot_width}" height="54" '
        'fill="#f3f4f6" opacity="0.9"/>'
    )

    for tick in [0.1, 1, 10, 100, 1000, 10000]:
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
        f'y2="{x_axis_y}" stroke="#111827" stroke-width="1.75"/>'
    )
    add(
        f'<line x1="{margin_left}" y1="{x_axis_y}" '
        f'x2="{width - margin_right}" y2="{x_axis_y}" '
        'stroke="#111827" stroke-width="3"/>'
    )
    anchor_x = x_pos(anchor_index)
    add(
        f'<line x1="{anchor_x:.2f}" y1="{margin_top}" x2="{anchor_x:.2f}" '
        f'y2="{height - margin_bottom}" stroke="#9ca3af" stroke-width="1.5" '
        'stroke-dasharray="6 6"/>'
    )
    add(
        f'<text x="{anchor_x + 10:.2f}" y="{margin_top + 18:.2f}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="14" '
        'font-weight="700" fill="#4b5563">Simulated 1-hour anchor</text>'
    )

    for index, (label, seconds) in enumerate(DURATIONS):
        x = x_pos(index)
        add(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{x_axis_y}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        add(
            f'<line x1="{x:.2f}" y1="{x_axis_y - 10}" x2="{x:.2f}" '
            f'y2="{x_axis_y + 10}" stroke="#111827" stroke-width="1.5"/>'
        )
        add(
            f'<text x="{x:.2f}" y="{x_axis_y + 31}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="700" '
            f'fill="#374151">{escape(label)}</text>'
        )

    axis_y = margin_top + plot_height / 2
    add(
        f'<text x="32" y="{axis_y:.2f}" transform="rotate(-90 32 {axis_y:.2f})" '
        'text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="16" fill="#111827">Estimated execution time '
        "(seconds, log scale)</text>"
    )
    add(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 40}" '
        'text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="17" font-weight="700" fill="#111827">Exact whole-file FFT size at 384 kHz</text>'
    )

    end_labels: list[tuple[float, str, str, float]] = []
    for name, (values, color) in series.items():
        points = " ".join(
            f"{x_pos(index):.2f},{y_pos(value):.2f}"
            for index, value in enumerate(values)
        )
        dash = line_styles.get(name, "")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        add(
            f'<polyline fill="none" stroke="{color}" stroke-width="4" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} points="{points}"/>'
        )
        for index in (anchor_index, len(values) - 1):
            value = values[index]
            x = x_pos(index)
            y = y_pos(value)
            add(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" '
                'stroke="#ffffff" stroke-width="2"/>'
            )
        end_labels.append((y_pos(values[-1]), name, color, values[-1]))

    label_line_x = width - margin_right + 18
    label_text_x = label_line_x + 14
    for y, name, color, value in relax_label_positions(end_labels):
        end_x = x_pos(len(DURATIONS) - 1)
        end_y = y_pos(series[name][0][-1])
        add(
            f'<line x1="{end_x + 8:.2f}" y1="{end_y:.2f}" x2="{label_line_x:.2f}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        add(
            f'<text x="{label_text_x:.2f}" y="{y - 4:.2f}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="15" '
            'font-weight="700" '
            f'fill="{color}">{escape(name)}</text>'
        )
        add(
            f'<text x="{label_text_x:.2f}" y="{y + 14:.2f}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="13" '
            f'fill="#4b5563">72 hr est. = {value:.1f} s</text>'
        )

    callout_x = 110
    callout_y = 790
    callout_lines = [
        "Samples at 384 kHz: 1 min = 23.04M, 1 hr = 1.382B, 24 hr = 33.178B, 72 hr = 99.533B",
        f"Simulated anchor point: {ANCHOR_DURATION_LABEL}, 1.382B samples, derived from the measured 48 kHz benchmark",
        "Values away from that point are modeled estimates, not directly measured runs",
    ]
    add(
        f'<rect x="{callout_x}" y="{callout_y}" width="1180" height="108" rx="12" '
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
