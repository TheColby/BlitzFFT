#!/usr/bin/env python3
"""Generate a one-page PDF map of the BlitzFFT CLI architecture."""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "blitzfft_system_block_diagram.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

pdfmetrics.registerFont(TTFont("Verdana", str(FONT_DIR / "Verdana.ttf")))
pdfmetrics.registerFont(TTFont("Verdana-Bold", str(FONT_DIR / "Verdana Bold.ttf")))
pdfmetrics.registerFont(TTFont("DIN", str(FONT_DIR / "DIN Alternate Bold.ttf")))

PAGE_W, PAGE_H = 1200, 760
INK = HexColor("#183436")
MUTED = HexColor("#5C7272")
PAPER = HexColor("#F4F5F0")
PANEL = HexColor("#E7ECE7")
WHITE = HexColor("#FFFFFF")
TEAL = HexColor("#087E76")
ORANGE = HexColor("#D46A3A")
GOLD = HexColor("#B58A2B")
LINE = HexColor("#A8B8B3")


def label(c, x, y, text, size=10, color=INK, font="Verdana"):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, y, text)


def panel(c, x, y, w, h, number, title, subtitle, accent):
    c.setFillColor(PANEL)
    c.roundRect(x, y, w, h, 17, fill=1, stroke=0)
    c.setFillColor(accent)
    c.roundRect(x + 18, y + h - 43, 28, 22, 7, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("DIN", 11)
    c.drawCentredString(x + 32, y + h - 37, number)
    label(c, x + 55, y + h - 37, title, 15, INK, "DIN")
    label(c, x + w - 365, y + h - 35, subtitle, 8.5, MUTED)


def card(c, x, y, w, h, kicker, title, lines, accent):
    c.setFillColor(WHITE)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 11, fill=1, stroke=1)
    c.setFillColor(accent)
    c.roundRect(x + 12, y + h - 13, 27, 3.5, 1.5, fill=1, stroke=0)
    label(c, x + 13, y + h - 30, kicker, 8.2, accent, "Verdana-Bold")
    compact = h < 110
    label(c, x + 13, y + h - (43 if compact else 52), title, 13, INK, "DIN")
    text_y = y + h - (60 if compact else 72)
    for line in lines:
        if pdfmetrics.stringWidth(line, "Verdana", 8.9) > w - 25:
            raise ValueError(f"Text exceeds card width: {line}")
        label(c, x + 13, text_y, line, 8.9, MUTED)
        text_y -= 14.5


def arrow(c, x1, y1, x2, y2, color=TEAL):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(2)
    c.line(x1, y1, x2, y2)
    if x2 > x1:
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - 7, y2 + 4)
        p.lineTo(x2 - 7, y2 - 4)
    elif y2 < y1:
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - 4, y2 + 7)
        p.lineTo(x2 + 4, y2 + 7)
    else:
        return
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def draw():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("BlitzFFT system block diagram")
    c.setAuthor("BlitzFFT")

    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(INK)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    label(c, 48, 701, "BLITZFFT", 31, INK, "DIN")
    label(c, 219, 701, "/ SYSTEM BLOCK DIAGRAM", 18, TEAL, "DIN")
    label(c, 49, 674, "Native Rust audio FFT CLI | framed analysis + exact whole-file benchmarking", 10.5, MUTED)
    label(c, 1000, 701, "CODE SNAPSHOT  d2af0bc", 9, MUTED, "Verdana-Bold")

    # Shared entry column.
    c.setFillColor(INK)
    c.roundRect(48, 106, 238, 526, 17, fill=1, stroke=0)
    label(c, 66, 604, "COMMON ENTRY", 16, WHITE, "DIN")
    label(c, 66, 586, "One input path, two execution modes", 8.7, HexColor("#AFCBC6"))
    card(c, 66, 495, 202, 83, "01 / SOURCE", "Audio input", ["WAV PCM / float32", "or generated sine"], TEAL)
    card(c, 66, 398, 202, 83, "02 / INGEST", "Decode + channel", ["hound WAV reader", "avg / left / right / index"], TEAL)
    card(c, 66, 280, 202, 103, "03 / PREPARE", "Mono samples", ["f32 ingest; f64 by default", "binary128 opt-in on nightly", "optional full-signal window"], TEAL)
    card(c, 66, 172, 202, 92, "04 / ROUTE", "Execution mode", ["Framed analysis or", "whole-file benchmark"], ORANGE)
    arrow(c, 167, 495, 167, 484, HexColor("#80C4B9"))
    arrow(c, 167, 398, 167, 386, HexColor("#80C4B9"))
    arrow(c, 167, 280, 167, 267, HexColor("#80C4B9"))

    # Route outputs remain in the gutter, clear of the two lane panels.
    c.setStrokeColor(ORANGE)
    c.setLineWidth(2)
    c.line(268, 205, 310, 205)
    c.line(310, 205, 310, 485)
    arrow(c, 310, 485, 350, 485, ORANGE)
    arrow(c, 310, 205, 350, 205, ORANGE)

    panel(c, 330, 385, 823, 247, "A", "FRAMED ANALYSIS", "STFT-style: many power-of-two FFTs", TEAL)
    card(c, 350, 420, 175, 129, "A1 / FRAME", "Window + hop", ["Choose N and hop", "Hann or other window", "Window each frame"], TEAL)
    card(c, 545, 420, 195, 129, "A2 / TRANSFORM", "Backend engine", ["CPU f32: paired real FFT", "CPU f64: SIMD native", "CUDA / Metal f32 optional", "binary128 CPU opt-in"], TEAL)
    card(c, 760, 420, 175, 129, "A3 / REDUCE", "Spectral bins", ["Positive frequencies", "Peak summary or top bins", "Optional Hz band filter"], TEAL)
    card(c, 955, 420, 178, 129, "A4 / DELIVER", "Output", ["text / CSV / JSON", "raw f32 binary", "stdout or file"], TEAL)
    for x1, x2 in [(525, 545), (740, 760), (935, 955)]:
        arrow(c, x1 + 2, 484, x2 - 3, 484)
    label(c, 350, 401, "f32 --benchmark compares selected backend with the CPU baseline.", 8.4, MUTED)

    panel(c, 330, 106, 823, 247, "B", "WHOLE-FILE BENCHMARK", "One exact FFT across the full signal", ORANGE)
    card(c, 350, 141, 175, 129, "B1 / PREPARE", "Full waveform", ["No frame or hop", "Optional full Hann", "Power-of-two or", "arbitrary length"], ORANGE)
    card(c, 545, 141, 195, 129, "B2 / COMPARE", "FFT engines", ["BlitzFFT f32 / f64", "RealFFT; RustFFT complex", "Optional FFTW / KissFFT", "and PocketFFT (f32)"], ORANGE)
    card(c, 760, 141, 175, 129, "B3 / MEASURE", "Results", ["Setup + execution time", "Peak bin + estimated Hz", "CLI comparison table"], ORANGE)
    card(c, 955, 141, 178, 129, "B4 / REPRODUCE", "Benchmark runner", ["measure_whole_fft.py", "Release build + trials", "Machine + Git JSON"], ORANGE)
    for x1, x2 in [(525, 545), (740, 760), (935, 955)]:
        arrow(c, x1 + 2, 205, x2 - 3, 205, ORANGE)
    label(c, 350, 122, "Whole-file mode supports f32/f64; binary128 is currently framed CPU only.", 8.4, MUTED)

    c.setStrokeColor(LINE)
    c.line(48, 75, 1153, 75)
    label(c, 48, 50, "SOLID PATH", 8.5, TEAL, "Verdana-Bold")
    label(c, 123, 50, "default native CPU + standard CLI flow", 8.5, MUTED)
    label(c, 530, 50, "OPTIONAL", 8.5, ORANGE, "Verdana-Bold")
    label(c, 601, 50, "CUDA, Metal, binary128 and foreign FFT comparisons", 8.5, MUTED)
    label(c, 1052, 50, "01 / 01", 8.5, MUTED, "Verdana-Bold")
    c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    draw()
