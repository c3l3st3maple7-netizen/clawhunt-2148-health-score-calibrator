"""Render a text log as a terminal-style PNG (evidence screenshot for AC2).

Usage::

    python scripts/render_evidence.py --title "AC2 — annual churn rate" \
        --out deliver/ac2-statistics.png deliver/run1.log deliver/acceptance.log

This is a presentation helper only; it is not part of the generator and needs
Pillow (a test-time/extra dependency, not a runtime one).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = (
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
)

BACKGROUND = (15, 17, 21)
CHROME = (26, 29, 36)
TEXT = (214, 222, 235)
MUTED = (120, 130, 148)
GREEN = (126, 214, 143)
RED = (232, 120, 120)
YELLOW = (240, 200, 120)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("no monospace font found; edit FONT_PATHS")


def colour_for(line: str) -> tuple[int, int, int]:
    if "PASS" in line or "all acceptance criteria passed" in line:
        return GREEN
    if "FAIL" in line:
        return RED
    if line.startswith("$") or line.startswith("=="):
        return YELLOW
    if not line.strip():
        return MUTED
    return TEXT


def gather(paths: list[str], title: str) -> list[str]:
    lines = [f"$ {title}", ""]
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            lines.append(line.rstrip())
        lines.append("")
    return lines


def render(lines: list[str], title: str, out: str, size: int = 22) -> None:
    font = load_font(size)
    small = load_font(max(12, size - 6))
    line_height = int(size * 1.55)
    padding = 34
    width_chars = max(len(line) for line in lines) if lines else 60
    char_width = font.getlength("M")
    width = int(width_chars * char_width) + padding * 2
    chrome = 46
    height = chrome + padding + line_height * len(lines) + padding

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, chrome], fill=CHROME)
    for index, colour in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        draw.ellipse([18 + index * 26, 16, 34 + index * 26, 32], fill=colour)
    draw.text((width / 2, chrome / 2), title, font=small, fill=MUTED, anchor="mm")

    y = chrome + padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=colour_for(line))
        y += line_height

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"wrote {out} ({width}x{height})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="text logs to render, in order")
    parser.add_argument("--title", default="calib evidence")
    parser.add_argument("--out", default="deliver/evidence.png")
    parser.add_argument("--size", type=int, default=22)
    args = parser.parse_args(argv)
    render(gather(args.files, args.title), args.title, args.out, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
