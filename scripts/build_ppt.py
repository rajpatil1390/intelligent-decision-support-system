"""Build the first project deck (Step 12) -> docs/presentation/IDSS_Presentation_v1_backup.pptx (16:9), .pdf + PNGs.

Kept as a backup: the presentation itself is now built on the college template by scripts/build_ppt_v2.py (Step 12C).

The deck is generated with python-pptx: purple -> blue gradient title and section-divider slides, white content slides
with a thin gradient top bar, a footer and the slide number, the dashboard palette and Segoe UI. Numbers are never typed
in: evaluation results come from docs/results/evaluation.json (3-seed means), test counts and coverage from
docs/testing.md and docs/report/ch6_testing.md, and the few single-demo-run values (risk tiers, drivers, what-if on the
churn data, seed 42) must appear verbatim in docs/report/ch7_results.md. Charts are native PowerPoint charts; icons are
the poster's line icons rendered to PNG with Playwright Chromium; diagrams and screenshots come from docs/.

Speaker notes: each presenter gives one contiguous block of slides; the first line of every slide's notes is
"Presenter: <name>" and the script after it must be 60-120 words. The build prints each presenter's slide range and an
estimated speaking time. The last slide ("Live Demo Steps") is a hidden backup slide for the live demo.

Export: PowerPoint (COM) opens the deck, checks every text box (text must fit its box, stay on the slide and be at
least 18 pt for body text, 28 pt for titles, 14 pt for small notes and 10 pt for the footer), saves the PDF and exports
one PNG per slide to docs/presentation/slides/ for visual QA. Without PowerPoint, LibreOffice converts the PDF (no
layout check) and PyMuPDF renders the PNGs.

Usage: .venv\\Scripts\\python.exe scripts\\build_ppt.py [--no-export] [--draft]
Setup: pip install -r requirements-docs.txt; python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from lxml import etree
from PIL import Image
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from build_poster import trim_white
from report_common import ACADEMIC_YEAR, COLLEGE, DEPARTMENT, GUIDE, PROGRAMME, PROJECT_TITLE, ROOT, STUDENTS, \
    _find_soffice

OUT_DIR = ROOT / "docs" / "presentation"
PPTX_OUT = OUT_DIR / "IDSS_Presentation_v1_backup.pptx"
PDF_OUT = OUT_DIR / "IDSS_Presentation_v1_backup.pdf"
PNG_DIR = OUT_DIR / "slides" / "v1_backup"  # QA renders (gitignored)
EVALUATION = ROOT / "docs" / "results" / "evaluation.json"
TESTING = ROOT / "docs" / "testing.md"
CH6 = ROOT / "docs" / "report" / "ch6_testing.md"
CH7 = ROOT / "docs" / "report" / "ch7_results.md"
LOGO = ROOT / "ui" / "assets" / "idss_logo.svg"

# ---------------------------------------------------------------- design tokens

PURPLE, BLUE, GREEN, PINK, YELLOW, ORANGE = "6C5CE7", "0984E3", "00B894", "FD79A8", "FDCB6E", "E17055"
INK, MUTED, WHITE, PAGE = "2D3436", "5B6472", "FFFFFF", "F8F9FE"
ACCENT_ON_DARK = "FFE3A3"  # the poster's title accent on the gradient
BODY, SEMI, BLACK = "Segoe UI", "Segoe UI Semibold", "Segoe UI Black"
FOOTER = f"IDSS · {COLLEGE}"

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.6
CONTENT_W = SLIDE_W - 2 * MARGIN
TOP = 1.8  # first content row below the title and the subtitle

MIN_PT = {"Title": 28, "Footer": 10, "Small": 14}  # by shape-name prefix; everything else is body text (18 pt)
BODY_MIN_PT = 18
CHART_PT = 16

# speaker notes: one contiguous block of slides per presenter (roles as in the README)
RAJ, BHAVESH, CHAITANY, VARUN = (name for name, _ in STUDENTS)
NOTE_WORDS = (60, 120)  # words in each slide's script, after the "Presenter:" line
SPEAKING_WPM = 130  # for the time estimate
LIVE_DEMO_MINUTES = 2.0  # time-boxed live demo during the dashboard slide

# 24 x 24 line icons (same drawing style and most shapes as the poster)
ICONS = {
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
    "layers": '<path d="M12 3l9 4.5-9 4.5-9-4.5z"/><path d="M3 12l9 4.5 9-4.5"/><path d="M3 16.5L12 21l9-4.5"/>',
    "flow": '<circle cx="5" cy="6" r="2.5"/><circle cx="19" cy="18" r="2.5"/>'
            '<path d="M7.5 6H15a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h7.5"/>',
    "upload": '<path d="M12 15V4M7.5 8.5L12 4l4.5 4.5"/>'
              '<path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"/>',
    "chat": '<path d="M4 5h16v11H10l-6 4z"/><path d="M8 9h8M8 12.5h5"/>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="2.8"/><path d="M5 6v12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8V6"/>'
                '<path d="M5 12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8"/>',
    "sliders": '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    "shield": '<path d="M12 3l7 3v5.5c0 4.3-3 7.7-7 9.5-4-1.8-7-5.2-7-9.5V6z"/><path d="M9 12l2.2 2.2L15.5 10"/>',
    "bars": '<path d="M4 20h16"/><path d="M7 16v-5M12 16V5M17 16v-8"/>',
    "bulb": '<path d="M9 18h6M10 21h4"/>'
            '<path d="M12 3a6 6 0 0 0-3.8 10.6c.6.6.8 1.3.8 2.4h6c0-1.1.2-1.8.8-2.4A6 6 0 0 0 12 3z"/>',
    "doc": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
    "monitor": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    "trophy": '<path d="M8 21h8M12 17v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0z"/>'
              '<path d="M7 6H4.5a2.5 2.5 0 0 0 2.6 3.4M17 6h2.5a2.5 2.5 0 0 1-2.6 3.4"/>',
    "code": '<path d="M8 8l-4 4 4 4M16 8l4 4-4 4M14 5l-4 14"/>',
    "flag": '<path d="M5 21V4h11l-2 4 2 4H5"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "branch": '<path d="M6 3v5a4 4 0 0 0 4 4h8"/><path d="M6 21v-5a4 4 0 0 1 4-4"/><path d="M15 9l3 3-3 3"/>',
    "sparkle": '<path d="M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8z"/><path d="M5 18l2 2 4-4"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "box": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    "users": '<circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.5"/>'
             '<path d="M15.5 14.2c3 .2 5.5 2.6 5.5 5.8"/>',
    "cycle": '<path d="M20 11a8 8 0 0 0-14.3-4.9L4 8"/><path d="M4 3v5h5"/><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16"/>'
             '<path d="M20 21v-5h-5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "flask": '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3"/><path d="M7.5 15h9"/>',
    "server": '<rect x="4" y="4" width="16" height="7" rx="1.5"/><rect x="4" y="13" width="16" height="7" rx="1.5"/>'
              '<path d="M8 7.5h.01M8 16.5h.01"/>',
    "cluster": '<circle cx="7" cy="8" r="2.5"/><circle cx="16" cy="7" r="2.5"/><circle cx="10" cy="17" r="2.5"/>'
               '<circle cx="18" cy="16" r="2"/>',
    "tag": '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.5"/>',
    "truck": '<path d="M3 6h11v10H3zM14 10h4l3 3v3h-7"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/>',
    "cart": '<path d="M3 4h2l2.4 11h11L21 8H6.2"/><circle cx="9" cy="19.5" r="1.5"/><circle cx="17" cy="19.5" r="1.5"/>',
    "user_minus": '<circle cx="10" cy="8" r="3.5"/><path d="M3 20c0-3.6 3.1-6.5 7-6.5 1.6 0 3 .4 4.2 1.2"/>'
                  '<path d="M16 17.5h6"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1'
            'M4.9 19.1L7 17M17 7l2.1-2.1"/>',
}


def _guid(key: str) -> str:
    """A stable GUID per key, so rebuilding the deck does not change its XML."""
    return "{" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"idss-presentation/{key}")).upper() + "}"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def mix(color: str, other: str, share_other: float) -> str:
    """Blend two hex colours; share_other = weight of `other` (0..1)."""
    a = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(other[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x + (y - x) * share_other):02X}" for x, y in zip(a, b, strict=True))


def tint(color: str, white: float = 0.88) -> str:
    return mix(color, WHITE, white)


def shade(color: str, black: float = 0.25) -> str:
    return mix(color, "000000", black)


def on(color: str) -> str:
    """Text/icon colour on a filled shape: dark on yellow, white on the other palette colours."""
    return INK if color == YELLOW else WHITE


# ---------------------------------------------------------------- facts (all numbers on the slides)

def _match(pattern: str, text: str, source: Path) -> str:
    found = re.search(pattern, text)
    if not found:
        raise SystemExit(f"Pattern {pattern!r} not found in {source.relative_to(ROOT)}")
    return found.group(1)


def load_facts() -> dict[str, Any]:
    data = json.loads(EVALUATION.read_text(encoding="utf-8"))
    summary = data["summary"]
    bench = summary["benchmark"]
    ablation = {row["variant"]: row for row in summary["ablation"]}
    clean_single = [r for r in data["runs"] if r["variant"] == "single_model" and r["perturbation"] == "clean"]
    model_names = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}
    unverified = [(r["case"], r["seed"], model_names.get(r["best_model"], r["best_model"].replace("_", " ")))
                  for r in data["runs"] if r["variant"] == "full" and r["perturbation"] == "clean"
                  and not r["verification_passed"]]
    robust = summary["robustness"]
    perturbed = [row[k] for row in robust for k in ("missing", "noise")]

    testing = TESTING.read_text(encoding="utf-8")
    ch6 = CH6.read_text(encoding="utf-8")
    table = ch6.split("**Table 6.3:**", 1)[1].split("\n\n", 2)[1]
    coverage = []
    for line in table.splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0].startswith("**"):
            continue
        coverage.append((cells[0].strip("`"), float(cells[3].rstrip("%"))))

    # single demo run (seed 42, churn): each value must appear in Chapter 7 exactly as written there
    ch7 = re.sub(r"\s+", " ", CH7.read_text(encoding="utf-8"))
    demo = {"high": 900, "medium": 447, "low": 1653, "high_cut": "0.4609", "threshold": "0.32",
            "drivers": [("contract_type", 34), ("tenure_months", 20), ("payment_method", 13)],
            "whatif_before": "0.651", "whatif_after": "0.193"}
    for snippet in ("holds 900 customers (30.0%)", "447 (14.9%)", "1,653 (55.1%)", "at a probability of 0.4609",
                    "`contract_type` (34% of the explanation)", "`tenure_months` (20%)", "`payment_method` (13%)",
                    "from 0.651 to 0.193", "the same single demo run, seed 42"):
        if snippet not in ch7:
            raise SystemExit(f"Demo-run fact {snippet!r} not found in {CH7.relative_to(ROOT)}")

    return {
        "seeds": len(data["seeds"]),
        "bench": bench,
        "overall": summary["paper"]["Overall"],
        "verified": (sum(b["verified"] for b in bench), sum(b["runs"] for b in bench)),
        "beats": (sum(b["nps"][0] > b["baseline_nps"][0] for b in bench), len(bench)),
        "ablation": ablation,
        "single_verified": (sum(r["verification_passed"] for r in clean_single), len(clean_single)),
        "unverified": unverified,
        "robust": robust,
        "perturbed_verified": (sum(p["verified"] for p in perturbed), sum(p["runs"] for p in perturbed)),
        "tests": int(_match(r"\*\*(\d+) tests\*\*", testing, TESTING)),
        "tests_offline": int(_match(r"\*\*(\d+) run offline\*\*", testing, TESTING)),
        "tests_passed": int(_match(r"\*\*(\d+) passed, \d+ deselected\*\*", testing, TESTING)),
        "coverage_total": int(_match(r"\*\*Total: (\d+)%\*\*", testing, TESTING)),
        "coverage": coverage,
        "demo": demo,
    }


# ---------------------------------------------------------------- rendering helpers (icons, logo, images)

class Renderer:
    """Renders SVG icons and the logo to transparent PNGs with one Chromium page."""

    def __init__(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        self._page = self._browser.new_page()
        self._cache: dict[tuple[str, str], bytes] = {}

    def _svg(self, markup: str) -> bytes:
        self._page.set_content(f'<html><body style="margin:0;background:transparent">{markup}</body></html>')
        return self._page.locator("svg").first.screenshot(omit_background=True)

    def icon(self, name: str, color: str) -> bytes:
        if (name, color) not in self._cache:
            self._cache[(name, color)] = self._svg(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 24 24" fill="none" '
                f'stroke="#{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
        return self._cache[(name, color)]

    def logo(self) -> bytes:
        """The UI logo, clipped to its rounded badge (its decorative circles overhang) with a light outline."""
        svg = LOGO.read_text(encoding="utf-8").replace('width="280" height="72"', 'width="1120" height="288"')
        svg = svg.replace("</defs>", '<clipPath id="badge"><rect width="280" height="72" rx="18"/></clipPath></defs>'
                                     '<g clip-path="url(#badge)">', 1).replace("</svg>", "</g>"
                          '<rect x="1" y="1" width="278" height="70" rx="17" fill="none" stroke="#ffffff" '
                          'stroke-opacity="0.55" stroke-width="2"/></svg>')
        return self._svg(svg)

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()


def image_bytes(path: Path, crop: tuple[float, float, float, float] = (0, 0, 1, 1), *, trim: bool = False,
                aspect: float | None = None, max_width: int = 2400) -> tuple[BytesIO, float]:
    """Crop (fractions of the source), optionally trim white margins or cut to an aspect ratio from the top-left,
    downscale, and return the PNG and its width / height."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    left, top, right, bottom = crop
    image = image.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))
    if trim:
        image = trim_white(image, pad=round(0.01 * image.width))
    if aspect:
        image = image.crop((0, 0, image.width, min(image.height, round(image.width / aspect))))
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer, image.width / image.height


# ---------------------------------------------------------------- low-level shape helpers

Content = str | Sequence[Any]  # a paragraph (str or list of runs) or a list of paragraphs


def _set_alpha(color_parent: Any, alpha: float) -> None:
    clr = color_parent.find(qn("a:srgbClr"))
    etree.SubElement(clr, qn("a:alpha")).set("val", str(round(alpha * 100000)))


def fill_text(tf: Any, content: Content, *, size: float = BODY_MIN_PT, color: str = INK, font: str = BODY,
              bold: bool = False, align: PP_ALIGN = PP_ALIGN.LEFT, anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
              space_after: float = 0, line_spacing: float | None = None, bullets: str | None = None) -> None:
    """Paragraphs: a str, or a list of paragraphs; each paragraph is a str or a list of runs (str or (str, style))."""
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paragraphs = [content] if isinstance(content, str) else list(content)
    for index, paragraph in enumerate(paragraphs):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after and index < len(paragraphs) - 1:
            p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        if bullets:
            _bullet(p, bullets, size)
        for segment in [paragraph] if isinstance(paragraph, str) else paragraph:
            text, style = (segment, {}) if isinstance(segment, str) else segment
            run = p.add_run()
            run.text = text
            f = run.font
            f.size = Pt(style.get("size", size))
            f.bold = style.get("bold", bold)
            f.name = style.get("font", font)
            f.color.rgb = rgb(style.get("color", color))
            if "alpha" in style:
                _set_alpha(run._r.get_or_add_rPr().find(qn("a:solidFill")), style["alpha"])
            if "spacing" in style:
                run._r.get_or_add_rPr().set("spc", str(style["spacing"]))


def _bullet(paragraph: Any, color: str, size: float) -> None:
    indent = round(Pt(size) * 1.25)
    p_pr = paragraph._p.get_or_add_pPr()
    p_pr.set("marL", str(indent))
    p_pr.set("indent", str(-indent))
    etree.SubElement(p_pr, qn("a:buClr")).append(etree.fromstring(
        f'<a:srgbClr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{color}"/>'))
    etree.SubElement(p_pr, qn("a:buSzPct")).set("val", "110000")
    etree.SubElement(p_pr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(p_pr, qn("a:buChar")).set("char", "•")


def text(slide: Any, x: float, y: float, w: float, h: float, content: Content, *, name: str = "Body",
         **style: Any) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    fill_text(box.text_frame, content, **style)
    return box


def shape(slide: Any, x: float, y: float, w: float, h: float, fill: str | None = None, *,
          kind: MSO_SHAPE = MSO_SHAPE.RECTANGLE, radius: float | None = None, line: str | None = None,
          line_w: float = 1.0, shadow: bool = False, alpha: float | None = None,
          gradient: tuple[str, str, float] | None = None, name: str = "Decor") -> Any:
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    if gradient:
        start, end, angle = gradient
        s.fill.gradient()
        s.fill.gradient_angle = angle
        stops = s.fill.gradient_stops
        stops[0].color.rgb, stops[0].position = rgb(start), 0.0
        stops[1].color.rgb, stops[1].position = rgb(end), 1.0
    elif fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
        if alpha is not None:
            _set_alpha(s.fill._xPr.find(qn("a:solidFill")), alpha)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shadow:
        effects = s._element.spPr.find(qn("a:effectLst"))
        outer = etree.SubElement(effects, qn("a:outerShdw"), blurRad=str(Pt(14)), dist=str(Pt(3)), dir="5400000",
                                 algn="t", rotWithShape="0")
        etree.SubElement(etree.SubElement(outer, qn("a:srgbClr"), val=INK), qn("a:alpha"), val="13000")
    return s


def card(slide: Any, x: float, y: float, w: float, h: float, fill: str = WHITE, **kw: Any) -> Any:
    return shape(slide, x, y, w, h, fill, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16, shadow=True,
                 name=kw.pop("name", "Card"), **kw)


def pill(slide: Any, x: float, y: float, w: float, h: float, label: Content, fill: str, *, color: str = INK,
         size: float = BODY_MIN_PT, font: str = SEMI, line: str | None = None, name: str = "Pill") -> Any:
    s = shape(slide, x, y, w, h, fill, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=h / 2, line=line, name=name)
    fill_text(s.text_frame, label, size=size, color=color, font=font, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return s


def picture(slide: Any, data: BytesIO | bytes, x: float, y: float, w: float | None = None, h: float | None = None, *,
            name: str = "Picture", rounded: float | None = None, line: str | None = None) -> Any:
    stream = BytesIO(data) if isinstance(data, bytes) else data
    pic = slide.shapes.add_picture(stream, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    pic.name = name
    if rounded:
        pic.auto_shape_type = MSO_SHAPE.ROUNDED_RECTANGLE
        geom = pic._element.spPr.find(qn("a:prstGeom"))
        av = geom.find(qn("a:avLst"))
        if av is None:
            av = etree.SubElement(geom, qn("a:avLst"))
        smaller = min(pic.width, pic.height) / 914400
        etree.SubElement(av, qn("a:gd"), name="adj", fmla=f"val {round(min(0.5, rounded / smaller) * 100000)}")
    if line:
        pic.line.color.rgb = rgb(line)
        pic.line.width = Pt(1)
    return pic


def fit_picture(slide: Any, data: BytesIO, ratio: float, x: float, y: float, w: float, h: float, **kw: Any) -> Any:
    """Largest picture of the given aspect ratio inside the box, centred."""
    pw, ph = (w, w / ratio) if w / ratio <= h else (h * ratio, h)
    return picture(slide, data, x + (w - pw) / 2, y + (h - ph) / 2, pw, ph, **kw)


# ---------------------------------------------------------------- deck

class Deck:
    def __init__(self, facts: dict[str, Any], renderer: Renderer) -> None:
        self.facts = facts
        self.render = renderer
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
        self.layout = self.prs.slide_layouts[5]  # "Title Only": every slide has a real title placeholder
        self.sections: list[tuple[str, list[Any]]] = []
        self.speaker: str | None = None
        self.talk: list[dict[str, Any]] = []  # per slide: presenter, words, extra minutes, hidden
        self._apply_theme()
        props = self.prs.core_properties
        props.title, props.subject = PROJECT_TITLE, f"Project presentation, {COLLEGE}"
        props.author = ", ".join(name for name, _ in STUDENTS)

    def _apply_theme(self) -> None:
        """Palette and Segoe UI in the theme, so charts, new shapes and edits in PowerPoint pick them up."""
        part = self.prs.slide_master.part.part_related_by(RT.THEME)
        root = etree.fromstring(part.blob)
        a = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        scheme = root.find(".//a:clrScheme", a)
        scheme.set("name", "IDSS")
        for tag, value in {"dk2": INK, "lt2": PAGE, "accent1": PURPLE, "accent2": BLUE, "accent3": GREEN,
                           "accent4": PINK, "accent5": YELLOW, "accent6": ORANGE, "hlink": BLUE,
                           "folHlink": PURPLE}.items():
            slot = scheme.find(f"a:{tag}", a)
            for child in list(slot):
                slot.remove(child)
            etree.SubElement(slot, qn("a:srgbClr"), val=value)
        fonts = root.find(".//a:fontScheme", a)
        fonts.set("name", "IDSS")
        fonts.find("a:majorFont/a:latin", a).set("typeface", SEMI)
        fonts.find("a:minorFont/a:latin", a).set("typeface", BODY)
        part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    # ---- slide frames

    def section(self, name: str) -> None:
        self.sections.append((name, []))

    def presenter(self, name: str) -> None:
        """The presenter of the following slides; each presenter gets exactly one contiguous block."""
        if name not in (student for student, _ in STUDENTS) or any(s["presenter"] == name for s in self.talk):
            raise SystemExit(f"Presenter {name!r} is unknown or already had a block of slides")
        self.speaker = name

    def _new(self, notes: str, *, extra_minutes: float = 0.0, hidden: bool = False) -> Any:
        words = len(notes.split())
        if self.speaker is None or not NOTE_WORDS[0] <= words <= NOTE_WORDS[1]:
            raise SystemExit(f"Slide {len(self.prs.slides) + 1}: notes need a presenter and {NOTE_WORDS[0]}-"
                             f"{NOTE_WORDS[1]} words, got {words} ({notes[:50]!r})")
        slide = self.prs.slides.add_slide(self.layout)
        self.sections[-1][1].append(slide)
        slide.notes_slide.notes_text_frame.text = f"Presenter: {self.speaker}\n{notes}"
        if hidden:
            slide._element.set("show", "0")
        self.talk.append({"presenter": self.speaker, "words": words, "extra": extra_minutes, "hidden": hidden})
        return slide

    def timing(self) -> list[tuple[str, int, int, int, float]]:
        """Per presenter: name, first and last slide number, words and estimated minutes (hidden slides excluded)."""
        rows = []
        for name, _ in STUDENTS:
            numbers = [i for i, s in enumerate(self.talk, start=1) if s["presenter"] == name]
            shown = [self.talk[i - 1] for i in numbers if not self.talk[i - 1]["hidden"]]
            words = sum(s["words"] for s in shown)
            rows.append((name, numbers[0], numbers[-1], words,
                         words / SPEAKING_WPM + sum(s["extra"] for s in shown)))
        return sorted(rows, key=lambda row: row[1])  # speaking order

    def _title(self, slide: Any, title: str, x: float, y: float, w: float, h: float, **style: Any) -> None:
        ph = slide.shapes.title
        ph.left, ph.top, ph.width, ph.height = Inches(x), Inches(y), Inches(w), Inches(h)
        ph.name = "Title"
        fill_text(ph.text_frame, title, **style)

    def _footer(self, slide: Any, dark: bool) -> None:
        color = WHITE if dark else MUTED
        text(slide, MARGIN, 7.0, 9.5, 0.3, FOOTER, name="Footer", size=11, color=color, anchor=MSO_ANCHOR.MIDDLE)
        box = text(slide, SLIDE_W - MARGIN - 0.8, 7.0, 0.8, 0.3, [[]], name="Footer number", size=11,
                   align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        p = box.text_frame.paragraphs[0]._p
        number = len(self.prs.slides)
        fld = etree.Element(qn("a:fld"), id=_guid(f"slide-number-{number}"), type="slidenum")
        end = p.find(qn("a:endParaRPr"))
        if end is None:
            p.append(fld)
        else:
            end.addprevious(fld)
        r_pr = etree.SubElement(fld, qn("a:rPr"), lang="en-US", sz="1100", b="1")
        etree.SubElement(etree.SubElement(r_pr, qn("a:solidFill")), qn("a:srgbClr"), val=WHITE if dark else PURPLE)
        etree.SubElement(r_pr, qn("a:latin"), typeface=SEMI)
        etree.SubElement(fld, qn("a:t")).text = str(number)

    def _gradient_background(self, slide: Any, pink_x: float = 5.2) -> None:
        layers = [shape(slide, 0, 0, SLIDE_W, SLIDE_H, gradient=(PURPLE, BLUE, 330), name="Decor background"),
                  shape(slide, 9.6, -2.6, 5.6, 5.6, WHITE, kind=MSO_SHAPE.OVAL, alpha=0.10),
                  shape(slide, pink_x, 5.9, 3.6, 3.6, PINK, kind=MSO_SHAPE.OVAL, alpha=0.30),
                  shape(slide, -1.2, -1.4, 3.0, 3.0, WHITE, kind=MSO_SHAPE.OVAL, alpha=0.06)]
        tree = slide.shapes._spTree
        for index, layer in enumerate(layers):  # behind the title placeholder, which the layout puts first
            tree.insert(2 + index, layer._element)

    def content(self, title: str, subtitle: str | None, notes: str, **kw: Any) -> Any:
        slide = self._new(notes, **kw)
        shape(slide, 0, 0, SLIDE_W, 0.09, gradient=(PURPLE, BLUE, 0), name="Decor top bar")
        self._title(slide, title, MARGIN, 0.34, CONTENT_W, 0.72, size=32, font=BODY, bold=True, color=INK,
                    anchor=MSO_ANCHOR.MIDDLE)
        if subtitle:
            text(slide, MARGIN, 1.06, CONTENT_W, 0.42, subtitle, name="Subtitle", size=19, color=MUTED)
        self._footer(slide, dark=False)
        return slide

    def divider(self, number: int, title: str, subtitle: str, topics: list[str], notes: str) -> Any:
        slide = self._new(notes)
        self._gradient_background(slide)
        text(slide, 0.9, 1.05, 4.0, 1.9, f"{number:02d}", name="Stat number", size=110, font=BLACK, color=WHITE,
             anchor=MSO_ANCHOR.BOTTOM)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(WHITE)
        _set_alpha(slide.shapes[-1].text_frame.paragraphs[0].runs[0]._r.rPr.find(qn("a:solidFill")), 0.35)
        self._title(slide, title, 0.9, 3.05, 11.5, 0.95, size=48, font=BODY, bold=True, color=WHITE,
                    anchor=MSO_ANCHOR.MIDDLE)
        text(slide, 0.9, 4.0, 11.5, 0.5, subtitle, name="Subtitle", size=22, color=WHITE)
        x, y = 0.9, 4.95
        for topic in topics:
            width = 0.55 + 0.135 * len(topic)
            if x + width > SLIDE_W - MARGIN:
                x, y = 0.9, y + 0.72
            pill(slide, x, y, width, 0.55, topic, WHITE, color=shade(PURPLE, 0.15), name="Pill topic")
            x += width + 0.2
        self._footer(slide, dark=True)
        return slide

    def icon(self, slide: Any, name: str, x: float, y: float, d: float, color: str, *,
             kind: MSO_SHAPE = MSO_SHAPE.OVAL) -> None:
        shape(slide, x, y, d, d, color, kind=kind, radius=d * 0.28, name="Icon background")
        pad = d * 0.22
        picture(slide, self.render.icon(name, on(color)), x + pad, y + pad, d - 2 * pad, d - 2 * pad, name="Icon")

    def finish(self) -> None:
        """Write the PowerPoint sections (p14:sectionLst) so the deck opens organised by section."""
        absent = [name for name, _ in STUDENTS if all(s["presenter"] != name for s in self.talk)]
        if absent:
            raise SystemExit(f"No slides for {', '.join(absent)}")
        p14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
        presentation = self.prs.part._element
        ids = {slide.slide_id: slide for slide in self.prs.slides}
        section_list = etree.Element(f"{{{p14}}}sectionLst", nsmap={"p14": p14})
        for name, slides in self.sections:
            sec = etree.SubElement(section_list, f"{{{p14}}}section", name=name,
                                   id=_guid(f"section-{name}"))
            id_list = etree.SubElement(sec, f"{{{p14}}}sldIdLst")
            for slide in slides:
                assert slide.slide_id in ids
                etree.SubElement(id_list, f"{{{p14}}}sldId", id=str(slide.slide_id))
        ext_list = presentation.find(qn("p:extLst"))
        if ext_list is None:
            ext_list = etree.SubElement(presentation, qn("p:extLst"))
        ext = etree.SubElement(ext_list, qn("p:ext"), uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
        ext.append(section_list)


# ---------------------------------------------------------------- charts

def style_chart(chart: Any, *, legend: bool = False) -> None:
    chart.font.name = BODY
    chart.font.size = Pt(CHART_PT)
    chart.font.color.rgb = rgb(INK)
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(CHART_PT)
    # no chart-area fill or border
    space = chart._chartSpace
    sp_pr = etree.fromstring('<c:spPr xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                             'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                             '<a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>')
    space.find(qn("c:chart")).addnext(sp_pr)


def quiet_axes(chart: Any, *, value_max: float | None = None, value_min: float | None = None,
               category_line: bool = False) -> None:
    value = chart.value_axis
    value.visible = False
    value.has_major_gridlines = False
    value.has_minor_gridlines = False
    if value_max is not None:
        value.maximum_scale = value_max
    if value_min is not None:
        value.minimum_scale = value_min
    category = chart.category_axis
    category.major_tick_mark = XL_TICK_MARK.NONE
    category.has_major_gridlines = False
    category.tick_labels.font.size = Pt(CHART_PT)
    if category_line:
        category.format.line.color.rgb = rgb("B8BED3")
        category.format.line.width = Pt(1)
    else:
        category.format.line.fill.background()


def bar_labels(plot: Any, number_format: str, *, position: XL_LABEL_POSITION = XL_LABEL_POSITION.OUTSIDE_END,
               bold: bool = True) -> None:
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = number_format
    labels.number_format_is_linked = False
    labels.position = position
    labels.font.size = Pt(CHART_PT)
    labels.font.bold = bold
    labels.font.color.rgb = rgb(INK)


def color_series(series: Any, color: str) -> None:
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = rgb(color)
    series.invert_if_negative = False


# ---------------------------------------------------------------- slides

def slide_title(deck: Deck) -> None:
    slide = deck._new(
        f"Good morning. We are Raj, Bhavesh, Chaitany and Varun, and our guide is {GUIDE}. Our project is IDSS, the "
        "Intelligent Decision Support System. A user types a business question in plain language and uploads a data "
        "table; IDSS returns a verified model with explained decisions and a PDF report. I will cover the problem, our "
        "objectives and the agent design.")
    deck._gradient_background(slide)
    picture(slide, deck.render.logo(), 0.8, 0.55, 2.6, 2.6 * 72 / 280, name="Logo")
    text(slide, 4.2, 0.5, 8.33, 0.45, COLLEGE, name="Header college", size=20, font=SEMI, color=WHITE,
         align=PP_ALIGN.RIGHT)
    text(slide, 4.2, 0.98, 8.33, 0.4, [[(DEPARTMENT.upper(), {"spacing": 200})]], name="Header department",
         size=18, color=WHITE, align=PP_ALIGN.RIGHT)
    first, rest = PROJECT_TITLE.split(" Using ", 1)
    accent = {"color": ACCENT_ON_DARK}
    deck._title(slide, [first, ["Using ", ("Generative AI", accent), " and"], [("Machine Learning", accent)]],
                0.8, 1.85, 11.7, 2.25, size=42, font=BODY, bold=True, color=WHITE, line_spacing=0.95)
    assert rest == "Generative AI and Machine Learning"
    text(slide, 0.8, 4.18, 11.7, 0.45, f"Final-year project · B.Tech {PROGRAMME}", name="Subtitle", size=20,
         color=WHITE)
    text(slide, 0.8, 4.85, 6.8, 0.4, [[("PRESENTED BY", {"spacing": 200})]], name="Label", size=18, font=SEMI,
         color=WHITE)
    for i, (name, _) in enumerate(STUDENTS):
        pill(slide, 0.8 + (i % 2) * 3.55, 5.35 + (i // 2) * 0.7, 3.35, 0.55, name, WHITE,
             color=shade(PURPLE, 0.2), name="Pill student")
    shape(slide, 8.55, 4.85, 3.98, 1.75, WHITE, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18, alpha=0.15,
          line=WHITE, name="Card guide")
    text(slide, 8.75, 5.0, 3.58, 1.45,
         [[("PROJECT GUIDE", {"spacing": 150, "font": SEMI})], [(GUIDE, {"size": 24, "bold": True})],
          f"Academic Year {ACADEMIC_YEAR}"],
         name="Guide", size=18, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=2)


def slide_agenda(deck: Deck) -> None:
    slide = deck.content("Agenda", "Five parts, four presenters and a live demo",
                         "The talk has five parts and a short live demo. I present the problem, our objectives, the "
                         "agent pipeline and our safety rules. Chaitany then covers the architecture, the backend and "
                         "the testing. Bhavesh explains the AutoML engine and the evaluation results. Varun shows the "
                         "dashboard live and gives the conclusion. All four of us will answer your questions at the "
                         "end.")
    items = [("Introduction", "Problem and objectives", "target", PURPLE),
             ("System design", "Agents, safeguards and architecture", "layers", BLUE),
             ("Implementation", "Technology stack and testing", "code", GREEN),
             ("AutoML and results", "Engine, decisions and evaluation", "trophy", ORANGE),
             ("Demo and conclusion", "Live dashboard and future scope", "flag", PINK)]
    h, gap = 0.82, 0.18
    for i, (title, desc, icon, color) in enumerate(items):
        y = TOP - 0.05 + i * (h + gap)
        card(slide, MARGIN, y, CONTENT_W, h)
        text(slide, MARGIN + 0.3, y, 1.0, h, f"{i + 1:02d}", name="Stat number", size=30, font=BLACK,
             color=shade(color, 0.1), anchor=MSO_ANCHOR.MIDDLE)
        deck.icon(slide, icon, MARGIN + 1.35, y + 0.13, 0.56, color)
        text(slide, MARGIN + 2.2, y, 3.6, h, title, name="Heading", size=22, font=SEMI, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, MARGIN + 5.9, y, CONTENT_W - 6.2, h, desc, size=19, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)


def slide_problem(deck: Deck) -> None:
    slide = deck.content("Problem statement", "Teams have questions and data, but no data scientists",
                         "We saw three problems. First, small businesses have data and questions but rarely a data "
                         "scientist to build and validate a model. Second, AutoML tools stop at a score, and a score "
                         "does not tell a manager what to do. Third, LLMs are weak with numbers: LLM-written training "
                         "code can fail, leak test data or report numbers that were never computed. So our idea: LLM "
                         "agents plan and explain; tested code computes every number.")
    pains = [("No data scientists", "Business teams cannot build, tune and validate models themselves", "users",
              ORANGE),
             ("AutoML stops at a model", "A score is not a decision: no reasons, no what-if, no actions", "bars",
              PINK),
             ("LLM code can invent numbers", "Scripts written by an LLM may fail or report unchecked results",
              "code", YELLOW)]
    gap, w = 0.35, (CONTENT_W - 2 * 0.35) / 3
    for i, (title, desc, icon, color) in enumerate(pains):
        x = MARGIN + i * (w + gap)
        card(slide, x, TOP + 0.05, w, 3.25)
        deck.icon(slide, icon, x + 0.35, TOP + 0.3, 0.8, color)
        text(slide, x + 0.35, TOP + 1.2, w - 0.6, 0.8, title, name="Heading", size=22, font=SEMI,
             anchor=MSO_ANCHOR.BOTTOM)
        text(slide, x + 0.35, TOP + 2.1, w - 0.6, 1.1, desc, size=18, color=MUTED)
    shape(slide, MARGIN, 5.4, CONTENT_W, 1.25, tint(PURPLE, 0.9), kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16,
          name="Callout")
    deck.icon(slide, "bulb", MARGIN + 0.35, 5.6, 0.85, PURPLE)
    text(slide, MARGIN + 1.5, 5.4, CONTENT_W - 1.9, 1.25,
         [[("Our idea: ", {"bold": True, "color": shade(PURPLE, 0.2)}),
           "LLM agents plan and explain, while tested code computes every number"]],
         size=24, anchor=MSO_ANCHOR.MIDDLE)


def slide_objectives(deck: Deck) -> None:
    slide = deck.content("Objectives", "Eight objectives, all met (Chapter 8)",
                         "We set eight objectives: a multi-agent design with an offline fallback for every agent; "
                         "plain-language requirements; AutoML within time budgets; verification of every model; "
                         "explained, actionable decisions; a REST API and a web interface; privacy and safety by "
                         "design; and an honest evaluation with automated tests. Chapter 8 of our report shows that all "
                         "eight were met. Next, the system design that meets them.")
    goals = [("Agents with offline fallbacks", "layers", PURPLE),
             ("Plain-language requirements", "chat", BLUE),
             ("AutoML within budgets", "sliders", ORANGE),
             ("Verify every model", "shield", GREEN),
             ("Explained, actionable decisions", "bulb", YELLOW),
             ("REST API and web interface", "monitor", PINK),
             ("Private and safe by design", "lock", BLUE),
             ("Honest evaluation and tests", "trophy", PURPLE)]
    gap, w, h = 0.3, (CONTENT_W - 3 * 0.3) / 4, 2.3
    for i, (title, icon, color) in enumerate(goals):
        x, y = MARGIN + (i % 4) * (w + gap), TOP + 0.05 + (i // 4) * (h + 0.25)
        card(slide, x, y, w, h)
        deck.icon(slide, icon, x + 0.3, y + 0.3, 0.72, color)
        text(slide, x + w - 0.75, y + 0.3, 0.45, 0.45, str(i + 1), name="Stat number", size=20, font=BLACK,
             color="C3C8DA", align=PP_ALIGN.RIGHT)
        text(slide, x + 0.3, y + 1.15, w - 0.5, 1.05, title, name="Heading", size=20, font=SEMI)


def slide_architecture(deck: Deck) -> None:
    slide = deck.content("System architecture", "Six layers: LLM agents plan and explain, deterministic engines "
                         "compute", "Thank you, Raj. The system has six layers. At the top, the presentation layer is the "
                         "Streamlit web interface. Below it, the service layer is a FastAPI backend. The orchestration "
                         "layer is the state machine that drives each run. The agent layer holds seven agents, each with "
                         "one job and an offline fallback. The engine layer does the real computation: data "
                         "preparation, AutoML, explanations and reports. At the bottom, the infrastructure layer stores "
                         "runs on disk and in SQLite. The LLM provider and web search are optional external services, "
                         "so the whole pipeline also runs offline.")
    data, ratio = image_bytes(ROOT / "docs" / "diagrams" / "system_architecture.png", (0, 0.105, 1, 0.9445),
                              trim=True)
    fit_picture(slide, data, ratio, MARGIN, TOP - 0.05, 8.55, 5.0, name="Picture architecture")
    stats = [("6", "layers", "web UI down to storage", PURPLE),
             ("7", "agents", "each with an offline fallback", BLUE),
             ("13", "states", "resumable state machine", GREEN)]
    for i, (number, label, desc, color) in enumerate(stats):
        y = TOP + 0.05 + i * 1.68
        card(slide, 9.45, y, 3.28, 1.45)
        text(slide, 9.7, y + 0.12, 1.25, 1.2, number, name="Stat number", size=44, font=BLACK, color=shade(color, 0.1),
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, 10.9, y + 0.17, 1.75, 1.15, [[(label, {"font": SEMI, "size": 22})], desc], size=18,
             color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(INK)


def slide_methodology(deck: Deck) -> None:
    slide = deck.content("Methodology", "One run: eight steps from upload to report",
                         "One run has eight steps. The Requirement Agent checks that the question fits the data, turns "
                         "it into a typed requirement and asks clarifying questions when needed. The Data Agent "
                         "profiles, cleans and splits the data 70/20/10. The Model Agent ranks models from a fixed "
                         "list, using best-practice notes from the Knowledge Agent, and the AutoML engine trains them. "
                         "The Verifier runs seven checks. SHAP and what-if analysis explain the model, and the Insight "
                         "Agent writes a grounded brief for the report.")
    steps = [("Upload", "CSV, Excel, JSON or Parquet", "upload", PURPLE),
             ("Requirement Agent", "question → validated requirement", "chat", BLUE),
             ("Data Agent", "profile, clean, split 70/20/10", "database", GREEN),
             ("AutoML", "model search + Optuna tuning", "sliders", ORANGE),
             ("Verifier", "7 checks, one re-plan", "shield", PINK),
             ("SHAP + What-if", "drivers, scenarios, risk tiers", "bars", YELLOW),
             ("Insight Agent", "grounded plain-language brief", "bulb", PURPLE),
             ("Report", "PDF + Markdown in the web UI", "doc", BLUE)]
    gap, w, h = 0.42, (CONTENT_W - 3 * 0.42) / 4, 2.32
    for i, (title, desc, icon, color) in enumerate(steps):
        col, row = i % 4, i // 4
        x, y = MARGIN + col * (w + gap), TOP + 0.05 + row * (h + 0.22)
        card(slide, x, y, w, h)
        deck.icon(slide, icon, x + 0.25, y + 0.22, 0.56, color)
        text(slide, x + w - 0.85, y + 0.16, 0.6, 0.6, str(i + 1), name="Stat number", size=28, font=BLACK,
             color=shade(color, 0.1) if color != YELLOW else shade(YELLOW, 0.35), align=PP_ALIGN.RIGHT)
        # titles end on the same line in every card, whether they take one line or two
        text(slide, x + 0.25, y + 0.82, w - 0.4, 0.7, title, name="Heading", size=20, font=SEMI,
             anchor=MSO_ANCHOR.BOTTOM)
        text(slide, x + 0.25, y + 1.54, w - 0.4, 0.64, desc, size=18, color=MUTED)
        if col < 3:
            shape(slide, x + w + 0.1, y + h / 2 - 0.16, 0.22, 0.32, "C3C8DA", kind=MSO_SHAPE.CHEVRON,
                  name="Decor arrow")


def slide_orchestrator(deck: Deck) -> None:
    slide = deck.content("Resumable orchestrator", "A state machine coordinates the agents and never crashes",
                         "The Orchestrator is a state machine with 13 states: INIT, then parse, validate, profile, plan, "
                         "train, verify, explain, insight and report, then DONE, plus NEEDS_INPUT and FAILED. After "
                         "every transition the run is saved to a state.json file and mirrored as an event in the SQLite "
                         "database, so a run can resume after a restart. If the request is incomplete, the run waits in "
                         "NEEDS_INPUT with clarifying questions, for at most two rounds. If verification fails, it "
                         "re-plans once with the full budget. Any error moves the run to FAILED with a message, so the "
                         "server never crashes.")
    data, ratio = image_bytes(ROOT / "docs" / "diagrams" / "agent_workflow.png", (0, 0.12, 1, 1), trim=True)
    fit_picture(slide, data, ratio, MARGIN, TOP - 0.05, 7.6, 4.95, name="Picture state machine")
    card(slide, 8.5, TOP + 0.05, 4.23, 4.85)
    points = ["13 states, saved after every transition: runs resume",
              "Clarifying questions when the request is incomplete (at most 2 rounds)",
              "Failed verification: one re-plan with the full budget",
              "Any error ends in FAILED with a message, never a crash"]
    text(slide, 8.8, TOP + 0.35, 3.7, 4.3, points, size=19, bullets=PURPLE, space_after=12)


def slide_engine(deck: Deck) -> None:
    slide = deck.content("AutoML engine and verification", "The LLM ranks models from a fixed zoo; tested code "
                         "trains, tunes and checks them",
                         "The LLM never writes training code; it only ranks model names from a fixed zoo for four task "
                         "families. The engine tunes each candidate with Optuna within a time budget: the fast budget "
                         "is three models with fifteen trials each. For binary classification it also tunes the "
                         "decision threshold on the validation split. The test split is read exactly once, and that is "
                         "logged. The Verifier's seven rule-based checks are on the right.")
    left_w = 5.75
    card(slide, MARGIN, TOP + 0.05, left_w, 4.85)
    deck.icon(slide, "sliders", MARGIN + 0.3, TOP + 0.3, 0.7, ORANGE)
    text(slide, MARGIN + 1.15, TOP + 0.3, left_w - 1.4, 0.7, "AutoML engine", name="Heading", size=24, font=SEMI,
         anchor=MSO_ANCHOR.MIDDLE)
    families = [("Classification", PURPLE), ("Regression", BLUE), ("Clustering", GREEN), ("Time series", ORANGE)]
    for i, (label, color) in enumerate(families):
        pill(slide, MARGIN + 0.3 + (i % 2) * 2.62, TOP + 1.25 + (i // 2) * 0.66, 2.45, 0.52, label,
             tint(color, 0.85), color=shade(color, 0.3), name="Pill family")
    text(slide, MARGIN + 0.3, TOP + 2.75, left_w - 0.6, 2.0,
         ["Optuna TPE search with pruning, within budgets", "Decision threshold tuned on the validation split",
          "Test split read exactly once (audited)"], size=18, bullets=ORANGE, space_after=8)
    x2 = MARGIN + left_w + 0.35
    w2 = SLIDE_W - MARGIN - x2
    card(slide, x2, TOP + 0.05, w2, 4.85)
    deck.icon(slide, "shield", x2 + 0.3, TOP + 0.3, 0.7, GREEN)
    text(slide, x2 + 1.15, TOP + 0.3, w2 - 1.4, 0.7, "Verifier: 7 rule-based checks", name="Heading", size=24,
         font=SEMI, anchor=MSO_ANCHOR.MIDDLE)
    checks = ["Metric targets met", "Beats a trivial baseline", "Latency, time and explainability limits",
              "At least 30 test rows", "Test split used once", "No leakage warnings", "Train–test gap below 0.15"]
    for i, check in enumerate(checks):
        y = TOP + 1.22 + i * 0.52
        deck.icon(slide, "check", x2 + 0.35, y + 0.06, 0.36, GREEN)
        text(slide, x2 + 0.9, y, w2 - 1.2, 0.48, check, size=18, anchor=MSO_ANCHOR.MIDDLE)


def slide_decisions(deck: Deck) -> None:
    demo = deck.facts["demo"]
    tiers = [("High", demo["high"], ORANGE), ("Medium", demo["medium"], YELLOW), ("Low", demo["low"], GREEN)]
    total = sum(n for _, n, _ in tiers)
    (d1, s1), (d2, s2), (d3, s3) = ((d.replace("_months", "").replace("_", " "), s) for d, s in demo["drivers"])
    slide = deck.content("From model to decisions", f"Churn example: single demo run (seed 42), all {total:,} "
                         "customers scored",
                         f"This is a single demo run on churn with seed 42. All {total:,} customers are scored and put "
                         f"into risk tiers: {demo['high']:,} high, {demo['medium']:,} medium and {demo['low']:,} low. "
                         f"High risk starts at {demo['high_cut']}, the larger of the tuned threshold "
                         f"{demo['threshold']} and the 70th percentile. SHAP shows the drivers: {d1} accounts for "
                         f"{s1}%, {d2} {s2}% and {d3} {s3}%. Moving high-risk customers to a two-year contract lowers "
                         f"their average predicted churn from {demo['whatif_before']} to {demo['whatif_after']}, "
                         "according to the model, not as a guaranteed effect.")
    w, gap = (CONTENT_W - 2 * 0.35) / 3, 0.35
    # risk tiers: doughnut + own legend
    x = MARGIN
    card(slide, x, TOP + 0.05, w, 4.85)
    text(slide, x + 0.3, TOP + 0.25, w - 0.6, 0.45, "Risk tiers", name="Heading", size=22, font=SEMI)
    data = CategoryChartData()
    data.categories = [t for t, _, _ in tiers]
    data.add_series("Customers", [n for _, n, _ in tiers])
    donut = 2.3
    frame = slide.shapes.add_chart(XL_CHART_TYPE.DOUGHNUT, Inches(x + (w - donut) / 2), Inches(TOP + 0.72),
                                   Inches(donut), Inches(donut), data)
    frame.name = "Chart risk tiers"
    chart = frame.chart
    style_chart(chart)
    plot = chart.plots[0]
    hole = plot._element.find(qn("c:holeSize"))
    if hole is not None:
        hole.set("val", "58")
    for point, (_, _, color) in zip(plot.series[0].points, tiers, strict=True):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = rgb(color)
        point.format.line.color.rgb = rgb(WHITE)
    text(slide, x + w / 2 - 0.8, TOP + 0.72 + donut / 2 - 0.27, 1.6, 0.54, f"{total:,}", name="Stat number", size=22,
         font=BLACK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    cutoffs = [f"≥ {demo['high_cut']}", f"≥ {demo['threshold']}", f"< {demo['threshold']}"]
    for i, ((label, n, color), cutoff) in enumerate(zip(tiers, cutoffs, strict=True)):
        y = TOP + 3.15 + i * 0.5
        shape(slide, x + 0.35, y + 0.12, 0.24, 0.24, color, kind=MSO_SHAPE.OVAL, name="Decor legend")
        text(slide, x + 0.75, y, 2.0, 0.48, [[(label, {"font": SEMI}), (f"  {cutoff}", {"color": MUTED})]],
             size=18, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + w - 1.25, y, 0.95, 0.48, f"{n:,}", size=18, font=SEMI, align=PP_ALIGN.RIGHT,
             anchor=MSO_ANCHOR.MIDDLE)
    # drivers: bar chart
    x = MARGIN + w + gap
    card(slide, x, TOP + 0.05, w, 4.85)
    text(slide, x + 0.3, TOP + 0.25, w - 0.6, 0.45, "Key drivers (SHAP)", name="Heading", size=22, font=SEMI)
    data = CategoryChartData()
    data.categories = [d.replace("_", " ").capitalize().replace(" months", " (months)") for d, _ in demo["drivers"]]
    data.add_series("Share of the explanation", [s / 100 for _, s in demo["drivers"]])
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x + 0.15), Inches(TOP + 0.8),
                                   Inches(w - 0.3), Inches(2.55), data)
    frame.name = "Chart drivers"
    chart = frame.chart
    style_chart(chart)
    quiet_axes(chart, value_max=0.5, value_min=0)
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 55
    color_series(plot.series[0], BLUE)
    bar_labels(plot, "0%")
    text(slide, x + 0.3, TOP + 3.5, w - 0.6, 1.2, "Share of the explanation: contract type alone is a third",
         size=18, color=MUTED)
    # what-if
    x = MARGIN + 2 * (w + gap)
    card(slide, x, TOP + 0.05, w, 4.85)
    text(slide, x + 0.3, TOP + 0.25, w - 0.6, 0.45, "What-if", name="Heading", size=22, font=SEMI)
    text(slide, x + 0.3, TOP + 0.8, w - 0.6, 0.95, "High-risk customers move to a two-year contract", size=18,
         color=MUTED)
    box_w = 1.45
    for i, (value, label, color) in enumerate(((demo["whatif_before"], "now", ORANGE),
                                               (demo["whatif_after"], "after", GREEN))):
        bx = x + 0.2 if i == 0 else x + w - 0.2 - box_w
        text(slide, bx, TOP + 1.85, box_w, 0.8, value, name="Stat number", size=32, font=BLACK,
             color=shade(color, 0.15), align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, bx, TOP + 2.68, box_w, 0.4, label, size=18, color=MUTED, align=PP_ALIGN.CENTER)
    shape(slide, x + w / 2 - 0.15, TOP + 2.06, 0.3, 0.38, "C3C8DA", kind=MSO_SHAPE.CHEVRON, name="Decor arrow")
    text(slide, x + 0.3, TOP + 3.3, w - 0.6, 1.4, "Average predicted churn probability (model estimate)", size=18,
         color=MUTED)


def slide_trust(deck: Deck) -> None:
    slide = deck.content("Trustworthy by design", "The LLM writes words; code computes every number",
                         "An LLM can invent numbers, so we use four guards. One: the Insight Agent sees only computed "
                         "facts, and any number in its text that matches no fact is rejected; after one retry a "
                         "template is used. Two: the only LLM-written code, in Ask Your Data, runs in a sandbox. Three: "
                         "the LLM gets aggregates, never raw rows, and keys stay in the .env file. Four: every LLM call "
                         "has a timeout and at most three retries. Over to Chaitany.")
    items = [("Grounded insights", "Every number in the brief must match a computed fact, else the template is used",
              "sparkle", PURPLE),
             ("Sandboxed analysis", "LLM-written code runs in an isolated child process: import allowlist, "
                                    "no secrets, timeout", "box", BLUE),
             ("Private by default", "Only aggregates reach the LLM, never raw rows; API keys stay in .env",
              "lock", GREEN),
             ("Bounded and audited", "At most 3 retries with timeouts; the test split is read exactly once",
              "clock", ORANGE)]
    gap, w, h = 0.35, (CONTENT_W - 0.35) / 2, 2.25
    for i, (title, desc, icon, color) in enumerate(items):
        x, y = MARGIN + (i % 2) * (w + gap), TOP + 0.05 + (i // 2) * (h + 0.35)
        card(slide, x, y, w, h)
        deck.icon(slide, icon, x + 0.35, y + 0.35, 0.9, color)
        text(slide, x + 1.55, y + 0.3, w - 1.85, 0.5, title, name="Heading", size=22, font=SEMI)
        text(slide, x + 1.55, y + 0.85, w - 1.85, 1.25, desc, size=18, color=MUTED)


def slide_stack(deck: Deck) -> None:
    slide = deck.content("Technology stack", "Python 3.12 and open-source libraries; runs on a laptop",
                         "The ML engine uses scikit-learn, XGBoost and LightGBM, with Optuna for tuning and SHAP for "
                         "explanations. The LLM can be any OpenAI-compatible endpoint, Pydantic v2 validates every "
                         "structured message, and retrieval uses a TF-IDF index over ten best-practice notes. The "
                         "backend is FastAPI with 17 endpoints for datasets, runs, results, what-if, questions and "
                         "reports; one background worker runs one training job at a time. SQLAlchemy stores datasets, "
                         "runs and events in SQLite. Errors always come back in one format, an error code and a "
                         "detail, and a second training request while one is running gets a clear busy reply. The "
                         "interface is Streamlit with Plotly, and reportlab builds the PDF. Every library is pinned in "
                         "requirements.txt.")
    columns = [("ML engine", "sliders", ORANGE, ["scikit-learn", "XGBoost", "LightGBM", "Optuna", "SHAP"]),
               ("GenAI and knowledge", "sparkle", PURPLE,
                ["LLM (OpenAI API)", "Pydantic v2", "TF-IDF retrieval", "Code sandbox"]),
               ("Backend and data", "server", BLUE, ["FastAPI", "SQLAlchemy 2.0", "SQLite", "pandas"]),
               ("Interface and reports", "monitor", GREEN, ["Streamlit", "Plotly", "reportlab", "pytest"])]
    gap, w = 0.3, (CONTENT_W - 3 * 0.3) / 4
    for i, (title, icon, color, chips) in enumerate(columns):
        x = MARGIN + i * (w + gap)
        card(slide, x, TOP + 0.05, w, 4.85)
        deck.icon(slide, icon, x + 0.3, TOP + 0.3, 0.7, color)
        text(slide, x + 0.3, TOP + 1.12, w - 0.6, 0.75, title, name="Heading", size=20, font=SEMI)
        for j, chip in enumerate(chips):
            pill(slide, x + 0.3, TOP + 1.95 + j * 0.58, w - 0.6, 0.48, chip, tint(color, 0.87),
                 color=shade(color, 0.35) if color != YELLOW else INK, name="Pill tech")


def slide_dashboard(deck: Deck) -> None:
    slide = deck.content("Web dashboard", "Streamlit UI (8 pages) on a FastAPI backend (17 endpoints)",
                         "The interface has eight Streamlit pages and talks to the backend only through the API. Here "
                         "you see Home with Demo Mode, Results with the leaderboard and checks, and the Decision Center "
                         "with the brief and risk tiers. Now I switch to the browser. Demo Mode uploads the churn "
                         "sample and starts a fast run of about one to three minutes, and the Run Monitor follows it "
                         "live.",
                         extra_minutes=LIVE_DEMO_MINUTES)
    shots = [("Home", "01_home.png", (370 / 1600, 12 / 1008, 1530 / 1600, 1), "Demo Mode and KPIs"),
             ("Results", "05_results.png", (370 / 1600, 12 / 1306, 1530 / 1600, 1), "Leaderboard and checks"),
             ("Decision Center", "06_decision_center.png", (370 / 1600, 318 / 2412, 1530 / 1600, 1),
              "Brief and risk tiers")]
    gap, w = 0.35, (CONTENT_W - 2 * 0.35) / 3
    bar_h, img_h = 0.42, 3.55
    for i, (page, file, crop, caption) in enumerate(shots):
        x = MARGIN + i * (w + gap)
        shape(slide, x, TOP + 0.05, w, bar_h + img_h, WHITE, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12,
              line="D6DAF0", shadow=True, name="Frame")
        shape(slide, x + 0.02, TOP + 0.07, w - 0.04, bar_h - 0.02, "E9ECFB", kind=MSO_SHAPE.ROUNDED_RECTANGLE,
              radius=0.1, name="Decor browser bar")
        for k, dot in enumerate((ORANGE, YELLOW, GREEN)):
            shape(slide, x + 0.18 + k * 0.2, TOP + 0.05 + bar_h / 2 - 0.06, 0.12, 0.12, dot, kind=MSO_SHAPE.OVAL)
        text(slide, x + 0.85, TOP + 0.05, w - 1.0, bar_h, page, name="Heading", size=18, font=SEMI,
             anchor=MSO_ANCHOR.MIDDLE)
        data, ratio = image_bytes(ROOT / "docs" / "screenshots" / file, crop, aspect=(w - 0.04) / img_h,
                                  max_width=1600)
        picture(slide, data, x + 0.02, TOP + 0.05 + bar_h, w - 0.04, img_h - 0.02, name="Picture screenshot")
        text(slide, x, TOP + 0.25 + bar_h + img_h, w, 0.45, caption, size=18, color=MUTED, align=PP_ALIGN.CENTER)


def slide_testing(deck: Deck) -> None:
    f = deck.facts
    slide = deck.content("Testing and quality", f"{f['tests']} automated tests; all {f['tests_passed']} offline "
                         f"tests pass; {f['coverage_total']}% code coverage",
                         f"We wrote {f['tests']} automated tests. {f['tests_offline']} of them run offline and all "
                         f"{f['tests_passed']} pass; the other {f['tests'] - f['tests_offline']} are live LLM tests that "
                         "run only on request. In the tests the LLM is replaced by a fake client and the workspace by a "
                         "temporary folder, so they never touch the network. Statement coverage is "
                         f"{f['coverage_total']}%. The chart shows coverage per package; the sandbox runner looks low "
                         "only because it runs in a child process that the coverage tool cannot trace. Ruff keeps the "
                         "code style clean, and the evaluation script is our system-level test. Now Bhavesh will "
                         "explain the AutoML engine and the results.")
    stats = [(str(f["tests"]), "automated tests", PURPLE),
             (str(f["tests_passed"]), "offline tests, all passing", GREEN),
             (f"{f['coverage_total']}%", "statement coverage", BLUE),
             (str(f["tests"] - f["tests_offline"]), "live LLM tests, run on request", ORANGE)]
    for i, (value, label, color) in enumerate(stats):
        x, y = MARGIN + (i % 2) * 2.95, TOP + 0.05 + (i // 2) * 2.5
        card(slide, x, y, 2.7, 2.2)
        text(slide, x + 0.25, y + 0.3, 2.2, 0.95, value, name="Stat number", size=44, font=BLACK,
             color=shade(color, 0.1), anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.25, y + 1.3, 2.2, 0.8, label, size=18, color=MUTED)
    x = MARGIN + 6.0
    w = SLIDE_W - MARGIN - x
    card(slide, x, TOP + 0.05, w, 4.7)
    text(slide, x + 0.3, TOP + 0.22, w - 0.6, 0.45, "Coverage per package", name="Heading", size=22, font=SEMI)
    coverage = sorted(f["coverage"], key=lambda c: -c[1])
    data = CategoryChartData()
    data.categories = [name for name, _ in coverage]
    data.add_series("Coverage", [value / 100 for _, value in coverage])
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x + 0.15), Inches(TOP + 0.7),
                                   Inches(w - 0.3), Inches(3.55), data)
    frame.name = "Chart coverage"
    chart = frame.chart
    style_chart(chart)
    quiet_axes(chart, value_max=1.28, value_min=0)
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 40
    color_series(plot.series[0], BLUE)
    for point, (name, _) in zip(plot.series[0].points, coverage, strict=True):
        if name == "idss/execution":
            point.format.fill.solid()
            point.format.fill.fore_color.rgb = rgb(tint(BLUE, 0.6))
    bar_labels(plot, "0.0%")
    text(slide, x + 0.3, TOP + 4.25, w - 0.6, 0.4, "idss/execution: the sandbox runner is a child process",
         name="Small note", size=14, color=MUTED)


def slide_setup(deck: Deck) -> None:
    f = deck.facts
    slide = deck.content("Evaluation setup", f"{len(f['bench'])} synthetic datasets × {f['seeds']} seeds, offline, "
                         "fast budget; the test split is read once",
                         f"We generated {len(f['bench'])} synthetic datasets, one per task type: churn, crop price, store "
                         "sales, delivery status with three classes, and customer segments. Each ran offline with seeds "
                         "42, 43 and 44 and the fast budget; we report the mean. The metrics come from AutoML-Agent: SR "
                         "is the success rate, NPS a normalised 0 to 1 score, and CS their average. For clustering we "
                         "use the adjusted Rand index.")
    icons = {"Churn": ("user_minus", PURPLE), "Crop price": ("tag", GREEN), "Store sales": ("cart", BLUE),
             "Delivery": ("truck", ORANGE), "Segments": ("cluster", PINK)}
    metric_names = {"f1": "F1", "rmse": "RMSE", "adjusted_rand_index": "ARI"}
    gap, w = 0.25, (CONTENT_W - 4 * 0.25) / 5
    for i, b in enumerate(f["bench"]):
        icon, color = icons[b["dataset"]]
        x = MARGIN + i * (w + gap)
        card(slide, x, TOP + 0.05, w, 3.3)
        deck.icon(slide, icon, x + 0.28, TOP + 0.3, 0.75, color)
        task = b["task"].replace("time-series forecasting", "forecasting")
        text(slide, x + 0.28, TOP + 1.2, w - 0.45, 2.0,
             [[(b["dataset"], {"font": SEMI, "size": 22})], [(task, {"color": MUTED})],
              [(f"{b['rows']:,} rows", {"color": MUTED})], [(f"{metric_names[b['primary_metric']]} ", {"font": SEMI}),
                                                            ("↑" if b["direction"] == "maximize" else "↓", {})]],
             size=18, space_after=2)
    shape(slide, MARGIN, 5.4, CONTENT_W, 1.2, tint(PURPLE, 0.9), kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16,
          name="Callout")
    deck.icon(slide, "trophy", MARGIN + 0.3, 5.58, 0.84, PURPLE)
    text(slide, MARGIN + 1.4, 5.4, CONTENT_W - 1.7, 1.2,
         [[("Metrics from AutoML-Agent", {"font": SEMI})],
          "SR = success rate  ·  NPS = normalised score  ·  CS = 0.5 · SR + 0.5 · NPS"],
         size=20, anchor=MSO_ANCHOR.MIDDLE)


def slide_kpis(deck: Deck) -> None:
    f = deck.facts
    o = f["overall"]
    (case, seed, model), = f["unverified"]  # the KPI text below assumes exactly one unverified clean run
    slide = deck.content("Results at a glance", f"Mean over {len(f['bench'])} datasets × {f['seeds']} seeds",
                         f"Across {f['verified'][1]} runs the success rate is {o['sr'][0]:.2f}: every run finished "
                         "with a PDF report, because tested code does the computing. The mean NPS is "
                         f"{o['nps'][0]:.3f} and CS is {o['cs'][0]:.3f}. {f['verified'][0]} of {f['verified'][1]} runs "
                         f"passed all checks. The one failure was {case} with seed {seed}: the selected {model} model "
                         "over-fitted, the re-plan could not fix it, and the system labelled it not verified instead "
                         "of hiding it.")
    stats = [(f"{o['sr'][0]:.2f}", "SR", "every run completed", GREEN),
             (f"{o['nps'][0]:.3f}", "NPS", "normalised performance", BLUE),
             (f"{o['cs'][0]:.3f}", "CS", "comprehensive score", PURPLE),
             (f"{f['verified'][0]}/{f['verified'][1]}", "Verified", "runs passed all checks", ORANGE),
             (f"{f['beats'][0]}/{f['beats'][1]}", "Datasets", "beat the trivial baseline", PINK)]
    gap, w = 0.25, (CONTENT_W - 4 * 0.25) / 5
    for i, (value, label, desc, color) in enumerate(stats):
        x = MARGIN + i * (w + gap)
        card(slide, x, TOP + 0.25, w, 3.2)
        shape(slide, x + 0.3, TOP + 0.5, 0.6, 0.12, color, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        text(slide, x + 0.2, TOP + 0.8, w - 0.4, 1.0, value, name="Stat number", size=44, font=BLACK,
             color=shade(color, 0.15) if color != YELLOW else shade(YELLOW, 0.4), align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.2, TOP + 1.9, w - 0.4, 1.4, [[(label, {"font": SEMI, "size": 22})], desc], size=18,
             color=MUTED, align=PP_ALIGN.CENTER)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(INK)
    shape(slide, MARGIN, 5.4, CONTENT_W, 1.2, tint(GREEN, 0.9), kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16,
          name="Callout")
    deck.icon(slide, "check", MARGIN + 0.3, 5.58, 0.84, GREEN)
    text(slide, MARGIN + 1.4, 5.4, CONTENT_W - 1.7, 1.2,
         "Every run finished because the LLM plans while tested code trains, verifies and explains",
         size=20, anchor=MSO_ANCHOR.MIDDLE)


def slide_baseline(deck: Deck) -> None:
    f = deck.facts
    gain = {b["dataset"]: b["improvement_pct"][0] for b in f["bench"]}
    assert gain["Churn"] is None and gain["Segments"] is None  # baselines at chance level (see the notes)
    slide = deck.content("Our model vs trivial baseline", "Normalised test score (0–1), mean of 3 seeds: higher is "
                         "better",
                         "Each model is compared with a trivial baseline on the same test split. RMSE is "
                         f"{gain['Crop price']:.1f}% lower on crop price and {gain['Store sales']:.1f}% lower on store "
                         f"sales, and delivery F1 is {gain['Delivery']:.1f}% higher. For churn and segments the "
                         "baseline is at chance level: the majority class never predicts churn, so its F1 is zero, and "
                         "random clusters have an ARI near zero, so a percentage makes no sense. Our model wins on "
                         f"{f['beats'][0]} of {f['beats'][1]} datasets.")
    data = CategoryChartData()
    data.categories = [b["dataset"] for b in f["bench"]]
    model, base = [], []
    for b in f["bench"]:
        key, base_key = ("score", "baseline") if b["direction"] == "maximize" else ("nps", "baseline_nps")
        model.append(b[key][0])
        base.append(max(b[base_key][0], 0.0))
    data.add_series("Our model (IDSS)", model)
    data.add_series("Trivial baseline", base)
    card(slide, MARGIN, TOP + 0.05, 8.1, 4.85)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(MARGIN + 0.2), Inches(TOP + 0.2),
                                   Inches(7.7), Inches(4.55), data)
    frame.name = "Chart baseline"
    chart = frame.chart
    style_chart(chart, legend=True)
    quiet_axes(chart, value_max=1.13, value_min=0)
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 45
    plot.overlap = -5
    color_series(plot.series[0], PURPLE)
    color_series(plot.series[1], ORANGE)
    bar_labels(plot, "0.000")
    x = MARGIN + 8.45
    w = SLIDE_W - MARGIN - x
    gains = [b for b in f["bench"] if b["improvement_pct"][0] is not None]
    for i, b in enumerate(gains):
        y = TOP + 0.05 + i * 1.22
        error = b["direction"] == "minimize"
        color = [GREEN, BLUE, ORANGE][i % 3]
        card(slide, x, y, w, 1.05)
        text(slide, x + 0.2, y, 1.75, 1.05, f"{b['improvement_pct'][0]:.1f}%", name="Stat number", size=30,
             font=BLACK, color=shade(color, 0.15), anchor=MSO_ANCHOR.MIDDLE)
        metric = "RMSE" if error else "F1"
        text(slide, x + 1.95, y, w - 2.1, 1.05, [f"{'lower' if error else 'higher'} {metric}",
                                                 [(b["dataset"], {"font": SEMI, "color": INK})]],
             size=18, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    chance = [b["dataset"] for b in f["bench"] if b["improvement_pct"][0] is None]
    text(slide, x, TOP + 0.15 + len(gains) * 1.22, w, 1.15,
         f"{' and '.join(chance)}: the baseline is at chance level (F1 = 0, ARI ≈ 0), so no % gain",
         size=18, color=MUTED)


def slide_ablation(deck: Deck) -> None:
    f = deck.facts
    ab = f["ablation"]
    full = ab["full"]["cs"][0]
    churn_fixed = ab["no_threshold"]["per_case"]["Churn"]["score"][0]
    churn_full = ab["full"]["per_case"]["Churn"]["score"][0]
    sv, sr = f["single_verified"]
    replanned = sum(b["replanned"] for b in f["bench"])
    assert ab["no_replan"]["cs"][0] == full  # re-planning did not change the score (see the notes)
    slide = deck.content("Ablation: what each component adds", f"CS lost when one component is removed "
                         f"(full system: CS = {full:.3f})",
                         "We removed one component at a time. Threshold tuning matters most on imbalanced data: "
                         f"without it, churn F1 drops from {churn_full:.3f} to {churn_fixed:.3f}. Without "
                         f"hyperparameter tuning, CS falls to {ab['no_hpo']['cs'][0]:.3f}, mainly because clustering "
                         "picks the wrong number of groups. A single default model gives CS "
                         f"{ab['single_model']['cs'][0]:.3f} instead of {full:.3f}, and only {sv} of its {sr} runs pass "
                         f"verification. Re-planning ran in {replanned} of {f['verified'][1]} clean runs and did not "
                         "change the score.")
    variants = [("no_threshold", "− threshold tuning"), ("no_hpo", "− hyperparameter tuning"),
                ("no_replan", "− verification re-planning"), ("single_model", "Single default model")]
    data = CategoryChartData()
    # the CS that remains goes into the category label: PowerPoint wraps long data labels
    data.categories = [f"{label}\nCS {ab[key]['cs'][0]:.3f}" for key, label in variants]
    losses = [max(full - ab[key]["cs"][0], 0.0) for key, _ in variants]
    data.add_series("CS lost", losses)
    card(slide, MARGIN, TOP + 0.05, 7.6, 4.85)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(MARGIN + 0.2), Inches(TOP + 0.3),
                                   Inches(7.2), Inches(4.35), data)
    frame.name = "Chart ablation"
    chart = frame.chart
    style_chart(chart)
    quiet_axes(chart, value_max=max(losses) * 1.9, value_min=0)
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 60
    color_series(plot.series[0], ORANGE)
    bar_labels(plot, '"−"0.000;"−"0.000;"±0"')  # values are losses (>= 0), shown as a change of CS
    x = MARGIN + 7.95
    w = SLIDE_W - MARGIN - x
    card(slide, x, TOP + 0.05, w, 2.6)
    deck.icon(slide, "sliders", x + 0.3, TOP + 0.3, 0.7, PURPLE)
    text(slide, x + 1.15, TOP + 0.3, w - 1.4, 0.7, "Threshold tuning", name="Heading", size=22, font=SEMI,
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.3, TOP + 1.1, w - 0.6, 0.85, f"{churn_fixed:.3f} → {churn_full:.3f}", name="Stat number",
         size=36, font=BLACK, color=shade(PURPLE, 0.15), anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.3, TOP + 1.95, w - 0.6, 0.6, "churn F1 vs a 0.5 cut-off", size=18, color=MUTED)
    card(slide, x, TOP + 2.9, w, 2.0)
    text(slide, x + 0.3, TOP + 3.1, w - 0.6, 0.8, f"{sv}/{sr}", name="Stat number", size=36, font=BLACK,
         color=shade(ORANGE, 0.15), anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.3, TOP + 3.9, w - 0.6, 0.9,
         f"single-model runs verified (full system: {f['verified'][0]}/{f['verified'][1]})", size=18, color=MUTED)


def slide_robustness(deck: Deck) -> None:
    f = deck.facts
    robust = f["robust"]
    noise_max = max(abs(r["noise"]["change_pct"][0]) for r in robust)
    worst = min(robust, key=lambda r: r["missing"]["change_pct"][0])
    pv, pr = f["perturbed_verified"]
    metric = {"f1": "F1", "rmse": "RMSE", "adjusted_rand_index": "ARI"}[
        next(b["primary_metric"] for b in f["bench"] if b["dataset"] == worst["dataset"])]
    slide = deck.content("Robustness to dirty data", "Change of the test score vs clean data (negative = worse), "
                         "mean of 3 seeds",
                         "Finally, robustness. We added five noise columns, or blanked 10% of the cells at random. "
                         f"Noise changed every score by at most {noise_max:.1f}%. Missing values hurt more, most of all "
                         f"on {worst['dataset'].lower()}, where {metric} got "
                         f"{abs(worst['missing']['change_pct'][0]):.1f}% worse, probably because simple imputation "
                         f"cannot recover the lost information. Still, {pv} of {pr} perturbed runs passed "
                         "verification. Now Varun will show the system live.")
    data = CategoryChartData()
    data.categories = [r["dataset"] for r in robust]
    data.add_series("+10% missing values", [r["missing"]["change_pct"][0] / 100 for r in robust])
    data.add_series("+5 noise columns", [r["noise"]["change_pct"][0] / 100 for r in robust])
    card(slide, MARGIN, TOP + 0.05, 8.1, 4.85)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(MARGIN + 0.2), Inches(TOP + 0.2),
                                   Inches(7.7), Inches(4.55), data)
    frame.name = "Chart robustness"
    chart = frame.chart
    style_chart(chart, legend=True)
    quiet_axes(chart, value_max=0.1, value_min=-0.75, category_line=True)
    chart.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 45
    plot.overlap = -5
    color_series(plot.series[0], ORANGE)
    color_series(plot.series[1], BLUE)
    bar_labels(plot, '+0.0%;"−"0.0%;0.0%')
    x = MARGIN + 8.45
    w = SLIDE_W - MARGIN - x
    for i, (value, desc, color) in enumerate(((f"≤ {noise_max:.1f}%", "change with 5 noise columns", BLUE),
                                             (f"−{abs(worst['missing']['change_pct'][0]):.1f}%",
                                              f"{worst['dataset']} with 10% missing values", ORANGE),
                                             (f"{pv}/{pr}", "perturbed runs still verified", GREEN))):
        y = TOP + 0.05 + i * 1.67
        card(slide, x, y, w, 1.51)
        text(slide, x + 0.25, y + 0.08, w - 0.5, 0.66, value, name="Stat number", size=32, font=BLACK,
             color=shade(color, 0.15), anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.25, y + 0.76, w - 0.5, 0.68, desc, size=18, color=MUTED)


def slide_conclusion(deck: Deck) -> None:
    f = deck.facts
    o = f["overall"]
    slide = deck.content("Conclusion", "A question and a table in; a verified model and explained decisions out",
                         "To conclude: IDSS turns a question and a table into a verified model with explained "
                         f"decisions. Every run finished (SR {o['sr'][0]:.2f}), the model beat the baseline on "
                         f"{f['beats'][0]} of {f['beats'][1]} datasets with CS {o['cs'][0]:.3f}, and {f['verified'][0]} "
                         f"of {f['verified'][1]} runs were verified, with over-fitting flagged, not hidden. Our limits: "
                         "one table at a time, an offline evaluation on synthetic data, sensitivity to missing values, "
                         "and what-if results that are correlation, not causation.")
    rows = [("Every run finishes", f"SR = {o['sr'][0]:.2f}: the LLM plans, tested code computes", "check", GREEN),
            ("Better than trivial everywhere", f"{f['beats'][0]}/{f['beats'][1]} datasets, CS = {o['cs'][0]:.3f}",
             "trophy", PURPLE),
            ("Honest self-checks", f"{f['verified'][0]}/{f['verified'][1]} runs verified; over-fitting is flagged, "
                                   "not hidden", "shield", BLUE),
            ("Decisions, not only scores", "Drivers, what-if, risk tiers and a grounded PDF brief", "bulb", ORANGE)]
    for i, (title, desc, icon, color) in enumerate(rows):
        y = TOP + 0.05 + i * 1.22
        card(slide, MARGIN, y, CONTENT_W, 1.05)
        deck.icon(slide, icon, MARGIN + 0.25, y + 0.17, 0.72, color)
        text(slide, MARGIN + 1.25, y, 4.4, 1.05, title, name="Heading", size=22, font=SEMI, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, MARGIN + 5.7, y, CONTENT_W - 5.95, 1.05, desc, size=20, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)


def slide_future(deck: Deck) -> None:
    slide = deck.content("Future scope", "Next steps follow from the limitations",
                         "Our future work follows from these limits: studies on real data with a user study; "
                         "benchmarking the LLM path, which our offline evaluation did not measure; stronger AutoML with "
                         "cross-validation, ensembling and better imputation; and causal decisions with uplift models "
                         "and cost-aware actions. We also plan smarter re-planning, multi-user deployment with "
                         "monitoring, and reports in regional languages such as Marathi and Hindi.")
    items = [("Real-data studies", "Public benchmarks, a partner's data and a user study", "users", PURPLE),
             ("Benchmark the LLM path", "Parsing accuracy, brief faithfulness and cost per run", "sparkle", BLUE),
             ("Stronger AutoML", "Cross-validation, ensembling, better imputation", "sliders", ORANGE),
             ("Causal decisions", "Uplift models and cost-aware, feasible actions", "branch", GREEN)]
    gap, w = 0.3, (CONTENT_W - 3 * 0.3) / 4
    for i, (title, desc, icon, color) in enumerate(items):
        x = MARGIN + i * (w + gap)
        card(slide, x, TOP + 0.05, w, 3.35)
        deck.icon(slide, icon, x + 0.3, TOP + 0.3, 0.8, color)
        text(slide, x + 0.3, TOP + 1.25, w - 0.5, 0.8, title, name="Heading", size=22, font=SEMI,
             anchor=MSO_ANCHOR.BOTTOM)
        text(slide, x + 0.3, TOP + 2.15, w - 0.5, 1.1, desc, size=18, color=MUTED)
    shape(slide, MARGIN, 5.5, CONTENT_W, 1.15, tint(BLUE, 0.9), kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16,
          name="Callout")
    deck.icon(slide, "trend", MARGIN + 0.3, 5.66, 0.84, BLUE)
    text(slide, MARGIN + 1.4, 5.5, CONTENT_W - 1.7, 1.15,
         [[("Also planned: ", {"font": SEMI}), "smarter re-planning, multi-user deployment with monitoring, and "
                                               "reports in regional languages"]], size=20, anchor=MSO_ANCHOR.MIDDLE)


def slide_thanks(deck: Deck) -> None:
    slide = deck._new(f"Thank you for your attention, and thank you to our guide, {GUIDE}, for the support. In one "
                      "line: the LLM plans and explains, and tested code computes every number. We are happy to take "
                      "your questions. Raj will answer on the problem and the agents, Chaitany on the backend and "
                      "testing, Bhavesh on the models and results, and I will answer on the interface and the demo.")
    deck._gradient_background(slide, pink_x=10.4)
    deck._title(slide, "Thank you", 0.8, 1.2, 11.7, 1.3, size=66, font=BLACK, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 0.8, 2.55, 11.7, 0.6, "Questions and discussion", name="Subtitle", size=26, color=WHITE)
    w, h = 5.6, 0.6
    for i, (name, roll) in enumerate(STUDENTS):
        y = 3.4 + i * (h + 0.16)
        shape(slide, 0.8, y, w, h, WHITE, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=h / 2, alpha=0.15, line=WHITE,
              name="Card student")
        text(slide, 1.1, y, 3.6, h, name, name="Pill student name", size=18, font=SEMI, color=WHITE,
             anchor=MSO_ANCHOR.MIDDLE)
        if roll:
            text(slide, 4.7, y, 1.4, h, [[(roll, {"alpha": 0.85})]], name="Pill student roll", size=18, color=WHITE,
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    picture(slide, deck.render.logo(), 7.3, 3.4, 3.4, 3.4 * 72 / 280, name="Logo")
    text(slide, 7.3, 4.5, 5.23, 1.6,
         [[("PROJECT GUIDE", {"spacing": 150, "font": SEMI})], [(GUIDE, {"size": 22, "bold": True})], DEPARTMENT,
          f"Academic Year {ACADEMIC_YEAR}"], name="Guide", size=18, color=WHITE, space_after=2)
    deck._footer(slide, dark=True)


def slide_demo_steps(deck: Deck) -> None:
    """Hidden backup slide: the path through the live demo, and what to show if it fails."""
    slide = deck.content("Live Demo Steps", "Backup slide, hidden in the slideshow: the path through the live demo",
                         "Backup slide, hidden during the slideshow. Before the talk, run powershell -ExecutionPolicy "
                         "Bypass -File scripts\\run_all.ps1 and do one Demo Mode run as a rehearsal. In the demo, click "
                         "Demo Mode on Home and show the Run Monitor. If the run is still training after a minute, pick "
                         "the rehearsal run in the run picker. Then show Results, the Decision Center, one what-if "
                         "change, and download the PDF on Reports. If the backend fails, do not debug in front of the "
                         "audience: open docs/screenshots and walk through the same pages.",
                         hidden=True)
    code = {"font": "Consolas", "color": INK}
    steps = [("Start the servers", [("scripts\\run_all.ps1", code), " (API + UI)"], PURPLE),
             ("Open the dashboard", [("localhost:8501", code), " in the browser"], BLUE),
             ("Demo Mode", "Home page: churn sample, fast run", GREEN),
             ("Run Monitor", "Pipeline stages update live", ORANGE),
             ("Results", "Leaderboard and verification checks", PINK),
             ("Decision Center", "Brief, risk tiers and key drivers", YELLOW),
             ("What-if", "Edit inputs, compare before / after", PURPLE),
             ("Reports", "Download the PDF report", BLUE)]
    gap, w, h, row_gap = 0.35, (CONTENT_W - 0.35) / 2, 0.78, 0.14
    for i, (title, desc, color) in enumerate(steps):
        x, y = MARGIN + (i // 4) * (w + gap), TOP + 0.05 + (i % 4) * (h + row_gap)  # down the left column first
        card(slide, x, y, w, h)
        pill(slide, x + 0.2, y + (h - 0.5) / 2, 0.5, 0.5, str(i + 1), color, color=on(color), font=BLACK,
             name="Pill step")
        text(slide, x + 0.9, y, w - 1.1, h, [[(title, {"font": SEMI, "size": 20, "color": INK})],
                                             [desc] if isinstance(desc, str) else desc],
             size=18, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    shape(slide, MARGIN, 5.6, CONTENT_W, 1.05, tint(ORANGE, 0.88), kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16,
          name="Callout")
    deck.icon(slide, "flag", MARGIN + 0.3, 5.715, 0.82, ORANGE)
    text(slide, MARGIN + 1.4, 5.6, CONTENT_W - 1.7, 1.05,
         [[("If the demo fails: ", {"font": SEMI, "color": shade(ORANGE, 0.35)}), "show the screenshots in ",
           ("docs/screenshots/", code), " instead"]], size=20, anchor=MSO_ANCHOR.MIDDLE)


def build(facts: dict[str, Any]) -> Deck:
    renderer = Renderer()
    try:
        deck = Deck(facts, renderer)
        # Raj: introduction, problem, objectives and the GenAI agents
        deck.section("Opening")
        deck.presenter(RAJ)
        slide_title(deck)
        slide_agenda(deck)
        deck.section("Introduction")
        deck.divider(1, "Introduction", "Why decision support needs more than a model", ["Problem", "Objectives"],
                     "Let us start with the problem. Many organisations already have data in spreadsheets and clear "
                     "questions, such as which customers are likely to leave. What they lack is the skill to turn that "
                     "data into a reliable model, and then the model into a clear decision. This first part explains "
                     "that gap and the eight objectives we set to close it.")
        slide_problem(deck)
        slide_objectives(deck)
        deck.section("System design")
        deck.divider(2, "System design", "LLM agents plan and explain; deterministic engines compute",
                     ["Methodology", "Trust", "Architecture", "Orchestrator"],
                     "Now the system design. Our key rule: language work goes to the LLM, and number work goes to "
                     "deterministic, tested code. I will first walk through one complete run and the agents in it, and "
                     "then explain the four guards that keep the results honest and the data private. After that, "
                     "Chaitany will show the layered architecture and the resumable orchestrator.")
        slide_methodology(deck)
        slide_trust(deck)
        # Chaitany: architecture, backend / API / database and testing
        deck.presenter(CHAITANY)
        slide_architecture(deck)
        slide_orchestrator(deck)
        deck.section("Implementation")
        deck.divider(3, "Implementation", "The technology stack and how we tested it", ["Technology stack", "Testing"],
                     "Next, the implementation. Everything is written in Python 3.12 with open-source libraries, so no "
                     "paid licence is needed, and it runs on a normal laptop. I will show the technology stack, "
                     "including the backend, the API "
                     "and the database, and then how we tested the system. Our rule was that tests never call the "
                     "network, so the whole test suite runs offline and gives the same result every time.")
        slide_stack(deck)
        slide_testing(deck)
        # Bhavesh: AutoML engine, algorithms, results and ablation
        deck.section("AutoML and results")
        deck.presenter(BHAVESH)
        deck.divider(4, "AutoML and results", "How models are built and checked, and how well they work",
                     ["AutoML + Verifier", "Decisions", "Setup", "Overview", "Baselines", "Ablation", "Robustness"],
                     "Thank you, Chaitany. In this part I explain how the models are built and checked, and how well "
                     "the system performs: the AutoML engine and the verifier, how a model becomes a decision, the "
                     "evaluation setup, the overall results, the baselines, the ablation study and the robustness "
                     "study. Every number on these slides comes from our evaluation files, not from the LLM.")
        slide_engine(deck)
        slide_decisions(deck)
        slide_setup(deck)
        slide_kpis(deck)
        slide_baseline(deck)
        slide_ablation(deck)
        slide_robustness(deck)
        # Varun: dashboard, live demo, limitations, conclusion and questions
        deck.section("Demo and conclusion")
        deck.presenter(VARUN)
        deck.divider(5, "Demo and conclusion", "The dashboard live, what we achieved and what comes next",
                     ["Dashboard", "Live demo", "Conclusion", "Future scope"],
                     "Thank you, Bhavesh. In this last part I will show the web dashboard with a short live demo on the "
                     "churn sample, and then summarise our results, the limitations and the future scope. The "
                     "dashboard is what a business user sees: simple pages to upload data, ask a question, watch the "
                     "run and read the decisions. No coding is needed at all.")
        slide_dashboard(deck)
        slide_conclusion(deck)
        slide_future(deck)
        slide_thanks(deck)
        deck.section("Backup")
        slide_demo_steps(deck)
        deck.finish()
    finally:
        renderer.close()
    return deck


# ---------------------------------------------------------------- export + layout check (PowerPoint COM)

LAYOUT_TOLERANCE_PT = 1.5


def _check_presentation(pres: Any) -> list[str]:
    """Text must fit its box and the slide, with a minimum size per shape role (see MIN_PT)."""
    width, height = pres.PageSetup.SlideWidth, pres.PageSetup.SlideHeight
    problems: list[str] = []

    def shapes(collection: Any) -> Any:
        for item in collection:
            if item.Type == 6:  # msoGroup
                yield from shapes(item.GroupItems)
            else:
                yield item

    for slide in pres.Slides:
        for item in shapes(slide.Shapes):
            name = item.Name
            where = f"slide {slide.SlideIndex} '{name}'"
            if not name.startswith("Decor") and (item.Left < -LAYOUT_TOLERANCE_PT or item.Top < -LAYOUT_TOLERANCE_PT
                                                 or item.Left + item.Width > width + LAYOUT_TOLERANCE_PT
                                                 or item.Top + item.Height > height + LAYOUT_TOLERANCE_PT):
                problems.append(f"{where} extends beyond the slide")
            if not item.HasTextFrame or not item.TextFrame.HasText:
                continue
            frame = item.TextFrame
            rng = frame.TextRange
            if rng.BoundHeight > item.Height + LAYOUT_TOLERANCE_PT \
                    or rng.BoundTop < item.Top - LAYOUT_TOLERANCE_PT \
                    or rng.BoundTop + rng.BoundHeight > item.Top + item.Height + LAYOUT_TOLERANCE_PT:
                problems.append(f"{where}: text is {rng.BoundHeight:.0f} pt high in a {item.Height:.0f} pt box "
                                f"({rng.Text[:40]!r})")
            # PowerPoint's bound width includes the space at a wrap, so the width test is only meaningful without
            # wrapping; wrapped text can only overflow sideways by breaking a word, which the line test catches
            if not frame.WordWrap and rng.BoundLeft + rng.BoundWidth > item.Left + item.Width + LAYOUT_TOLERANCE_PT:
                problems.append(f"{where}: text is wider than its box ({rng.Text[:40]!r})")
            lines = []
            while len(lines) < 40 and (line := rng.Lines(len(lines) + 1, 1)).Length > 0:
                lines.append(line.Text)
            for before, after in zip(lines, lines[1:]):
                if before[-1:].isalnum() and after[:1].isalnum():
                    problems.append(f"{where}: word broken across lines ({before[-12:]!r} | {after[:12]!r})")
            if name.startswith("Pill") and len(lines) > rng.Paragraphs().Count:  # chips and pills: one line each
                problems.append(f"{where}: wraps to {len(lines)} lines ({rng.Text[:40]!r})")
            minimum = next((size for prefix, size in MIN_PT.items() if name.startswith(prefix)), BODY_MIN_PT)
            for i in range(1, rng.Runs().Count + 1):
                run = rng.Runs(i)
                if run.Text.strip() and run.Font.Size < minimum:
                    problems.append(f"{where}: {run.Font.Size:g} pt < {minimum} pt ({run.Text[:30]!r})")
    return problems


def export_powerpoint(pptx: Path, pdf: Path, png_dir: Path) -> list[str]:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(str(pptx), -1, 0, 0)  # ReadOnly, Untitled = no, WithWindow = no
        try:
            problems = _check_presentation(pres)
            pres.SaveAs(str(pdf), 32)  # ppSaveAsPDF
            png_dir.mkdir(parents=True, exist_ok=True)
            for old in png_dir.glob("slide_*.png"):
                old.unlink()
            for slide in pres.Slides:
                slide.Export(str(png_dir / f"slide_{slide.SlideIndex:02d}.png"), "PNG", 1920, 1080)
        finally:
            pres.Close()
    finally:
        if app.Presentations.Count == 0:  # never close a PowerPoint the user is working in
            app.Quit()
        pythoncom.CoUninitialize()
    return problems


def export_libreoffice(pptx: Path, pdf: Path, png_dir: Path) -> bool:
    soffice = _find_soffice()
    if not soffice:
        return False
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(pdf.parent), str(pptx)],
                   check=True, timeout=600)
    import pymupdf  # requirements-docs.txt

    png_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as doc:
        for index, page in enumerate(doc, start=1):
            page.get_pixmap(dpi=144).save(png_dir / f"slide_{index:02d}.png")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-export", action="store_true", help="only write the .pptx (no PDF / PNG)")
    parser.add_argument("--draft", action="store_true", help="keep the exports even if the layout check fails")
    args = parser.parse_args()

    facts = load_facts()
    deck = build(facts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deck.prs.save(PPTX_OUT)
    print(f"Saved {PPTX_OUT.relative_to(ROOT)}: {len(deck.prs.slides)} slides in {len(deck.sections)} sections "
          f"({PPTX_OUT.stat().st_size / 1024:,.0f} KB)")
    hidden = [i for i, s in enumerate(deck.talk, start=1) if s["hidden"]]
    print(f"Presenters (notes at {SPEAKING_WPM} words/min, live demo {LIVE_DEMO_MINUTES:g} min; hidden slides "
          f"{', '.join(map(str, hidden))} not counted):")
    for name, first, last, words, minutes in deck.timing():
        print(f"  {name:<22} slides {first:>2}-{last:<2} {words:>4} words  ~{minutes:.1f} min")
    if args.no_export:
        return
    try:
        problems = export_powerpoint(PPTX_OUT, PDF_OUT, PNG_DIR)
    except ImportError as exc:  # pywin32 missing / not Windows
        print(f"PowerPoint automation not available: {exc}")
        problems = None
    except Exception as exc:  # COM errors (no PowerPoint, file locked by a viewer, ...)
        print(f"PowerPoint export failed: {exc}")
        problems = None
    if problems is None:
        if export_libreoffice(PPTX_OUT, PDF_OUT, PNG_DIR):
            print("PDF converted with LibreOffice (no layout check); slide PNGs rendered with PyMuPDF.")
        else:
            raise SystemExit("Neither PowerPoint nor LibreOffice is available: open the .pptx in PowerPoint and "
                             "use File > Export > PDF.")
    else:
        for problem in problems:
            print(f"  LAYOUT: {problem}")
        if problems and not args.draft:
            raise SystemExit(f"Layout check failed ({len(problems)} problems); fix them or rerun with --draft.")
        print("Layout check passed: all text fits, stays on the slides and meets the minimum sizes."
              if not problems else "Layout problems kept (--draft).")
    print(f"Saved {PDF_OUT.relative_to(ROOT)} ({PDF_OUT.stat().st_size / 1024:,.0f} KB) and "
          f"{len(list(PNG_DIR.glob('slide_*.png')))} slide PNGs in {PNG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
