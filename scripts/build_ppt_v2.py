"""Build the presentation on the official college template (Step 12C) -> docs/presentation/IDSS_Presentation.pptx + .pdf.

The deck starts from docs/format/ppt_template.pptx (G H Raisoni slide master: college logo top-left, "raisoni EDUCATION"
top-right, orange -> purple bar at the bottom; the template itself is not committed). Its empty sample slide is removed
and exactly 14 slides are added with the template's own layouts (Title Slide, Title and Content, Two Content, Title
Only), so every slide inherits the logos and the bar. The title and body placeholders of the master and the layouts are
moved into the safe area (left/right margin 0.6 in, top >= 1.2 in below the logos, bottom <= 6.95 in above the bar) and
restyled: Calibri, titles 32 pt bold purple #670087, body >= 20 pt dark grey #333333, orange #E0620D for highlights and
numbers, light purple #F3E8F7 card fills. Simple English, few words, pictures first.

Checks (the build fails on any problem):
- numbers: every number in the slide text and the speaker notes must be found in docs/results/evaluation.json (summary,
  3-seed means) or docs/testing.md. Exceptions: roll numbers, the academic year, reference years (which must appear in
  docs/report/references.md), step badges and the 0 / 1 bounds of the score scale.
- speaker notes: first line "Presenter: <name>", then 50-90 words; Raj 1-4, Chaitany 5-8, Varun 9-10 and 13-14,
  Bhavesh 11-12.
- no empty placeholder, no stretched picture, text contrast >= 4.5:1 (3:1 for large text), exactly 14 slides.
- PowerPoint (COM): every shape inside the safe area, text fits its box, no word broken across lines, chips on one line,
  text >= 20 pt and titles >= 32 pt. Then the PDF and 1920 x 1080 PNGs (docs/presentation/slides/, gitignored) are
  exported. Without PowerPoint: LibreOffice PDF + PyMuPDF PNGs (no layout check).

Usage: .venv\\Scripts\\python.exe scripts\\build_ppt_v2.py [--no-export] [--draft]
Setup: pip install -r requirements-docs.txt; python -m playwright install chromium (icons)
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.enum.dml import MSO_FILL
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from build_ppt import EVALUATION, TESTING, Renderer, export_libreoffice, fill_text, fit_picture, image_bytes, \
    load_facts, picture, rgb, shape
from report_common import ACADEMIC_YEAR, COLLEGE, DEPARTMENT, GUIDE, PROJECT_TITLE, ROOT, STUDENTS

TEMPLATE = ROOT / "docs" / "format" / "ppt_template.pptx"
OUT_DIR = ROOT / "docs" / "presentation"
PPTX_OUT = OUT_DIR / "IDSS_Presentation.pptx"
PDF_OUT = OUT_DIR / "IDSS_Presentation.pdf"
PNG_DIR = OUT_DIR / "slides"  # QA renders (gitignored)
REFERENCES = ROOT / "docs" / "report" / "references.md"
SCREENS = ROOT / "docs" / "screenshots"
ARCHITECTURE = ROOT / "docs" / "diagrams" / "system_architecture.png"
FONT_FILES = {False: Path(r"C:\Windows\Fonts\calibri.ttf"), True: Path(r"C:\Windows\Fonts\calibrib.ttf")}

# ---------------------------------------------------------------- design tokens (college template)

PURPLE, ORANGE, GREY, LIGHT, WHITE = "670087", "E0620D", "333333", "F3E8F7", "FFFFFF"
FRAME = "C9B3D3"  # thin outline around screenshots
FONT = "Calibri"
TITLE_PT, BODY_PT = 32, 20

SLIDE_W, SLIDE_H = 13.333, 7.5
LEFT, TOP, BOTTOM = 0.6, 1.2, 6.95  # safe area: side margins, below the logos, above the gradient bar
RIGHT = SLIDE_W - LEFT
CONTENT_W = RIGHT - LEFT
Y0 = 2.05  # first content row, below the title
Y1 = BOTTOM - 0.1  # last content row ends here
GAP = 0.3
HALF_W = (CONTENT_W - GAP) / 2
TITLE_BOX = (LEFT, TOP, CONTENT_W, 0.7)
BODY_BOX = (LEFT, Y0, CONTENT_W, Y1 - Y0)

SLIDES = 14
RAJ, BHAVESH, CHAITANY, VARUN = (name for name, _ in STUDENTS)
PRESENTERS = {1: RAJ, 2: RAJ, 3: RAJ, 4: RAJ, 5: CHAITANY, 6: CHAITANY, 7: CHAITANY, 8: CHAITANY, 9: VARUN, 10: VARUN,
              11: BHAVESH, 12: BHAVESH, 13: VARUN, 14: VARUN}
NOTE_WORDS = (50, 90)  # words in each slide's script, after the "Presenter:" line
SPEAKING_WPM = 130


# ---------------------------------------------------------------- text helpers

def write(tf: Any, content: Any, **style: Any) -> None:
    """build_ppt.fill_text with the template's font, body size and colour as defaults."""
    fill_text(tf, content, **({"font": FONT, "size": BODY_PT, "color": GREY} | style))


def text(slide: Any, x: float, y: float, w: float, h: float, content: Any, *, name: str = "Body",
         **style: Any) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    write(box.text_frame, content, **style)
    return box


@cache
def _font(bold: bool) -> ImageFont.FreeTypeFont | None:
    path = FONT_FILES[bold]
    return ImageFont.truetype(str(path), 400) if path.exists() else None


def text_width(label: str, size: float, bold: bool = False) -> float:
    """Width of one line of Calibri text in inches (measured with the installed font)."""
    font = _font(bold)
    if font is None:
        return len(label) * size * 0.52 / 72
    return font.getlength(label) / 400 * size / 72


def card(slide: Any, x: float, y: float, w: float, h: float, fill: str = LIGHT, *, name: str = "Card",
         **kw: Any) -> Any:
    return shape(slide, x, y, w, h, fill, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14, name=name, **kw)


def tile(slide: Any, x: float, y: float, w: float, h: float, *, name: str = "Tile") -> Any:
    """White card with a thin outline: orange numbers need white behind them (only 3:1 on the light purple)."""
    return shape(slide, x, y, w, h, WHITE, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14, line=FRAME, line_w=1.25,
                 name=name)


def chip(slide: Any, x: float, y: float, label: str, *, h: float = 0.55, fill: str = LIGHT, color: str = GREY) -> float:
    """A one-line rounded label; returns its width."""
    w = text_width(label, BODY_PT) + 0.5
    s = shape(slide, x, y, w, h, fill, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=h / 2, name=f"Chip {label}")
    write(s.text_frame, label, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return w


def badge(slide: Any, x: float, y: float, d: float, number: int) -> None:
    """Orange step number in a circle (left out of the number check: it is a position, not a result)."""
    s = shape(slide, x, y, d, d, ORANGE, kind=MSO_SHAPE.OVAL, line=WHITE, line_w=2, name=f"Step {number}")
    write(s.text_frame, str(number), color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def arrow(slide: Any, x: float, y: float, w: float, h: float, *, down: bool = False) -> None:
    shape(slide, x, y, w, h, ORANGE, kind=MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW, name="Arrow")


def numbered(tf: Any, size: float = BODY_PT) -> None:
    """Turn the paragraphs of a text frame into an automatic 1. 2. 3. list (numbers are not part of the text)."""
    indent = round(Pt(size) * 1.6)
    for p in tf.paragraphs:
        p_pr = p._p.get_or_add_pPr()
        p_pr.set("marL", str(indent))
        p_pr.set("indent", str(-indent))
        etree.SubElement(p_pr, qn("a:buClr")).append(etree.fromstring(
            f'<a:srgbClr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{PURPLE}"/>'))
        etree.SubElement(p_pr, qn("a:buFont")).set("typeface", FONT)
        etree.SubElement(p_pr, qn("a:buAutoNum")).set("type", "arabicPeriod")


# ---------------------------------------------------------------- deck

class Deck:
    def __init__(self, facts: dict[str, Any], renderer: Renderer) -> None:
        if not TEMPLATE.exists():
            raise SystemExit(f"College template not found: {TEMPLATE.relative_to(ROOT)} (copy it there; it is not "
                             "committed)")
        self.facts = facts
        self.render = renderer
        self.prs = Presentation(str(TEMPLATE))
        size = (round(self.prs.slide_width / 914400, 3), round(self.prs.slide_height / 914400, 3))
        if size != (SLIDE_W, SLIDE_H):
            raise SystemExit(f"Template is {size[0]} x {size[1]} in, expected {SLIDE_W} x {SLIDE_H} (16:9)")
        self._drop_sample_slides()
        self._theme()
        self._restyle_placeholders()
        self.layouts = {layout.name: layout for layout in self.prs.slide_layouts}
        self.talk: list[dict[str, Any]] = []  # per slide: title, presenter, words
        props = self.prs.core_properties
        props.title, props.subject = PROJECT_TITLE, f"Project presentation, {COLLEGE}"
        props.author = ", ".join(name for name, _ in STUDENTS)
        props.last_modified_by = props.author

    def _drop_sample_slides(self) -> None:
        """The template ships one empty sample slide; it is removed, so the deck holds only our slides."""
        id_list = self.prs.slides._sldIdLst
        for sld_id in list(id_list):
            rel_id = sld_id.rId
            id_list.remove(sld_id)
            self.prs.part.drop_rel(rel_id)

    def _theme(self) -> None:
        """Template colours in the theme, so charts and shapes added later in PowerPoint pick them up."""
        part = self.prs.slide_master.part.part_related_by(RT.THEME)
        root = etree.fromstring(part.blob)
        a = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        scheme = root.find(".//a:clrScheme", a)
        scheme.set("name", "GHRCEM")
        for tag, value in {"dk2": GREY, "lt2": LIGHT, "accent1": PURPLE, "accent2": ORANGE, "accent3": "9B4DB5",
                           "accent4": "F29B5B", "accent5": "5C5C5C", "accent6": "C79BD6", "hlink": PURPLE,
                           "folHlink": ORANGE}.items():
            slot = scheme.find(f"a:{tag}", a)
            for child in list(slot):
                slot.remove(child)
            etree.SubElement(slot, qn("a:srgbClr"), val=value)
        part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    def _restyle_placeholders(self) -> None:
        """Title and body placeholders of the master and every layout: inside the safe area, Calibri, template colours
        (the template places its title over the logo and uses Times New Roman)."""
        left_box = (LEFT, Y0, HALF_W, Y1 - Y0)
        right_box = (LEFT + HALF_W + GAP, Y0, HALF_W, Y1 - Y0)
        for owner in [self.prs.slide_master, *self.prs.slide_layouts]:
            for ph in owner.placeholders:
                kind, idx = ph.placeholder_format.type, ph.placeholder_format.idx
                if kind == PP_PLACEHOLDER.CENTER_TITLE:
                    box, style = (LEFT, 1.45, CONTENT_W, 1.35), (36, True, PURPLE)
                elif kind == PP_PLACEHOLDER.TITLE:
                    box, style = TITLE_BOX, (TITLE_PT, True, PURPLE)
                elif kind == PP_PLACEHOLDER.SUBTITLE:
                    box, style = (LEFT, 2.9, CONTENT_W, 0.5), (24, True, ORANGE)
                elif kind in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
                    box = {2: left_box, 3: right_box}.get(idx, BODY_BOX)
                    style = (BODY_PT, False, GREY)
                else:  # footer, date and slide number sit on the gradient bar; they are never used
                    continue
                ph.left, ph.top, ph.width, ph.height = (Inches(v) for v in box)
                _placeholder_style(ph, *style)

    # ---- slides

    def new(self, layout: str, title: Any, presenter: str, notes: str, **title_style: Any) -> Any:
        number = len(self.prs.slides) + 1
        words = len(notes.split())
        if PRESENTERS.get(number) != presenter:
            raise SystemExit(f"Slide {number} must be presented by {PRESENTERS.get(number)}, not {presenter}")
        if not NOTE_WORDS[0] <= words <= NOTE_WORDS[1]:
            raise SystemExit(f"Slide {number}: notes need {NOTE_WORDS[0]}-{NOTE_WORDS[1]} words, got {words} "
                             f"({notes[:50]!r})")
        slide = self.prs.slides.add_slide(self.layouts[layout])
        slide.notes_slide.notes_text_frame.text = f"Presenter: {presenter}\n{notes}"
        ph = slide.shapes.title
        ph.name = "Title"
        write(ph.text_frame, title, **({"size": TITLE_PT, "bold": True, "color": PURPLE, "anchor": MSO_ANCHOR.MIDDLE}
                                       | title_style))
        flat = title if isinstance(title, str) else " ".join(t if isinstance(t, str) else "".join(
            s if isinstance(s, str) else s[0] for s in t) for t in title)
        self.talk.append({"title": flat, "presenter": presenter, "words": words})
        return slide

    def placeholder(self, slide: Any, idx: int, name: str) -> Any:
        ph = next(p for p in slide.placeholders if p.placeholder_format.idx == idx)
        ph.name = name
        return ph

    def icon(self, slide: Any, name: str, x: float, y: float, d: float, color: str = PURPLE) -> None:
        """A white line icon in a filled circle."""
        shape(slide, x, y, d, d, color, kind=MSO_SHAPE.OVAL, name="Icon circle")
        pad = d * 0.23
        picture(slide, self.render.icon(name, WHITE), x + pad, y + pad, d - 2 * pad, d - 2 * pad, name="Icon")


def _placeholder_style(ph: Any, size: float, bold: bool, color: str) -> None:
    """Level-1 run defaults of a layout/master placeholder (replaces the template's Times New Roman)."""
    tx_body = ph._element.txBody
    lst = tx_body.find(qn("a:lstStyle"))
    if lst is None:
        lst = etree.Element(qn("a:lstStyle"))
        tx_body.find(qn("a:bodyPr")).addnext(lst)
    lvl1 = lst.find(qn("a:lvl1pPr"))
    if lvl1 is None:
        lvl1 = etree.SubElement(lst, qn("a:lvl1pPr"))
        lst.insert(0, lvl1)
    rpr = lvl1.find(qn("a:defRPr"))
    if rpr is None:
        rpr = etree.SubElement(lvl1, qn("a:defRPr"))
    for child in list(rpr):
        rpr.remove(child)
    rpr.set("sz", str(round(size * 100)))
    rpr.set("b", "1" if bold else "0")
    etree.SubElement(etree.SubElement(rpr, qn("a:solidFill")), qn("a:srgbClr"), val=color)
    etree.SubElement(rpr, qn("a:latin"), typeface=FONT)
    etree.SubElement(rpr, qn("a:cs"), typeface=FONT)


# ---------------------------------------------------------------- slides 1-4: Raj

def slide_title(deck: Deck) -> None:
    first, rest = PROJECT_TITLE.split(" Using ", 1)
    slide = deck.new(
        "Title Slide", [f"{first} (IDSS)", f"Using {rest}"], RAJ,
        f"Good morning, respected guide and examiners. We are Raj, Chaitany, Varun and Bhavesh, and our guide is "
        f"{GUIDE}. Our project is IDSS, the Intelligent Decision Support System. In simple words: a user asks a business "
        "question in plain English and uploads a data table. IDSS builds and checks a model by itself, explains the "
        "answer and suggests what to do next. I will start with the problem.",
        size=36, align=PP_ALIGN.CENTER)
    write(deck.placeholder(slide, 4, "Subtitle").text_frame, DEPARTMENT, size=24, bold=True, color=ORANGE,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, LEFT, 3.55, CONTENT_W, 0.4, "Presented by", align=PP_ALIGN.CENTER)
    w = (CONTENT_W - 3 * 0.25) / 4
    for i, (name, roll) in enumerate(STUDENTS):
        x = LEFT + i * (w + 0.25)
        box = card(slide, x, 4.05, w, 1.3, name=f"Card {name}")
        write(box.text_frame, [[(name, {"bold": True, "color": PURPLE})], *([roll] if roll else [])], align=PP_ALIGN.CENTER,
              anchor=MSO_ANCHOR.MIDDLE)
        box.text_frame.margin_left = box.text_frame.margin_right = Inches(0.12)
    text(slide, LEFT, 5.65, CONTENT_W, 0.45, [[("Project Guide: ", {"color": PURPLE}), (GUIDE, {"color": GREY})]],
         size=24, bold=True, align=PP_ALIGN.CENTER)
    text(slide, LEFT, 6.15, CONTENT_W, 0.4, f"Academic Year {ACADEMIC_YEAR}", align=PP_ALIGN.CENTER)


def slide_problem(deck: Deck) -> None:
    slide = deck.new(
        "Title and Content", "Introduction & Problem", RAJ,
        "Every business today collects data: sales, customers, deliveries. But turning that data into a good decision "
        "is hard. Building a machine learning model needs special skills, and data experts are costly and hard to find. "
        "Even when a model exists, a single score does not tell a manager what to do next. So many small businesses "
        "still decide by guesswork. We wanted a tool that goes from a plain question straight to a clear, explained "
        "decision.")
    body = deck.placeholder(slide, 1, "Body bullets")
    body.left, body.top, body.width, body.height = Inches(LEFT), Inches(Y0 + 0.15), Inches(6.6), Inches(2.85)
    write(body.text_frame, ["Businesses collect lots of data every day",
                            "Building ML models needs rare, costly experts",
                            "A model score alone does not say what to do",
                            "So many decisions are still guesswork"],
          size=22, bullets=PURPLE, space_after=14)
    text(slide, LEFT, 5.2, 6.6, 0.45, [[("ML", {"bold": True, "color": PURPLE}),
                                        " = machine learning: computers learn from data"]])
    card(slide, LEFT, 5.95, 6.6, 0.9, WHITE, line=ORANGE, line_w=1.5, name="Callout")
    deck.icon(slide, "target", LEFT + 0.2, 6.07, 0.66, ORANGE)
    text(slide, LEFT + 1.05, 5.95, 5.4, 0.9, "Our aim: plain question → clear decision", size=22, bold=True,
         color=PURPLE, anchor=MSO_ANCHOR.MIDDLE)
    # right: the gap between data and decisions
    x = 7.6
    card(slide, x, Y0, RIGHT - x, Y1 - Y0, name="Card gap")
    rows = [("database", "Lots of data", "Sales, customers, deliveries", PURPLE),
            ("users", "Experts needed", "Slow and costly to hire", ORANGE),
            ("bulb", "Clear decision?", "Often still a guess", PURPLE)]
    for i, (icon, head, sub, color) in enumerate(rows):
        y = 2.35 + i * 1.5
        deck.icon(slide, icon, x + 0.35, y, 0.9, color)
        text(slide, x + 1.5, y - 0.05, RIGHT - x - 1.7, 1.0,
             [[(head, {"bold": True, "size": 24, "color": PURPLE})], sub], anchor=MSO_ANCHOR.MIDDLE)
        if i < len(rows) - 1:
            arrow(slide, x + 0.62, y + 1.0, 0.36, 0.4, down=True)


def slide_objectives(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Objectives", RAJ,
        "We set four objectives. First, understand a business question written in plain English, with no coding. "
        "Second, build and tune machine learning models automatically and pick the best one. Third, check every result "
        "on unseen data before showing it, so the user can trust it. Fourth, explain why the model decided and suggest "
        "clear actions, including what-if tests where the user changes an input and sees the new result.")
    items = [("chat", "Understand the question", "Read a business question in plain English"),
             ("gear", "Build models automatically", "Try and tune many models, pick the best"),
             ("shield", "Check every result", "Test on unseen data before showing it"),
             ("bulb", "Explain and advise", "Show why, suggest actions, write a report")]
    w = (CONTENT_W - 3 * GAP) / 4
    for i, (icon, head, desc) in enumerate(items):
        x = LEFT + i * (w + GAP)
        card(slide, x, Y0, w, 3.85, name=f"Card {head}")
        deck.icon(slide, icon, x + 0.25, Y0 + 0.3, 0.95, ORANGE if i % 2 else PURPLE)
        text(slide, x + 0.25, Y0 + 1.45, w - 0.45, 0.8, head, size=22, bold=True, color=PURPLE,
             anchor=MSO_ANCHOR.BOTTOM)
        text(slide, x + 0.25, Y0 + 2.4, w - 0.45, 1.3, desc)
    card(slide, LEFT, 6.1, CONTENT_W, 0.75, WHITE, line=ORANGE, line_w=1.5, name="Callout")
    text(slide, LEFT + 0.3, 6.1, CONTENT_W - 0.6, 0.75, "Goal: decisions that anyone can trust and understand",
         size=22, bold=True, color=PURPLE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_literature(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Literature Review", RAJ,
        "We studied related work. Auto-sklearn finds good models automatically, but the user must write code. Data "
        "Interpreter is an AI agent that writes data-science code. AutoML-Agent, from ICML 2025, inspired our design: a "
        "team of AI agents builds models from start to end. SHAP explains why a model decided. But no single tool goes "
        "from a plain question to a checked model and an explained business decision. IDSS fills that gap. Now Chaitany "
        "will explain our system.")
    rows = [("Auto-sklearn", "Feurer et al., NIPS 2015", "Finds the best model automatically",
             "Needs coding; no plain-English questions"),
            ("Data Interpreter", "Hong et al., arXiv 2024", "AI agent writes data-science code",
             "No what-if or business advice"),
            ("AutoML-Agent", "Trirat et al., ICML 2025", "AI agents build models from start to end",
             "Stops at the model; no decisions"),
            ("SHAP", "Lundberg & Lee, NIPS 2017", "Shows why a model decided",
             "Only explains; does not build or advise")]
    widths = (3.45, 4.25, CONTENT_W - 3.45 - 4.25)
    frame = slide.shapes.add_table(len(rows) + 1, 3, Inches(LEFT), Inches(Y0), Inches(CONTENT_W),
                                   Inches(0.5 + 0.84 * len(rows)))
    frame.name = "Table literature"
    table = frame.table
    for col, width in zip(table.columns, widths, strict=True):
        col.width = Inches(width)
    table.rows[0].height = Inches(0.5)
    for row in list(table.rows)[1:]:
        row.height = Inches(0.84)

    def cell(r: int, c: int, content: Any, fill: str, **style: Any) -> None:
        target = table.cell(r, c)
        target.fill.solid()
        target.fill.fore_color.rgb = rgb(fill)
        target.margin_left = target.margin_right = Inches(0.12)
        target.margin_top = target.margin_bottom = Inches(0.04)
        target.vertical_anchor = MSO_ANCHOR.MIDDLE
        write(target.text_frame, content, **style)

    for c, head in enumerate(("Work", "What it does", "What is missing")):
        cell(0, c, head, PURPLE, bold=True, color=WHITE)
    for r, (work, source, does, missing) in enumerate(rows, start=1):
        fill = LIGHT if r % 2 else WHITE
        cell(r, 0, [[(work, {"bold": True, "color": PURPLE})], source], fill)
        cell(r, 1, does, fill)
        cell(r, 2, missing, fill)
    card(slide, LEFT, 6.1, CONTENT_W, 0.75, WHITE, line=ORANGE, line_w=1.5, name="Callout")
    text(slide, LEFT + 0.3, 6.1, CONTENT_W - 0.6, 0.75,
         [[("Gap: ", {"color": ORANGE}), ("no tool goes from a plain question to an explained decision",
                                          {"color": PURPLE})]],
         size=22, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------- slides 5-8: Chaitany

def slide_architecture(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Proposed System", CHAITANY,
        "This is our system architecture, in six layers. On top, the user works in a Streamlit web dashboard. It talks "
        "to a FastAPI server, which starts each run. The orchestrator is the manager: it moves the run step by step and "
        "saves progress after every step. Then come the AI agents, and below them the engines that train, explain and "
        "report. Our key rule: the AI plans and explains, but tested code does all the maths. Only summaries go to the "
        "AI, never raw data rows.")
    data, ratio = image_bytes(ARCHITECTURE, (0, 0.105, 1, 1))  # without the diagram's own title
    h = Y1 - Y0
    w = h * ratio
    picture(slide, data, LEFT, Y0, w, h, name="Picture architecture", line=FRAME)
    x = LEFT + w + 0.35
    cw = RIGHT - x
    points = [("sparkle", "AI agents plan", "LLM: AI that reads and writes text", PURPLE),
              ("check", "Code does the maths", "Same input, same answer", ORANGE),
              ("lock", "Data stays private", "AI sees only summaries", PURPLE)]
    ch = (h - 2 * 0.25) / 3
    for i, (icon, head, sub, color) in enumerate(points):
        y = Y0 + i * (ch + 0.25)
        card(slide, x, y, cw, ch, name=f"Card {head}")
        deck.icon(slide, icon, x + 0.2, y + 0.25, 0.7, color)
        text(slide, x + 1.05, y + 0.15, cw - 1.2, ch - 0.3, [[(head, {"bold": True, "color": PURPLE, "size": 22})], sub],
             anchor=MSO_ANCHOR.MIDDLE)


def slide_flow(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "How It Works", CHAITANY,
        "Here is one run in six simple steps. The user uploads a data file and asks a question in plain English. The AI "
        "understands what to predict and what kind of problem it is. Then the engine tries several models and tunes "
        "them automatically. The verifier checks the results on data the model has never seen. Finally, the system "
        "explains the main reasons and suggests actions. If no AI is connected, every step still works with safe rules.")
    steps = [("upload", "Upload data", "CSV or Excel file"),
             ("chat", "Ask a question", "In plain English"),
             ("sparkle", "AI understands", "Finds what to predict"),
             ("gear", "Auto-train models", "Tries many models"),
             ("shield", "Check results", "Tests on unseen data"),
             ("bulb", "Explain & recommend", "Reasons and actions")]
    gap = 0.34
    w = (CONTENT_W - 5 * gap) / 6
    y, h = Y0 + 0.05, 3.55
    for i, (icon, head, desc) in enumerate(steps):
        x = LEFT + i * (w + gap)
        card(slide, x, y, w, h, name=f"Card {head}")
        d = 1.0
        deck.icon(slide, icon, x + (w - d) / 2, y + 0.3, d, ORANGE if i % 2 else PURPLE)
        badge(slide, x + (w - d) / 2 + d - 0.32, y + 0.14, 0.46, i + 1)
        text(slide, x + 0.1, y + 1.45, w - 0.2, 0.75, head, bold=True, color=PURPLE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.BOTTOM)
        text(slide, x + 0.1, y + 2.35, w - 0.2, 1.05, desc, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            arrow(slide, x + w + 0.06, y + 0.62, gap - 0.12, 0.36)
    card(slide, LEFT, 6.0, CONTENT_W, 0.85, WHITE, line=ORANGE, line_w=1.5, name="Callout")
    deck.icon(slide, "check", LEFT + 0.25, 6.08, 0.68, ORANGE)
    text(slide, LEFT + 1.15, 6.0, CONTENT_W - 1.4, 0.85, "No AI connected? Every step still works with safe rules",
         size=22, bold=True, color=PURPLE, anchor=MSO_ANCHOR.MIDDLE)


def slide_team(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Modules & Team Roles", CHAITANY,
        "We split the work into four modules, one per member. Raj led the AI agents: they understand the question, plan "
        "the work and write the insights. Bhavesh built the machine learning engine that trains, tunes and checks the "
        "models. I worked on the backend: the API server, the database and the background runs. Varun built the web "
        "dashboard and led testing and documentation. We met every week and joined our modules through clear "
        "interfaces.")
    modules = {RAJ: ("sparkle", "GenAI agents", "Understand the question, plan, write insights"),
               BHAVESH: ("sliders", "ML engine", "Train, tune and check models automatically"),
               CHAITANY: ("server", "Backend, API & database", "API server, database, background runs"),
               VARUN: ("monitor", "Dashboard, testing & docs", "Web screens, tests and reports")}
    h = (Y1 - Y0 - GAP) / 2
    for i, (name, roll) in enumerate(STUDENTS):
        icon, module, desc = modules[name]
        x, y = LEFT + (i % 2) * (HALF_W + GAP), Y0 + (i // 2) * (h + GAP)
        color = PURPLE if i in (0, 3) else ORANGE
        card(slide, x, y, HALF_W, h, name=f"Card {name}")
        band = shape(slide, x, y, HALF_W, 0.8, color, kind=MSO_SHAPE.ROUND_2_SAME_RECTANGLE, name=f"Band {name}")
        band.adjustments[0] = 0.14 / 0.8
        picture(slide, deck.render.icon(icon, WHITE), x + 0.25, y + 0.14, 0.52, 0.52, name="Icon")
        text(slide, x + 0.95, y, HALF_W - 1.2, 0.8, [[(name, {}), *([(f"   {roll}", {})] if roll else [])]], size=22,
             bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.3, y + 0.95, HALF_W - 0.6, h - 1.1, [[(module, {"bold": True, "color": PURPLE, "size": 24})],
                                                              desc], anchor=MSO_ANCHOR.MIDDLE, space_after=4)


def slide_stack(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Technology Stack", CHAITANY,
        "Our technology stack uses open-source libraries, all in Python. For machine learning we use scikit-learn, "
        "XGBoost and LightGBM, with Optuna to tune settings and SHAP to explain results. The AI language model can be "
        "OpenAI, Groq, Gemini or a local Ollama model. The backend uses FastAPI with a SQLite database, the dashboard "
        "uses Streamlit and Plotly, and reports are made with ReportLab. We test with pytest. Now Varun will show the "
        "dashboard.")
    rows = [("code", "Language & data", ["Python", "pandas", "NumPy"]),
            ("bars", "Machine learning", ["scikit-learn", "XGBoost", "LightGBM", "Optuna", "SHAP"]),
            ("sparkle", "AI model (LLM)", ["OpenAI", "Groq", "Gemini", "Ollama"]),
            ("server", "Server & database", ["FastAPI", "SQLAlchemy", "SQLite"]),
            ("monitor", "Dashboard", ["Streamlit", "Plotly"]),
            ("doc", "Reports & testing", ["ReportLab", "pytest", "Ruff"])]
    pitch = (Y1 - Y0 - 0.6) / (len(rows) - 1)
    for i, (icon, label, tools) in enumerate(rows):
        y = Y0 + i * pitch
        deck.icon(slide, icon, LEFT, y, 0.6, ORANGE if i % 2 else PURPLE)
        text(slide, LEFT + 0.8, y, 2.6, 0.6, label, bold=True, color=PURPLE, anchor=MSO_ANCHOR.MIDDLE)
        x = LEFT + 3.5
        for tool in tools:
            x += chip(slide, x, y + 0.025, tool) + 0.2


# ---------------------------------------------------------------- slides 9-10: Varun (dashboard)

def two_screens(slide: Any, screens: list[tuple[str, tuple[float, float, float, float], str, str]]) -> None:
    """Two screenshots side by side with a caption under each; both keep their aspect ratio."""
    box_h = 3.95
    for i, (file, crop, head, desc) in enumerate(screens):
        x = LEFT + i * (HALF_W + GAP)
        data, ratio = image_bytes(SCREENS / file, crop)
        fit_picture(slide, data, ratio, x, Y0, HALF_W, box_h, name=f"Picture {file}", line=FRAME)
        text(slide, x, Y0 + box_h + 0.12, HALF_W, 0.75, [[(head, {"bold": True, "color": PURPLE, "size": 22})], desc],
             align=PP_ALIGN.CENTER)


def slide_dashboard(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Dashboard", VARUN,
        "This is the dashboard a business user sees. On the left is the Home page: one click on Demo Mode uploads a "
        "sample customer file, asks a churn question and starts a fast run. Simple step cards guide a new user. On the "
        "right is the Results page: it shows the best model, its test score, whether it passed the checks, and a chart "
        "comparing all the models tried. No coding is needed at any point.")
    two_screens(slide, [("01_home.png", (0, 0, 1, 1000 / 1008), "Home", "Start a demo with one click"),
                        ("05_results.png", (0, 0, 1, 1000 / 1306), "Results", "Best model and its quality checks")])


def slide_decisions(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "Decision Center & What-If", VARUN,
        "The Decision Center turns the model into action. On the left is the decision brief: a short summary, the main "
        "reasons behind churn, and risk groups, high, medium and low, so the team knows whom to call first. On the right "
        "is the what-if simulator: the user changes an input, such as the contract type, and sees how the chance of "
        "churn changes. These screens come from one demo run. Now Bhavesh will share the results.")
    left = 300 / 1600
    two_screens(slide, [("06_decision_center.png", (left, 255 / 2412, 1, 1330 / 2412), "Decision Center",
                         "Summary, risk groups and actions"),
                        ("07_what_if.png", (left, 255 / 1470, 1, 1330 / 1470), "What-if",
                         "Change an input, see the new result")])


# ---------------------------------------------------------------- slides 11-12: Bhavesh

def slide_results(deck: Deck) -> None:
    f = deck.facts
    bench = f["bench"]
    ab = f["ablation"]
    fixed, tuned = (ab[key]["per_case"]["Churn"]["score"][0] for key in ("no_threshold", "full"))
    beats, total = f["beats"]
    slide = deck.new(
        "Title Only", "Results", BHAVESH,
        f"We tested IDSS on {len(bench)} synthetic datasets, {f['seeds']} runs each. The chart compares our model, in "
        "purple, with a simple guess, in orange, such as always giving the most common answer. Higher is better, and our "
        f"model wins on {beats} of {total} datasets. One small idea helped a lot: tuning the yes or no cut-off. For "
        f"churn, it raised the F1 score from {fixed:.3f} to {tuned:.3f}. F1 shows how well we catch the customers who "
        "really leave.")
    chart_w = 7.75
    text(slide, LEFT, Y0, chart_w, 0.42, [[("Score from 0 to 1", {"bold": True, "color": PURPLE}),
                                           f" · higher is better · mean of {f['seeds']} runs"]], name="Body chart note")
    data = CategoryChartData()
    data.categories = [b["dataset"] for b in bench]
    data.add_series("Our model (IDSS)", [b["nps"][0] for b in bench])
    data.add_series("Simple guess (baseline)", [max(b["baseline_nps"][0], 0.0) for b in bench])
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(LEFT), Inches(Y0 + 0.5), Inches(chart_w),
                                   Inches(Y1 - Y0 - 0.5), data)
    frame.name = "Chart results"
    chart = frame.chart
    chart.font.name, chart.font.size = FONT, Pt(BODY_PT)
    chart.font.color.rgb = rgb(GREY)
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(BODY_PT)
    sp_pr = etree.fromstring('<c:spPr xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                             'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                             '<a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>')
    chart._chartSpace.find(qn("c:chart")).addnext(sp_pr)
    value = chart.value_axis
    value.visible = False
    value.has_major_gridlines = False
    value.maximum_scale, value.minimum_scale = 1.18, 0.0
    category = chart.category_axis
    category.reverse_order = True
    category.major_tick_mark = XL_TICK_MARK.NONE
    category.tick_labels.font.size = Pt(BODY_PT)
    category.format.line.color.rgb = rgb(FRAME)
    plot = chart.plots[0]
    plot.gap_width, plot.overlap = 55, -8
    for series, color in zip(plot.series, (PURPLE, ORANGE), strict=True):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = rgb(color)
        series.invert_if_negative = False
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = "0.00", False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(BODY_PT)
    labels.font.color.rgb = rgb(GREY)
    # right: two callouts
    x = LEFT + chart_w + 0.35
    w = RIGHT - x
    h = (Y1 - Y0 - GAP) / 2
    for i, (icon, head, value_text, desc) in enumerate((
            ("trophy", "Our model wins", f"{beats} / {total}", "datasets beat the simple guess"),
            ("sliders", "Tuning the cut-off", f"{fixed:.3f} → {tuned:.3f}", "churn F1 (catch score)"))):
        y = Y0 + i * (h + GAP)
        tile(slide, x, y, w, h, name="Tile result")
        deck.icon(slide, icon, x + 0.25, y + 0.2, 0.6, PURPLE if i == 0 else ORANGE)
        text(slide, x + 1.0, y + 0.2, w - 1.15, 0.6, head, name="Heading", size=22, bold=True, color=PURPLE,
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.25, y + 0.85, w - 0.45, 0.85, value_text, name="Stat value", size=40, bold=True,
             color=ORANGE, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.25, y + 1.7, w - 0.45, 0.45, desc)


def slide_testing(deck: Deck) -> None:
    f = deck.facts
    o = f["overall"]
    if f["tests_passed"] != f["tests_offline"]:
        raise SystemExit("docs/testing.md: not every offline test passed")
    verified, runs = f["verified"]
    slide = deck.new(
        "Title Only", "Testing & Evaluation", BHAVESH,
        f"Here are our key numbers. The success rate, SR, is {o['sr'][0]:.2f}: every one of the {o['runs']} runs "
        f"finished. The model score, NPS, measures quality from 0 to 1; we got {o['nps'][0]:.3f}. The combined score, "
        f"CS, is the average of both: {o['cs'][0]:.3f}. For software quality, we wrote {f['tests']} automated tests; "
        f"{f['tests_offline']} run offline and all pass. They cover {f['coverage_total']}% of our code. Also, "
        f"{verified} of {runs} runs passed every verifier check. Now Varun will conclude.")
    groups = [(f"Evaluation · {len(f['bench'])} synthetic datasets, {f['seeds']} runs each",
               [(f"{o['sr'][0]:.2f}", "Success rate (SR)", "runs that finished"),
                (f"{o['nps'][0]:.3f}", "Model score (NPS)", "quality from 0 to 1"),
                (f"{o['cs'][0]:.3f}", "Combined score (CS)", "average of SR and NPS")]),
              ("Software testing",
               [(f"{f['tests']}", "Automated tests", f"{f['tests_offline']} run offline, all pass"),
                (f"{f['coverage_total']}%", "Code coverage", "code lines run by the tests")])]
    y = Y0
    tile_h = 1.85
    for label, tiles in groups:
        text(slide, LEFT, y, CONTENT_W, 0.42, label, bold=True, color=PURPLE)
        y += 0.47
        w = (CONTENT_W - (len(tiles) - 1) * GAP) / len(tiles)
        for i, (number, head, desc) in enumerate(tiles):
            x = LEFT + i * (w + GAP)
            tile(slide, x, y, w, tile_h, name=f"Tile {head}")
            text(slide, x + 0.2, y + 0.12, w - 0.4, 0.85, number, name="Stat value", size=48, bold=True, color=ORANGE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            text(slide, x + 0.2, y + 0.98, w - 0.4, 0.8, [[(head, {"bold": True, "color": PURPLE})], desc],
                 align=PP_ALIGN.CENTER)
        y += tile_h + 0.2


# ---------------------------------------------------------------- slides 13-14: Varun

def slide_conclusion(deck: Deck) -> None:
    f = deck.facts
    o = f["overall"]
    verified, runs = f["verified"]
    beats, total = f["beats"]
    if beats != total:
        raise SystemExit("Slide 13 says the model beats a simple guess on every dataset; evaluation.json disagrees")
    slide = deck.new(
        "Two Content", "Conclusion & Future Scope", VARUN,
        f"To conclude. IDSS lets a user ask a question in plain English and get an explained decision. All {o['runs']} "
        f"runs finished, our models beat a simple guess on all {total} datasets, and {verified} of {runs} runs passed "
        "every check. For the future, we want to test with real company data and real users. When a user says missing "
        "a case is costly, the system should tune the cut-off for recall, using the F2 score. We also plan reports in "
        "Hindi and Marathi.")
    columns = [(2, "trophy", "What we achieved", PURPLE,
                ["Ask in plain English, get an explained decision",
                 f"All {o['runs']} runs finished; {verified} passed every check",
                 f"Beat a simple guess on all {total} datasets"]),
               (3, "trend", "Future scope", ORANGE,
                ["Test on real company data with real users",
                 "Catch more cases: tune cut-off for recall (F2) on request",
                 "Reports in Hindi and Marathi"])]
    for i, (idx, icon, head, color, bullets) in enumerate(columns):
        x = LEFT + i * (HALF_W + GAP)
        card(slide, x, Y0, HALF_W, Y1 - Y0, name=f"Card {head}")
        deck.icon(slide, icon, x + 0.3, Y0 + 0.3, 0.8, color)
        text(slide, x + 1.3, Y0 + 0.3, HALF_W - 1.6, 0.8, head, name="Heading", size=26, bold=True, color=PURPLE,
             anchor=MSO_ANCHOR.MIDDLE)
        body = deck.placeholder(slide, idx, f"Body {head}")
        body.left, body.top = Inches(x + 0.35), Inches(Y0 + 1.4)
        body.width, body.height = Inches(HALF_W - 0.65), Inches(Y1 - Y0 - 1.65)
        write(body.text_frame, bullets, size=22, bullets=color, space_after=16)
        slide.shapes._spTree.append(body._element)  # the layout put it first: bring it in front of the card


def slide_references(deck: Deck) -> None:
    slide = deck.new(
        "Title Only", "References", VARUN,
        "These are our main references. AutoML-Agent from ICML 2025 inspired our agent design. Auto-sklearn, Data "
        "Interpreter and SHAP are the related works we compared, and Optuna is the tuning library we use. Thank you to "
        f"our guide, {GUIDE}, and to the department for the support. Thank you all for listening. We are happy to take "
        "your questions: Raj on the agents, Chaitany on the backend, Bhavesh on the models and results, and I on the "
        "dashboard.")
    refs = [("P. Trirat et al., ", "AutoML-Agent", ", ICML 2025"),
            ("M. Feurer et al., ", "Auto-sklearn", ", NIPS 2015"),
            ("S. Hong et al., ", "Data Interpreter", ", arXiv 2024"),
            ("S. M. Lundberg, S.-I. Lee, ", "SHAP", ", NIPS 2017"),
            ("T. Akiba et al., ", "Optuna", ", KDD 2019")]
    w = 6.95
    box = text(slide, LEFT, Y0 + 0.25, w, Y1 - Y0 - 0.5,
               [[before, (name, {"bold": True, "color": PURPLE}), after] for before, name, after in refs],
               name="Body references", space_after=24)
    numbered(box.text_frame)
    x = LEFT + w + 0.4
    card(slide, x, Y0, RIGHT - x, Y1 - Y0, PURPLE, name="Card thanks")
    deck.icon(slide, "chat", x + (RIGHT - x - 1.1) / 2, Y0 + 0.65, 1.1, ORANGE)
    text(slide, x + 0.2, Y0 + 2.0, RIGHT - x - 0.4, 0.95, "Thank You", name="Stat thanks", size=48, bold=True,
         color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.2, Y0 + 3.0, RIGHT - x - 0.4, 0.7, "Questions?", size=32, color=LIGHT, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)


def build(facts: dict[str, Any]) -> Deck:
    renderer = Renderer()
    try:
        deck = Deck(facts, renderer)
        for make in (slide_title, slide_problem, slide_objectives, slide_literature,  # Raj
                     slide_architecture, slide_flow, slide_team, slide_stack,  # Chaitany
                     slide_dashboard, slide_decisions,  # Varun
                     slide_results, slide_testing,  # Bhavesh
                     slide_conclusion, slide_references):  # Varun
            make(deck)
    finally:
        renderer.close()
    return deck


# ---------------------------------------------------------------- checks before saving

NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:,\d{3})*(?:\.\d+)?")


def _shapes(collection: Any) -> Iterator[Any]:
    for item in collection:
        if item.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _shapes(item.shapes)
        else:
            yield item


def _formats(value: float) -> set[str]:
    if isinstance(value, int):
        return {str(value), f"{value:,}"}
    found = {f"{value:.{digits}f}" for digits in (1, 2, 3)}
    if abs(value) >= 10:
        found |= {f"{value:.0f}", f"{value:,.0f}"}
    return found


def source_numbers() -> tuple[set[str], set[str]]:
    """Numbers that may appear on the slides: (results from evaluation.json + testing.md, structural exceptions)."""
    data = json.loads(EVALUATION.read_text(encoding="utf-8"))
    summary = data["summary"]
    results: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, int | float):
            results.update(_formats(node))
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(summary)
    bench = summary["benchmark"]
    derived = [len(data["seeds"]), len(bench), sum(b["verified"] for b in bench), sum(b["runs"] for b in bench),
               sum(b["nps"][0] > b["baseline_nps"][0] for b in bench)]
    for value in derived:
        results.update(_formats(value))
    for number in NUMBER.findall(TESTING.read_text(encoding="utf-8")):
        results.update({number, number.replace(",", "")})
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", REFERENCES.read_text(encoding="utf-8")))
    structural = ({roll for _, roll in STUDENTS if roll} | set(re.findall(r"\d{4}", ACADEMIC_YEAR)) | years
                  | {"0", "1"})  # 0 and 1 = bounds of the 0-1 score scale
    return results, structural


def check_numbers(prs: Any) -> list[str]:
    results, structural = source_numbers()
    allowed = results | structural
    problems = []
    for number, slide in enumerate(prs.slides, start=1):
        texts = [("notes", slide.notes_slide.notes_text_frame.text)]
        for item in _shapes(slide.shapes):
            if item.name.startswith("Step"):
                continue
            if item.has_text_frame:
                texts.append((item.name, item.text_frame.text))
            if item.has_table:
                texts.extend((item.name, c.text_frame.text) for row in item.table.rows for c in row.cells)
        for where, content in texts:
            for found in NUMBER.findall(content):
                if found not in allowed:
                    problems.append(f"slide {number} '{where}': {found} is not in evaluation.json / testing.md")
    return problems


def check_notes(deck: Deck) -> list[str]:
    problems = []
    if len(deck.prs.slides) != SLIDES:
        problems.append(f"{len(deck.prs.slides)} slides, expected exactly {SLIDES}")
    for number, slide in enumerate(deck.prs.slides, start=1):
        notes = slide.notes_slide.notes_text_frame.text
        first, _, script = notes.partition("\n")
        if first != f"Presenter: {PRESENTERS[number]}":
            problems.append(f"slide {number}: notes start with {first!r}")
        if not NOTE_WORDS[0] <= len(script.split()) <= NOTE_WORDS[1]:
            problems.append(f"slide {number}: {len(script.split())} words of notes")
    return problems


def _solid(item: Any) -> str | None:
    try:
        if item.fill.type == MSO_FILL.SOLID:
            return str(item.fill.fore_color.rgb)
    except (AttributeError, TypeError):
        pass
    return None


def _luminance(color: str) -> float:
    channels = [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _runs_contrast(tf: Any, background: str, where: str) -> list[str]:
    problems = []
    for p in tf.paragraphs:
        for r in p.runs:
            if not r.text.strip() or r.font.color.type is None:
                continue
            size = r.font.size.pt if r.font.size else BODY_PT
            large = size >= 24 or (bool(r.font.bold) and size >= 18.66)
            need = 3.0 if large else 4.5
            ratio = contrast(str(r.font.color.rgb), background)
            if ratio < need:
                problems.append(f"{where}: contrast {ratio:.2f} < {need} ({r.text[:30]!r} on #{background})")
    return problems


def check_visuals(prs: Any) -> list[str]:
    """Text contrast against what lies behind it, no text hidden under a later shape, no stretched pictures, no empty
    placeholders."""
    problems = []
    for number, slide in enumerate(prs.slides, start=1):
        items = list(_shapes(slide.shapes))
        filled: list[tuple[float, float, float, float, str]] = []
        for z, item in enumerate(items):
            where = f"slide {number} '{item.name}'"
            box = (item.left, item.top, item.left + item.width, item.top + item.height)
            own = _solid(item)
            if item.is_placeholder and (not item.has_text_frame or not item.text_frame.text.strip()):
                problems.append(f"{where}: empty placeholder")
            if item.has_text_frame and item.text_frame.text.strip():
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                behind = next((f[4] for f in reversed(filled) if f[0] <= cx <= f[2] and f[1] <= cy <= f[3]), WHITE)
                problems += _runs_contrast(item.text_frame, own or behind, where)
                above = next((o.name for o in items[z + 1:] if _solid(o) and o.left <= cx <= o.left + o.width
                              and o.top <= cy <= o.top + o.height), None)
                if above:
                    problems.append(f"{where}: text is hidden under '{above}'")
            if item.has_table:
                for row in item.table.rows:
                    for c in row.cells:
                        problems += _runs_contrast(c.text_frame, _solid(c) or WHITE, where)
            if item.shape_type == MSO_SHAPE_TYPE.PICTURE:
                px_w, px_h = item.image.size
                image_ratio = px_w * (1 - item.crop_left - item.crop_right) / (px_h * (1 - item.crop_top
                                                                                    - item.crop_bottom))
                if abs(image_ratio / (item.width / item.height) - 1) > 0.01:
                    problems.append(f"{where}: picture stretched ({image_ratio:.3f} vs {item.width / item.height:.3f})")
            if own:
                filled.append((*box, own))
    return problems


# ---------------------------------------------------------------- export + layout check (PowerPoint COM)

TOLERANCE_PT = 1.5


def check_layout(pres: Any) -> list[str]:
    """Inside the safe area, text fits its box, no broken words, chips on one line, minimum text sizes."""
    problems: list[str] = []
    if pres.Slides.Count != SLIDES:
        problems.append(f"{pres.Slides.Count} slides, expected exactly {SLIDES}")
    left, top, right, bottom = (v * 72 for v in (LEFT, TOP, RIGHT, BOTTOM))

    def shapes(collection: Any) -> Iterator[Any]:
        for item in collection:
            if item.Type == 6:  # msoGroup
                yield from shapes(item.GroupItems)
            else:
                yield item

    def sizes(rng: Any, minimum: float, where: str) -> None:
        for i in range(1, rng.Runs().Count + 1):
            run = rng.Runs(i)
            if run.Text.strip() and run.Font.Size < minimum:
                problems.append(f"{where}: {run.Font.Size:g} pt < {minimum} pt ({run.Text[:30]!r})")

    for slide in pres.Slides:
        for item in shapes(slide.Shapes):
            where = f"slide {slide.SlideIndex} '{item.Name}'"
            if (item.Left < left - TOLERANCE_PT or item.Top < top - TOLERANCE_PT
                    or item.Left + item.Width > right + TOLERANCE_PT or item.Top + item.Height > bottom + TOLERANCE_PT):
                problems.append(f"{where} is outside the safe area ({item.Left / 72:.2f}, {item.Top / 72:.2f}, "
                                f"{(item.Left + item.Width) / 72:.2f}, {(item.Top + item.Height) / 72:.2f} in)")
            if item.Type == 14 and (not item.HasTextFrame or not item.TextFrame.HasText):  # msoPlaceholder
                problems.append(f"{where}: empty placeholder")
            if item.HasTable:
                for r in range(1, item.Table.Rows.Count + 1):
                    for c in range(1, item.Table.Columns.Count + 1):
                        sizes(item.Table.Cell(r, c).Shape.TextFrame.TextRange, BODY_PT, f"{where} cell {r},{c}")
            if not item.HasTextFrame or not item.TextFrame.HasText:
                continue
            frame = item.TextFrame
            rng = frame.TextRange
            if rng.BoundHeight > item.Height + TOLERANCE_PT or rng.BoundTop < item.Top - TOLERANCE_PT \
                    or rng.BoundTop + rng.BoundHeight > item.Top + item.Height + TOLERANCE_PT:
                problems.append(f"{where}: text is {rng.BoundHeight:.0f} pt high in a {item.Height:.0f} pt box "
                                f"({rng.Text[:40]!r})")
            lines = []
            while len(lines) < 40 and (line := rng.Lines(len(lines) + 1, 1)).Length > 0:
                lines.append(line.Text)
            for before, after in zip(lines, lines[1:]):
                if before[-1:].isalnum() and after[:1].isalnum():
                    problems.append(f"{where}: word broken across lines ({before[-12:]!r} | {after[:12]!r})")
            if item.Name.startswith(("Chip", "Step")) and len(lines) > 1:
                problems.append(f"{where}: wraps to {len(lines)} lines ({rng.Text[:40]!r})")
            sizes(rng, TITLE_PT if item.Name.startswith("Title") else BODY_PT, where)
    return problems


def export_powerpoint(pptx: Path, pdf: Path, png_dir: Path) -> list[str]:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(str(pptx), -1, 0, 0)  # ReadOnly, Untitled = no, WithWindow = no
        try:
            problems = check_layout(pres)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-export", action="store_true", help="only write the .pptx (no PDF / PNG)")
    parser.add_argument("--draft", action="store_true", help="keep the exports even if the layout check fails")
    args = parser.parse_args()

    deck = build(load_facts())
    numbers = check_numbers(deck.prs)
    for problem in numbers:
        print(f"  NUMBER: {problem}")
    if numbers:
        raise SystemExit(f"Number check failed ({len(numbers)} numbers not found in the result files)")
    problems = check_notes(deck) + check_visuals(deck.prs)
    for problem in problems:
        print(f"  CHECK: {problem}")
    if problems and not args.draft:
        raise SystemExit(f"Deck check failed ({len(problems)} problems); fix them or rerun with --draft.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deck.prs.save(str(PPTX_OUT))
    print(f"Saved {PPTX_OUT.relative_to(ROOT)}: {len(deck.prs.slides)} slides on {TEMPLATE.name} "
          f"({PPTX_OUT.stat().st_size / 1024:,.0f} KB); every number found in evaluation.json / testing.md")
    for number, talk in enumerate(deck.talk, start=1):
        print(f"  {number:>2}  {talk['title']:<60} {talk['presenter']:<22} {talk['words']:>3} words")
    for name, _ in STUDENTS:
        mine = [i for i, t in enumerate(deck.talk, start=1) if t["presenter"] == name]
        words = sum(deck.talk[i - 1]["words"] for i in mine)
        print(f"  {name:<22} slides {', '.join(map(str, mine)):<14} {words:>4} words  ~{words / SPEAKING_WPM:.1f} min")
    if args.no_export:
        return
    try:
        layout = export_powerpoint(PPTX_OUT, PDF_OUT, PNG_DIR)
    except ImportError as exc:  # pywin32 missing / not Windows
        print(f"PowerPoint automation not available: {exc}")
        layout = None
    except Exception as exc:  # COM errors (no PowerPoint, file locked by a viewer, ...)
        print(f"PowerPoint export failed: {exc}")
        layout = None
    if layout is None:
        if export_libreoffice(PPTX_OUT, PDF_OUT, PNG_DIR):
            print("PDF converted with LibreOffice (no layout check); slide PNGs rendered with PyMuPDF.")
        else:
            raise SystemExit("Neither PowerPoint nor LibreOffice is available: open the .pptx in PowerPoint and "
                             "use File > Export > PDF.")
    else:
        for problem in layout:
            print(f"  LAYOUT: {problem}")
        if layout and not args.draft:
            raise SystemExit(f"Layout check failed ({len(layout)} problems); fix them or rerun with --draft.")
        print("Layout check passed: everything inside the safe area, text fits and is at least 20 pt (titles 32 pt)."
              if not layout else "Layout problems kept (--draft).")
    print(f"Saved {PDF_OUT.relative_to(ROOT)} ({PDF_OUT.stat().st_size / 1024:,.0f} KB) and "
          f"{len(list(PNG_DIR.glob('slide_*.png')))} slide PNGs in {PNG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
