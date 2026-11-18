"""Build the final project report in the college format (Step 10B).

Reads the Step 10A content in docs/report/*.md and writes docs/report/final/IDSS_Project_Report.docx in the
layout of docs/format/Project_Report_Format_new.docx (G H Raisoni College of Engineering and Management,
Jalgaon): A4; margins top 1", bottom 1.15", left 1.5", right 0.6"; Times New Roman with 1.5 line spacing;
chapter titles right-aligned 16 pt bold above a 2 1/4 pt rule; Garamond bold captions; page numbers centred
in the footer, lower-case roman up to the abbreviations and Arabic from the abstract.

The page index and the lists of figures and tables are tables whose page numbers are PAGEREF fields pointing
at bookmarks on the headings and captions. Word fills them in: export_pdfs() updates every field twice,
saves the .docx and exports IDSS_Project_Report.pdf (LibreOffice or manual steps as fallback). The PDF is then
checked: every row of the page index and of the lists of figures and tables must show the printed page number of
the page where that heading or caption actually is.

Usage: .venv\\Scripts\\python.exe scripts\\build_report.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Mm, Pt
from PIL import Image

from report_common import (
    ACADEMIC_YEAR,
    BODY_FONT,
    CAPTION_FONT,
    CITY_LINE,
    CODE_FONT,
    COLLEGE,
    DEGREE,
    DEPARTMENT,
    FINAL_DIR,
    GUIDE,
    HOD,
    PROGRAMME,
    PROJECT_TITLE,
    REPORT_DIR,
    STUDENTS,
    Block,
    Numbering,
    add_bookmark,
    add_field,
    add_inline,
    export_pdfs,
    paragraph_border,
    parse_markdown,
    plain,
    set_cell_shading,
    set_page_numbering,
    set_row_flags,
    set_run_font,
    set_style_font,
    set_table_borders,
    set_table_width,
    student_label,
)

CHAPTER_FILES = [
    "ch1_introduction.md",
    "ch2_literature_review.md",
    "ch3_planning_management.md",
    "ch4_analysis_design.md",
    "ch5_implementation.md",
    "ch6_testing.md",
    "ch7_results.md",
    "ch8_conclusion.md",
]
APPENDIX_FILES = ["appendix_a_user_manual.md", "appendix_b_weekly_progress.md"]
OUTPUT = FINAL_DIR / "IDSS_Project_Report.docx"

PAGE_WIDTH_IN, MARGIN_LEFT_IN, MARGIN_RIGHT_IN = 210 / 25.4, 1.5, 0.6
TEXT_WIDTH_IN = PAGE_WIDTH_IN - MARGIN_LEFT_IN - MARGIN_RIGHT_IN  # 6.17 in
TEXT_WIDTH_TWIPS = round(TEXT_WIDTH_IN * 1440)
MAX_FIGURE_HEIGHT_IN = 7.3  # figure + caption stay on one page (text height is 9.54 in)
MAX_IMAGE_PX = 2000  # downscale 300-dpi diagrams: still ~320 dpi at full text width
CODE_CHARS = 78  # Consolas 10 pt characters per line inside the code box
CHAR_EM, CELL_PAD_IN = 0.55, 0.2  # average Times New Roman character width (em); cell margins + slack (in)
KEEP_TABLE_MAX_IN = 4.0  # tables estimated below this height never split across pages
HEADER_FILL, CODE_FILL = "D9D9D9", "F2F2F2"
_CHAPTER = re.compile(r"^(Chapter|Appendix) ([0-9A-Z]+):\s*(.+)$")
_TRAILING_COMMENT = re.compile(r"^(?P<code>.*?\S)\s{2,}(?P<comment>(//|#)\s.*)$")


@dataclass
class IndexEntry:
    kind: str  # front | chapter | section
    label: str  # chapter number or appendix letter (chapter rows only)
    title: str
    anchor: str


# ---------------------------------------------------------------- helpers

def read_md(name: str) -> list[Block]:
    return parse_markdown((REPORT_DIR / name).read_text(encoding="utf-8"))


def split_by_title(blocks: list[Block]) -> dict[str, list[Block]]:
    """Front matter: the blocks under each level-1 heading, keyed by the heading text."""
    parts: dict[str, list[Block]] = {}
    current: list[Block] | None = None
    for block in blocks:
        if block.kind == "heading" and block.level == 1:
            current = parts.setdefault(block.text, [])
        elif current is not None:
            current.append(block)
    return parts


def column_widths(rows: list[list[str]], total_twips: int, size_pt: float) -> list[int]:
    """Column widths from content length: every column gets room for its longest word (capped), the rest of
    the width goes to the columns with the longest text."""
    n = max(len(r) for r in rows)
    char_in = size_pt * CHAR_EM / 72
    texts = [[plain(r[c]) if c < len(r) else "" for r in rows] for c in range(n)]
    natural = [min(max(len(t) for t in col), 70) * char_in + CELL_PAD_IN for col in texts]
    minimum = [min(max((len(w) for t in col for w in t.split(" ")), default=1), 14) * char_in + CELL_PAD_IN
               for col in texts]  # split on plain spaces only: no-break spaces join words
    total_in = total_twips / 1440
    if sum(natural) <= total_in:
        widths = [w * total_in / sum(natural) for w in natural]
    elif sum(minimum) >= total_in:
        widths = [w * total_in / sum(minimum) for w in minimum]
    else:
        extra = [max(nat - mn, 0.0) for nat, mn in zip(natural, minimum)]
        rest = total_in - sum(minimum)
        widths = [mn + rest * e / sum(extra) for mn, e in zip(minimum, extra)]
    twips = [round(w * 1440) for w in widths]
    twips[-1] += total_twips - sum(twips)
    return twips


def table_height_in(rows: list[list[str]], widths_twips: list[int], size_pt: float) -> float:
    """Rough height of a table: wrapped lines per cell from the text length and the column width."""
    char_in = size_pt * CHAR_EM / 72
    height = 0.0
    for row in rows:
        lines = 1
        for text, width in zip(row, widths_twips):
            per_line = max(int((width / 1440 - CELL_PAD_IN) / char_in), 1)
            lines = max(lines, -(-len(plain(text)) // per_line))
        height += lines * size_pt * 1.15 / 72 + 2 / 72
    return height


def wrap_code(lines: list[str], width: int = CODE_CHARS) -> list[str]:
    """Fit code lines into the box: shorten comment padding, move a long trailing comment above its line,
    then wrap at a space with a 4-space continuation indent."""
    out: list[str] = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        match = _TRAILING_COMMENT.match(line)
        if match:
            compact = f"{match['code']}  {match['comment']}"
            if len(compact) <= width:
                out.append(compact)
                continue
            out.append(" " * indent + match["comment"])
            line = match["code"]
        continuation = " " * (indent + 4)
        while len(line) > width:
            cut = line.rfind(" ", indent + 1, width)
            if cut <= indent:
                cut = width
            out.append(line[:cut].rstrip())
            line = continuation + line[cut:].lstrip()
        out.append(line)
    return out


def load_image(path: Path) -> tuple[BytesIO, tuple[int, int]]:
    """PNG stream of an image, downscaled to MAX_IMAGE_PX wide (keeps the .docx small), and its size."""
    with Image.open(path) as image:
        width, height = image.size
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        if width > MAX_IMAGE_PX:
            image = image.resize((MAX_IMAGE_PX, round(height * MAX_IMAGE_PX / width)), Image.LANCZOS)
        stream = BytesIO()
        image.save(stream, format="PNG", optimize=True)
    stream.seek(0)
    return stream, (width, height)


# ---------------------------------------------------------------- builder

class ReportBuilder:
    def __init__(self) -> None:
        self.doc = Document()
        self._configure_page(self.doc.sections[0])
        self._setup_styles()
        self.numbering = Numbering(self.doc)
        self.index: list[IndexEntry] = []
        self.figures: list[tuple[str, str, str]] = []  # number, caption, anchor
        self.tables: list[tuple[str, str, str]] = []
        self.first_after_heading = False
        props = self.doc.core_properties
        props.title = PROJECT_TITLE
        props.author = ", ".join(name for name, _ in STUDENTS)
        props.subject = f"B.Tech {PROGRAMME} project report, {COLLEGE}"
        props.comments = "Generated by scripts/build_report.py from docs/report/*.md"

    # ---- page set-up and styles

    @staticmethod
    def _configure_page(section) -> None:
        section.page_width, section.page_height = Mm(210), Mm(297)
        section.top_margin, section.bottom_margin = Inches(1.0), Inches(1.15)
        section.left_margin, section.right_margin = Inches(MARGIN_LEFT_IN), Inches(MARGIN_RIGHT_IN)
        section.header_distance, section.footer_distance = Inches(0.5), Inches(0.4)

    def _setup_styles(self) -> None:
        styles = self.doc.styles
        defaults = styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
        fonts = defaults.find(qn("w:rFonts"))
        for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn(f"w:{attr}"), BODY_FONT)
        for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn(f"w:{attr}"), None)
        lang = defaults.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            defaults.append(lang)
        lang.set(qn("w:val"), "en-GB")

        normal = styles["Normal"]
        set_style_font(normal, BODY_FONT, 12)
        pf = normal.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.0
        pf.widow_control = True

        def para(name: str, base: str = "Normal", *, font: str | None = None, size: float | None = None,
                 bold: bool | None = None, italic: bool | None = None, align=None, line: float | None = None,
                 before: float | None = None, after: float | None = None, first: Cm | None = None,
                 left=None, right=None, keep_next: bool | None = None, keep_lines: bool | None = None,
                 page_break: bool | None = None):
            style = styles[name] if name in styles else styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style.base_style = styles[base]
            style.quick_style = True
            if font:
                set_style_font(style, font)
            if size is not None:
                style.font.size = Pt(size)
            if bold is not None:
                style.font.bold = bold
            if italic is not None:
                style.font.italic = italic
            fmt = style.paragraph_format
            if align is not None:
                fmt.alignment = align
            if line is not None:
                fmt.line_spacing = line
            if before is not None:
                fmt.space_before = Pt(before)
            if after is not None:
                fmt.space_after = Pt(after)
            if first is not None:
                fmt.first_line_indent = first
            if left is not None:
                fmt.left_indent = left
            if right is not None:
                fmt.right_indent = right
            if keep_next is not None:
                fmt.keep_with_next = keep_next
            if keep_lines is not None:
                fmt.keep_together = keep_lines
            if page_break is not None:
                fmt.page_break_before = page_break
            return style

        justify, center, right_align = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT
        para("IDSS Body", align=justify, line=1.5, first=Cm(1))
        para("IDSS Body First", "IDSS Body", first=Cm(0))
        para("IDSS Plain", align=justify, line=1.5, after=12)
        para("IDSS List", align=justify, line=1.5)
        para("IDSS Quote", align=justify, line=1.5, left=Cm(1), right=Cm(1), before=6, after=6)
        para("IDSS Table", size=11, align=WD_ALIGN_PARAGRAPH.LEFT, before=1, after=1)
        para("IDSS Index", size=12, before=2, after=2)
        para("IDSS Code", font=CODE_FONT, size=10)
        para("IDSS Caption", font=CAPTION_FONT, size=12, bold=True, align=center, before=6, after=6,
             keep_lines=True)
        para("IDSS Figure", align=center, before=12, keep_next=True)
        para("IDSS Chapter Label", size=16, bold=True, align=right_align, after=6, keep_next=True, page_break=True)
        para("IDSS Front Title", size=20, bold=True, align=center, after=18, keep_next=True)
        para("IDSS Reference", align=justify, line=1.5, left=Inches(0.5), first=Inches(-0.5), after=6)
        para("IDSS Spacer", size=6)
        self._outline(styles["IDSS Front Title"], 0)

        for level, size, align in ((1, 16, right_align), (2, 18, None), (3, 14, None)):
            style = styles[f"Heading {level}"]
            set_style_font(style, BODY_FONT, size)
            style.font.bold, style.font.italic = True, False
            rpr = style.element.get_or_add_rPr()
            color = rpr.find(qn("w:color"))
            if color is not None:
                rpr.remove(color)
            fmt = style.paragraph_format
            fmt.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.LEFT
            fmt.space_before = Pt(0 if level == 1 else 12)  # one blank line (12 pt) before each heading
            fmt.space_after = Pt(24 if level == 1 else 6)
            fmt.line_spacing, fmt.keep_with_next, fmt.keep_together = 1.0, True, True
            self._outline(style, level - 1)

    @staticmethod
    def _outline(style, level: int) -> None:
        ppr = style.element.get_or_add_pPr()
        outline = ppr.find(qn("w:outlineLvl"))
        if outline is None:
            outline = OxmlElement("w:outlineLvl")
            ppr.append(outline)
        outline.set(qn("w:val"), str(level))

    # ---- sections and footers

    def new_section(self):
        """Start a new section on a new page. The closing section break goes into the last paragraph, so a
        full last page never spills an empty paragraph onto an extra blank page."""
        body = self.doc.element.body
        sentinel = body.sectPr
        last = sentinel.getprevious()
        if last is None or last.tag != qn("w:p") or (last.pPr is not None and last.pPr.sectPr is not None):
            last = self.doc.add_paragraph(style="IDSS Spacer")._p
        last.set_sectPr(copy.deepcopy(sentinel))
        for ref in sentinel.xpath("w:headerReference|w:footerReference"):
            sentinel.remove(ref)  # the new section starts linked to the previous footer
        section = self.doc.sections[-1]
        section.start_type = WD_SECTION.NEW_PAGE
        self._configure_page(section)
        return section

    @staticmethod
    def set_footer(section, with_number: bool) -> None:
        footer = section.footer
        footer.is_linked_to_previous = False
        name = footer.paragraphs[0]
        name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(name.add_run(COLLEGE), BODY_FONT, 10)
        if with_number:
            number = footer.add_paragraph()
            number.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_field(number, "PAGE", "1", size=12)

    # ---- small building blocks

    def p(self, text: str = "", style: str = "Normal", **kwargs):
        paragraph = self.doc.add_paragraph(style=style)
        if text:
            add_inline(paragraph, text, **kwargs)
        return paragraph

    def line(self, text: str, size: float, *, bold: bool = False, italic: bool = False,
             align=WD_ALIGN_PARAGRAPH.CENTER, before: float = 0, after: float = 0, spacing: float = 1.15):
        paragraph = self.doc.add_paragraph()
        fmt = paragraph.paragraph_format
        paragraph.alignment = align
        fmt.line_spacing, fmt.space_before, fmt.space_after = spacing, Pt(before), Pt(after)
        if text:
            set_run_font(paragraph.add_run(text), BODY_FONT, size, bold, italic)
        return paragraph

    def front_title(self, text: str, anchor: str | None = None, *, new_page: bool = True, size: float = 20,
                    before: float = 0):
        paragraph = self.doc.add_paragraph(text, style="IDSS Front Title")
        fmt = paragraph.paragraph_format
        fmt.page_break_before = new_page
        if size != 20:  # the ABSTRACT heading is 14 pt
            for run in paragraph.runs:
                run.font.size = Pt(size)
        if before:
            fmt.space_before = Pt(before)
        if anchor:
            add_bookmark(paragraph, anchor)
        self.first_after_heading = True
        return paragraph

    def spacer(self) -> None:
        self.doc.add_paragraph(style="IDSS Spacer")

    def borderless_table(self, rows: int, widths_in: list[float]):
        table = self.doc.add_table(rows=rows, cols=len(widths_in))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, "nil")
        set_table_width(table, [round(w * 1440) for w in widths_in])
        return table

    # ---- content collection (index entries need every heading and caption before the index is written)

    def collect(self, files: list[list[Block]], first_file_no: int = 0) -> None:
        """Set bookmark names on chapter / section headings and captions and record them for the index."""
        for file_no, blocks in enumerate(files, start=first_file_no):
            section_no = 0
            for block in blocks:
                if block.kind == "heading" and block.level == 1:
                    match = _CHAPTER.match(block.text)
                    if not match:
                        raise ValueError(f"Unexpected chapter heading: {block.text}")
                    block.anchor = f"_IDX_C_{match.group(2)}"
                    self.index.append(IndexEntry("chapter", match.group(2), match.group(3), block.anchor))
                elif block.kind == "heading" and block.level == 2:
                    section_no += 1
                    block.anchor = f"_IDX_S_{file_no}_{section_no}"
                    self.index.append(IndexEntry("section", "", plain(block.text), block.anchor))
                elif block.kind == "caption":
                    block.anchor = f"_IDX_{block.caption_kind}_{block.number.replace('.', '_')}"
                    entry = (block.number, plain(block.text), block.anchor)
                    if block.caption_kind == "Figure":
                        self.figures.append(entry)
                    elif block.caption_kind == "Table":
                        self.tables.append(entry)

    # ---- front matter

    def title_page(self) -> None:
        self.line("A", 12)
        self.line("Project Report", 16, bold=True)
        self.line("on", 12, after=6)
        self.line(PROJECT_TITLE.upper(), 20, bold=True, after=12)
        self.line("submitted for Partial Fulfillment of", 12)
        self.line("the Requirements of the Degree", 12)
        self.line("of", 12)
        self.line(DEGREE, 16, bold=True)
        self.line("in", 12)
        self.line(PROGRAMME, 16, bold=True)
        self.line("to", 12)
        self.line(COLLEGE, 16, bold=True, after=12)
        self.line("Submitted by", 12, after=4)
        for name, roll in STUDENTS:
            self.line(student_label(name, roll), 14, bold=True)
        self.line("Under the Guidance of", 12, before=12, after=4)
        self.line(GUIDE, 14, bold=True)
        logo = self.line("", 12)  # empty centred space for the college logo
        logo.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        logo.paragraph_format.line_spacing = Pt(100)
        self.line(DEPARTMENT, 12, bold=True)
        self.line("G H Raisoni College of Engineering and Management,", 14)
        self.line(CITY_LINE, 14)
        self.line(ACADEMIC_YEAR, 14)

    def certificate(self) -> None:
        self.line(COLLEGE, 16, bold=True, spacing=1.0)
        self.line(DEPARTMENT, 12, bold=True, before=6, spacing=1.0)
        self.line("CERTIFICATE", 20, bold=True, before=30, after=24, spacing=1.0)
        intro = self.p(style="IDSS Plain")
        intro.paragraph_format.space_after = Pt(6)
        set_run_font(intro.add_run("This is to certify that the project entitled “"), BODY_FONT, 12)
        set_run_font(intro.add_run(PROJECT_TITLE), BODY_FONT, 12, bold=True, italic=True)
        set_run_font(intro.add_run("”, submitted by"), BODY_FONT, 12)
        for name, roll in STUDENTS:
            self.line(student_label(name, roll), 12, bold=True, spacing=1.3)
        body = self.p(style="IDSS Plain")
        body.paragraph_format.space_before = Pt(6)
        set_run_font(body.add_run("in partial fulfillment of the degree of "), BODY_FONT, 12)
        set_run_font(body.add_run(DEGREE), BODY_FONT, 12, italic=True)
        set_run_font(body.add_run(" in "), BODY_FONT, 12)
        set_run_font(body.add_run(PROGRAMME), BODY_FONT, 12, italic=True)
        set_run_font(body.add_run(" has been satisfactorily carried out under my guidance as per the requirement"
                                  f" of {COLLEGE}."), BODY_FONT, 12)
        for label, value in (("Date: ", "____________________"), ("Place: ", "Jalgaon")):
            paragraph = self.line("", 12, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.5)
            set_run_font(paragraph.add_run(label), BODY_FONT, 12, bold=True)
            set_run_font(paragraph.add_run(value), BODY_FONT, 12)
        self.line("", 12, after=60)
        signatures = self.borderless_table(4, [TEXT_WIDTH_IN / 2] * 2)
        cells = [("______________________", "______________________"), (HOD, GUIDE),
                 ("Head of Department", "Project Guide"), (DEPARTMENT, DEPARTMENT)]
        for r, values in enumerate(cells):
            for c, value in enumerate(values):
                paragraph = signatures.rows[r].cells[c].paragraphs[0]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_run_font(paragraph.add_run(value), BODY_FONT, 12, bold=(r == 1))

    def acknowledgement(self, blocks: list[Block]) -> None:
        self.front_title("Acknowledgement", new_page=False)
        for block in blocks:
            if block.kind == "para":
                self.p(block.text, "IDSS Plain")
        self.line("", 12, after=12)
        names = self.borderless_table(len(STUDENTS), [3.4, TEXT_WIDTH_IN - 3.4])
        for row, (name, _) in zip(names.rows, STUDENTS):
            paragraph = row.cells[1].paragraphs[0]
            paragraph.paragraph_format.line_spacing = 1.3
            set_run_font(paragraph.add_run(name), BODY_FONT, 12)

    def _index_table(self, header: tuple[str, str, str], rows: list[tuple[str, str, str, bool, bool]],
                     widths_in: list[float]) -> None:
        """Borderless index table (as in the template); rows = (label, title, anchor, bold, indent)."""
        table = self.borderless_table(len(rows) + 1, widths_in)
        aligns = (WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT)
        for c, text in enumerate(header):
            paragraph = table.rows[0].cells[c].paragraphs[0]
            paragraph.style = "IDSS Index"
            paragraph.alignment = aligns[c]
            set_run_font(paragraph.add_run(text), BODY_FONT, 12, bold=True)
        set_row_flags(table.rows[0], header=True)
        for r, (label, title, anchor, bold, indent) in enumerate(rows, start=1):
            row = table.rows[r]
            set_row_flags(row)
            cells = [row.cells[c].paragraphs[0] for c in range(3)]
            for c, paragraph in enumerate(cells):
                paragraph.style = "IDSS Index"
                paragraph.alignment = aligns[c]
            set_run_font(cells[0].add_run(label), BODY_FONT, 12, bold=bold)
            if indent:
                cells[1].paragraph_format.left_indent = Inches(0.3)
            add_inline(cells[1], title, size=12, bold=bold)
            add_field(cells[2], f"PAGEREF {anchor} \\h", "–", size=12, bold=bold)

    def page_index(self) -> None:
        self.front_title("Page Index", "_IDX_F_PageIndex")
        rows = []
        for entry in self.index:
            if entry.kind == "front":
                rows.append(("", entry.title, entry.anchor, True, False))
            elif entry.kind == "chapter":
                rows.append((entry.label, entry.title, entry.anchor, True, False))
            else:
                rows.append(("", entry.title, entry.anchor, False, True))
        self._index_table(("Chapter", "Title", "Page No."), rows, [0.85, TEXT_WIDTH_IN - 1.75, 0.9])

    def list_of(self, title: str, anchor: str, column: str, items: list[tuple[str, str, str]]) -> None:
        self.front_title(title, anchor)
        rows = [(number, caption, item_anchor, False, False) for number, caption, item_anchor in items]
        self._index_table(("Sr. No.", column, "Page No."), rows, [0.85, TEXT_WIDTH_IN - 1.75, 0.9])

    def symbol_table(self, title: str, anchor: str | None, table: Block, *, new_page: bool, before: float = 0):
        self.front_title(title, anchor, new_page=new_page, before=before)
        header, *rows = table.rows
        grid = self.borderless_table(len(table.rows), [1.5, TEXT_WIDTH_IN - 1.5])
        for r, cells in enumerate([header, *rows]):
            set_row_flags(grid.rows[r], header=(r == 0))
            for c, text in enumerate(cells):
                paragraph = grid.rows[r].cells[c].paragraphs[0]
                paragraph.style = "IDSS Index"
                add_inline(paragraph, text, size=12, bold=(r == 0) or None)

    def abstract(self, blocks: list[Block]) -> None:
        title = self.front_title("ABSTRACT", "_IDX_F_Abstract", new_page=False, size=14)
        title.paragraph_format.space_after = Pt(0)
        for _ in range(2):  # two blank lines after the heading
            self.line("", 12, spacing=1.0)
        text = next(b.text for b in blocks if b.kind == "para" and not b.text.startswith("**Keywords"))
        keywords = next(b.text for b in blocks if b.kind == "para" and b.text.startswith("**Keywords"))
        words = len(plain(text).split())
        if words > 150:
            print(f"WARNING: the abstract has {words} words (limit 150).")
        self.p(text, "IDSS Plain", italic=True)
        paragraph = self.p(style="IDSS Plain")
        set_run_font(paragraph.add_run("Keywords: "), BODY_FONT, 12, bold=True)
        add_inline(paragraph, keywords.replace("**Keywords:**", "").strip())
        print(f"Abstract: {words} words.")

    # ---- chapters

    def chapter(self, blocks: list[Block]) -> None:
        for block in blocks:
            render = getattr(self, f"_render_{block.kind}")
            render(block)

    def _render_heading(self, block: Block) -> None:
        if block.level == 1:
            match = _CHAPTER.match(block.text)
            self.p(f"{match.group(1)} {match.group(2)}", "IDSS Chapter Label")
            heading = self.p(match.group(3), "Heading 1")
            paragraph_border(heading, "bottom", 18, 6)  # one thick line (2 1/4 pt) after the chapter name
        else:
            heading = self.p(block.text, f"Heading {min(block.level, 3)}")
        if block.anchor:
            add_bookmark(heading, block.anchor)
        self.first_after_heading = True

    def _render_para(self, block: Block) -> None:
        self.p(block.text, "IDSS Body First" if self.first_after_heading else "IDSS Body")
        self.first_after_heading = False

    def _render_quote(self, block: Block) -> None:
        self.p(block.text, "IDSS Quote")

    def _render_list(self, block: Block) -> None:
        num_id = self.numbering.new_list(block.ordered, block.start)
        for item in block.items:
            paragraph = self.p(item, "IDSS List")
            Numbering.apply(paragraph, num_id)

    def _render_caption(self, block: Block) -> None:
        paragraph = self.doc.add_paragraph(style="IDSS Caption")
        set_run_font(paragraph.add_run(f"{block.caption_kind} {block.number}: "), CAPTION_FONT, 12, bold=True)
        add_inline(paragraph, block.text, font=CAPTION_FONT, size=12, bold=True)
        fmt = paragraph.paragraph_format
        if block.caption_kind == "Figure":  # below the figure
            fmt.space_after = Pt(12)
        else:  # above the table, algorithm or listing
            fmt.space_before, fmt.keep_with_next = Pt(12), True
        if block.anchor:
            add_bookmark(paragraph, block.anchor)

    def _render_image(self, block: Block) -> None:
        path = (REPORT_DIR / block.path).resolve()
        stream, (width, height) = load_image(path)
        width_in = TEXT_WIDTH_IN
        if width_in * height / width > MAX_FIGURE_HEIGHT_IN:
            width_in = MAX_FIGURE_HEIGHT_IN * width / height
        paragraph = self.doc.add_paragraph(style="IDSS Figure")
        paragraph.add_run().add_picture(stream, width=Inches(width_in))

    def _render_table(self, block: Block) -> None:
        n = max(len(r) for r in block.rows)
        size = 11 if n <= 6 else 10 if n <= 8 else 9
        # "0.800 ± 0.011" may wrap only after the "±" (never leaving "±" alone on a line)
        rows = [[cell.replace(" ± ", " ± ") for cell in row] for row in block.rows]
        table = self.doc.add_table(rows=len(rows), cols=n)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, "single", 4)
        widths = column_widths(rows, TEXT_WIDTH_TWIPS, size)
        set_table_width(table, widths)
        aligns = {"left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT,
                  "center": WD_ALIGN_PARAGRAPH.CENTER}
        # short tables never split across pages; long ones split between rows with the header repeated
        keep_together = table_height_in(rows, widths, size) <= KEEP_TABLE_MAX_IN
        for r, cells in enumerate(rows):
            row = table.rows[r]
            set_row_flags(row, header=(r == 0))
            for c in range(n):
                text = cells[c] if c < len(cells) else ""
                cell = row.cells[c]
                paragraph = cell.paragraphs[0]
                paragraph.style = "IDSS Table"
                paragraph.alignment = aligns[block.aligns[c] if c < len(block.aligns) else "left"]
                # the header row always stays with the first data row (no caption + header orphan at a page end)
                paragraph.paragraph_format.keep_with_next = (keep_together or r == 0) and r < len(rows) - 1
                add_inline(paragraph, text, size=size, bold=True if r == 0 else None)
                if r == 0:
                    set_cell_shading(cell, HEADER_FILL)
        self.spacer()

    def _render_code(self, block: Block) -> None:
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, "single", 4, "808080")
        set_table_width(table, [TEXT_WIDTH_TWIPS])
        set_row_flags(table.rows[0])  # the whole listing stays on one page
        cell = table.rows[0].cells[0]
        set_cell_shading(cell, CODE_FILL)
        for i, text in enumerate(wrap_code(block.items)):
            paragraph = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
            paragraph.style = "IDSS Code"
            set_run_font(paragraph.add_run(text), CODE_FONT, 10)
        self.spacer()

    def _render_pagebreak(self, block: Block) -> None:
        pass  # chapters start on a new page anyway

    # ---- back matter

    def references(self, blocks: list[Block]) -> None:
        self.front_title("References", "_IDX_F_References")
        for block in blocks:
            match = re.match(r"^\[(\d+)\]\s+(.*)$", block.text) if block.kind == "para" else None
            if not match:
                continue
            paragraph = self.doc.add_paragraph(style="IDSS Reference")
            set_run_font(paragraph.add_run(f"[{match.group(1)}]\t"), BODY_FONT, 12)
            add_inline(paragraph, match.group(2))

    def publications(self, blocks: list[Block]) -> None:
        self.front_title("Publications and Certificates", "_IDX_F_Publications")
        for block in blocks:
            if block.kind == "para":
                self.p(block.text, "IDSS Body First")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))


# ---------------------------------------------------------------- checks

def check_index_json(builder: ReportBuilder) -> None:
    """Compare the captions found in the chapters with docs/report/figures_tables_index.json."""
    index = json.loads((REPORT_DIR / "figures_tables_index.json").read_text(encoding="utf-8"))
    for key, found in (("figures", builder.figures), ("tables", builder.tables)):
        expected = {item["number"]: plain(item["caption"]) for item in index[key]}
        actual = {number: caption for number, caption, _ in found}
        for number in sorted(set(expected) | set(actual)):
            if expected.get(number) != actual.get(number):
                print(f"WARNING: {key} {number}: index says {expected.get(number)!r}, text has {actual.get(number)!r}")


def check_output(doc) -> None:
    """Fail loudly on leftovers: unresolved markers, template sample text or unparsed Markdown."""
    texts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                texts.extend(p.text for p in cell.paragraphs if p.style.name != "IDSS Code")
    problems = [t for t in texts if "TO VERIFY" in t or "MANET" in t or "**" in t or re.search(r"(?<!\w)`", t)]
    if problems:
        raise SystemExit("Unexpected text in the output:\n" + "\n".join(problems[:10]))


_ROMAN = re.compile(r"^(?=[ivxlc]+$)c{0,3}(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")


def _is_page_number(token: str) -> bool:
    return token.isdigit() or bool(_ROMAN.match(token))


def _index_rows(doc, first: int, stop: int) -> list[tuple[str, str]]:
    """(row text, page no.) of an index table on 0-based PDF pages first..stop-1. A row starts on the line with a
    number in the right-hand page column; lines without one continue the previous row's wrapped title."""
    rows: list[list[str]] = []
    for page in (doc[i] for i in range(first, stop)):
        lines: dict[int, list] = {}
        for word in page.get_text("words"):
            if word[1] > page.rect.height * 0.9 or word[3] < page.rect.height * 0.08:
                continue  # footer, running header
            lines.setdefault(round(word[3]), []).append(word)
        for _, words in sorted(lines.items()):
            words.sort(key=lambda w: w[0])
            text = " ".join(w[4] for w in words)
            if text.startswith(("Chapter Title", "Sr. No.")) or text in ("Page Index", "List of Figures",
                                                                          "List of Tables"):
                continue
            last = words[-1]
            if _is_page_number(last[4]) and last[0] > page.rect.width * 0.78:
                rows.append([" ".join(w[4] for w in words[:-1]), last[4]])
            elif rows:
                rows[-1][0] += " " + text
    return [(text, page) for text, page in rows]


def check_pdf_index(pdf: Path) -> list[str]:
    """Compare the page index and the lists of figures and tables in the exported PDF with the printed page
    numbers (footer) of the pages that hold each heading (PDF bookmarks) and each caption."""
    import pymupdf  # requirements-docs.txt

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("­", "").replace("’", "'")).strip().lower()

    problems: list[str] = []
    with pymupdf.open(pdf) as doc:
        printed = []
        for page in doc:
            footer = [w[4] for w in page.get_text("words") if w[1] > page.rect.height * 0.9]
            printed.append(next((t for t in footer if _is_page_number(t)), None))
        toc = doc.get_toc()
        start = {norm(title): page - 1 for level, title, page in toc if level == 1}
        marks = [(norm(title), page - 1, level) for level, title, page in toc]
        aliases = {"abbreviations and nomenclature": "abbreviations"}
        rows = _index_rows(doc, start["page index"], start["list of figures"])
        for text, page in rows:
            entry = aliases.get(norm(text), norm(text))
            hits = [m for m in marks if m[0] == entry] or \
                [m for m in marks if m[2] == 1 and m[0] == re.sub(r"^([0-9]+|[a-z])\s+", "", entry)]
            if len(hits) != 1:
                problems.append(f"Page index row {text!r} matches {len(hits)} headings")
            elif printed[hits[0][1]] != page:
                problems.append(f"Page index: {text!r} says {page}, the heading is on page {printed[hits[0][1]]}")
        for title, word, end in (("list of figures", "Figure", "list of tables"),
                                 ("list of tables", "Table", "abbreviations")):
            for text, page in _index_rows(doc, start[title], start[end]):
                number = re.match(r"([0-9A-Z]+\.\d+)\s", text)
                label = f"{word} {number.group(1)}:" if number else text
                pages = [i for i in range(start["abstract"], doc.page_count) if label in doc[i].get_text()]
                if len(pages) != 1:
                    problems.append(f"{title}: caption {label!r} found on {len(pages)} pages")
                elif printed[pages[0]] != page:
                    problems.append(f"{title}: {label} listed on {page}, the caption is on page {printed[pages[0]]}")
        print(f"Index check: {len(rows)} page-index rows, lists of figures and tables against {doc.page_count} "
              f"PDF pages: {'all match' if not problems else f'{len(problems)} problems'}.")
    return problems


# ---------------------------------------------------------------- main

def build(output: Path = OUTPUT) -> ReportBuilder:
    front = split_by_title(read_md("00_front_matter.md"))
    chapters = [(name, read_md(name)) for name in CHAPTER_FILES]
    appendices = [(name, read_md(name)) for name in APPENDIX_FILES]
    for name, _ in STUDENTS:
        if name not in (REPORT_DIR / "00_front_matter.md").read_text(encoding="utf-8"):
            raise SystemExit(f"{name} is missing from 00_front_matter.md")

    builder = ReportBuilder()
    for title, anchor in (("Page Index", "_IDX_F_PageIndex"), ("List of Figures", "_IDX_F_Figures"),
                          ("List of Tables", "_IDX_F_Tables"), ("Abbreviations and Nomenclature", "_IDX_F_Abbrev"),
                          ("Abstract", "_IDX_F_Abstract")):
        builder.index.append(IndexEntry("front", "", title, anchor))
    builder.collect([blocks for _, blocks in chapters])
    builder.index.append(IndexEntry("front", "", "References", "_IDX_F_References"))
    builder.index.append(IndexEntry("front", "", "Publications and Certificates", "_IDX_F_Publications"))
    builder.collect([blocks for _, blocks in appendices], first_file_no=len(chapters))
    check_index_json(builder)

    # Title page: no page number
    builder.title_page()
    # Certificate: counted as page i, number not printed (as in the template)
    section = builder.new_section()
    builder.set_footer(section, with_number=False)
    set_page_numbering(section, "lowerRoman", 1)
    builder.certificate()
    # Acknowledgement … Nomenclature: lower-case roman numbers
    section = builder.new_section()
    builder.set_footer(section, with_number=True)
    set_page_numbering(section, "lowerRoman")
    builder.acknowledgement(front["Acknowledgement"])
    builder.page_index()
    builder.list_of("List of Figures", "_IDX_F_Figures", "Figure Name", builder.figures)
    builder.list_of("List of Tables", "_IDX_F_Tables", "Table Name", builder.tables)
    abbreviations = next(b for b in front["Abbreviations"] if b.kind == "table")
    nomenclature = next(b for b in front["Nomenclature"] if b.kind == "table")
    builder.symbol_table("Abbreviations", "_IDX_F_Abbrev", abbreviations, new_page=True)
    builder.symbol_table("Nomenclature", None, nomenclature, new_page=False, before=24)
    # Abstract … Appendices: Arabic numbers from 1
    section = builder.new_section()
    set_page_numbering(section, "decimal", 1)
    builder.abstract(front["Abstract"])
    for _, blocks in chapters:
        builder.chapter(blocks)
    builder.references(read_md("references.md"))
    builder.publications(read_md("publications.md"))
    for _, blocks in appendices:
        builder.chapter(blocks)

    check_output(builder.doc)
    builder.save(output)
    print(f"Saved {output.relative_to(REPORT_DIR.parents[1])}: {len(builder.figures)} figures,"
          f" {len(builder.tables)} tables, {sum(e.kind == 'chapter' for e in builder.index)} chapters/appendices.")
    return builder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdf", action="store_true", help="only write the .docx (no field update / PDF)")
    args = parser.parse_args()
    build()
    if not args.no_pdf:
        for pdf in export_pdfs([OUTPUT]):
            print(f"Saved {pdf.relative_to(REPORT_DIR.parents[1])}")
            if problems := check_pdf_index(pdf):
                raise SystemExit("The page index does not match the PDF:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
