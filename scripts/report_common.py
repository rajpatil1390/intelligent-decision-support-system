"""Shared helpers for the Step 10B report builders (build_report.py, build_weekly_reports.py).

Holds the project / team details, a small Markdown parser for docs/report/*.md, python-docx helpers
(fonts, borders, bookmarks, fields, numbering) and the PDF export (Microsoft Word COM, else LibreOffice).
Install the extra packages first: .venv\\Scripts\\python.exe -m pip install -r requirements-docs.txt
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "report"
FINAL_DIR = REPORT_DIR / "final"
FORMAT_DIR = ROOT / "docs" / "format"

# ---- Project and team (same details as docs/report/00_front_matter.md) ----
PROJECT_TITLE = "Intelligent Decision Support System Using Generative AI and Machine Learning"
COLLEGE = "G H Raisoni College of Engineering and Management, Jalgaon"
DEPARTMENT = "Department of Data Science"
PROGRAMME = "CSE (Data Science)"
DEGREE = "Bachelor of Technology"
CITY_LINE = "Jalgaon – 425002 (MS)"
ACADEMIC_YEAR = "2026–2027"
GUIDE = "Dr. Arjun Jadhaw"
HOD = "Dipak Patil"
# (name, roll number). Roll numbers are left out of the public release; fill them in locally to print them in the
# college documents. A blank roll number prints the name alone.
STUDENTS: list[tuple[str, str]] = [
    ("Raj Vishwas Patil", ""),
    ("Bhavesh Vinod Patil", ""),
    ("Chaitany Rajesh Naval", ""),
    ("Varun Darshan Patil", ""),
]


def student_label(name: str, roll: str) -> str:
    """'Name (roll number)' for the college documents, or just the name when the roll number is blank."""
    return f"{name} ({roll})" if roll else name


BODY_FONT = "Times New Roman"
CAPTION_FONT = "Garamond"
CODE_FONT = "Consolas"


# ---------------------------------------------------------------- Markdown parsing

@dataclass
class Block:
    """One block of a Markdown file: heading, para, list, table, image, caption, code, quote or pagebreak."""

    kind: str
    text: str = ""
    level: int = 0
    items: list[str] = field(default_factory=list)
    ordered: bool = False
    start: int = 1
    rows: list[list[str]] = field(default_factory=list)
    aligns: list[str] = field(default_factory=list)
    lang: str = ""
    path: str = ""
    caption_kind: str = ""
    number: str = ""
    anchor: str = ""  # bookmark name (set by the report builder for indexed headings and captions)


_CAPTION = re.compile(r"^\*\*(Figure|Table|Algorithm|Listing) ([0-9A-Z]+\.[0-9]+):\*\*\s*(.*)$", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_OLIST = re.compile(r"^(\d+)\.\s+(.*)$")
_ULIST = re.compile(r"^[-*]\s+(.*)$")
_PAGEBREAK = "\x00PAGEBREAK"


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def parse_markdown(text: str) -> list[Block]:
    """Parse the subset of Markdown used in docs/report/ into blocks (comments dropped, page breaks kept)."""
    text = text.replace("<!-- page-break -->", f"\n{_PAGEBREAK}\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    blocks: list[Block] = []
    para: list[str] = []

    def flush() -> None:
        if not para:
            return
        joined = " ".join(s.strip() for s in para)
        match = _CAPTION.match(joined)
        if match:
            blocks.append(Block("caption", text=match.group(3).strip(), caption_kind=match.group(1),
                                number=match.group(2)))
        else:
            blocks.append(Block("para", text=joined))
        para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped == _PAGEBREAK:
            flush()
            blocks.append(Block("pagebreak"))
            i += 1
            continue
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            blocks.append(Block("code", lang=lang, items=code))
            i += 1
            continue
        heading = _HEADING.match(stripped)
        if heading and not para:
            blocks.append(Block("heading", text=heading.group(2).strip(), level=len(heading.group(1))))
            i += 1
            continue
        image = _IMAGE.match(stripped)
        if image:
            flush()
            blocks.append(Block("image", text=image.group(1), path=image.group(2)))
            i += 1
            continue
        if stripped.startswith("|"):
            flush()
            rows: list[list[str]] = []
            aligns: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _split_row(lines[i])
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    aligns = ["right" if c.endswith(":") and not c.startswith(":") else
                              "center" if c.startswith(":") and c.endswith(":") else "left" for c in cells]
                else:
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", rows=rows, aligns=aligns or ["left"] * len(rows[0])))
            continue
        if stripped.startswith(">"):
            flush()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(Block("quote", text=" ".join(quote)))
            continue
        olist, ulist = _OLIST.match(stripped), _ULIST.match(stripped)
        if (olist or ulist) and not para:
            ordered = bool(olist)
            start = int(olist.group(1)) if olist else 1
            items: list[str] = []
            while i < len(lines):
                cur = lines[i]
                m = _OLIST.match(cur.strip()) if ordered else _ULIST.match(cur.strip())
                if m and not cur.startswith(" "):
                    items.append(m.group(2) if ordered else m.group(1))
                elif cur.strip() and cur.startswith(" ") and items:
                    items[-1] += " " + cur.strip()
                else:
                    break
                i += 1
            blocks.append(Block("list", items=items, ordered=ordered, start=start))
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


_INLINE = re.compile(r"(\*\*|\*|`|</?sup>|</?sub>)")


@dataclass
class Segment:
    text: str
    bold: bool
    italic: bool
    code: bool
    vert: str | None


def inline_segments(text: str) -> list[Segment]:
    """Split Markdown inline text into formatted segments (bold, italic, code, superscript, subscript)."""
    segments: list[Segment] = []
    bold = italic = code = False
    vert: str | None = None
    for part in _INLINE.split(text):
        if not part:
            continue
        if code:
            if part == "`":
                code = False
            else:
                segments.append(Segment(part, bold, italic, True, vert))
            continue
        if part == "`":
            code = True
        elif part == "**":
            bold = not bold
        elif part == "*":
            italic = not italic
        elif part in ("<sup>", "<sub>"):
            vert = "superscript" if part == "<sup>" else "subscript"
        elif part in ("</sup>", "</sub>"):
            vert = None
        else:
            segments.append(Segment(part, bold, italic, False, vert))
    return segments


def plain(text: str) -> str:
    """Markdown inline text without its markup."""
    return "".join(s.text for s in inline_segments(text))


# ---------------------------------------------------------------- python-docx helpers

def set_run_font(run, name: str | None = None, size: float | None = None, bold: bool | None = None,
                 italic: bool | None = None) -> None:
    """Set a run's font on every script slot (ascii, hAnsi, eastAsia, cs) so no theme font leaks in."""
    if name:
        rpr = run._r.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.insert(0, fonts)
        for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn(f"w:{attr}"), name)
        for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn(f"w:{attr}"), None)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def set_style_font(style, name: str, size: float | None = None) -> None:
    """Like set_run_font, for a style's run properties."""
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), name)
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{attr}"), None)
    if size is not None:
        style.font.size = Pt(size)


def add_inline(paragraph: Paragraph, text: str, *, font: str | None = None, size: float | None = None,
               bold: bool | None = None, italic: bool | None = None) -> None:
    """Add Markdown inline text to a paragraph; `bold`/`italic` force the style for every run."""
    for seg in inline_segments(text):
        run = paragraph.add_run(seg.text)
        set_run_font(run, font, size, True if bold else (seg.bold or None),
                     True if italic else (seg.italic or None))
        if seg.vert == "superscript":
            run.font.superscript = True
        elif seg.vert == "subscript":
            run.font.subscript = True


def paragraph_border(paragraph: Paragraph, side: str = "bottom", size_eighths: int = 18, space: int = 4,
                     color: str = "000000") -> None:
    """Add a single border line to one side of a paragraph (size in eighths of a point: 18 = 2 1/4 pt)."""
    ppr = paragraph._p.get_or_add_pPr()
    borders = ppr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        # pBdr must precede shd/tabs/spacing/ind/jc in pPr
        successor = None
        for tag in ("w:shd", "w:tabs", "w:suppressAutoHyphens", "w:spacing", "w:ind", "w:jc", "w:outlineLvl",
                    "w:rPr", "w:sectPr"):
            successor = ppr.find(qn(tag))
            if successor is not None:
                break
        if successor is not None:
            successor.addprevious(borders)
        else:
            ppr.append(borders)
    line = OxmlElement(f"w:{side}")
    line.set(qn("w:val"), "single")
    line.set(qn("w:sz"), str(size_eighths))
    line.set(qn("w:space"), str(space))
    line.set(qn("w:color"), color)
    borders.append(line)


_bookmark_id = [100]


def add_bookmark(paragraph: Paragraph, name: str) -> None:
    """Wrap a paragraph's content in a bookmark (target of a PAGEREF field)."""
    _bookmark_id[0] += 1
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(_bookmark_id[0]))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(_bookmark_id[0]))
    p = paragraph._p
    ppr = p.pPr
    if ppr is not None:
        ppr.addnext(start)
    else:
        p.insert(0, start)
    p.append(end)


def add_field(paragraph: Paragraph, instruction: str, placeholder: str = "", *, font: str = BODY_FONT,
              size: float | None = None, bold: bool | None = None) -> None:
    """Insert a Word field (PAGE, PAGEREF …); Word fills in the result when fields are updated."""
    def fld_char(kind: str) -> None:
        run = paragraph.add_run()
        set_run_font(run, font, size, bold)
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        run._r.append(el)

    fld_char("begin")
    run = paragraph.add_run()
    set_run_font(run, font, size, bold)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    run._r.append(instr)
    fld_char("separate")
    run = paragraph.add_run(placeholder)
    set_run_font(run, font, size, bold)
    fld_char("end")


def set_cell_shading(cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def set_table_borders(table, value: str = "single", size_eighths: int = 4, color: str = "000000") -> None:
    """Set all borders of a table (value 'nil' removes them)."""
    tblpr = table._tbl.tblPr
    existing = tblpr.find(qn("w:tblBorders"))
    if existing is not None:
        tblpr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), value)
        if value != "nil":
            el.set(qn("w:sz"), str(size_eighths))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), color)
        borders.append(el)
    # tblBorders comes after tblW/jc/tblCellSpacing/tblInd in tblPr
    anchor = None
    for tag in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook"):
        anchor = tblpr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(borders)
    else:
        tblpr.append(borders)


def set_row_flags(row, *, header: bool = False, cant_split: bool = True) -> None:
    """Repeat a row as the table header on every page and/or keep the row on one page."""
    trpr = row._tr.get_or_add_trPr()
    if cant_split:
        trpr.append(OxmlElement("w:cantSplit"))
    if header:
        trpr.append(OxmlElement("w:tblHeader"))


def set_table_width(table, widths_twips: list[int]) -> None:
    """Fixed layout with explicit column widths (twips) on the grid and on every cell."""
    tblpr = table._tbl.tblPr
    tblw = tblpr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.append(tblw)
    tblw.set(qn("w:w"), str(sum(widths_twips)))
    tblw.set(qn("w:type"), "dxa")
    layout = tblpr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblpr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = table._tbl.tblGrid
    for col, width in zip(grid.findall(qn("w:gridCol")), widths_twips):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths_twips):
            tcpr = cell._tc.get_or_add_tcPr()
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.insert(0, tcw)
            tcw.set(qn("w:w"), str(width))
            tcw.set(qn("w:type"), "dxa")


def set_page_numbering(section, fmt: str, start: int | None = None) -> None:
    """Set the page-number format (decimal, lowerRoman) and an optional restart value of a section."""
    sectpr = section._sectPr
    old = sectpr.find(qn("w:pgNumType"))
    if old is not None:
        sectpr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(qn("w:fmt"), fmt)
    if start is not None:
        el.set(qn("w:start"), str(start))
    # schema order: ... pgMar, paperSrc, pgBorders, lnNumType, pgNumType, cols ...
    anchor = None
    for tag in ("w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg", "w:textDirection", "w:bidi",
                "w:rtlGutter", "w:docGrid"):
        anchor = sectpr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(el)
    else:
        sectpr.append(el)


class Numbering:
    """Creates list definitions: one bullet list type and a restartable decimal list type."""

    BULLET, DECIMAL = 900, 901

    def __init__(self, doc, left_twips: int = 680, hanging_twips: int = 340) -> None:
        self._numbering = doc.part.numbering_part.element
        self._next = 900
        for abs_id, fmt, text in ((self.BULLET, "bullet", "•"), (self.DECIMAL, "decimal", "%1.")):
            xml = (f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{abs_id}">'
                   '<w:multiLevelType w:val="singleLevel"/>'
                   f'<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/>'
                   f'<w:lvlText w:val="{text}"/><w:lvlJc w:val="left"/>'
                   f'<w:pPr><w:ind w:left="{left_twips}" w:hanging="{hanging_twips}"/></w:pPr>'
                   f'<w:rPr><w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{BODY_FONT}"'
                   f' w:cs="{BODY_FONT}" w:hint="default"/></w:rPr>'
                   '</w:lvl></w:abstractNum>')
            first_num = self._numbering.find(qn("w:num"))
            el = parse_xml(xml)
            if first_num is not None:
                first_num.addprevious(el)
            else:
                self._numbering.append(el)

    def new_list(self, ordered: bool, start: int = 1) -> int:
        """A new w:num instance (numbering restarts at `start` for ordered lists)."""
        self._next += 1
        abs_id = self.DECIMAL if ordered else self.BULLET
        xml = (f'<w:num {nsdecls("w")} w:numId="{self._next}"><w:abstractNumId w:val="{abs_id}"/>'
               f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="{start}"/></w:lvlOverride></w:num>')
        self._numbering.append(parse_xml(xml))
        return self._next

    @staticmethod
    def apply(paragraph: Paragraph, num_id: int) -> None:
        ppr = paragraph._p.get_or_add_pPr()
        numpr = parse_xml(f'<w:numPr {nsdecls("w")}><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>')
        anchor = None
        for tag in ("w:suppressLineNumbers", "w:pBdr", "w:shd", "w:tabs", "w:spacing", "w:ind", "w:jc"):
            anchor = ppr.find(qn(tag))
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addprevious(numpr)
        else:
            ppr.append(numpr)


def centered(paragraph: Paragraph) -> Paragraph:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return paragraph


# ---------------------------------------------------------------- PDF export

class WordUnavailable(RuntimeError):
    pass


def _export_with_word(docx_paths: list[Path]) -> list[Path]:
    """Open each .docx in Word, update all fields twice, save, and export a PDF next to it."""
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:  # not on Windows or pywin32 missing
        raise WordUnavailable(f"pywin32 not importable: {exc}") from exc

    pdfs: list[Path] = []
    for path in docx_paths:
        last_error: Exception | None = None
        for attempt in range(3):  # a fresh Word instance can drop the COM connection right after start-up
            word = None
            pythoncom.CoInitialize()
            try:
                word = win32com.client.DispatchEx("Word.Application")
                word.Visible = False
                word.DisplayAlerts = 0  # wdAlertsNone
                doc = word.Documents.Open(str(path), False, False, False)  # ConfirmConversions, ReadOnly, AddToRecent
                try:
                    for _ in range(2):  # twice: page numbers in the index move when the index itself grows
                        for story in doc.StoryRanges:
                            story.Fields.Update()
                        for toc in doc.TablesOfContents:
                            toc.Update()
                        doc.Repaginate()
                    doc.Save()
                    pdf = path.with_suffix(".pdf")
                    # positional arguments: file, PDF format (17), do not open, optimise for print, whole
                    # document, content only, document properties and PDF bookmarks from the headings
                    doc.ExportAsFixedFormat(str(pdf), 17, False, 0, 0, 1, 1, 0, True, True, 1, True, True, False)
                finally:
                    doc.Close(0)
                pdfs.append(pdf)
                last_error = None
                break
            except Exception as exc:  # COM errors: retry with a new instance
                last_error = exc
                time.sleep(3)
            finally:
                if word is not None:
                    try:
                        word.Quit(0)
                    except Exception:  # the instance may already be gone
                        pass
                pythoncom.CoUninitialize()
        if last_error is not None:
            raise WordUnavailable(f"Word failed on {path.name}: {last_error}") from last_error
    return pdfs


def _find_soffice() -> str | None:
    found = shutil.which("soffice")
    if found:
        return found
    for candidate in (Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
                      Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe")):
        if candidate.exists():
            return str(candidate)
    return None


def export_pdfs(docx_paths: list[Path]) -> list[Path]:
    """Update fields and export PDFs with Word; else convert with LibreOffice; else print manual steps."""
    try:
        pdfs = _export_with_word(docx_paths)
        print("Fields updated and PDF exported with Microsoft Word.")
        return pdfs
    except WordUnavailable as exc:
        print(f"Microsoft Word automation not available: {exc}")
    soffice = _find_soffice()
    if soffice:
        pdfs = []
        for path in docx_paths:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(path.parent), str(path)],
                           check=True, timeout=600)
            pdfs.append(path.with_suffix(".pdf"))
        print("PDF converted with LibreOffice (page numbers in the index may need a field update in Word).")
        return pdfs
    print("Neither Word nor LibreOffice found. Open each .docx in Word, press Ctrl+A then F9 (update fields),"
          " save, and use File > Save As > PDF.")
    return []
