"""Build the eight project weekly reports in the department format (Step 10B).

One A4 page per week, following docs/format/Project Weekly Report format.docx: the template's college header
(name, logo and rule) with "Department of Data Science", "PROJECT WEEKLY REPORT", the project name,
"Work done during week: Week N – From ________ To ________" (dates left blank for the team), the five points
of docs/report/appendix_b_weekly_progress.md, the group members and the guide / HOD signature lines.
The template's header block is reused when the template is present (docs/format/ is not in Git); otherwise a
plain text header is built. Output: docs/report/final/IDSS_Weekly_Reports.docx (+ .pdf via Word).

Usage: .venv\\Scripts\\python.exe scripts\\build_weekly_reports.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import copy
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt

from report_common import (
    BODY_FONT,
    COLLEGE,
    DEPARTMENT,
    FINAL_DIR,
    FORMAT_DIR,
    GUIDE,
    HOD,
    PROJECT_TITLE,
    REPORT_DIR,
    STUDENTS,
    add_inline,
    export_pdfs,
    paragraph_border,
    parse_markdown,
    set_run_font,
    set_style_font,
    set_table_borders,
    set_table_width,
    student_label,
)

TEMPLATE = FORMAT_DIR / "Project Weekly Report format.docx"
OUTPUT = FINAL_DIR / "IDSS_Weekly_Reports.docx"
_WEEK = re.compile(r"^Week (\d+)\s+—\s+(.+)$")
_VML = "{urn:schemas-microsoft-com:vml}"
_O_SPID = "{urn:schemas-microsoft-com:office:office}spid"


def read_weeks() -> list[tuple[int, str, list[str]]]:
    """(week number, title, five points) from appendix_b_weekly_progress.md."""
    blocks = parse_markdown((REPORT_DIR / "appendix_b_weekly_progress.md").read_text(encoding="utf-8"))
    weeks: list[tuple[int, str, list[str]]] = []
    for i, block in enumerate(blocks):
        match = _WEEK.match(block.text) if block.kind == "heading" and block.level == 2 else None
        if match:
            points = next(b.items for b in blocks[i + 1:] if b.kind == "list")
            if len(points) != 5:
                raise SystemExit(f"Week {match.group(1)} has {len(points)} points, expected 5")
            weeks.append((int(match.group(1)), match.group(2), points))
    if [w[0] for w in weeks] != list(range(1, 9)):
        raise SystemExit(f"Expected weeks 1-8, found {[w[0] for w in weeks]}")
    return weeks


class WeeklyBuilder:
    def __init__(self) -> None:
        if TEMPLATE.exists():
            self.doc = Document(str(TEMPLATE))
            self.header = self._take_template_header()
        else:
            print(f"Template not found ({TEMPLATE.name}); building a plain header without the logo.")
            self.doc = Document()
            for side in ("top", "bottom", "left", "right"):
                setattr(self.doc.sections[0], f"{side}_margin", Inches(1))
            self.header = self._plain_header()
        section = self.doc.sections[0]
        section.page_width, section.page_height = Mm(210), Mm(297)  # A4 (the template is US Letter)
        self.text_twips = round((section.page_width - section.left_margin - section.right_margin) / 635)
        set_style_font(self.doc.styles["Normal"], BODY_FONT, 12)  # paragraph marks too (template: Calibri)
        self.body = self.doc.element.body
        self._copies = 0

    def _take_template_header(self) -> list:
        """The template's first three paragraphs (college name + logo, rule, department), body emptied."""
        body = self.doc.element.body
        children = [el for el in body if el.tag != qn("w:sectPr")]
        header = children[:3]
        text = ["".join(t.text or "" for t in el.iter(qn("w:t"))) for el in header]
        if "Raisoni" not in text[0] or "Department of" not in text[2]:
            raise SystemExit("Unexpected weekly template layout: header block not found")
        for t in header[2].iter(qn("w:t")):
            if t.text and set(t.text.strip()) == {"_"}:
                t.text = DEPARTMENT.removeprefix("Department of ")
        for el in children:
            body.remove(el)
        return header

    def _plain_header(self) -> list:
        name = self.doc.add_paragraph()
        name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(name.add_run(COLLEGE), BODY_FONT, 16, bold=True)
        paragraph_border(name, "bottom", 12, 4, "7030A0")
        department = self.doc.add_paragraph()
        department.alignment = WD_ALIGN_PARAGRAPH.CENTER
        department.paragraph_format.space_before = Pt(12)
        set_run_font(department.add_run(DEPARTMENT), BODY_FONT, 14, bold=True)
        elements = [name._p, department._p]
        for el in elements:
            self.doc.element.body.remove(el)
        return elements

    def _add_header(self, new_page: bool) -> None:
        """A copy of the header block with unique drawing / shape ids (Word rejects duplicates)."""
        self._copies += 1
        sect_pr = self.body.sectPr
        for i, original in enumerate(self.header):
            el = copy.deepcopy(original)
            for attr in list(el.attrib):
                if attr.endswith(("}paraId", "}textId")):
                    del el.attrib[attr]
            for doc_pr in el.iter(qn("wp:docPr")):
                doc_pr.set("id", str(1000 + 10 * self._copies + i))
                doc_pr.set("name", f"Picture {1000 + 10 * self._copies + i}")
            for shape in el.iter():
                if isinstance(shape.tag, str) and shape.tag.startswith(_VML):
                    if shape.get("id"):
                        shape.set("id", f"Shape {self._copies}-{i}")
                    if shape.get(_O_SPID):
                        shape.set(_O_SPID, f"_x0000_s{2000 + 10 * self._copies + i}")
            if new_page and i == 0:
                ppr = el.find(qn("w:pPr"))
                page_break = ppr.makeelement(qn("w:pageBreakBefore"), {})
                ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, page_break)
            sect_pr.addprevious(el)

    def _para(self, *, align=WD_ALIGN_PARAGRAPH.LEFT, before: float = 0, after: float = 0, line: float = 1.0):
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = align
        fmt = paragraph.paragraph_format
        fmt.space_before, fmt.space_after, fmt.line_spacing = Pt(before), Pt(after), line
        return paragraph

    def week(self, number: int, points: list[str]) -> None:
        self._add_header(new_page=number > 1)
        title = self._para(align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=18)
        set_run_font(title.add_run("PROJECT WEEKLY REPORT"), BODY_FONT, 14, bold=True)

        name = self._para(after=18, line=1.15)
        set_run_font(name.add_run("Project Name: "), BODY_FONT, 14, bold=True)
        set_run_font(name.add_run(PROJECT_TITLE), BODY_FONT, 14)

        work = self._para(after=8)
        set_run_font(work.add_run(f"Work done during week: Week {number} – From ________ To ________"),
                     BODY_FONT, 14, bold=True)
        for i, point in enumerate(points, start=1):
            item = self._para(align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, line=1.15)
            item.paragraph_format.left_indent = Inches(0.35)
            item.paragraph_format.first_line_indent = Inches(-0.35)
            set_run_font(item.add_run(f"{i}.\t"), BODY_FONT, 12, bold=True)
            add_inline(item, point, font=BODY_FONT, size=12)

        members = self._para(before=18, after=6)
        set_run_font(members.add_run("Group Members :"), BODY_FONT, 14, bold=True)
        for i, (student, roll) in enumerate(STUDENTS, start=1):
            row = self._para(after=2, line=1.15)
            row.paragraph_format.left_indent = Inches(0.35)
            set_run_font(row.add_run(f"{i}.  {student_label(student, roll)}"), BODY_FONT, 12)

        self._para(after=72)  # room for the signatures
        table = self.doc.add_table(rows=2, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, "nil")
        guide_twips = round(self.text_twips * 0.62)  # the guide label is longer than the HOD label
        set_table_width(table, [guide_twips, self.text_twips - guide_twips])
        cells = (("______________________________", "______________________"),
                 (f"Guide Name & Sign: {GUIDE}", f"HOD: {HOD}"))
        for r, values in enumerate(cells):
            for c, value in enumerate(values):
                paragraph = table.rows[r].cells[c].paragraphs[0]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                paragraph.paragraph_format.space_after = Pt(0)
                set_run_font(paragraph.add_run(value), BODY_FONT, 12 if r == 0 else 13, bold=(r == 1))

    def save(self) -> None:
        props = self.doc.core_properties
        props.title = f"{PROJECT_TITLE} – Project Weekly Reports"
        props.author = ", ".join(name for name, _ in STUDENTS)
        props.comments = "Generated by scripts/build_weekly_reports.py from appendix_b_weekly_progress.md"
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(OUTPUT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdf", action="store_true", help="only write the .docx (no PDF export)")
    args = parser.parse_args()
    builder = WeeklyBuilder()
    for number, _, points in read_weeks():
        builder.week(number, points)
    builder.save()
    print(f"Saved {OUTPUT.relative_to(REPORT_DIR.parents[1])}: 8 weekly reports.")
    if not args.no_pdf:
        for pdf in export_pdfs([OUTPUT]):
            print(f"Saved {pdf.relative_to(REPORT_DIR.parents[1])}")


if __name__ == "__main__":
    main()
