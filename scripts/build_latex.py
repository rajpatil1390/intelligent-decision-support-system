"""Build the LaTeX version of the final project report (Step 14).

Converts the Step 10A content in docs/report/*.md (the same text as IDSS_Project_Report.docx) into
docs/report_latex/: main.tex (preamble in the college format), chapters/*.tex and figures/ (every image the report
uses, flattened onto white and at most 2000 px wide), then packs IDSS_Report_Overleaf.zip (main.tex, chapters/,
figures/) for Overleaf with the pdfLaTeX compiler. When latexmk or pdflatex is on the PATH the report is also compiled
to IDSS_Project_Report_LaTeX.pdf and the log is checked for undefined references and overfull boxes.

Conversion: "# Chapter n" / "## n.m" / "### n.m.k" -> \\chapter / \\section / \\subsection with labels from their
numbers (checked against LaTeX's own numbering); paragraphs, bold, italic, code, super- and subscripts, bullet and
numbered lists, quotes; tables -> longtable (bold shaded header row repeated on every page, column widths as in the
Word report, short tables never split); figures with the caption below; algorithms (algorithm + algpseudocode) and
listings with the caption above; "Table 4.1" / "Section 2.7" -> \\ref and "[9]" -> \\cite. LaTeX special characters
and every non-ASCII character of the text are escaped (_ASCII, _UNICODE); an unknown character stops the build.

Static checks (no TeX needed): every .tex file is ASCII with balanced braces and environments, no unescaped special
character, every \\ref / \\cite / image resolves, 23 figures, 40 tables, 8 algorithms and 5 listings, and each chapter
has the same numbers and words as its Markdown file.

Usage: .venv\\Scripts\\python.exe scripts\\build_latex.py [--no-compile]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from build_report import (
    APPENDIX_FILES,
    CHAPTER_FILES,
    KEEP_TABLE_MAX_IN,
    MAX_FIGURE_HEIGHT_IN,
    MAX_IMAGE_PX,
    TEXT_WIDTH_TWIPS,
    column_widths,
    read_md,
    split_by_title,
    table_height_in,
    wrap_code,
)
from report_common import (
    ACADEMIC_YEAR,
    CITY_LINE,
    COLLEGE,
    DEGREE,
    DEPARTMENT,
    GUIDE,
    HOD,
    PROGRAMME,
    PROJECT_TITLE,
    REPORT_DIR,
    ROOT,
    STUDENTS,
    Block,
    inline_segments,
    plain,
    student_label,
)

OUT_DIR = ROOT / "docs" / "report_latex"
CHAPTER_DIR = OUT_DIR / "chapters"
FIGURE_DIR = OUT_DIR / "figures"
PDF_NAME = "IDSS_Project_Report_LaTeX.pdf"
ZIP_NAME = "IDSS_Report_Overleaf.zip"
EXPECTED = {"figure": 23, "table": 40, "algorithm": 8, "listing": 5}
LOGO_SPACE = "3cm"  # empty space on the title page for the college logo

# ---------------------------------------------------------------- character escaping

_ASCII = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
    "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
    ">": r"\textgreater{}", "|": r"\textbar{}",
}
_UNICODE = {
    "\u00a0": "~", "±": r"\ensuremath{\pm}", "—": "---", "–": "--", "−": r"\ensuremath{-}",
    "←": r"\ensuremath{\leftarrow}", "→": r"\ensuremath{\rightarrow}", "↑": r"\ensuremath{\uparrow}",
    "↓": r"\ensuremath{\downarrow}", "“": "``", "”": "''", "‘": "`", "’": "'", "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}", "×": r"\ensuremath{\times}", "·": r"\ensuremath{\cdot}", "…": r"\ldots{}",
    "≈": r"\ensuremath{\approx}", "≠": r"\ensuremath{\neq}", "∈": r"\ensuremath{\in}", "∪": r"\ensuremath{\cup}",
    "∅": r"\ensuremath{\emptyset}", "÷": r"\ensuremath{\div}", "√": r"\ensuremath{\surd}",
    "Δ": r"\ensuremath{\Delta}", "Σ": r"\ensuremath{\Sigma}", "θ": r"\ensuremath{\theta}",
    "φ": r"\ensuremath{\varphi}", "ŷ": r"\ensuremath{\hat{y}}", "′": r"\ensuremath{{}'}",
    "¹": r"\textsuperscript{1}", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}", "₹": r"\rupee{}",
    "✓": r"\ding{51}", "✗": r"\ding{55}", "é": r"\'{e}", "É": r"\'{E}", "ü": r'\"{u}', "ä": r'\"{a}',
}
_HEADING_DASHES = {"—": r"\textemdash{}", "–": r"\textendash{}"}  # PDF bookmarks do not join -- / ---

_REF_PREFIX = {"Figure": "fig", "Table": "tab", "Section": "sec", "Chapter": "ch", "Algorithm": "alg",
               "Listing": "lst", "Appendix": "ch"}
_NUM = r"(?:\d+(?:\.\d+)*|[A-H](?:\.\d+)*(?![A-Za-z]))"
_SEP = r"(?:,\s*|\s+and\s+|\s+to\s+)"
_SPECIAL = re.compile(
    r"(?P<url>https?://[^\s*]+)"
    r"|\[(?P<cite>\d{1,2})\]"
    rf"|\b(?P<kind>{'|'.join(_REF_PREFIX)})(?P<plural>s?)\s+(?P<first>{_NUM})(?P<rest>(?:{_SEP}{_NUM})*)")


def chars_tex(text: str, prev: str = " ", heading: bool = False) -> str:
    """Escape plain text: LaTeX specials, non-ASCII characters, straight quotes turned into `` and ''."""
    text = text.replace(" ± ", " ±\u00a0")  # "0.800 ± 0.011" never leaves "±" alone at a line end
    out: list[str] = []
    for i, ch in enumerate(text):
        before = text[i - 1] if i else prev
        if ch == '"':
            out.append("``" if before in " ([{/\u2014\u2013" else "''")
        elif ch == "'":
            out.append("`" if before in " ([{" else "'")
        elif ch == "-" and text[i + 1:i + 2] == "-":
            out.append("-{}")  # no en-dash ligature
        elif ch in _ASCII:
            out.append(_ASCII[ch])
        elif ord(ch) < 128:
            out.append(ch)
        elif heading and ch in _HEADING_DASHES:
            out.append(_HEADING_DASHES[ch])
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        else:
            raise SystemExit(f"No LaTeX mapping for {ch!r} (U+{ord(ch):04X}) in: {text}")
    return "".join(out).replace("```", "``{}`")


def _special_tex(match: re.Match) -> str:
    if match["url"]:
        return rf"\url{{{match['url']}}}"
    if match["cite"]:
        return rf"\cite{{ref{match['cite']}}}"
    prefix = _REF_PREFIX[match["kind"]]
    out = rf"{match['kind']}{match['plural']}~\ref{{{prefix}:{match['first']}}}"
    for sep, number in re.findall(rf"({_SEP})({_NUM})", match["rest"]):
        out += rf"{sep.rstrip()}~\ref{{{prefix}:{number}}}"
    return out


def text_tex(text: str, prev: str = " ", heading: bool = False) -> str:
    """Plain text with cross-references (\\ref), citations (\\cite) and URLs (\\url) recognised."""
    out: list[str] = []
    pos = 0
    for match in _SPECIAL.finditer(text):
        out.append(chars_tex(text[pos:match.start()], text[pos - 1] if pos else prev, heading))
        out.append(_special_tex(match))
        pos = match.end()
    out.append(chars_tex(text[pos:], text[pos - 1] if pos else prev, heading))
    return "".join(out)


def code_tex(text: str) -> str:
    """Inline code in \\code{} (typewriter), breakable after an inner / or _ (long spans also after . - , =)."""
    breaks = "/_.-,=" if len(text) > 18 else "/_"
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch in _ASCII:
            out.append(_ASCII[ch])
        elif ch == '"':
            out.append(r"\textquotedbl{}")
        elif ch == "-" and text[i + 1:i + 2] == "-":
            out.append("-{}")
        elif ord(ch) < 128:
            out.append(ch)
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        else:
            raise SystemExit(f"No LaTeX mapping for {ch!r} in code: {text}")
        if ch in breaks and 2 <= i <= len(text) - 3:
            out.append(r"\allowbreak{}")
    return r"\code{" + "".join(out) + "}"


def inline(text: str, heading: bool = False) -> str:
    """Markdown inline text (bold, italic, code, <sup>, <sub>) as LaTeX."""
    out: list[str] = []
    prev = " "
    for seg in inline_segments(text):
        body = code_tex(seg.text) if seg.code else text_tex(seg.text, prev, heading)
        prev = seg.text[-1:] or prev
        if seg.vert == "superscript":
            body = rf"\textsuperscript{{{body}}}"
        elif seg.vert == "subscript":
            body = rf"\textsubscript{{{body}}}"
        if seg.italic and not seg.code:
            body = rf"\textit{{{body}}}"
        if seg.bold:
            body = rf"\textbf{{{body}}}"
        out.append(body)
    result = "".join(out)
    return "{}" + result if result.startswith("[") else result  # never read as an optional argument


# ---------------------------------------------------------------- pseudo-code and listings

_ALG_FIRST = {"procedure", "repeat", "for", "try", "catch", "if", "else", "then", "return"}
_ALG_CONTROL = {"if", "then", "else", "return", "do", "while", "catch"}  # bold once a line has a control keyword


def _alg_text(text: str, keywords: bool) -> str:
    parts: list[str] = []
    first, control = True, False
    for token in re.split(r"(\s+)", text):
        if not token or token.isspace():
            parts.append(token)
            continue
        if keywords and ((first and token in _ALG_FIRST) or (token in _ALG_CONTROL and (control or token == "if"))):
            control = True
            parts.append(rf"\textbf{{{token}}}")
        elif token == "\\":
            parts.append(r"\ensuremath{\setminus}")  # "actionable \ Δ"
        else:
            parts.append(chars_tex(token))
        first = False
    return "".join(parts)


def algorithm_line(line: str) -> str:
    """One pseudo-code line as \\State: indentation kept (4 spaces = 1.5 em), keywords bold, // -> \\Comment."""
    if not line.strip():
        return r"\Statex"
    indent = len(line) - len(line.lstrip(" "))
    body = re.sub(r"\s+", " ", line.strip())
    quotes = iter("“”" * body.count('"'))  # string literals: quotes alternate open / close
    body = re.sub('"', lambda _: next(quotes), body)
    comment = None
    match = re.search(r"\s*//\s*(.*)$", body)
    if match:
        body, comment = body[:match.start()], match.group(1)
    tex = r"\State "
    if indent:
        tex += rf"\hspace*{{{indent / 4 * 1.5:.2f}em}}"
    tex += _alg_text(body, keywords=True)
    if comment:
        tex += rf" \Comment{{{_alg_text(comment, keywords=False)}}}"
    return tex


def listing_line(line: str) -> str:
    """A code line inside lstlisting: non-ASCII characters go through the (*@ ... @*) escape."""
    return "".join(f"(*@{_UNICODE[ch]}@*)" if ord(ch) > 127 else ch for ch in line)


# ---------------------------------------------------------------- chapter conversion

_HEADING_NO = re.compile(r"^([0-9]+|[A-Z])\.([0-9]+)(?:\.([0-9]+))?\s+(.+)$")
_CHAPTER = re.compile(r"^(Chapter|Appendix) ([0-9A-Z]+):\s*(.+)$")


class ChapterWriter:
    """Converts the blocks of one chapter / appendix file into LaTeX, checking every number on the way."""

    def __init__(self, counts: Counter, labels: set[str], figures: list[Path]) -> None:
        self.counts = counts
        self.labels = labels
        self.figures = figures
        self.chapter = ""
        self.section = self.subsection = 0
        self.float_no: Counter = Counter()
        self.lines: list[str] = []

    def label(self, name: str) -> str:
        if name in self.labels:
            raise SystemExit(f"Duplicate label {name}")
        self.labels.add(name)
        return rf"\label{{{name}}}"

    def expect_number(self, kind: str, number: str) -> None:
        """A caption number must be the next one LaTeX will give (chapter.n, numbered by chapter)."""
        self.float_no[kind] += 1
        expected = f"{self.chapter}.{self.float_no[kind]}"
        if number != expected:
            raise SystemExit(f"{kind} {number} is out of sequence (LaTeX would number it {expected})")
        self.counts[kind] += 1

    def convert(self, blocks: list[Block]) -> str:
        i = 0
        while i < len(blocks):
            block = blocks[i]
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if block.kind == "image":
                if not (nxt and nxt.kind == "caption" and nxt.caption_kind == "Figure"):
                    raise SystemExit(f"Image {block.path} is not followed by a figure caption")
                self.figure(block, nxt)
                i += 2
                continue
            if block.kind == "caption":
                if block.caption_kind == "Table" and nxt and nxt.kind == "table":
                    self.table(nxt, block)
                elif block.caption_kind in ("Algorithm", "Listing") and nxt and nxt.kind == "code":
                    self.code(nxt, block)
                else:
                    raise SystemExit(f"Caption {block.caption_kind} {block.number} has no matching block")
                i += 2
                continue
            getattr(self, f"_render_{block.kind}")(block)
            i += 1
        return "\n".join(self.lines).rstrip() + "\n"

    # ---- blocks

    def _render_heading(self, block: Block) -> None:
        if block.level == 1:
            match = _CHAPTER.match(block.text)
            if not match:
                raise SystemExit(f"Unexpected chapter heading: {block.text}")
            self.chapter, self.section, self.subsection = match.group(2), 0, 0
            self.float_no.clear()
            self.lines += [rf"\chapter{{{inline(match.group(3), heading=True)}}}{self.label('ch:' + self.chapter)}",
                           ""]
            return
        match = _HEADING_NO.match(block.text)
        if not match:  # unnumbered section (Appendix B weeks): in the page index, without a number
            if block.level != 2:
                raise SystemExit(f"Unnumbered heading below section level: {block.text}")
            self.lines += [r"\FloatBarrier", r"\setcounter{secnumdepth}{0}",
                           rf"\section{{{inline(block.text, heading=True)}}}", r"\setcounter{secnumdepth}{2}", ""]
            return
        chapter, sec, sub, title = match.groups()
        if chapter != self.chapter:
            raise SystemExit(f"Heading {block.text!r} is not in chapter {self.chapter}")
        if block.level == 2 and sub is None:
            self.section, self.subsection = self.section + 1, 0
            if int(sec) != self.section:
                raise SystemExit(f"Section {block.text!r} out of sequence (expected {chapter}.{self.section})")
            self.lines += [r"\FloatBarrier",
                           rf"\section{{{inline(title, heading=True)}}}{self.label(f'sec:{chapter}.{sec}')}", ""]
        elif block.level == 3 and sub is not None:
            self.subsection += 1
            if int(sec) != self.section or int(sub) != self.subsection:
                raise SystemExit(f"Subsection {block.text!r} out of sequence")
            self.lines += [rf"\subsection{{{inline(title, heading=True)}}}"
                           rf"{self.label(f'sec:{chapter}.{sec}.{sub}')}", ""]
        else:
            raise SystemExit(f"Heading level and number do not match: {'#' * block.level} {block.text}")

    def _render_para(self, block: Block) -> None:
        self.lines += [inline(block.text), ""]

    def _render_quote(self, block: Block) -> None:
        self.lines += [r"\begin{quote}", inline(block.text), r"\end{quote}", ""]

    def _render_list(self, block: Block) -> None:
        if block.ordered:
            self.lines.append(r"\begin{enumerate}" + (f"[start={block.start}]" if block.start != 1 else ""))
        else:
            self.lines.append(r"\begin{itemize}")
        self.lines += [rf"  \item {inline(item)}" for item in block.items]
        self.lines += [r"\end{enumerate}" if block.ordered else r"\end{itemize}", ""]

    def _render_pagebreak(self, block: Block) -> None:
        pass  # chapters start on a new page anyway

    def _render_code(self, block: Block) -> None:
        self.code(block)

    def figure(self, image: Block, caption: Block) -> None:
        self.expect_number("figure", caption.number)
        source = (REPORT_DIR / image.path).resolve()
        if not source.exists():
            raise SystemExit(f"Missing image {source}")
        if any(f.name == source.name and f != source for f in self.figures):
            raise SystemExit(f"Two different images are named {source.name}")
        self.figures.append(source)
        self.lines += [
            r"\begin{figure}[htbp]",
            r"  \centering",
            rf"  \includegraphics[width=\linewidth,height={MAX_FIGURE_HEIGHT_IN}in,keepaspectratio]"
            rf"{{figures/{source.name}}}",
            rf"  \caption{{{inline(caption.text)}}}{self.label('fig:' + caption.number)}",
            r"\end{figure}",
            "",
        ]

    def table(self, table: Block, caption: Block) -> None:
        self.expect_number("table", caption.number)
        self.lines += table_tex(table.rows, table.aligns,
                                rf"\caption{{{inline(caption.text)}}}{self.label('tab:' + caption.number)}")
        self.lines.append("")

    def code(self, block: Block, caption: Block | None = None) -> None:
        if caption is not None and caption.caption_kind == "Algorithm":
            self.expect_number("algorithm", caption.number)
            self.lines += [
                r"\begin{algorithm}[H]",
                rf"  \caption{{{inline(caption.text)}}}{self.label('alg:' + caption.number)}",
                r"  \noindent\fcolorbox{codeframe}{codebg}{\begin{minipage}"
                r"{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}",
                r"  \linespread{1}\small",
                r"  \begin{algorithmic}",
                *(f"    {algorithm_line(line)}" for line in block.items),
                r"  \end{algorithmic}",
                r"  \end{minipage}}",
                r"\end{algorithm}",
                "",
            ]
            return
        options = ""
        if caption is not None:
            self.expect_number("listing", caption.number)
            name = "lst:" + caption.number
            self.label(name)
            options = f"[caption={{{inline(caption.text)}}},label={{{name}}}]"
        self.lines += [r"\begin{lstlisting}" + options, *(listing_line(line) for line in wrap_code(block.items)),
                       r"\end{lstlisting}", ""]


def table_tex(rows: list[list[str]], aligns: list[str], caption: str) -> list[str]:
    """A Markdown table in the Word report's layout: grid, bold shaded header row, column widths from the content,
    11 / 10 / 9 pt by column count. Short tables (estimated below KEEP_TABLE_MAX_IN, as in the Word report) are a
    non-floating table that never splits; longer ones are a longtable whose header row repeats on every page."""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    aligns = (aligns + ["left"] * n)[:n]
    size = 11 if n <= 6 else 10 if n <= 8 else 9
    size_cmd = {11: r"\small", 10: r"\footnotesize", 9: r"\fontsize{9}{11}\selectfont"}[size]
    widths = column_widths(rows, TEXT_WIDTH_TWIPS, size)
    keep = table_height_in(rows, widths, size) <= KEEP_TABLE_MAX_IN
    align_cmd = {"left": r"\raggedright", "right": r"\raggedleft", "center": r"\centering"}
    spec = "|".join(rf">{{{align_cmd[a]}\arraybackslash\hspace{{0pt}}}}"
                    rf"p{{\dimexpr {w / sum(widths):.4f}\linewidth-2\tabcolsep-2\arrayrulewidth\relax}}"
                    for w, a in zip(widths, aligns))
    header = r"\rowcolor{tableheader}" + " & ".join(rf"\textbf{{{inline(c)}}}" if c else "" for c in rows[0])
    body = [" & ".join(inline(c) for c in row) + r" \\ \hline" for row in rows[1:]]
    style = [rf"\linespread{{1}}{size_cmd}", rf"\setlength{{\tabcolsep}}{{{3 if n >= 9 else 4}pt}}"]
    if keep:
        return [r"\begin{table}[H]", r"\centering", *style, caption, rf"\begin{{tabular}}{{|{spec}|}}", r"\hline",
                header + r" \\", r"\hline", *body, r"\end{tabular}", r"\end{table}"]
    # a page break falls inside an \hline and longtable draws the rule on both pages, so the repeated
    # head has no rule below its header row (the row's own rule closes it)
    return [r"\begingroup", *style, rf"\begin{{longtable}}{{|{spec}|}}", rf"{caption}\\", r"\hline",
            header + r" \\", r"\hline", r"\endfirsthead", r"\hline", header + r" \\", r"\endhead", *body,
            r"\end{longtable}", r"\endgroup"]


# ---------------------------------------------------------------- front and back matter

def frontmatter_tex(front: dict[str, list[Block]]) -> str:
    """Title page, certificate, acknowledgement, page index, lists, abbreviations / nomenclature and abstract, in
    the page sequence of the Word report (title page not numbered, certificate = page i without a number, roman up
    to the nomenclature, Arabic from the abstract)."""
    def size(pt: int, text: str, bold: bool = False) -> str:
        series = r"\bfseries" if bold else ""
        return rf"{{\fontsize{{{pt}}}{{{round(pt * 1.2)}}}\selectfont{series} {text}\par}}"

    names = [chars_tex(student_label(name, roll)) for name, roll in STUDENTS]
    college = chars_tex(COLLEGE)
    out = [
        "% Front matter in the page sequence of the Word report. Generated by scripts/build_latex.py",
        "% from docs/report/00_front_matter.md and the team details in scripts/report_common.py.",
        "",
        "% ---- title page (not numbered)",
        r"\pagenumbering{Alph}",
        r"\begin{titlepage}",
        r"\centering",
        r"\linespread{1.15}\selectfont",
        size(12, "A"),
        size(16, "Project Report", True),
        size(12, "on"),
        r"\vspace{6pt}",
        size(20, chars_tex(PROJECT_TITLE.upper()), True),
        r"\vspace{12pt}",
        size(12, "submitted for Partial Fulfillment of"),
        size(12, "the Requirements of the Degree"),
        size(12, "of"),
        size(16, chars_tex(DEGREE), True),
        size(12, "in"),
        size(16, chars_tex(PROGRAMME), True),
        size(12, "to"),
        size(16, college, True),
        r"\vspace{12pt}",
        size(12, "Submitted by"),
        r"\vspace{4pt}",
        *(size(14, name, True) for name in names),
        r"\vspace{12pt}",
        size(12, "Under the Guidance of"),
        r"\vspace{4pt}",
        size(14, chars_tex(GUIDE), True),
        r"\vfill",
        "% College logo: put the image in figures/, uncomment the next line and delete the \\vspace* below it.",
        rf"% \includegraphics[height={LOGO_SPACE}]{{figures/college_logo.png}}\par",
        rf"\vspace*{{{LOGO_SPACE}}}",
        r"\vfill",
        size(12, chars_tex(DEPARTMENT), True),
        size(14, chars_tex(COLLEGE.split(", Jalgaon")[0]) + ","),
        size(14, chars_tex(CITY_LINE)),
        size(14, chars_tex(ACADEMIC_YEAR)),
        r"\end{titlepage}",
        "",
        "% ---- certificate (page i, number not printed)",
        r"\pagestyle{idss}",
        r"\pagenumbering{roman}",
        r"\thispagestyle{collegeonly}",
        r"\begin{center}",
        r"\linespread{1}\selectfont",
        size(16, college, True),
        r"\vspace{6pt}",
        size(12, chars_tex(DEPARTMENT), True),
        r"\vspace{30pt}",
        size(20, "CERTIFICATE", True),
        r"\end{center}",
        r"\vspace{18pt}",
        r"\noindent This is to certify that the project entitled ``\textbf{\textit{" + chars_tex(PROJECT_TITLE)
        + r"}}'', submitted by\par",
        r"\vspace{6pt}",
        r"{\centering\linespread{1.3}\selectfont\bfseries",
        *(name + r"\par" for name in names),
        "}",
        r"\vspace{6pt}",
        rf"\noindent in partial fulfillment of the degree of \textit{{{chars_tex(DEGREE)}}} in "
        rf"\textit{{{chars_tex(PROGRAMME)}}} has been satisfactorily carried out under my guidance as per the "
        rf"requirement of {college}.\par",
        r"\vspace{12pt}",
        r"\noindent\textbf{Date:} \rule{4.5cm}{0.4pt}\par",
        r"\noindent\textbf{Place:} Jalgaon\par",
        r"\vspace{60pt}",
        r"\noindent\begin{tabular}{@{}>{\centering}p{0.48\linewidth}@{\hspace{0.04\linewidth}}"
        r">{\centering\arraybackslash}p{0.48\linewidth}@{}}",
        r"\rule{5.5cm}{0.4pt} & \rule{5.5cm}{0.4pt} \\",
        rf"\textbf{{{chars_tex(HOD)}}} & \textbf{{{chars_tex(GUIDE)}}} \\",
        r"Head of Department & Project Guide \\",
        rf"{chars_tex(DEPARTMENT)} & {chars_tex(DEPARTMENT)} \\",
        r"\end{tabular}",
        r"\clearpage",
        "",
        "% ---- acknowledgement (page ii)",
        r"\chapter*{Acknowledgement}",
        r"\begingroup",
        r"\setlength{\parindent}{0pt}\setlength{\parskip}{12pt}",
    ]
    for block in front["Acknowledgement"]:
        if block.kind == "para":
            out += [inline(block.text), ""]
    out += [
        r"\par\endgroup",
        r"\vspace{12pt}",
        r"{\renewcommand{\arraystretch}{1.3}",
        r"\noindent\hspace*{3.4in}\begin{tabular}[t]{@{}l@{}}",
        *(chars_tex(name) + r" \\" for name, _ in STUDENTS),
        r"\end{tabular}\par}",
        "",
        "% ---- page index and the lists of figures and tables (each starts a new page)",
        r"\tableofcontents",
        r"\listoffigures",
        r"\listoftables",
        "",
        "% ---- abbreviations and nomenclature",
        r"\chapter*{Abbreviations}",
        r"\frontentry{Abbreviations and Nomenclature}",
        *symbol_table(next(b for b in front["Abbreviations"] if b.kind == "table")),
        r"\vspace{24pt}",
        r"{\centering\normalfont\bfseries\fontsize{20}{24}\selectfont Nomenclature\par}",
        r"\vspace{18pt}",
        *symbol_table(next(b for b in front["Nomenclature"] if b.kind == "table")),
        "",
        "% ---- abstract (Arabic page numbers from here)",
        r"\clearpage",
        r"\pagenumbering{arabic}",
        r"\frontentry{Abstract}",
        r"\begin{center}",
        r"{\normalfont\bfseries\fontsize{14}{17}\selectfont ABSTRACT\par}",
        r"\end{center}",
        r"\vspace{28pt}",
    ]
    text = next(b.text for b in front["Abstract"] if b.kind == "para" and not b.text.startswith("**Keywords"))
    keywords = next(b.text for b in front["Abstract"] if b.kind == "para" and b.text.startswith("**Keywords"))
    words = len(plain(text).split())
    print(f"Abstract: {words} words.")
    if words > 150:
        print(f"WARNING: the abstract has {words} words (limit 150).")
    out += [r"\noindent " + inline(text) + r"\par", r"\vspace{12pt}", r"\noindent " + inline(keywords) + r"\par", ""]
    return "\n".join(out)


def symbol_table(table: Block) -> list[str]:
    """Borderless two-column longtable (abbreviations, nomenclature): bold header repeated on every page."""
    header = " & ".join(rf"\textbf{{{inline(c)}}}" for c in table.rows[0]) + r" \\[2pt]"
    return [
        r"\begingroup",
        r"\linespread{1}\selectfont\renewcommand{\arraystretch}{1.25}",
        r"\begin{longtable}{@{}p{1.5in}p{\dimexpr\linewidth-1.5in-2\tabcolsep\relax}@{}}",
        header, r"\endfirsthead", header, r"\endhead",
        *(" & ".join(inline(c) for c in row) + r" \\" for row in table.rows[1:]),
        r"\end{longtable}",
        r"\endgroup",
    ]


def references_tex(blocks: list[Block]) -> str:
    out = ["% References in the college style, numbered by first citation as in the Word report.",
           "% Generated by scripts/build_latex.py from docs/report/references.md.",
           r"\begin{thebibliography}{99}", r"\frontentry{References}"]
    expected = 0
    for block in blocks:
        match = re.match(r"^\[(\d+)\]\s+(.*)$", block.text) if block.kind == "para" else None
        if not match:
            continue
        expected += 1
        if int(match.group(1)) != expected:
            raise SystemExit(f"Reference [{match.group(1)}] out of order (expected [{expected}])")
        out += [rf"\bibitem{{ref{expected}}} {inline(match.group(2))}", ""]
    out.append(r"\end{thebibliography}")
    return "\n".join(out) + "\n"


def publications_tex(blocks: list[Block]) -> str:
    out = ["% Generated by scripts/build_latex.py from docs/report/publications.md.",
           r"\chapter*{Publications and Certificates}", r"\frontentry{Publications and Certificates}", ""]
    out += [r"\noindent " + inline(b.text) + "\n" for b in blocks if b.kind == "para"]
    return "\n".join(out)


MAIN_TEX = r"""% IDSS project report, LaTeX version (Step 14), in the format of the Word report (college format).
% Generated by scripts/build_latex.py from docs/report/*.md: edit the Markdown sources and re-run the script.
% Compile with pdfLaTeX: latexmk -pdf main.tex   (Overleaf: Menu > Compiler > pdfLaTeX)
\documentclass[12pt,a4paper,oneside]{report}

% ---- page layout
\usepackage[a4paper,top=1in,bottom=1.15in,left=1.5in,right=0.6in,footskip=0.7in]{geometry}

% ---- fonts: Times (newtx) for text and maths, EB Garamond for captions, Inconsolata for code
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext}
\usepackage[lining]{ebgaramond}  % after newtxtext (both define \sufigures); only used for captions
\renewcommand*{\rmdefault}{ntxtlf}  % ebgaramond switches the roman family: Times stays the body font
\usepackage{newtxmath}
\usepackage[varqu,scaled=0.95]{zi4}
\usepackage{pifont}  % \ding{51} check mark, \ding{55} cross
\IfFileExists{tfrupee.sty}{\usepackage{tfrupee}}{}
\providecommand{\rupee}{Rs.\,}  % fallback if the tfrupee font is not installed
\usepackage{microtype}

% ---- spacing: 1.5 lines, justified, 1 cm first-line indent (none after a heading)
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{1cm}
\setlength{\emergencystretch}{2em}

% ---- headings
\usepackage{titlesec}
\titleformat{\chapter}[display]
  {\normalfont\bfseries\fontsize{16}{19}\selectfont\filleft}
  {\chaptertitlename\ \thechapter}{6pt}{}
  [{\vspace{4pt}\titlerule[2.25pt]\vspace{6pt}}]
\titleformat{name=\chapter,numberless}[block]
  {\normalfont\bfseries\fontsize{20}{24}\selectfont\filcenter}{}{0pt}{}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}
\titleformat{\section}{\normalfont\bfseries\fontsize{18}{22}\selectfont}{\thesection}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 1pt}{6pt}
\titleformat{\subsection}{\normalfont\bfseries\fontsize{14}{17}\selectfont}{\thesubsection}{0.5em}{}
\titlespacing*{\subsection}{0pt}{12pt plus 2pt minus 1pt}{6pt}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{1}

% ---- lists
\usepackage{enumitem}
\setlist{itemsep=0pt,parsep=0pt,topsep=4pt,partopsep=0pt}

% ---- colours, tables, figures
\usepackage[table]{xcolor}
\definecolor{tableheader}{HTML}{D9D9D9}
\definecolor{codebg}{HTML}{F2F2F2}
\definecolor{codeframe}{HTML}{808080}
\usepackage{array}
\usepackage{longtable}
\setlength{\LTcapwidth}{\linewidth}
\usepackage{graphicx}
\usepackage{float}
\usepackage{placeins}  % \FloatBarrier before every section keeps figures in their section
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}

% ---- captions: EB Garamond bold 12 pt, centred, "Figure 4.1:"; figures below, tables above
\usepackage{caption}
\DeclareCaptionFont{idsscaption}{\ebgaramond}
\captionsetup{font={idsscaption,bf,normalsize,stretch=1},labelsep=colon,justification=centering,skip=6pt}
\captionsetup[figure]{position=bottom}
\captionsetup[table]{position=top}

% ---- code listings: Inconsolata 10 pt, single spacing, light grey box
\usepackage{listings}
\lstset{basicstyle=\linespread{1}\ttfamily\fontsize{10}{12}\selectfont, columns=fixed, basewidth=0.5em,
  keepspaces=true, showstringspaces=false, frame=single, framesep=4pt, framerule=0.4pt,
  rulecolor=\color{codeframe}, backgroundcolor=\color{codebg}, xleftmargin=4.4pt, xrightmargin=4.4pt,
  breaklines=true, escapeinside={(*@}{@*)}, aboveskip=8pt, belowskip=8pt, captionpos=t, tabsize=4}
\captionsetup[lstlisting]{position=top}
\newcommand{\code}[1]{\texttt{#1}}

% ---- footer: college name and the centred page number
\usepackage{fancyhdr}
\newcommand{\footerblock}[1]{\parbox[b]{\linewidth}{\centering\fontsize{10}{12}\selectfont
  <<COLLEGE>>\par\vspace{2pt}{\normalsize #1}}}
\fancypagestyle{idss}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}%
  \fancyfoot[C]{\footerblock{\thepage}}}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}%
  \fancyfoot[C]{\footerblock{\thepage}}}
\fancypagestyle{collegeonly}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}%
  \fancyfoot[C]{\footerblock{}}}

% ---- page index and lists of figures / tables as in the Word report: "Chapter | Title | Page No."
\usepackage{etoolbox}
\makeatletter
\patchcmd{\l@chapter}{1.5em}{5em}{}{}
\renewcommand*{\l@section}{\@dottedtocline{1}{6.8em}{2.5em}}
\renewcommand*{\l@figure}{\@dottedtocline{1}{0em}{5em}}
\renewcommand*{\l@table}{\@dottedtocline{1}{0em}{5em}}
\newcommand{\indexheader}[2]{\noindent\makebox[5em][l]{\textbf{#1}}\textbf{#2}\hfill\textbf{Page No.}\par
  \vspace{6pt}}
\newcommand{\frontentry}[1]{\phantomsection\addcontentsline{toc}{chapter}{\protect\numberline{}#1}}
\newcommand{\indexbody}[1]{{\linespread{1}\selectfont\setlength{\parskip}{2pt plus 1pt}\@starttoc{#1}}}
\renewcommand{\contentsname}{Page Index}
\renewcommand{\bibname}{References}
\renewcommand{\tableofcontents}{\chapter*{\contentsname}\frontentry{\contentsname}%
  \indexheader{Chapter}{Title}\indexbody{toc}}
\renewcommand{\listoffigures}{\chapter*{\listfigurename}\frontentry{\listfigurename}%
  \indexheader{Sr. No.}{Figure Name}\indexbody{lof}}
\renewcommand{\listoftables}{\chapter*{\listtablename}\frontentry{\listtablename}%
  \indexheader{Sr. No.}{Table Name}\indexbody{lot}}
\makeatother

% ---- links (no coloured boxes in print); pseudo-code after hyperref
\usepackage[hyphens]{url}
\urlstyle{same}
\usepackage[hidelinks,bookmarksnumbered]{hyperref}
\hypersetup{pdftitle={<<TITLE>>}, pdfauthor={<<AUTHORS>>}, pdfsubject={<<SUBJECT>>}}
\usepackage[chapter]{algorithm}
\floatstyle{plaintop}
\restylefloat{algorithm}
\captionsetup[algorithm]{position=top}
\usepackage{algpseudocode}

\begin{document}
\input{chapters/frontmatter}
<<CHAPTERS>>
\input{chapters/references}
\input{chapters/publications}
\appendix
<<APPENDICES>>
\end{document}
"""


def main_tex(chapters: list[str], appendices: list[str]) -> str:
    authors = ", ".join(name for name, _ in STUDENTS)
    replacements = {
        "<<COLLEGE>>": chars_tex(COLLEGE),
        "<<TITLE>>": PROJECT_TITLE,
        "<<AUTHORS>>": authors,
        "<<SUBJECT>>": f"B.Tech {PROGRAMME} project report, {COLLEGE}",
        "<<CHAPTERS>>": "\n".join(rf"\input{{chapters/{name}}}" for name in chapters),
        "<<APPENDICES>>": "\n".join(rf"\input{{chapters/{name}}}" for name in appendices),
    }
    text = MAIN_TEX
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


# ---------------------------------------------------------------- figures

def copy_figure(source: Path, target: Path) -> None:
    """Flatten transparency onto white (pdfTeX embeds RGB PNGs without re-encoding) and downscale to 2000 px."""
    with Image.open(source) as image:
        if "A" in image.getbands() or image.mode == "P":
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = image.convert("RGB")
        if image.width > MAX_IMAGE_PX:
            image = image.resize((MAX_IMAGE_PX, round(image.height * MAX_IMAGE_PX / image.width)), Image.LANCZOS)
        image.save(target, format="PNG", optimize=True)


# ---------------------------------------------------------------- static checks

_ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")


def _strip_verbatim(tex: str) -> str:
    tex = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", r"\\begin{lstlisting}\\end{lstlisting}", tex,
                 flags=re.S)
    return "\n".join(line for line in tex.splitlines() if not line.startswith("%"))


def check_tex(files: dict[str, str]) -> list[str]:
    """Static checks of the generated LaTeX (no TeX installation needed)."""
    problems: list[str] = []
    labels: set[str] = set()
    refs: list[tuple[str, str]] = []
    cites: list[tuple[str, str]] = []
    bibitems: set[str] = set()
    for name, tex in files.items():
        if any(ord(ch) > 127 for ch in tex):
            problems.append(f"{name}: non-ASCII characters")
        body = _strip_verbatim(tex)
        depth = 0
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "\\":
                i += 2
                continue
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if depth < 0:
                problems.append(f"{name}: unbalanced '}}' near {body[max(0, i - 60):i + 1]!r}")
                depth = 0
            i += 1
        if depth:
            problems.append(f"{name}: {depth} unclosed '{{'")
        stack: list[str] = []
        for kind, env in _ENV.findall(body):
            if kind == "begin":
                stack.append(env)
            elif not stack or stack.pop() != env:
                problems.append(f"{name}: \\end{{{env}}} does not match")
        if stack:
            problems.append(f"{name}: unclosed environments {stack}")
        plain_body = re.sub(r"\\(?:label|ref|input|includegraphics(?:\[[^\]]*\])?|url|cite|bibitem)\{[^}]*\}", "",
                            body)
        plain_body = re.sub(r"label=\{[^}]*\}", "", plain_body)
        for pattern, what in ((r"(?<!\\)[#$^]", "unescaped # $ ^"), (r"(?<!\\)_", "unescaped _"),
                              (r"(?<![\\\w])%(?!$)", "unescaped %")):
            for match in re.finditer(pattern, plain_body):
                if name != "main.tex":
                    problems.append(f"{name}: {what}: {plain_body[max(0, match.start() - 40):match.end() + 20]!r}")
        labels.update(re.findall(r"\\label\{([^}]+)\}", body))
        labels.update(re.findall(r"label=\{([^}]+)\}", body))
        refs += [(name, r) for r in re.findall(r"\\ref\{([^}]+)\}", body)]
        cites += [(name, c) for c in re.findall(r"\\cite\{([^}]+)\}", body)]
        bibitems.update(re.findall(r"\\bibitem\{([^}]+)\}", body))
        for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
            if not (OUT_DIR / path).exists():
                problems.append(f"{name}: missing image {path}")
    problems += [f"{name}: undefined reference {r}" for name, r in refs if r not in labels]
    problems += [f"{name}: undefined citation {c}" for name, c in cites if c not in bibitems]
    unused = sorted(bibitems - {c for _, c in cites})
    if unused:
        print(f"Note: references never cited in the text: {', '.join(unused)}")
    return problems


def _md_tokens(blocks: list[Block]) -> tuple[Counter, Counter]:
    texts: list[str] = []
    for block in blocks:
        if block.kind == "heading":
            match = _CHAPTER.match(block.text) if block.level == 1 else None
            texts += [match.group(2), match.group(3)] if match else [block.text]
        elif block.kind == "caption":
            texts += [block.number, block.text]
        elif block.kind in ("para", "quote"):
            texts.append(block.text)
        elif block.kind == "list":
            texts += block.items
        elif block.kind == "table":
            texts += [cell for row in block.rows for cell in row]
        elif block.kind == "code":
            texts += block.items
    pieces = []
    for text in texts:
        pieces += [seg.text for seg in inline_segments(text)] if "`" in text or "*" in text or "<" in text else [text]
    joined = " ".join(pieces).translate(str.maketrans("¹²³", "123"))
    joined = re.sub(r"(?<=[\s(])\[(\d+)\]", r" \1 ", joined)
    return _tokens(joined)


def _tex_tokens(tex: str) -> tuple[Counter, Counter]:
    listings: list[str] = []

    def keep_listing(match: re.Match) -> str:
        listings.append(match.group(2))
        return match.group(1)

    tex = re.sub(r"(\\begin\{lstlisting\}(?:\[[^\n]*\])?)\n(.*?)\\end\{lstlisting\}", keep_listing, tex, flags=re.S)
    tex = re.sub(r"\\endfirsthead.*?\\endhead", " ", tex, flags=re.S)  # the repeated table header
    tex = re.sub(r"^\s*(\\begin\{(?:long)?table\}|\\begin\{tabular\}|\\noindent\\fcolorbox|%).*$", " ", tex,
                 flags=re.M)
    tex = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\begin\{(?:figure|algorithm)\}\[[^\]]*\]|\\begin\{enumerate\}\[start=\d+\]", " ", tex)
    tex = re.sub(r"\\begin\{thebibliography\}\{99\}", " ", tex)
    tex = re.sub(r"\\(?:hspace\*?|vspace|fontsize|setlength|setcounter|linespread)\{[^}]*\}(?:\{[^}]*\})?", " ", tex)
    tex = re.sub(r"\\(?:label|ref)\{[a-z]+:([^}]*)\}|label=\{[a-z]+:([^}]*)\}", r" \1\2 ", tex)
    tex = re.sub(r"\\(?:cite|bibitem)\{ref(\d+)\}", r" \1 ", tex)
    tex = re.sub(r"\\(?:begin|end|rowcolor)\{[^}]*\}", " ", tex)
    accents = {"'": "\u0301", '"': "\u0308"}
    tex = re.sub(r"""\\(['"])\{([A-Za-z])\}""", lambda m: unicodedata.normalize("NFC", m.group(2) + accents[m.group(1)]),
                 tex)
    tex = re.sub(r"\\[A-Za-z@]+\*?", " ", tex).replace("caption=", " ")
    code = re.sub(r"\(\*@.*?@\*\)", " ", "\n".join(listings))
    return _tokens(re.sub(r"[{}~]", " ", tex) + "\n" + code)


def _tokens(text: str) -> tuple[Counter, Counter]:
    numbers = Counter(re.findall(r"\d+(?:[.,]\d+)*", text))
    words = Counter(w.lower() for w in re.findall(r"[A-Za-z]{4,}", text))
    return numbers, words


def compare_with_markdown(md_blocks: list[Block], tex: str, name: str) -> list[str]:
    """Same numbers and words in the .tex as in the Markdown (as multisets): nothing dropped, nothing changed."""
    md_numbers, md_words = _md_tokens(md_blocks)
    tex_numbers, tex_words = _tex_tokens(tex)
    problems = []
    for what, md, tx in (("numbers", md_numbers, tex_numbers), ("words", md_words, tex_words)):
        missing, extra = md - tx, tx - md
        if missing or extra:
            problems.append(f"{name}: {what} differ from the Markdown; missing {dict(missing)}, extra {dict(extra)}")
    return problems


# ---------------------------------------------------------------- compile

def compile_pdf() -> Path | None:
    """latexmk (or pdflatex three times) when installed; returns the PDF or None."""
    latexmk, pdflatex = shutil.which("latexmk"), shutil.which("pdflatex")
    if not (latexmk or pdflatex):
        print("No local LaTeX (latexmk / pdflatex not on the PATH): compile IDSS_Report_Overleaf.zip on Overleaf"
              " (New Project > Upload Project, Menu > Compiler: pdfLaTeX).")
        return None
    flags = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
    if latexmk:
        subprocess.run([latexmk, "-pdf", *flags, "main.tex"], cwd=OUT_DIR, timeout=1200, check=False)
    else:
        for _ in range(3):
            subprocess.run([pdflatex, *flags, "main.tex"], cwd=OUT_DIR, timeout=600, check=False)
    log_path, pdf = OUT_DIR / "main.log", OUT_DIR / "main.pdf"
    if not pdf.exists() or not log_path.exists():
        raise SystemExit("LaTeX did not produce main.pdf; see docs/report_latex/main.log")
    log = log_path.read_text(encoding="latin-1")
    errors = re.findall(r"^! .*$|^.*:\d+: .*$", log, flags=re.M)
    undefined = re.findall(r"(?:Reference|Citation) `[^']+' on page \S+ undefined", log)
    overfull = re.findall(r"Overfull \\hbox \((\d+\.?\d*)pt too wide\)[^\n]*lines? (\d+)", log)
    big = [(float(pt), line) for pt, line in overfull if float(pt) > 1.0]
    target = OUT_DIR / PDF_NAME
    shutil.copyfile(pdf, target)
    print(f"Compiled {target.relative_to(ROOT)}; {len(overfull)} overfull boxes ({len(big)} above 1 pt),"
          f" {len(undefined)} undefined references, {len(errors)} error lines.")
    for pt, line in big[:20]:
        print(f"  overfull {pt:.1f} pt at line {line}")
    if errors or undefined or "Label(s) may have changed" in log:
        raise SystemExit("LaTeX log has errors, undefined references or unresolved labels:\n  "
                         + "\n  ".join((errors + undefined)[:20]))
    return target


# ---------------------------------------------------------------- main

def build() -> dict[str, str]:
    front = split_by_title(read_md("00_front_matter.md"))
    counts: Counter = Counter()
    labels: set[str] = set()
    figures: list[Path] = []
    files: dict[str, str] = {}
    problems: list[str] = []
    for name in CHAPTER_FILES + APPENDIX_FILES:
        blocks = read_md(name)
        tex = ChapterWriter(counts, labels, figures).convert(blocks)
        stem = Path(name).stem if name in CHAPTER_FILES else "_".join(Path(name).stem.split("_")[:2])
        header = f"% Generated by scripts/build_latex.py from docs/report/{name}: edit the Markdown, not this file.\n"
        files[f"chapters/{stem}.tex"] = header + tex
        problems += compare_with_markdown(blocks, tex, name)
    references = read_md("references.md")
    files["chapters/references.tex"] = references_tex(references)
    problems += compare_with_markdown(references, files["chapters/references.tex"], "references.md")
    files["chapters/publications.tex"] = publications_tex(read_md("publications.md"))
    files["chapters/frontmatter.tex"] = frontmatter_tex(front)
    chapter_stems = [Path(n).stem for n in CHAPTER_FILES]
    appendix_stems = ["_".join(Path(n).stem.split("_")[:2]) for n in APPENDIX_FILES]
    files["main.tex"] = main_tex(chapter_stems, appendix_stems)

    for kind, number in EXPECTED.items():
        if counts[kind] != number:
            problems.append(f"{counts[kind]} {kind}s converted, {number} expected")

    # write the folder: stale chapters and figures are removed
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHAPTER_DIR.glob("*.tex"):
        if f"chapters/{old.name}" not in files:
            old.unlink()
    for old in FIGURE_DIR.glob("*.png"):
        if old.name not in {f.name for f in figures}:
            old.unlink()
    for relative, text in files.items():
        (OUT_DIR / relative).write_text(text, encoding="ascii", newline="\n")
    for source in figures:
        copy_figure(source, FIGURE_DIR / source.name)

    problems += check_tex(files)
    if problems:
        raise SystemExit("LaTeX build checks failed:\n  " + "\n  ".join(problems))
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}: main.tex, {len(files) - 1} chapter files, {len(figures)} figures;"
          f" {counts['figure']} figures, {counts['table']} tables, {counts['algorithm']} algorithms,"
          f" {counts['listing']} listings; {len(labels)} labels. Static checks passed.")
    return files


def write_zip(files: dict[str, str]) -> Path:
    target = OUT_DIR / ZIP_NAME
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            archive.write(OUT_DIR / relative, relative)
        for figure in sorted(FIGURE_DIR.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")
    print(f"Saved {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB) for Overleaf.")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-compile", action="store_true", help="write the .tex files and the zip only")
    args = parser.parse_args()
    files = build()
    write_zip(files)
    if not args.no_compile:
        pdf = compile_pdf()
        if pdf is not None:
            try:
                import pymupdf  # requirements-docs.txt

                with pymupdf.open(pdf) as doc:
                    print(f"{pdf.name}: {doc.page_count} pages.")
            except ImportError:
                pass


if __name__ == "__main__":
    main()
