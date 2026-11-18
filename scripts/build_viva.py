"""Build the viva preparation sheet (Step 12B): docs/presentation/viva_questions.md -> viva_questions.pdf (A4).

The Markdown holds 30 likely viva questions with short answers, grouped by presenter: one `##` section per student in
presenting order (as in the slides), 7-8 numbered `###` questions each, one answer paragraph per question. The build
checks that structure, checks that every number in the questions and answers (small counts up to 10 excepted) also
appears in the project's own documents (report chapters, evaluation, testing notes, architecture, README),
renders the PDF with Playwright Chromium and fails if an answer prints on fewer than 3 or more than 5 lines.

Usage: .venv\\Scripts\\python.exe scripts\\build_viva.py
Setup: pip install -r requirements-docs.txt; python -m playwright install chromium
"""

from __future__ import annotations

import html
import re

from playwright.sync_api import sync_playwright

from report_common import COLLEGE, ROOT, STUDENTS, Block, inline_segments, parse_markdown

SOURCE = ROOT / "docs" / "presentation" / "viva_questions.md"
PDF_OUT = SOURCE.with_suffix(".pdf")
FACT_SOURCES = [*sorted((ROOT / "docs" / "report").glob("*.md")), ROOT / "docs" / "results" / "evaluation.md",
                ROOT / "docs" / "testing.md", ROOT / "docs" / "architecture.md", ROOT / "README.md"]
PRESENTING_ORDER = [STUDENTS[i][0] for i in (0, 2, 1, 3)]  # Raj, Chaitany, Bhavesh, Varun: the slide order
QUESTIONS = 30
PER_PRESENTER = (7, 8)
ANSWER_LINES = (3, 5)
COLORS = ["#6C5CE7", "#0984E3", "#E17055", "#00B894"]  # one per presenter, from the dashboard palette

_QUESTION = re.compile(r"^(\d+)\.\s+(.+)$")
_DIGIT_GROUP = re.compile(r"(?<=\d)[,   ](?=\d{3}(?!\d))")  # 3,000 / 20 000 -> 3000 / 20000
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(_DIGIT_GROUP.sub("", text)))


def parse(blocks: list[Block]) -> tuple[str, list[str], list[tuple[str, list[tuple[int, str, str]]]]]:
    """Title, intro paragraphs and [(presenter heading, [(number, question, answer)])]; raises on any other shape."""
    if not blocks or blocks[0].kind != "heading" or blocks[0].level != 1:
        raise SystemExit("viva_questions.md must start with a '# ' title")
    title, intro, sections = blocks[0].text, [], []
    question: tuple[int, str] | None = None
    for block in blocks[1:]:
        if block.kind == "heading" and block.level == 2:
            sections.append((block.text, []))
        elif block.kind == "heading" and block.level == 3 and sections:
            match = _QUESTION.match(block.text)
            if not match or question is not None:
                raise SystemExit(f"Expected a numbered question after an answer: {block.text!r}")
            question = (int(match.group(1)), match.group(2))
        elif block.kind == "para" and question is not None:
            sections[-1][1].append((*question, block.text))
            question = None
        elif block.kind == "para" and not sections:
            intro.append(block.text)
        else:
            raise SystemExit(f"Unexpected {block.kind} block: {(block.text or block.items)!r:.60}")
    if question is not None:
        raise SystemExit(f"Question {question[0]} has no answer")
    return title, intro, sections


def check(sections: list[tuple[str, list[tuple[int, str, str]]]]) -> None:
    presenters = [heading.split(" – ")[0] for heading, _ in sections]
    if presenters != PRESENTING_ORDER:
        raise SystemExit(f"Sections must be {PRESENTING_ORDER} in this order, got {presenters}")
    numbered = [n for _, qas in sections for n, _, _ in qas]
    if numbered != list(range(1, QUESTIONS + 1)):
        raise SystemExit(f"Questions must be numbered 1..{QUESTIONS} in order, got {numbered}")
    for heading, qas in sections:
        if not PER_PRESENTER[0] <= len(qas) <= PER_PRESENTER[1]:
            raise SystemExit(f"{heading}: {len(qas)} questions, expected {PER_PRESENTER[0]}-{PER_PRESENTER[1]}")
    corpus = set().union(*(numbers(path.read_text(encoding="utf-8")) for path in FACT_SOURCES))
    unknown = sorted({(n, number) for _, qas in sections for n, question, answer in qas
                      for number in numbers(f"{question} {answer}")
                      if not (number.isdigit() and int(number) <= 10) and number not in corpus})
    if unknown:
        raise SystemExit("Numbers not found in the project documents: "
                         + ", ".join(f"Q{n}: {number}" for n, number in unknown))


def inline(text: str) -> str:
    out = []
    for seg in inline_segments(text):
        piece = html.escape(seg.text)
        if seg.code:
            piece = f"<code>{piece}</code>"
        if seg.bold:
            piece = f"<strong>{piece}</strong>"
        if seg.italic:
            piece = f"<em>{piece}</em>"
        if seg.vert:
            piece = f"<sup>{piece}</sup>" if seg.vert == "superscript" else f"<sub>{piece}</sub>"
        out.append(piece)
    return "".join(out)


CSS = """
@page { size: A4; margin: 15mm 17mm 17mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #2D3436; }
main { width: 176mm; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 1.5mm; color: #6C5CE7; }
.sub { margin: 0 0 3mm; color: #5B6472; font-size: 10pt; }
.intro { margin: 0 0 2mm; }
section { break-before: page; }
section:first-of-type { break-before: auto; margin-top: 5mm; }
h2 { margin: 0 0 4mm; padding: 2.2mm 4mm; border-radius: 3mm; color: #fff; font-size: 13pt; background: var(--c); }
.qa { break-inside: avoid; position: relative; margin: 0 0 3.6mm; padding-left: 4.4mm; }
.qa::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 1.2mm; border-radius: 0.6mm;
              background: var(--c); }
h3 { margin: 0 0 0.8mm; font-size: 11pt; line-height: 1.3; }
h3 .n { color: var(--c); margin-right: 1.5mm; }
.answer { margin: 0; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #EEF0FF; padding: 0 0.6mm; border-radius: 0.8mm; }
"""


def render_html(title: str, intro: list[str], sections: list[tuple[str, list[tuple[int, str, str]]]]) -> str:
    parts = [f"<h1>{inline(title)}</h1>", f'<p class="sub">{inline(intro[0])}</p>']
    parts += [f'<p class="intro">{inline(p)}</p>' for p in intro[1:]]
    for (heading, qas), color in zip(sections, COLORS, strict=True):
        parts.append(f'<section style="--c: {color}"><h2>{inline(heading)}</h2>')
        for n, question, answer in qas:
            parts.append(f'<div class="qa"><h3><span class="n">{n}.</span>{inline(question)}</h3>'
                         f'<p class="answer" data-q="{n}">{inline(answer)}</p></div>')
        parts.append("</section>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")


FOOTER = ('<div style="width:100%;font:8pt Segoe UI,Arial,sans-serif;color:#5B6472;padding:0 17mm;display:flex;'
          f'justify-content:space-between"><span>IDSS viva preparation · {html.escape(COLLEGE)}</span>'
          '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')


def main() -> None:
    title, intro, sections = parse(parse_markdown(SOURCE.read_text(encoding="utf-8")))
    check(sections)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.emulate_media(media="print")
        page.set_content(render_html(title, intro, sections), wait_until="load")
        lines = page.evaluate("""() => [...document.querySelectorAll('.answer')].map(p => [Number(p.dataset.q),
            Math.round(p.getBoundingClientRect().height / parseFloat(getComputedStyle(p).lineHeight))])""")
        bad = [f"Q{n}: {count} lines" for n, count in lines if not ANSWER_LINES[0] <= count <= ANSWER_LINES[1]]
        if bad:
            browser.close()
            raise SystemExit(f"Answers must print on {ANSWER_LINES[0]}-{ANSWER_LINES[1]} lines: {', '.join(bad)}")
        page.pdf(path=str(PDF_OUT), prefer_css_page_size=True, print_background=True, display_header_footer=True,
                 header_template="<span></span>", footer_template=FOOTER)
        browser.close()

    import pymupdf  # requirements-docs.txt

    with pymupdf.open(PDF_OUT) as doc:
        pages = doc.page_count
    counts = ", ".join(f"{heading.split(' – ')[0]} {len(qas)}" for heading, qas in sections)
    print(f"Checked {QUESTIONS} questions ({counts}); every number found in the project documents; answers "
          f"{min(c for _, c in lines)}-{max(c for _, c in lines)} lines.")
    print(f"Saved {PDF_OUT.relative_to(ROOT)} ({pages} pages, {PDF_OUT.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
