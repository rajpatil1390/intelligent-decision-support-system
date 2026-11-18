"""Build the one-page A4 project report (Step 13) -> docs/one_page/IDSS_One_Page_Report.pdf and .png.

docs/one_page/one_page_report.html is the source: one A4 portrait page in HTML + CSS whose layout and style follow
docs/format/one_page_report_reference.pdf (masthead with two logos, navy title banner, two columns with orange
section markers, navy table headers and shaded rows). This script
1. extracts the two logos from the college PowerPoint template (docs/format/ppt_template.pptx, gitignored) into
   docs/one_page/assets/: the GHR College logo and the "raisoni EDUCATION" logo, trimmed of their white margins
   (without the template the committed assets are used);
2. writes the results table (dataset, best model, score, verified runs) into the page between the
   <!-- table:results --> markers, from docs/results/evaluation.json;
3. checks the content: every element with a data-fact attribute must show the value computed from
   docs/results/evaluation.json or docs/testing.md, any other number on the page must be a reference year listed in
   docs/report/references.md or a module number, and the title, guide, team, college and department must match the
   report (scripts/report_common.py);
4. renders the page with Playwright Chromium to a vector PDF and a 300-dpi PNG (2480 x 3508 px) and fails if
   anything overflows the page, a column, a box or a table cell, text runs into a box's padding, text is below
   7.5 pt, a one-line item wraps, an image is missing, or the PDF is not exactly one A4 page with its text as real
   (selectable) text.

Fonts: Inter (static weights from jsDelivr) when reachable, otherwise Segoe UI.

Usage: .venv\\Scripts\\python.exe scripts\\build_one_page.py [--draft]
One-time setup: .venv\\Scripts\\python.exe -m playwright install chromium
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

from report_common import COLLEGE, DEPARTMENT, GUIDE, PROJECT_TITLE, ROOT, STUDENTS

PAGE_DIR = ROOT / "docs" / "one_page"
ASSETS_DIR = PAGE_DIR / "assets"
SOURCE = PAGE_DIR / "one_page_report.html"
PDF_OUT = PAGE_DIR / "IDSS_One_Page_Report.pdf"
PNG_OUT = PAGE_DIR / "IDSS_One_Page_Report.png"
TEMPLATE = ROOT / "docs" / "format" / "ppt_template.pptx"
EVALUATION = ROOT / "docs" / "results" / "evaluation.json"
TESTING = ROOT / "docs" / "testing.md"
REFERENCES = ROOT / "docs" / "report" / "references.md"

DPI = 300
A4_MM = (210, 297)
A4_CSS_PX = tuple(mm / 25.4 * 96 for mm in A4_MM)  # 793.7 x 1122.5 CSS px
A4_PNG_PX = tuple(round(mm / 25.4 * DPI) for mm in A4_MM)  # 2480 x 3508 px
MIN_PT = 7.5

# (media file in the template, output name, crop box in px (left, top, right, bottom) or None for the whole image)
LOGOS: list[tuple[str, str, tuple[int, int, int, int] | None]] = [
    ("ppt/media/image2.png", "ghr_college_logo.png", None),  # the master's top-left logo
    ("ppt/media/image1.jpg", "raisoni_education_logo.png", (1400, 0, 1920, 300)),  # top-right corner of the background
]

MODEL_NAMES = {"logistic_regression": "Logistic regression", "lightgbm": "LightGBM", "xgboost": "XGBoost",
               "ridge": "Ridge", "random_forest": "Random forest", "extra_trees": "Extra trees",
               "gaussian_mixture": "Gaussian mixture", "kmeans": "k-means", "seasonal_naive": "Seasonal naive"}
TABLE_BLOCK = re.compile(r"(<!-- table:(\w+) -->).*?(<!-- /table:\2 -->)", re.DOTALL)
TICK = ('<svg class="mk ok" viewBox="0 0 10 10" role="img" aria-label="verified"><path d="M2 5.4 4.2 7.5 8.1 2.7"/>'
        '</svg>')
CROSS = ('<svg class="mk bad" viewBox="0 0 10 10" role="img" aria-label="not verified"><path d="M2.7 2.7 7.3 7.3M7.3 '
         '2.7 2.7 7.3"/></svg>')

LAYOUT_JS = """() => {
  const problems = [];
  const mm = px => (px / 96 * 25.4).toFixed(1);
  const describe = el => (el.querySelector('h2, th, b') || el).textContent.trim().replace(/\\s+/g, ' ').slice(0, 40);
  const page = document.querySelector('.page'), pr = page.getBoundingClientRect();
  if (page.scrollHeight > page.clientHeight + 1 || page.scrollWidth > page.clientWidth + 1)
    problems.push(`page content ${mm(page.scrollWidth)} x ${mm(page.scrollHeight)} mm > A4`);
  for (const el of document.querySelectorAll('.col, section, table, td, th, .mhead, .mitem, .root, .banner, footer, li')) {
    if (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1)
      problems.push(`${el.tagName.toLowerCase()}.${el.className}: "${describe(el)}" overflows `
                    + `(${mm(el.scrollWidth)} x ${mm(el.scrollHeight)} > ${mm(el.clientWidth)} x ${mm(el.clientHeight)} mm)`);
    const r = el.getBoundingClientRect();
    if (r.bottom > pr.bottom - 4 / 25.4 * 96 || r.right > pr.right - 7 / 25.4 * 96 || r.left < pr.left + 7 / 25.4 * 96)
      problems.push(`${el.tagName.toLowerCase()}: "${describe(el)}" leaves the printable area`);
  }
  // text must stay inside each box's content area: scrollWidth includes the padding, so text running into a
  // cell's padding is only caught by measuring the text itself
  for (const el of document.querySelectorAll('td, th, .mhead, .mitem, .root, li, p, h2, h3')) {
    const s = getComputedStyle(el), r = el.getBoundingClientRect(), range = document.createRange();
    range.selectNodeContents(el);
    const t = range.getBoundingClientRect();
    const left = r.left + parseFloat(s.borderLeftWidth) + parseFloat(s.paddingLeft) - 0.5;
    const right = r.right - parseFloat(s.borderRightWidth) - parseFloat(s.paddingRight) + 0.5;
    if (t.width && (t.left < left || t.right > right))
      problems.push(`${el.tagName.toLowerCase()}: "${el.textContent.trim().replace(/\\s+/g, ' ').slice(0, 40)}" text `
                    + `runs ${mm(Math.max(left - t.left, t.right - right))} mm into its padding or beyond`);
  }
  // a column's sections must end above the column's bottom edge (the column clips its overflow)
  for (const col of document.querySelectorAll('.col')) {
    const last = [...col.children].at(-1).getBoundingClientRect(), cr = col.getBoundingClientRect();
    if (last.bottom > cr.bottom + 0.5) problems.push(`#${col.id} content ends ${mm(last.bottom - cr.bottom)} mm below the column`);
  }
  // one line per system module, bullet, diagram box and reference, unless a parent allows more (data-lines)
  for (const el of document.querySelectorAll('ol.modules p, ul.dots li, .mitem, .mhead, .root, .refs li')) {
    const style = getComputedStyle(el), lh = parseFloat(style.lineHeight);
    const lines = Math.round((el.getBoundingClientRect().height - parseFloat(style.paddingTop)
                              - parseFloat(style.paddingBottom)) / lh);
    const max = +(el.closest('[data-lines]')?.dataset.lines ?? 1);
    if (lines > max) problems.push(`"${el.textContent.trim().replace(/\\s+/g, ' ').slice(0, 50)}" wraps to ${lines} lines (max ${max})`);
  }
  for (const img of document.images)
    if (!img.complete || img.naturalWidth === 0) problems.push(`image not loaded: ${img.getAttribute('src')}`);
  return problems;
}"""

SMALL_TEXT_JS = """(minPt) => {
  const sizes = {};
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.textContent.trim()) continue;
    const pt = Math.round(parseFloat(getComputedStyle(node.parentElement).fontSize) * 0.75 * 10) / 10;
    if (pt < minPt) (sizes[pt] ??= []).push(node.textContent.trim().slice(0, 30));
  }
  return sizes;
}"""

# Free space at the bottom of each column (the columns spread it between their sections) and the gap it makes.
BALANCE_JS = """() => [...document.querySelectorAll('.col')].map(col => {
  const mm = px => +(px / 96 * 25.4).toFixed(1);
  const sections = [...col.children], cr = col.getBoundingClientRect();
  const heights = sections.map(s => s.getBoundingClientRect().height);
  const used = heights.reduce((sum, h) => sum + h, 0);
  const gap = (cr.height - used) / Math.max(1, sections.length - 1);
  return [col.id, mm(cr.height), mm(used), mm(gap),
          sections.map((s, i) => `${s.querySelector('h2').textContent} ${mm(heights[i])}`).join(', ')];
})"""


def trim_white(image: Image.Image, threshold: int = 12) -> Image.Image:
    """Crop away the white margin around the content."""
    rgb = image.convert("RGB")
    mask = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).convert("L")
    box = mask.point(lambda v: 255 if v > threshold else 0).getbbox()
    return rgb.crop(box) if box else rgb


def prepare_logos() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE.exists():
        missing = [name for _, name, _ in LOGOS if not (ASSETS_DIR / name).exists()]
        if missing:
            raise SystemExit(f"{TEMPLATE.relative_to(ROOT)} not found and logos missing: {missing}")
        print(f"  {TEMPLATE.name} not found: using the logos in {ASSETS_DIR.relative_to(ROOT)}")
        return
    with zipfile.ZipFile(TEMPLATE) as pptx:
        for member, name, box in LOGOS:
            image = Image.open(io.BytesIO(pptx.read(member)))
            if box:
                image = image.crop(box)
            image = trim_white(image)
            image.save(ASSETS_DIR / name, optimize=True)
            print(f"  assets/{name}: {image.width} x {image.height} px (from {member})")


def results_table(benchmark: list[dict]) -> str:
    """Dataset, best model, score (normalised performance score, 0-1) and one mark per run: verified or not.

    The marks are inline SVG: the tick and cross glyphs are missing from Inter, and the fallback font Chromium picks
    for them ends up as a Type3 font in the PDF.
    """
    rows = []
    for row in benchmark:
        model = MODEL_NAMES.get(row["best_model"], row["best_model"].replace("_", " ").capitalize())
        marks = (TICK * row["verified"] + CROSS * (row["runs"] - row["verified"]))
        rows.append(f'<tr><td>{html.escape(row["dataset"])}</td><td>{model}</td>'
                    f'<td data-fact="score:{html.escape(row["dataset"])}">{row["nps"][0]:.3f}</td>'
                    f'<td class="marks" data-marks="{row["verified"]}/{row["runs"]}">{marks}</td></tr>')
    return ('<table class="res"><colgroup><col><col><col><col></colgroup><thead><tr><th>Dataset</th>'
            '<th>Best model</th><th>Score</th><th>Verified</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def write_tables() -> None:
    """Replace each <!-- table:NAME --> ... <!-- /table:NAME --> block with a table drawn from evaluation.json."""
    summary = json.loads(EVALUATION.read_text(encoding="utf-8"))["summary"]
    blocks = {"results": results_table(summary["benchmark"])}
    found: set[str] = set()

    def fill(match: re.Match[str]) -> str:
        name = match.group(2)
        if name not in blocks:
            raise SystemExit(f"Unknown table {name!r} in {SOURCE.name}")
        found.add(name)
        return f"{match.group(1)}{blocks[name]}{match.group(3)}"

    page = SOURCE.read_bytes().decode("utf-8")
    updated = TABLE_BLOCK.sub(fill, page)
    if missing := blocks.keys() - found:
        raise SystemExit(f"Table markers missing in {SOURCE.name}: {sorted(missing)}")
    if updated != page:
        SOURCE.write_bytes(updated.encode("utf-8"))
    print(f"  results table drawn from evaluation.json{' (page updated)' if updated != page else ''}")


def expected_facts() -> dict[str, str]:
    """Every number the page may show, computed from the evaluation results and the testing notes."""
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    summary = evaluation["summary"]
    overall = summary["paper"]["Overall"]
    ablation = {row["variant"]: row for row in summary["ablation"]}
    benchmark = summary["benchmark"]
    testing = TESTING.read_text(encoding="utf-8")

    def from_testing(pattern: str) -> str:
        match = re.search(pattern, testing)
        if not match:
            raise SystemExit(f"Pattern {pattern!r} not found in {TESTING.name}")
        return match.group(1)

    return {
        "sr": f"{overall['sr'][0]:.2f}",
        "nps": f"{overall['nps'][0]:.3f}",
        "cs": f"{overall['cs'][0]:.3f}",
        "seeds": str(len(evaluation["seeds"])),
        "datasets": str(len(benchmark)),
        "beats_baseline": f"{sum(b['nps'][0] > b['baseline_nps'][0] for b in benchmark)}/{len(benchmark)}",
        "verified": f"{sum(b['verified'] for b in benchmark)}/{sum(b['runs'] for b in benchmark)}",
        "churn_f1": f"{ablation['full']['per_case']['Churn']['score'][0]:.3f}",
        "churn_f1_fixed": f"{ablation['no_threshold']['per_case']['Churn']['score'][0]:.3f}",
        "tests": from_testing(r"\*\*(\d+) tests\*\*"),
        "tests_passed": from_testing(r"\*\*(\d+) passed, \d+ deselected\*\*"),
        "coverage": from_testing(r"\*\*Total: (\d+)%\*\*") + "%",
        **{f"score:{b['dataset']}": f"{b['nps'][0]:.3f}" for b in benchmark},
    }


def visible_text(page_html: str) -> str:
    body = re.sub(r"<head>.*?</head>|<!--.*?-->", " ", page_html, flags=re.DOTALL)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body))).strip()


def check_content(page_html: str) -> None:
    facts, errors = expected_facts(), []
    shown = re.findall(r'data-fact="([^"]+)"[^>]*>([^<]*)<', page_html)
    if not shown:
        raise SystemExit(f"No data-fact elements found in {SOURCE.name}")
    for key, value in shown:
        value = html.unescape(value).strip()
        if key not in facts:
            errors.append(f"unknown fact {key!r}")
        elif value != facts[key]:
            errors.append(f"{key}: page shows {value!r}, sources give {facts[key]!r}")
    # verified marks: one green tick per verified run, one red cross per run that failed verification
    benchmark = {b["dataset"]: b for b in json.loads(EVALUATION.read_text(encoding="utf-8"))["summary"]["benchmark"]}
    for dataset, marks in re.findall(r"<tr><td>([^<]+)</td>.*?class=\"marks\"[^>]*>(.*?)</td>", page_html):
        row = benchmark.get(html.unescape(dataset))
        if row is None:
            errors.append(f"results table: unknown dataset {dataset!r}")
        elif (marks.count('class="mk ok"'), marks.count('class="mk bad"')) != (row["verified"],
                                                                              row["runs"] - row["verified"]):
            errors.append(f"results table: {dataset} marks do not match {row['verified']}/{row['runs']} verified runs")

    # every other number: a reference year (docs/report/references.md) or a module number 1..n
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", REFERENCES.read_text(encoding="utf-8")))
    modules = re.findall(r'<span class="num">(\d+)</span>', page_html)
    if modules != [str(i) for i in range(1, len(modules) + 1)]:
        errors.append(f"module numbers are not 1..n: {modules}")
    rest = re.sub(r'(data-fact="[^"]+"[^>]*>)[^<]*<', r"\1<", page_html)
    rest = re.sub(r'<span class="num">\d+</span>', "", rest)
    for number in re.findall(r"(?<![A-Za-z])\d+(?:[.,/]\d+)*%?", visible_text(rest)):  # not names like F1
        if number not in years:
            errors.append(f"number {number!r} is neither a checked fact nor a reference year")

    text = visible_text(page_html)
    required = [PROJECT_TITLE, GUIDE, COLLEGE.replace(" and ", " & "), DEPARTMENT.upper(),
                *(name for name, _ in STUDENTS)]
    errors += [f"missing text: {item!r}" for item in required if item not in text]
    if errors:
        raise SystemExit("Content check failed:\n  " + "\n  ".join(errors))
    print(f"  {len(shown)} numbers match evaluation.json / testing.md; other numbers are reference years or module"
          " numbers; title, guide, team, college and department match the report")


def check_pdf(path: Path) -> None:
    """Exactly one A4 page whose text is real text (not an image of text)."""
    import pymupdf  # requirements-docs.txt

    with pymupdf.open(path) as doc:
        if doc.page_count != 1:
            raise SystemExit(f"{path.name} has {doc.page_count} pages, expected 1")
        page = doc[0]
        width_mm, height_mm = (round(v / 72 * 25.4) for v in (page.rect.width, page.rect.height))
        if (width_mm, height_mm) != A4_MM:
            raise SystemExit(f"{path.name} page is {width_mm} x {height_mm} mm, expected A4")
        text = re.sub(r"\s+", " ", page.get_text())
        missing = [item for item in (PROJECT_TITLE.split(" Using")[0], "LITERATURE REVIEW", "SYSTEM MODULES")
                   if item.lower() not in text.lower()]
        if missing:
            raise SystemExit(f"{path.name}: text not found as vector text: {missing}")
        fonts = sorted({f"{font[3].split('+')[-1] or '(unnamed)'} ({font[2]})" for font in page.get_fonts()})
        images = len(page.get_images())
    print(f"  PDF: 1 A4 page, {len(text.split())} words of vector text, {images} images (logos), fonts: "
          + ", ".join(fonts))


def render(draft: bool = False) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=DPI / 96)
        page.goto(SOURCE.as_uri(), wait_until="networkidle", timeout=60_000)
        page.emulate_media(media="print")  # the PNG shows exactly what the PDF prints
        page.evaluate("document.fonts.ready.then(() => true)")
        web_fonts = page.evaluate("[...document.fonts].filter(f => f.family.includes('Inter')"
                                  " && f.status === 'loaded').length")
        print(f"  fonts: {f'Inter ({web_fonts} static weights)' if web_fonts else 'system fallback (Segoe UI)'}")
        problems = page.evaluate(LAYOUT_JS)
        small = page.evaluate(SMALL_TEXT_JS, MIN_PT)
        if small:
            problems.append(f"text below {MIN_PT} pt: {small}")
        for col, height, used, gap, sections in page.evaluate(BALANCE_JS):
            print(f"  {col}: {used} of {height} mm used, {gap} mm between sections ({sections})")
        if problems and not draft:
            browser.close()
            raise SystemExit("Layout check failed:\n  " + "\n  ".join(problems))
        for problem in problems:
            print(f"  LAYOUT: {problem}")
        page.pdf(path=str(PDF_OUT), prefer_css_page_size=True, print_background=True)
        page.screenshot(path=str(PNG_OUT), clip={"x": 0, "y": 0, "width": A4_CSS_PX[0], "height": A4_CSS_PX[1]})
        browser.close()

    check_pdf(PDF_OUT)
    with Image.open(PNG_OUT) as png:
        image = png.convert("RGB")
    if image.size != A4_PNG_PX:  # Chromium rounds the clip to whole device pixels
        image = image.resize(A4_PNG_PX, Image.Resampling.LANCZOS)
    image.save(PNG_OUT, dpi=(DPI, DPI), optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draft", action="store_true",
                        help="write the PDF and PNG even if the layout check fails (for editing the page)")
    args = parser.parse_args()
    print("Preparing logos:")
    prepare_logos()
    print("Writing the results table:")
    write_tables()
    print("Checking content:")
    check_content(SOURCE.read_text(encoding="utf-8"))
    print("Rendering with Chromium:")
    render(draft=args.draft)
    for path in (PDF_OUT, PNG_OUT):
        print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
