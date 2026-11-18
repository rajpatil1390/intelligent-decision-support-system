"""Build the A4 project poster (Step 11) -> docs/poster/IDSS_Poster_A4.pdf and IDSS_Poster_A4.png.

docs/poster/poster.html is the source: one A4 portrait page in HTML + CSS (@page A4, margin 0). This script
1. prepares the images the page uses in docs/poster/assets/: the architecture diagram without its title and the
   main area of two dashboard screenshots, trimmed of white margins and downscaled to at most the A4 width at
   300 dpi;
2. writes the two results charts into poster.html (between the <!-- chart:NAME --> markers) as inline-SVG bar
   charts drawn from docs/results/evaluation.json: our model vs the trivial baseline per dataset (with the
   improvement %) and the comprehensive score (CS) of each ablation configuration;
3. checks the page against the sources: every element with a data-fact attribute must show the value computed
   from docs/results/evaluation.json or docs/testing.md, and the title, college, team and guide must match the
   report (scripts/report_common.py);
4. renders the page with Playwright Chromium to a vector PDF and a 300-dpi PNG (2480 x 3508 px), and fails if
   anything overflows its card or the page, an image is missing, text outside the footer and the charts is below
   9 pt, chart text is below 8 pt, overlaps other chart text or leaves its bar, or the PDF has more than one page.

Fonts: Poppins and Inter from Google Fonts when reachable, otherwise the page falls back to Segoe UI.

Usage: .venv\\Scripts\\python.exe scripts\\build_poster.py [--draft]
One-time setup: .venv\\Scripts\\python.exe -m playwright install chromium
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

from report_common import ACADEMIC_YEAR, COLLEGE, DEPARTMENT, GUIDE, PROJECT_TITLE, ROOT, STUDENTS

POSTER_DIR = ROOT / "docs" / "poster"
ASSETS_DIR = POSTER_DIR / "assets"
SOURCE = POSTER_DIR / "poster.html"
PDF_OUT = POSTER_DIR / "IDSS_Poster_A4.pdf"
PNG_OUT = POSTER_DIR / "IDSS_Poster_A4.png"
EVALUATION = ROOT / "docs" / "results" / "evaluation.json"
TESTING = ROOT / "docs" / "testing.md"

DPI = 300
A4_MM = (210, 297)
A4_CSS_PX = tuple(mm / 25.4 * 96 for mm in A4_MM)  # 793.7 x 1122.5 CSS px
A4_PNG_PX = tuple(round(mm / 25.4 * DPI) for mm in A4_MM)  # 2480 x 3508 px
MAX_ASSET_PX = A4_PNG_PX[0]  # no image is printed wider than the page, so this keeps >= 300 dpi

# (source, output, crop box as fractions of the source (left, top, right, bottom))
ASSETS: list[tuple[Path, str, tuple[float, float, float, float]]] = [
    # without the title, subtitle, "External" label and the arrows footnote (the card states the same)
    (ROOT / "docs" / "diagrams" / "system_architecture.png", "architecture.png", (0, 0.105, 1, 0.9445)),
    # main area only (no sidebar): Results from the banner to the leaderboard, Decision Center brief to risk tiers
    (ROOT / "docs" / "screenshots" / "05_results.png", "screen_results.png",
     (370 / 1600, 12 / 1306, 1530 / 1600, 940 / 1306)),
    (ROOT / "docs" / "screenshots" / "06_decision_center.png", "screen_decision_center.png",
     (370 / 1600, 318 / 2412, 1530 / 1600, 1210 / 2412)),
]

OVERFLOW_JS = """() => {
  const problems = [];
  const describe = el => (el.querySelector('h2, h3, b, span') || el).textContent.trim().slice(0, 40);
  const poster = document.querySelector('.poster');
  if (poster.scrollHeight > poster.clientHeight + 1 || poster.scrollWidth > poster.clientWidth + 1)
    problems.push(`page content ${poster.scrollWidth}x${poster.scrollHeight} > ${poster.clientWidth}x${poster.clientHeight} px`);
  for (const el of document.querySelectorAll('.card, .step, .tile, .kpi, .chip, .shot, footer')) {
    if (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1)
      problems.push(`${el.className}: "${describe(el)}" ${el.scrollWidth}x${el.scrollHeight} > ${el.clientWidth}x${el.clientHeight}`);
    const r = el.getBoundingClientRect(), p = poster.getBoundingClientRect();
    if (r.bottom > p.bottom + 0.5 || r.right > p.right + 0.5)
      problems.push(`${el.className}: "${describe(el)}" extends beyond the page`);
  }
  for (const img of document.images)
    if (!img.complete || img.naturalWidth === 0) problems.push(`image not loaded: ${img.getAttribute('src')}`);
  return problems;
}"""

IMAGES_JS = """() => [...document.images].map(img => {
  const r = img.getBoundingClientRect();
  const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);  // object-fit: contain
  return [img.getAttribute('src'), Math.round(r.width / 96 * 25.4), Math.round(r.height / 96 * 25.4),
          Math.round(96 / scale)];
})"""

SMALL_TEXT_JS = """() => {
  const sizes = {};
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.textContent.trim() || node.parentElement.closest('svg, footer')) continue;
    const pt = Math.round(parseFloat(getComputedStyle(node.parentElement).fontSize) * 0.75 * 10) / 10;
    if (pt < 9) (sizes[pt] ??= []).push(node.textContent.trim().slice(0, 30));
  }
  return sizes;
}"""

# Chart text: >= 8 pt, inside the chart, no two labels overlapping (cap-height boxes, so labels on adjacent bars
# may share a line box), and every value drawn inside a bar (class "in") fits between the bar's ends.
CHART_JS = """() => {
  const problems = [];
  for (const svg of document.querySelectorAll('svg.chart')) {
    const frame = svg.getBoundingClientRect(), scale = svg.getScreenCTM().a, boxes = [];
    for (const t of svg.querySelectorAll('text')) {
      const px = parseFloat(getComputedStyle(t).fontSize) * scale, r = t.getBoundingClientRect();
      const label = `${svg.id}: "${t.textContent}"`;
      if (px * 0.75 < 8) problems.push(`${label} is ${(px * 0.75).toFixed(1)} pt (< 8 pt)`);
      if (r.left < frame.left - 0.5 || r.right > frame.right + 0.5 || r.top < frame.top - 0.5
          || r.bottom > frame.bottom + 0.5) problems.push(`${label} extends beyond the chart`);
      const mid = (r.top + r.bottom) / 2, box = {label, left: r.left, right: r.right,
                                                 top: mid - 0.36 * px, bottom: mid + 0.36 * px};
      for (const b of boxes)
        if (box.left < b.right && b.left < box.right && box.top < b.bottom && b.top < box.bottom)
          problems.push(`${label} overlaps ${b.label}`);
      boxes.push(box);
      if (t.classList.contains('in')) {
        const bar = t.previousElementSibling.getBoundingClientRect();
        if (r.left < bar.left + 0.5 || r.right > bar.right) problems.push(`${label} does not fit inside its bar`);
      }
    }
  }
  return problems;
}"""

# ---------- results charts: inline SVG in mm (1 user unit = 1 mm at the poster's layout size) ----------
CHART_FONT = 3.0  # mm = 8.5 pt at layout size; render() fails if any chart text ends up below 8 pt
PURPLE, BASELINE_TINT, LAVENDER = "#6C5CE7", "#F2B8A8", "#DCD7FB"  # tints keep dark value labels readable
INK, MUTED, GAIN_GREEN, AXIS = "#2D3436", "#4F5966", "#007A63", "#B8BED3"
ABLATION_LABELS = {"full": "Full system", "no_threshold": "− threshold", "no_hpo": "− HPO",
                   "no_replan": "− re-planning", "single_model": "Single model"}
CHART_BLOCK = re.compile(r"(<!-- chart:(\w+) -->).*?(<!-- /chart:\2 -->)", re.DOTALL)


def _svg_text(x: float, y: float, text: str, *, anchor: str = "start", fill: str = INK, bold: bool = False,
              cls: str = "") -> str:
    extra = (' font-weight="700"' if bold else "") + (f' class="{cls}"' if cls else "")
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" dominant-baseline="central" fill="{fill}"'
            f'{extra}>{html.escape(text)}</text>')


def _bar(x: float, y: float, length: float, height: float, fill: str, value: str, value_fill: str) -> str:
    """A bar with its value printed inside the bar's end, or just after the bar when the bar is too short."""
    rect = f'<rect x="{x:.2f}" y="{y:.2f}" width="{length:.2f}" height="{height:.2f}" rx="0.6" fill="{fill}"/>'
    if length >= 9.5:
        label = _svg_text(x + length - 0.8, y + height / 2, value, anchor="end", fill=value_fill, bold=True, cls="in")
    else:
        label = _svg_text(x + length + 0.8, y + height / 2, value, bold=True)
    return f"<g>{rect}{label}</g>"


def _svg(name: str, width: float, height: float, title: str, parts: list[str]) -> str:
    return (f'<svg class="chart" id="chart-{name}" viewBox="0 0 {width} {height}" font-size="{CHART_FONT}" '
            f'role="img" aria-label="{html.escape(title)}">{"".join(parts)}</svg>')


def baseline_chart(benchmark: list[dict]) -> str:
    """Per dataset: our model and the trivial baseline on a 0-1 score, plus the gain on the primary metric (%)."""
    width, height, label_x, x0, track, top, pitch, bar_h = 58.6, 35.4, 18.0, 19.0, 27.0, 0.2, 7.05, 2.95
    parts = [f'<line x1="{x0}" y1="{top - 0.4}" x2="{x0}" y2="{top + len(benchmark) * pitch - 0.4}" '
             f'stroke="{AXIS}" stroke-width="0.25"/>']
    for i, row in enumerate(benchmark):
        y = top + i * pitch
        # 0-1 metrics (F1, ARI) as they are; error metrics (RMSE) as the normalised score 1 / (1 + RMSLE) (NPS)
        key, base_key = ("score", "baseline") if row["direction"] == "maximize" else ("nps", "baseline_nps")
        model, base = row[key][0], row[base_key][0]
        parts.append(_bar(x0, y, model * track, bar_h, PURPLE, f"{model:.3f}", "#FFFFFF"))
        parts.append(_bar(x0, y + bar_h + 0.2, max(base, 0.0) * track, bar_h, BASELINE_TINT, f"{base:.3f}", INK))
        middle = y + bar_h + 0.1  # dataset name and improvement sit between the two bars
        parts.append(_svg_text(label_x, middle, row["dataset"], anchor="end"))
        gain = row["improvement_pct"][0]  # None when the baseline is at chance level (relative gain undefined)
        parts.append(_svg_text(width - 0.2, middle, "—" if gain is None else f"{gain:+.1f}%", anchor="end",
                               fill=MUTED if gain is None else GAIN_GREEN, bold=gain is not None))
    return _svg("baseline", width, height, "Our model vs the trivial baseline per dataset", parts)


def ablation_chart(ablation: list[dict]) -> str:
    """Overall comprehensive score (CS) of each ablation configuration, on a 0-1 axis."""
    width, height, label_x, x0, track, top, pitch, bar_h = 40.5, 35.4, 20.4, 21.1, 19.2, 0.3, 6.1, 4.4
    full_cs = next(row for row in ablation if row["variant"] == "full")["cs"][0]
    parts = []
    for i, row in enumerate(ablation):
        y, cs, full = top + i * pitch, row["cs"][0], row["variant"] == "full"
        parts.append(_bar(x0, y, cs * track, bar_h, PURPLE if full else LAVENDER, f"{cs:.3f}",
                          "#FFFFFF" if full else INK))
        parts.append(_svg_text(label_x, y + bar_h / 2, ABLATION_LABELS[row["variant"]], anchor="end"))
    bottom = top + len(ablation) * pitch - 0.4
    ref_x = x0 + full_cs * track
    parts.append(f'<line x1="{ref_x:.2f}" y1="{top - 0.4}" x2="{ref_x:.2f}" y2="{bottom:.2f}" stroke="{PURPLE}" '
                 f'stroke-width="0.3" stroke-dasharray="0.8 0.6"/>')
    parts.append(f'<line x1="{x0}" y1="{bottom:.2f}" x2="{x0 + track}" y2="{bottom:.2f}" stroke="{AXIS}" '
                 f'stroke-width="0.25"/>')
    for value, anchor in ((0.0, "start"), (0.5, "middle"), (1.0, "end")):
        parts.append(_svg_text(x0 + value * track, height - 2.2, f"{value:g}", anchor=anchor, fill=MUTED))
    return _svg("ablation", width, height, "Comprehensive score by ablation configuration", parts)


def write_charts() -> None:
    """Replace each <!-- chart:NAME --> ... <!-- /chart:NAME --> block in poster.html with a freshly drawn chart."""
    summary = json.loads(EVALUATION.read_text(encoding="utf-8"))["summary"]
    blocks = {
        "baseline": f'<div class="cap cap-split"><span><i style="background:{PURPLE}"></i>Our model vs<i '
                    f'style="background:{BASELINE_TINT}"></i>baseline</span><span class="cap-note">gain</span>'
                    f'</div>' + baseline_chart(summary["benchmark"]),
        "ablation": '<div class="cap">Ablation – CS by setup</div>' + ablation_chart(summary["ablation"]),
    }
    found: set[str] = set()

    def fill(match: re.Match[str]) -> str:
        name = match.group(2)
        if name not in blocks:
            raise SystemExit(f"Unknown chart {name!r} in {SOURCE.name}")
        found.add(name)
        return f"{match.group(1)}{blocks[name]}{match.group(3)}"

    page = SOURCE.read_bytes().decode("utf-8")
    updated = CHART_BLOCK.sub(fill, page)
    if missing := blocks.keys() - found:
        raise SystemExit(f"Chart markers missing in {SOURCE.name}: {sorted(missing)}")
    if updated != page:
        SOURCE.write_bytes(updated.encode("utf-8"))
    print(f"  {len(found)} charts drawn from evaluation.json{' (poster.html updated)' if updated != page else ''}")


def trim_white(image: Image.Image, pad: int) -> Image.Image:
    """Crop away the white margin around the content, keeping `pad` px."""
    rgb = image.convert("RGB")
    bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).point(lambda v: 255 if v > 12 else 0)
    box = bbox.getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    return image.crop((max(0, left - pad), max(0, top - pad), min(image.width, right + pad),
                       min(image.height, bottom + pad)))


def prepare_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for source, name, (left, top, right, bottom) in ASSETS:
        image = Image.open(source)
        w, h = image.size
        image = image.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))
        if "screen" not in name:  # screenshots keep their page background
            image = trim_white(image, pad=round(0.012 * image.width))
        if image.width > MAX_ASSET_PX:
            image = image.resize((MAX_ASSET_PX, round(image.height * MAX_ASSET_PX / image.width)),
                                 Image.Resampling.LANCZOS)
        image.convert("RGB").save(ASSETS_DIR / name, optimize=True)
        print(f"  assets/{name}: {image.width} x {image.height} px")


def expected_facts() -> dict[str, str]:
    """Every number the poster may show, computed from the evaluation results and the testing notes."""
    summary = json.loads(EVALUATION.read_text(encoding="utf-8"))["summary"]
    overall = summary["paper"]["Overall"]
    ablation = {row["variant"]: row for row in summary["ablation"]}
    churn = ablation["full"]["per_case"]["Churn"]
    churn_fixed = ablation["no_threshold"]["per_case"]["Churn"]
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
        "coverage": from_testing(r"\*\*Total: (\d+)%\*\*") + "%",
        "tests": from_testing(r"\*\*(\d+) tests\*\*"),
        "tests_passed": from_testing(r"\*\*(\d+) passed, \d+ deselected\*\*"),
        "datasets": str(len(benchmark)),
        "seeds": str(len(json.loads(EVALUATION.read_text(encoding="utf-8"))["seeds"])),
        "beats_baseline": f"{sum(b['nps'][0] > b['baseline_nps'][0] for b in benchmark)}/{len(benchmark)}",
        "verified": f"{sum(b['verified'] for b in benchmark)}/{sum(b['runs'] for b in benchmark)}",
        "churn_f1": f"{churn['score'][0]:.3f}",
        "churn_f1_fixed": f"{churn_fixed['score'][0]:.3f}",
        "threshold_change": f"{churn_fixed['change_vs_full_pct'][0]:.1f}%",
    }


def check_content(page_html: str) -> None:
    facts = expected_facts()
    shown = re.findall(r'data-fact="(\w+)"[^>]*>([^<]*)<', page_html)
    if not shown:
        raise SystemExit("No data-fact elements found in poster.html")
    errors = []
    for key, value in shown:
        value = html.unescape(value).strip().replace("−", "-")  # typographic minus on the page
        if key not in facts:
            errors.append(f"unknown fact {key!r}")
        elif value != facts[key]:
            errors.append(f"{key}: poster shows {value!r}, sources give {facts[key]!r}")
    text = html.unescape(re.sub(r"<[^>]+>", " ", page_html))
    text = re.sub(r"\s+", " ", text)
    for required in [PROJECT_TITLE, COLLEGE, DEPARTMENT, GUIDE, ACADEMIC_YEAR, *(name for name, _ in STUDENTS)]:
        if required not in text:
            errors.append(f"missing text: {required!r}")
    if errors:
        raise SystemExit("Poster content check failed:\n  " + "\n  ".join(errors))
    print(f"  {len(shown)} numbers match evaluation.json / testing.md; title, college and team match the report")


def pdf_page_count(path: Path) -> int:
    try:
        import pymupdf  # requirements-docs.txt
    except ImportError:
        return len(re.findall(rb"/Type\s*/Page(?!s)", path.read_bytes()))
    with pymupdf.open(path) as doc:
        return doc.page_count


def render(draft: bool = False) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=DPI / 96)
        page.goto(SOURCE.as_uri(), wait_until="networkidle", timeout=60_000)
        page.emulate_media(media="print")  # the PNG shows exactly what the PDF prints
        page.evaluate("document.fonts.ready.then(() => true)")
        web_fonts = page.evaluate("document.fonts.check('700 12px Poppins') && document.fonts.check('12px Inter')"
                                  " && [...document.fonts].some(f => f.family.includes('Poppins')"
                                  " && f.status === 'loaded')")
        print(f"  fonts: {'Poppins + Inter (Google Fonts)' if web_fonts else 'system fallback (Segoe UI)'}")
        problems = page.evaluate(OVERFLOW_JS)
        if problems and not draft:
            browser.close()
            raise SystemExit("Layout check failed:\n  " + "\n  ".join(problems))
        for problem in problems:
            print(f"  LAYOUT: {problem}")
        chart_problems = page.evaluate(CHART_JS)
        if chart_problems and not draft:
            browser.close()
            raise SystemExit("Chart check failed:\n  " + "\n  ".join(chart_problems))
        for problem in chart_problems:
            print(f"  CHART: {problem}")
        small = page.evaluate(SMALL_TEXT_JS)
        if small and not draft:
            browser.close()
            raise SystemExit(f"Text below 9 pt outside the footer: {small}")
        for src, w_mm, h_mm, dpi in page.evaluate(IMAGES_JS):
            print(f"  {src}: {w_mm} x {h_mm} mm, {dpi} dpi")
        page.pdf(path=str(PDF_OUT), prefer_css_page_size=True, print_background=True)
        page.screenshot(path=str(PNG_OUT), clip={"x": 0, "y": 0, "width": A4_CSS_PX[0], "height": A4_CSS_PX[1]})
        browser.close()

    pages = pdf_page_count(PDF_OUT)
    if pages != 1:
        raise SystemExit(f"{PDF_OUT.name} has {pages} pages, expected 1")
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
    print("Preparing images:")
    prepare_assets()
    print("Drawing charts:")
    write_charts()
    print("Checking content:")
    check_content(SOURCE.read_text(encoding="utf-8"))
    print("Rendering with Chromium:")
    render(draft=args.draft)
    for path in (PDF_OUT, PNG_OUT):
        print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
