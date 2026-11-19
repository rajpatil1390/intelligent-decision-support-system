"""Full-page UI screenshots for the report, poster and slides (Step 9) -> docs/screenshots/*.png.

Drives the running Streamlit UI with Playwright (Chromium, 1600 px wide) through a complete story: upload
store_sales.csv and start a forecast, then Demo Mode on customer_churn.csv, and captures every page once its
charts have rendered. Needs the API and the UI running on an EMPTY workspace (the run pickers default to the
newest runs), ideally offline, e.g. in two terminals:

    $env:WORKSPACE_DIR = "<empty folder>"; $env:LLM_MODEL = " "; $env:LLM_API_KEY = " "
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
    .venv\\Scripts\\python.exe -m streamlit run ui\\app.py --server.port 8501 --server.address localhost --server.headless true

then:  .venv\\Scripts\\python.exe scripts\\make_screenshots.py
       (--skip-runs re-captures 01 and 05-10 from the two runs a previous call made)
One-time setup: .venv\\Scripts\\python.exe -m playwright install chromium
"""

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
SAMPLES = ROOT / "data" / "samples"
WIDTH, HEIGHT = 1600, 1000
RUN_TIMEOUT_MS = 15 * 60 * 1000


def settle(page: Page, extra_ms: int = 1200) -> None:
    """Wait until the script run has finished and every Plotly chart has drawn."""
    page.wait_for_selector(".idss-hero", timeout=60_000)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=30_000)
    except Exception:  # noqa: BLE001 – the widget may never appear; nothing to wait for then
        pass
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-testid="stPlotlyChart"]')]
                 .every(el => el.querySelector('.main-svg'))""", timeout=30_000)
    page.wait_for_timeout(extra_ms)


def shoot(page: Page, name: str) -> None:
    """Full-page capture: Streamlit scrolls inside its main section, so grow the viewport to the content first."""
    settle(page)
    height = page.evaluate("""() => {
        const main = document.querySelector('[data-testid="stMain"]');
        return Math.ceil(main ? main.scrollHeight : document.body.scrollHeight);
    }""")
    page.set_viewport_size({"width": WIDTH, "height": max(HEIGHT, height + 8)})
    settle(page, 1500)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": WIDTH, "height": HEIGHT})
    print(f"  saved {path.relative_to(ROOT)}")


def go(page: Page, title: str) -> None:
    """Open a page through the sidebar navigation (keeps the session, so the current run stays selected)."""
    page.locator('[data-testid="stSidebarNav"]').get_by_text(title, exact=True).click()
    page.wait_for_timeout(800)
    settle(page)


def choose(page: Page, label: str, option: str) -> None:
    """Pick `option` (substring) in the select box labelled `label`."""
    box = page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(label, exact=True)).first
    box.locator('input[role="combobox"]').click()
    page.get_by_role("option", name=option).first.click()
    page.wait_for_timeout(800)
    settle(page)


def wait_finished(page: Page) -> None:
    page.get_by_text("The run finished").wait_for(timeout=RUN_TIMEOUT_MS)
    settle(page)


def make_runs(page: Page) -> None:
    """Upload store_sales.csv and run the sales forecast, then Demo Mode on customer churn (02, 03, 04)."""
    print("Upload store_sales.csv")
    go(page, "Upload Data")
    page.locator('[data-testid="stFileUploader"] input[type="file"]').set_input_files(str(SAMPLES / "store_sales.csv"))
    page.get_by_role("button", name="Upload and profile").click()
    page.get_by_text("Uploaded", exact=False).first.wait_for(timeout=60_000)
    shoot(page, "02_upload_profile")

    print("New analysis: sales forecast")
    go(page, "New Analysis")
    page.get_by_role("button", name="Example: Sales forecast").click()
    page.wait_for_timeout(800)
    shoot(page, "03_new_analysis")
    page.get_by_role("button", name="Start analysis").click()
    print("  waiting for the forecast run to finish…")
    wait_finished(page)

    print("Demo Mode: customer churn")
    go(page, "Home")
    page.get_by_role("button", name="Demo Mode").click()
    page.locator(".idss-stage.current").filter(has_text="Train").wait_for(timeout=RUN_TIMEOUT_MS)
    time.sleep(4)  # let a few more events arrive
    shoot(page, "04_run_monitor")
    print("  waiting for the churn run to finish…")
    wait_finished(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--skip-runs", action="store_true",
                        help="do not upload or train; capture the result pages of the existing runs")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.goto(args.base_url)
        settle(page)

        if not args.skip_runs:
            make_runs(page)

        go(page, "Home")
        shoot(page, "01_home")
        go(page, "Results")
        shoot(page, "05_results")
        go(page, "Decision Center")
        shoot(page, "06_decision_center")

        print("What-if simulation")
        page.get_by_role("tab", name="What-if simulator").click()
        page.wait_for_timeout(800)
        choose(page, "contract_type", "Two year")
        page.get_by_role("button", name="Simulate").click()
        page.get_by_text("Result", exact=True).wait_for(timeout=60_000)
        shoot(page, "07_what_if")

        print("Forecast (store sales run)")
        page.get_by_role("tab", name="Decision brief").click()
        choose(page, "Run", "Forecast daily store sales")
        shoot(page, "08_forecast")

        go(page, "Ask Your Data")
        shoot(page, "09_ask_your_data")
        go(page, "Reports")
        page.wait_for_timeout(1500)
        shoot(page, "10_reports")
        browser.close()
    print(f"Done: {len(list(OUT_DIR.glob('*.png')))} screenshots in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
