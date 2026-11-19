"""Record the README demo GIF -> docs/assets/demo.gif, and save the demo run's facts -> docs/assets/demo_run.json.

Starts its own offline API and UI (no LLM, web search off) on spare ports with a throw-away workspace, then drives
the UI with Playwright (Chromium): Home -> Demo Mode -> Run Monitor -> Results -> Decision Center what-if
(contract_type -> "Two year"). Each step is a captioned key frame; short cross-fades join them (~15 s, looping).
The run's requirement, metrics, leaderboard, drivers, risk tiers, recommendations and example what-if are written
to demo_run.json, the source of the README's example walkthrough.

    .venv\\Scripts\\python.exe scripts\\make_demo_gif.py           # ~2-4 min (one fast training run)

One-time setup: .venv\\Scripts\\python.exe -m playwright install chromium
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

from make_screenshots import choose, go, settle  # same folder: shared Playwright helpers

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets"
VIEWPORT = {"width": 1280, "height": 800}
GIF_WIDTH = 960
CAPTION_HEIGHT = 44
PURPLE = (108, 92, 231)
FADE_STEPS, FADE_MS = 3, 70
MAX_GIF_BYTES = 8 * 1024 * 1024
RUN_TIMEOUT_MS = 15 * 60 * 1000
FONT_CANDIDATES = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf",
                   Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]


# --------------------------------------------------------------------------- servers


def wait_http(url: str, timeout_s: float = 90) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{url} did not answer within {timeout_s:.0f} s")


Servers = list[tuple[subprocess.Popen[bytes], Path]]


def start_servers(workspace: Path, api_port: int, ui_port: int, log_dir: Path, servers: Servers) -> None:
    """Start the API and the UI (appended to `servers` as they start, so the caller can always stop them)."""
    env = os.environ | {
        "WORKSPACE_DIR": str(workspace), "LLM_MODEL": " ", "LLM_API_KEY": " ", "TAVILY_API_KEY": " ",
        "SERPAPI_KEY": " ", "IDSS_PUBLIC_DEMO": "0", "IDSS_API_URL": f"http://localhost:{api_port}",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    commands = {
        "api": [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
        "ui": [sys.executable, "-m", "streamlit", "run", str(Path("ui") / "app.py"), "--server.port", str(ui_port),
               "--server.address", "localhost", "--server.headless", "true"],
    }
    for name, command in commands.items():
        log_path = log_dir / f"{name}.log"
        with log_path.open("wb") as log:  # the child keeps its own handle
            servers.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT),
                            log_path))
    wait_http(f"http://127.0.0.1:{api_port}/health")
    wait_http(f"http://127.0.0.1:{ui_port}/_stcore/health")


def stop_servers(servers: Servers, *, show_logs: bool) -> None:
    for process, _ in servers:
        process.terminate()
    for process, log_path in servers:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        if show_logs and log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print(f"--- last lines of {log_path.name} ---", *tail, sep="\n")


# --------------------------------------------------------------------------- recording


def to_top(page: Page, locator_text: str, *, exact: bool = True) -> None:
    """Scroll Streamlit's main area so the element with this text sits near the top of the viewport."""
    page.get_by_text(locator_text, exact=exact).first.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(600)


def record(page: Page, base_url: str) -> list[tuple[Image.Image, str, int]]:
    """Drive the demo and return (screenshot, caption, hold ms) key frames."""
    frames: list[tuple[Image.Image, str, int]] = []

    def grab(caption: str, hold_ms: int) -> None:
        path = Path(tempfile.gettempdir()) / f"idss_demo_frame_{len(frames)}.png"
        page.screenshot(path=str(path))
        frames.append((Image.open(path).convert("RGB"), caption, hold_ms))
        path.unlink(missing_ok=True)
        print(f"  frame {len(frames)}: {caption}")

    page.goto(base_url)
    settle(page)
    grab("Home – one click on Demo Mode uploads the churn sample and asks the question", 2000)

    page.get_by_role("button", name="Demo Mode").click()
    page.locator(".idss-stage.current").filter(has_text="Train").wait_for(timeout=RUN_TIMEOUT_MS)
    page.wait_for_timeout(2500)
    grab("Run Monitor – question understood, data profiled; the AutoML engine tunes models", 1700)

    page.get_by_text("The run finished").wait_for(timeout=RUN_TIMEOUT_MS)
    settle(page)
    grab("Run Monitor – trained, verified, explained and reported", 1500)

    go(page, "Results")
    grab("Results – held-out test metrics, tuned threshold and verification", 1700)
    page.get_by_role("tab", name="Key drivers").click()
    page.wait_for_timeout(800)
    settle(page)
    page.get_by_role("tab", name="Key drivers").evaluate("e => e.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(800)
    grab("Results – key drivers from SHAP, per original column", 1600)

    go(page, "Decision Center")
    grab("Decision Center – plain-language brief, risk tiers and recommended actions", 1500)
    page.get_by_role("tab", name="What-if simulator").click()
    page.wait_for_timeout(800)
    choose(page, "contract_type", "Two year")
    to_top(page, "Levers", exact=False)
    grab("What-if – change a lever: contract_type → Two year", 1300)
    page.get_by_role("button", name="Simulate").click()
    page.get_by_text("Result", exact=True).wait_for(timeout=60_000)
    settle(page)
    to_top(page, "Result")
    grab("What-if – churn probability before vs after the change", 2600)
    return frames


# --------------------------------------------------------------------------- GIF


def font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def captioned(image: Image.Image, caption: str, step: int, total: int) -> Image.Image:
    height = round(image.height * GIF_WIDTH / image.width)
    canvas = Image.new("RGB", (GIF_WIDTH, height + CAPTION_HEIGHT), PURPLE)
    canvas.paste(image.resize((GIF_WIDTH, height), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, height + CAPTION_HEIGHT // 2), f"{step}/{total}   {caption}", fill="white", font=font(17),
              anchor="lm")
    return canvas


def build_gif(frames: list[tuple[Image.Image, str, int]], path: Path) -> None:
    keys = [(captioned(image, caption, i, len(frames)), hold) for i, (image, caption, hold) in enumerate(frames, 1)]
    images: list[Image.Image] = []
    durations: list[int] = []
    for index, (image, hold) in enumerate(keys):
        images.append(image)
        durations.append(hold)
        following = keys[(index + 1) % len(keys)][0]
        for step in range(1, FADE_STEPS + 1):
            images.append(Image.blend(image, following, step / (FADE_STEPS + 1)))
            durations.append(FADE_MS)
    palette = [im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
    palette[0].save(path, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True,
                    disposal=1)
    size = path.stat().st_size
    print(f"Saved {path.relative_to(ROOT)}: {len(images)} frames, {sum(durations) / 1000:.1f} s, "
          f"{size / 2**20:.2f} MB")
    if size > MAX_GIF_BYTES:
        raise SystemExit(f"The GIF is larger than {MAX_GIF_BYTES / 2**20:.0f} MB.")


# --------------------------------------------------------------------------- run facts


def public_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
    """The requirement without local paths: a dataset path/name becomes the uploaded file's own name."""
    clean = json.loads(json.dumps(requirement))
    for dataset in clean.get("datasets", []):
        for key in ("path", "name"):
            if dataset.get(key):
                dataset[key] = re.sub(r"^[0-9a-f]{32}_", "", Path(dataset[key]).name)
    return clean


def run_facts(api: str, workspace: Path) -> dict[str, Any]:
    """The demo run's numbers for the README walkthrough (read back through the API + the run's state.json)."""
    run = requests.get(f"{api}/runs", params={"limit": 1}, timeout=30).json()[0]
    run_id = run["run_id"]
    detail = requests.get(f"{api}/runs/{run_id}", timeout=30).json()
    board = requests.get(f"{api}/runs/{run_id}/leaderboard", timeout=30).json()
    explanation = requests.get(f"{api}/runs/{run_id}/explanations", timeout=30).json()
    recs = requests.get(f"{api}/runs/{run_id}/recommendations", timeout=30).json()
    insight = requests.get(f"{api}/runs/{run_id}/insight", timeout=30).json()
    state = json.loads((workspace / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    return {
        "note": "Offline demo run recorded by scripts/make_demo_gif.py (Demo Mode, fast budget, no LLM).",
        "question": detail["question"],
        "requirement": public_requirement(state["requirement"]),
        "task_type": detail["task_type"],
        "target": detail["target"],
        "best_model": detail["best_model"],
        "primary_metric": detail["primary_metric"],
        "rows": (state.get("profile") or {}).get("n_rows"),
        "test_metrics": detail["test_metrics"],
        "test_metrics_default_threshold": (state.get("training") or {}).get("test_metrics_default_threshold"),
        "decision_threshold": detail["decision_threshold"],
        "verification": {"passed": detail["verification"]["passed"],
                         "checks": [{"name": c["name"], "passed": c["passed"]} for c in detail["verification"]["checks"]]},
        "leaderboard": [{k: e.get(k) for k in ("rank", "model", "status", "valid_score", "train_time_s",
                                                "inference_ms_per_row", "model_size_kb", "n_trials", "selected")}
                        for e in board["entries"]],
        "drivers": explanation["global_explanation"],
        "risk_tiers": recs["risk_tiers"],
        "tier_cutoffs": recs["tier_cutoffs"],
        "recommendations": recs["recommendations"],
        "counterfactual_example": recs["counterfactual_example"],
        "what_if_example": recs["what_if_example"],
        "insight": insight,
        "report": {"pdf": bool(detail.get("report_pdf_url")), "markdown": bool(detail.get("report_md_url"))},
        "events": [e["message"] for e in detail["events"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--ui-port", type=int, default=8510)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # captions contain "–" and "→"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="idss_demo_gif_"))
    print(f"Starting the API (:{args.api_port}) and UI (:{args.ui_port}) offline on {workspace}")
    servers: Servers = []
    failed = True
    try:
        start_servers(workspace, args.api_port, args.ui_port, workspace, servers)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            frames = record(page, f"http://localhost:{args.ui_port}")
            browser.close()
        build_gif(frames, OUT_DIR / "demo.gif")
        facts = run_facts(f"http://127.0.0.1:{args.api_port}", workspace)
        (OUT_DIR / "demo_run.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {(OUT_DIR / 'demo_run.json').relative_to(ROOT)} (run of {facts['best_model']}, "
              f"{facts['primary_metric']} = {facts['test_metrics'].get(facts['primary_metric'])})")
        failed = False
    finally:
        stop_servers(servers, show_logs=failed)
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
