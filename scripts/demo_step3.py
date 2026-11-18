"""Step 3 demo: run the Orchestrator on the churn sample with the real LLM from .env.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\demo_step3.py
    .venv\\Scripts\\python.exe scripts\\demo_step3.py --question "Forecast next month's sales" --data data\\samples\\store_sales.csv
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings  # noqa: E402  (needs PROJECT_ROOT on sys.path)
from idss.agents.manager import MAX_REVISIONS, Orchestrator, RunContext, RunState  # noqa: E402
from idss.core.llm_client import LLMClient, LLMError  # noqa: E402
from idss.core.logger import setup_logging  # noqa: E402

DEFAULT_QUESTION = "Which customers are likely to churn and what should we do to retain them?"
DEFAULT_DATA = PROJECT_ROOT / "data" / "samples" / "customer_churn.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IDSS Step 3 demo: question + dataset -> validated requirement.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--verbose", action="store_true", help="also show pipeline log messages")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")  # LLM text may contain characters the console cannot encode
    setup_logging(level="INFO" if args.verbose else "WARNING", force=True)

    settings = get_settings()
    try:
        client = LLMClient(settings)
    except LLMError as exc:
        print(f"Cannot run the demo: {exc}")
        print("Create your .env first:  Copy-Item .env.example .env")
        print("Then set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY (no key needed for ollama) and run this again.")
        return 1

    print(f"LLM: {settings.llm_provider} / {settings.llm_model}")
    print(f"Question: {args.question}")
    print(f"Data: {args.data}\n")

    orchestrator = Orchestrator(client, budget_mode="fast")  # keep the demo to a few minutes at most
    run_id = orchestrator.start(args.question, args.data)
    ctx = orchestrator.run_until_blocked()
    for _ in range(MAX_REVISIONS):
        if ctx.state != RunState.NEEDS_INPUT:
            break
        answer = _ask(ctx)
        if answer is None:
            print(f"\nRun {run_id} is waiting for input; answer later with Orchestrator.provide_answer().")
            break
        orchestrator.provide_answer(run_id, answer)
        ctx = orchestrator.run_until_blocked()

    _report(ctx, client)
    return 0 if ctx.state == RunState.DONE else 2


def _ask(ctx: RunContext) -> str | None:
    print("The assistant needs more information:")
    for question in ctx.pending_questions:
        print(f"  - {question}")
    if not sys.stdin.isatty():
        return None
    try:
        answer = input("Your answer: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return answer or None


def _report(ctx: RunContext, client: LLMClient) -> None:
    print("\n== State transitions ==")
    for event in ctx.history:
        transition = f"{event.from_state} -> {event.state}" if event.from_state else f"{event.state}"
        print(f"{event.timestamp.astimezone():%H:%M:%S}  {transition:<24} {event.message}")
    if ctx.requirement is not None:
        print("\n== Parsed requirement ==")
        print(ctx.requirement.model_dump_json(indent=2))
    if ctx.summary:
        print("\n== Summary ==")
        print(ctx.summary)
    if ctx.data is not None:
        print("\n== Data summary ==")
        print(ctx.data.llm_summary)
    if ctx.message:
        print(f"\n== Result ({ctx.state}) ==")
        print(ctx.message)
    for error in ctx.errors:
        print(f"ERROR: {error}")
    tokens = sum(record.total_tokens or 0 for record in client.usage_log)
    latency = sum(record.latency_s for record in client.usage_log)
    print(f"\nRun {ctx.run_id}: {len(client.usage_log)} LLM calls, {tokens:,} tokens, {latency:.1f}s total LLM latency.")


if __name__ == "__main__":
    sys.exit(main())
