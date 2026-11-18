"""Count the project's source lines of code and apply the basic COCOMO model (organic mode) for the report.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\count_loc.py
A line counts as code when it holds at least one Python token that is not a comment or a docstring; blank lines,
comment-only lines and docstrings are reported separately. `reference/` and `.venv/` are never read.
"""

import io
import sys
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION = ("idss", "api", "ui", "configs")  # the running system
DELIVERED = (*APPLICATION, "scripts")  # plus evaluation / tooling scripts; tests are reported separately
PACKAGES = (*DELIVERED, "tests")
SKIP_TOKENS = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
               tokenize.ENCODING, tokenize.COMMENT}

# Basic COCOMO, organic mode (Boehm, 1981): E = a * KLOC^b person-months, D = c * E^d months.
COCOMO_A, COCOMO_B, COCOMO_C, COCOMO_D = 2.4, 1.05, 2.5, 0.38


def _is_docstring(tokens: list[tokenize.TokenInfo], index: int) -> bool:
    """A string token that forms a statement on its own (module, class or function docstring)."""
    before = index - 1
    while before >= 0 and tokens[before].type in (tokenize.NL, tokenize.COMMENT):
        before -= 1
    after = index + 1
    while after < len(tokens) and tokens[after].type in (tokenize.NL, tokenize.COMMENT):
        after += 1
    starts_statement = before < 0 or tokens[before].type in (
        tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING)
    ends_statement = after >= len(tokens) or tokens[after].type in (tokenize.NEWLINE, tokenize.ENDMARKER)
    return starts_statement and ends_statement


def count_file(path: Path) -> dict[str, int]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    code: set[int] = set()
    docs: set[int] = set()
    comments: set[int] = set()
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for index, token in enumerate(tokens):
        rows = range(token.start[0], token.end[0] + 1)
        if token.type == tokenize.COMMENT:
            comments.add(token.start[0])
        elif token.type == tokenize.STRING and _is_docstring(tokens, index):
            docs.update(rows)
        elif token.type not in SKIP_TOKENS:
            code.update(rows)
    code -= docs
    return {"physical": len(lines), "blank": sum(1 for line in lines if not line.strip()), "code": len(code),
            "docstring": len(docs - code), "comment": len(comments - code - docs)}


def count_package(name: str) -> dict[str, int]:
    totals = {"files": 0, "physical": 0, "blank": 0, "code": 0, "docstring": 0, "comment": 0}
    for path in sorted((PROJECT_ROOT / name).rglob("*.py")):
        totals["files"] += 1
        for key, value in count_file(path).items():
            totals[key] += value
    return totals


def cocomo_organic(kloc: float) -> tuple[float, float, float]:
    effort = COCOMO_A * kloc ** COCOMO_B
    duration = COCOMO_C * effort ** COCOMO_D
    return effort, duration, effort / duration


def main() -> int:
    results = {name: count_package(name) for name in PACKAGES}
    header = f"{'package':<10}{'files':>7}{'physical':>10}{'blank':>8}{'comment':>9}{'docstring':>11}{'code':>8}"
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        print(f"{name:<10}{row['files']:>7}{row['physical']:>10}{row['blank']:>8}{row['comment']:>9}"
              f"{row['docstring']:>11}{row['code']:>8}")
    delivered = {key: sum(results[name][key] for name in DELIVERED) for key in results["idss"]}
    print("-" * len(header))
    print(f"{'delivered':<10}{delivered['files']:>7}{delivered['physical']:>10}{delivered['blank']:>8}"
          f"{delivered['comment']:>9}{delivered['docstring']:>11}{delivered['code']:>8}")
    application = sum(results[name]["code"] for name in APPLICATION)
    for label, code_lines in (("delivered code: application + scripts, tests excluded", delivered["code"]),
                              ("application only: " + ", ".join(APPLICATION), application)):
        kloc = code_lines / 1000
        effort, duration, staff = cocomo_organic(kloc)
        print(f"\nBasic COCOMO, organic mode, KLOC = {kloc:.3f} ({label})")
        print(f"  Effort    E = {COCOMO_A} x KLOC^{COCOMO_B} = {effort:.2f} person-months")
        print(f"  Duration  D = {COCOMO_C} x E^{COCOMO_D} = {duration:.2f} months")
        print(f"  Staff     P = E / D = {staff:.2f} persons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
