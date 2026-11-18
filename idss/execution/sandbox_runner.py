"""Child-process side of the sandbox. Executed as a script (python -I sandbox_runner.py config.json).

Not imported by the application. Loads the data into `df`, runs the user code with restricted builtins
and an import allowlist, captures stdout (capped), and writes {ok, stdout, result, error, truncated} as
JSON to the output path from the config. Depends only on the standard library, pandas and numpy.
"""

import builtins
import datetime
import io
import json
import re
import sys
import traceback

import numpy as np
import pandas as pd


class CappedWriter(io.TextIOBase):
    """Keeps at most `limit` characters of everything written to it."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False

    def write(self, text: str) -> int:
        room = self.limit - self.size
        if room > 0:
            kept = text[:room]
            self.parts.append(kept)
            self.size += len(kept)
        if len(text) > max(room, 0):
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        return "".join(self.parts)


def to_jsonable(value: object, max_rows: int) -> object:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return json.loads(value.head(max_rows).to_json(orient="records", date_format="iso", default_handler=str))
    if isinstance(value, pd.Series):
        return json.loads(value.head(max_rows).to_json(date_format="iso", default_handler=str))
    if isinstance(value, np.ndarray):
        value = value.tolist()[:max_rows]

    def default(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (datetime.date, datetime.datetime, pd.Timestamp)):
            return item.isoformat()
        return str(item)

    return json.loads(json.dumps(value, default=default))


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as handle:
        config = json.load(handle)

    allowed = set(config["allowed_modules"])
    blocked = set(config["blocked_members"])
    io_member = re.compile(config["io_member_pattern"])
    real_import = builtins.__import__

    def is_blocked(member: str) -> bool:
        return member.startswith("_") or member in blocked or bool(io_member.match(member))

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002 (builtin signature)
        root, *rest = name.split(".")
        if level != 0 or root not in allowed or any(is_blocked(part) for part in [*rest, *(fromlist or ())]):
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {name: getattr(builtins, name) for name in config["safe_builtins"] if hasattr(builtins, name)}
    safe_builtins["__import__"] = guarded_import
    namespace = {"__builtins__": safe_builtins, "__name__": "__sandbox__",
                 "df": pd.read_parquet(config["data_path"]), "pd": pd, "np": np}

    response = {"ok": True, "error": None, "result": None}
    captured = CappedWriter(config["max_stdout_chars"])
    real_stdout = sys.stdout
    sys.stdout = captured
    try:
        exec(compile(config["code"], "<analysis>", "exec"), namespace)  # noqa: S102 (guarded sandbox execution)
        response["result"] = to_jsonable(namespace.get("result"), config["max_result_rows"])
        if len(json.dumps(response["result"])) > config["max_result_chars"]:
            response["result"] = str(response["result"])[: config["max_result_chars"]] + " ... (truncated)"
    except BaseException as exc:  # report every failure, including SystemExit, to the parent
        line = next((f.lineno for f in reversed(traceback.extract_tb(exc.__traceback__)) if f.filename == "<analysis>"), None)
        response["ok"] = False
        response["error"] = f"{type(exc).__name__}: {exc}" + (f" (line {line})" if line else "")
    finally:
        sys.stdout = real_stdout

    response["stdout"] = captured.getvalue()
    response["truncated"] = captured.truncated
    with open(config["output_path"], "w", encoding="utf-8") as handle:
        json.dump(response, handle, default=str)


if __name__ == "__main__":
    main()
