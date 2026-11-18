"""Data module tests: loader, profiler, preprocessing, splitting and the DataAgent (LLM mocked)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from idss.agents.data_agent import DataAgent
from idss.core.llm_client import LLMError
from idss.core.schemas import DatasetSpec, ProblemSpec, UserRequirement
from idss.ml.data_loader import DataLoadError, load_dataset
from idss.ml.preprocess import (
    MAX_ONEHOT_CATEGORIES,
    PREPROCESSOR_FILENAME,
    build_preprocessor,
    clean_dataset,
    load_preprocessor,
    save_preprocessor,
    split_data,
)
from idss.ml.profiler import profile_dataset
from idss.storage.runs import run_dir

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """120 rows covering every semantic type plus constant and mostly-missing columns."""
    rng = np.random.default_rng(0)
    n = 120
    mostly_missing = rng.normal(size=n)
    mostly_missing[:72] = np.nan  # 60% missing
    return pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "customer_code": [f"C{i:04d}" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50_000, 12_000, n).round(2),
        "plan": rng.choice(["basic", "plus", "pro"], n),
        "is_active": rng.choice(["yes", "no"], n),
        "joined": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
        "review": [f"Customer {i} wrote a fairly long review about service quality, pricing and support." for i in range(n)],
        "constant": "same",
        "mostly_missing": mostly_missing,
        "label": rng.choice(["good", "bad"], n),
    })


def make_classification_df(n: int = 1000, positive_share: float = 0.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: int(n * positive_share)] = 1
    rng.shuffle(y)
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.choice(["a", "b", "c"], n), "target": y})


# --------------------------------------------------------------------------- loader


def test_load_csv_detects_semicolon_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a;b;c\n1;2;x\n3;4;y\n", encoding="utf-8")
    df = load_dataset(path)
    assert list(df.columns) == ["a", "b", "c"] and df.shape == (2, 3)


def test_load_csv_falls_back_to_latin1(tmp_path: Path) -> None:
    path = tmp_path / "latin.csv"
    path.write_bytes("name,city\nJosé,Málaga\nZoë,Köln\n".encode("latin-1"))
    df = load_dataset(path)
    assert df["name"].tolist() == ["José", "Zoë"]


def test_load_csv_with_utf8_bom_and_tabs(tmp_path: Path) -> None:
    path = tmp_path / "tabs.csv"
    path.write_bytes("﻿col one\tcol two\n1\t2\n".encode("utf-8"))
    assert list(load_dataset(path).columns) == ["col one", "col two"]


def test_load_xlsx_json_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    df.to_excel(tmp_path / "d.xlsx", index=False)
    df.to_json(tmp_path / "d.json", orient="records")
    df.to_parquet(tmp_path / "d.parquet")
    for name in ("d.xlsx", "d.json", "d.parquet"):
        loaded = load_dataset(tmp_path / name)
        assert loaded["x"].tolist() == [1, 2, 3] and loaded["y"].tolist() == ["a", "b", "c"], name


def test_load_json_lines_with_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text('{"a": 1, "tags": ["x", "y"]}\n{"a": 2, "tags": []}\n', encoding="utf-8")
    df = load_dataset(path)
    assert df["tags"].tolist() == ['["x", "y"]', "[]"]  # nested values kept as JSON strings


@pytest.mark.parametrize(
    "filename, content, message",
    [
        ("missing.csv", None, "not found"),
        ("notes.txt", b"hello", "Unsupported file type"),
        ("empty.csv", b"", "empty"),
        ("header_only.csv", b"a,b,c\n", "no data rows"),
        ("corrupt.xlsx", b"this is not a zip archive", "Could not read"),
        ("binary.csv", b"\x00\x01\x02\x03garbage", "binary"),
        ("corrupt.parquet", b"PAR1 broken", "Could not read"),
    ],
)
def test_load_errors(tmp_path: Path, filename: str, content: bytes | None, message: str) -> None:
    path = tmp_path / filename
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(DataLoadError, match=message):
        load_dataset(path)


def test_load_max_rows_samples_and_keeps_order(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    pd.DataFrame({"i": range(100), "v": range(100)}).to_csv(path, index=False)
    df = load_dataset(path, max_rows=10)
    assert len(df) == 10 and df["i"].is_monotonic_increasing


# --------------------------------------------------------------------------- profiler


def test_semantic_types(mixed_df: pd.DataFrame) -> None:
    profile = profile_dataset(mixed_df, target="label")
    types = {col.name: col.semantic_type for col in profile.columns}
    assert types == {
        "record_id": "id",
        "customer_code": "id",
        "age": "numeric",
        "income": "numeric",
        "plan": "categorical",
        "is_active": "boolean",
        "joined": "datetime",
        "review": "text",
        "constant": "categorical",
        "mostly_missing": "numeric",
        "label": "categorical",
    }
    assert profile.column("mostly_missing").missing_pct == 60.0
    assert profile.n_rows == 120 and profile.n_cols == 11


def test_datetime_strings_are_detected_but_numbers_and_month_names_are_not() -> None:
    df = pd.DataFrame({
        "when": ["2024-01-05", "2024-02-11", "05/03/2024", "2024-04-20"] * 10,
        "year": ["2019", "2020", "2021", "2022"] * 10,
        "month": ["Jan", "Feb", "Mar", "Apr"] * 10,
        "y": range(40),
    })
    types = {col.name: col.semantic_type for col in profile_dataset(df, target="y").columns}
    assert types["when"] == "datetime" and types["year"] == "categorical" and types["month"] == "categorical"


def test_target_suggested_by_keyword(mixed_df: pd.DataFrame) -> None:
    assert profile_dataset(mixed_df).suggested_target == "label"


def test_target_falls_back_to_last_eligible_column() -> None:
    df = pd.DataFrame({"a": np.arange(50) % 7, "b": np.linspace(0, 1, 50), "note_id": range(50)})
    profile = profile_dataset(df)
    assert profile.suggested_target == "b"  # note_id is an identifier, so it is skipped


def test_explicit_target_is_case_insensitive_and_validated(mixed_df: pd.DataFrame) -> None:
    assert profile_dataset(mixed_df, target="  LABEL ").suggested_target == "label"
    with pytest.raises(ValueError, match="not found"):
        profile_dataset(mixed_df, target="does_not_exist")


@pytest.mark.parametrize(
    "target_values, expected",
    [
        (["a", "b", "c"] * 20, "tabular_classification"),
        (list(range(5)) * 12, "tabular_classification"),  # <= 20 unique integers
        (list(range(0, 120, 2)), "tabular_regression"),  # many unique integers (step 2, so not an ID sequence)
        (list(np.linspace(0.5, 99.5, 60)), "tabular_regression"),
    ],
)
def test_task_type_suggestion(target_values: list, expected: str) -> None:
    df = pd.DataFrame({"feature": np.arange(60) % 9, "outcome": target_values})
    assert profile_dataset(df, target="outcome").suggested_task_type == expected


def test_task_type_clustering_and_time_series() -> None:
    df = pd.DataFrame({"f1": np.arange(60) % 9, "f2": np.linspace(0, 1, 60)})
    assert profile_dataset(df, suggest_target=False).suggested_task_type == "tabular_clustering"

    ts = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D").strftime("%Y-%m-%d"),
        "sales": np.linspace(100, 200, 60),
    })
    assert profile_dataset(ts).suggested_task_type == "time_series_forecasting"


def test_profile_warnings() -> None:
    rng = np.random.default_rng(1)
    n = 200
    target = np.zeros(n, dtype=int)
    target[:10] = 1  # 5% minority class
    df = pd.DataFrame({
        "mostly_missing": np.where(rng.random(n) < 0.5, np.nan, 1.0) + rng.normal(size=n),
        "constant": 7,
        "city": [f"city_{i % 60}" for i in range(n)],  # 60 categories
        "leaky": target * 10.0 + rng.normal(0, 0.01, n),
        "x": rng.normal(size=n),
        "target": target,
    })
    text = " | ".join(profile_dataset(df, target="target").warnings)
    assert "'mostly_missing' is" in text and "% missing" in text
    assert "'constant' is constant" in text
    assert "imbalanced" in text and "5.0%" in text
    assert "'city' has 60 categories" in text
    assert "'leaky' is almost perfectly correlated" in text
    assert "'x'" not in text


@pytest.mark.parametrize(
    "filename, target, task_type",
    [
        ("customer_churn.csv", "churn", "tabular_classification"),
        ("crop_price.csv", "price", "tabular_regression"),
        ("store_sales.csv", "sales", "time_series_forecasting"),
    ],
)
def test_sample_datasets_profile_as_expected(filename: str, target: str, task_type: str) -> None:
    profile = profile_dataset(load_dataset(SAMPLES_DIR / filename))
    assert (profile.suggested_target, profile.suggested_task_type) == (target, task_type)


# --------------------------------------------------------------------------- preprocessing


def test_preprocessor_shapes_and_dropped_columns(mixed_df: pd.DataFrame) -> None:
    profile = profile_dataset(mixed_df, target="label")
    features = mixed_df.drop(columns=["label"])
    preprocessor = build_preprocessor(profile, "label").fit(features)
    out = preprocessor.transform(features)
    names = list(preprocessor.get_feature_names_out())

    # 3 numeric + plan (3) + is_active (2) one-hot + 4 date parts = 12
    assert out.shape == (120, 12) and len(names) == 12
    assert not np.isnan(out).any()
    assert {"date__joined_year", "date__joined_dayofweek", "num__age"} <= set(names)
    for dropped in ("record_id", "customer_code", "review", "constant"):
        assert not any(dropped in name for name in names)


def test_preprocessor_ignores_unknown_categories(mixed_df: pd.DataFrame) -> None:
    profile = profile_dataset(mixed_df, target="label")
    features = mixed_df.drop(columns=["label"])
    preprocessor = build_preprocessor(profile, "label").fit(features)

    unseen = features.head(1).copy()
    unseen["plan"] = "enterprise"
    out = preprocessor.transform(unseen)
    plan_columns = [i for i, name in enumerate(preprocessor.get_feature_names_out()) if name.startswith("cat__plan_")]
    assert len(plan_columns) == 3 and np.all(out[0, plan_columns] == 0)


def test_preprocessor_caps_onehot_categories() -> None:
    df = pd.DataFrame({"city": [f"c{i % 30}" for i in range(300)], "y": np.arange(300) % 2})
    profile = profile_dataset(df, target="y")
    preprocessor = build_preprocessor(profile, "y").fit(df[["city"]])
    assert preprocessor.transform(df[["city"]]).shape[1] <= MAX_ONEHOT_CATEGORIES


def test_save_and_load_preprocessor(tmp_path: Path, mixed_df: pd.DataFrame) -> None:
    profile = profile_dataset(mixed_df, target="label")
    features = mixed_df.drop(columns=["label"])
    preprocessor = build_preprocessor(profile, "label").fit(features)

    path = save_preprocessor(preprocessor, "test-run", runs_dir=tmp_path)
    assert path == tmp_path / "test-run" / PREPROCESSOR_FILENAME and path.is_file()
    np.testing.assert_allclose(load_preprocessor(path).transform(features), preprocessor.transform(features))


@pytest.mark.parametrize("bad_run_id", ["../escape", "a/b", "", "run id"])
def test_run_dir_rejects_unsafe_ids(tmp_path: Path, bad_run_id: str) -> None:
    with pytest.raises(ValueError):
        run_dir(bad_run_id, tmp_path)


# --------------------------------------------------------------------------- cleaning & splitting


def test_split_ratios_cover_all_rows_once() -> None:
    df = make_classification_df(1000)
    train, valid, test = split_data(df, "target", "tabular_regression")
    assert (len(train), len(valid), len(test)) == (700, 200, 100)
    combined = pd.concat([train, valid, test]).index
    assert combined.is_unique and set(combined) == set(df.index)


def test_split_is_stratified_for_classification() -> None:
    df = make_classification_df(1000, positive_share=0.2)
    for part in split_data(df, "target", "tabular_classification"):
        assert part["target"].mean() == pytest.approx(0.2, abs=0.01)


def test_split_falls_back_when_a_class_is_too_rare() -> None:
    df = make_classification_df(100)
    df.loc[0, "target"] = 2  # a class with a single member cannot be stratified
    train, valid, test = split_data(df, "target", "tabular_classification")
    assert len(train) + len(valid) + len(test) == 100


def test_split_time_series_is_chronological() -> None:
    df = pd.DataFrame({"t": range(100), "y": np.arange(100.0)})
    train, valid, test = split_data(df, "y", "time_series_forecasting")
    assert (len(train), len(valid), len(test)) == (70, 20, 10)
    assert train["t"].max() < valid["t"].min() and valid["t"].max() < test["t"].min()


def test_split_rejects_missing_target_and_tiny_data() -> None:
    df = make_classification_df(50).astype({"target": float})
    df.loc[3, "target"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        split_data(df, "target", "tabular_classification")
    with pytest.raises(ValueError, match="at least"):
        split_data(df.head(5), "target", "tabular_classification")


def test_clean_dataset_removes_duplicates_and_missing_targets() -> None:
    df = pd.DataFrame({"x": [1, 1, 2, 3, 4], "y": [0.0, 0.0, np.nan, 1.0, 0.0]})
    profile = profile_dataset(df, target="y")
    cleaned, steps = clean_dataset(df, "y", "tabular_classification", profile)
    assert cleaned["x"].tolist() == [1, 3, 4]
    assert any("1 duplicate" in step for step in steps) and any("missing target" in step for step in steps)


# --------------------------------------------------------------------------- DataAgent


def test_data_agent_end_to_end_with_llm_summary(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.chat.return_value = "  This customer data shows who is likely to leave.  "
    result = DataAgent(llm, runs_dir=tmp_path).run(SAMPLES_DIR / "customer_churn.csv", run_id="churn-run")

    assert result.summary_source == "llm" and result.llm_summary == "This customer data shows who is likely to leave."
    assert (result.target, result.task_type) == ("churn", "tabular_classification")
    sizes = result.split_sizes
    assert sizes.train + sizes.valid + sizes.test == result.profile.n_rows == 3000
    assert set(result.artifacts) == {"preprocessor", "train", "valid", "test"}
    assert all(Path(p).is_file() for p in result.artifacts.values())
    assert Path(result.artifacts["preprocessor"]).parent == tmp_path / "churn-run"
    assert any("customer_id" in step and "identifier" in step for step in result.cleaning_steps)

    prompt = llm.chat.call_args.args[0][1]["content"]
    assert "churn" in prompt and "CUST-00001" not in prompt  # aggregates only, no raw records


@pytest.mark.parametrize("failure", [LLMError("service down", retryable=True), RuntimeError("unexpected")])
def test_data_agent_falls_back_to_template_when_llm_fails(tmp_path: Path, failure: Exception) -> None:
    llm = MagicMock()
    llm.chat.side_effect = failure
    result = DataAgent(llm, runs_dir=tmp_path).run(SAMPLES_DIR / "store_sales.csv")

    assert result.summary_source == "template"
    assert "730 rows" in result.llm_summary and "'sales'" in result.llm_summary
    assert result.task_type == "time_series_forecasting"
    assert (result.split_sizes.train, result.split_sizes.valid, result.split_sizes.test) == (511, 146, 73)


def test_data_agent_follows_user_requirement(tmp_path: Path) -> None:
    requirement = UserRequirement(
        problem=ProblemSpec(task_type="tabular_regression", description="Predict how many months customers stay."),
        datasets=[DatasetSpec(target_column="TENURE_MONTHS")],
        confidence=0.9,
    )
    result = DataAgent(None, runs_dir=tmp_path).run(SAMPLES_DIR / "customer_churn.csv", requirement)
    assert (result.target, result.task_type, result.summary_source) == ("tenure_months", "tabular_regression", "template")

    clustering = UserRequirement(
        problem=ProblemSpec(task_type="tabular_clustering", description="Segment crops."), confidence=0.8
    )
    result = DataAgent(None, runs_dir=tmp_path).run(SAMPLES_DIR / "crop_price.csv", clustering)
    assert result.target is None and result.task_type == "tabular_clustering"
