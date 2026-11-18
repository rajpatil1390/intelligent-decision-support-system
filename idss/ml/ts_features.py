"""Leakage-free lag, rolling and calendar features for time-series forecasting.

Every target-derived feature for row t uses only rows strictly before t (shift >= 1), so evaluation is
honest one-step-ahead forecasting. The transformer keeps the tail of the series it was fitted on and
prepends it when transforming a later window (valid/test), so the first rows of that window get real
lags without the window itself ever being seen during fitting.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from idss.ml.profiler import parse_datetimes

FEATURE_PREFIX = "ts_"
CALENDAR_PARTS = ("dayofyear", "weekofyear", "quarter", "is_weekend")


def infer_seasonal_period(times: pd.Series) -> int:
    """Season length implied by the sampling interval: hourly→24, daily→7, weekly→52, monthly→12, quarterly→4."""
    parsed = parse_datetimes(times).dropna().sort_values()
    if len(parsed) < 3:
        return 1
    step = parsed.diff().dropna().median()
    for limit, period in ((pd.Timedelta(hours=1), 24), (pd.Timedelta(days=1), 7), (pd.Timedelta(days=7), 52),
                          (pd.Timedelta(days=31), 12), (pd.Timedelta(days=92), 4)):
        if step <= limit:
            return period
    return 1


def choose_lags(period: int, n_rows: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Lags (recent values and one/two seasons back) and rolling windows that fit the history length."""
    limit = max(1, n_rows // 3)
    lags = tuple(lag for lag in sorted({1, 2, 3, period, 2 * period}) if lag <= limit)
    window = period if 1 < period <= limit else min(3, limit)
    return lags or (1,), (max(window, 2),)


class TimeSeriesFeatures(TransformerMixin, BaseEstimator):
    """Adds ts_lag_*, ts_rolling_mean_*, ts_rolling_std_* and calendar columns; drops the raw target column.

    Input frames must contain `time_column` and `target` (the target supplies history; NaN for rows to
    forecast is allowed). Output keeps all other columns, in the input row order.
    """

    def __init__(self, time_column: str, target: str, lags: tuple[int, ...] = (1,), windows: tuple[int, ...] = (3,)) -> None:
        self.time_column = time_column
        self.target = target
        self.lags = lags
        self.windows = windows

    def feature_names(self) -> list[str]:
        names = [f"{FEATURE_PREFIX}lag_{lag}" for lag in self.lags]
        for window in self.windows:
            names += [f"{FEATURE_PREFIX}rolling_mean_{window}", f"{FEATURE_PREFIX}rolling_std_{window}"]
        return names + [f"{FEATURE_PREFIX}{part}" for part in CALENDAR_PARTS]

    def fit(self, X: pd.DataFrame, y: object = None) -> "TimeSeriesFeatures":
        frame = self._sorted(pd.DataFrame(X))
        keep = max([*self.lags, *self.windows, 1])
        self.history_ = frame[[self.time_column, self.target]].tail(keep).reset_index(drop=True)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = np.asarray(
            [str(col) for col in frame.columns if col != self.target] + self.feature_names(), dtype=object
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X)
        ordered = self._sorted(frame)
        times = parse_datetimes(ordered[self.time_column])

        history = self.history_
        if len(history) and times.min() <= parse_datetimes(history[self.time_column]).max():
            history = history.iloc[0:0]  # overlapping window (e.g. re-transforming training data): no prepend
        series = pd.concat([history[self.target], ordered[self.target]], ignore_index=True).astype("float64")

        features: dict[str, pd.Series] = {}
        for lag in self.lags:
            features[f"{FEATURE_PREFIX}lag_{lag}"] = series.shift(lag)
        past = series.shift(1)
        for window in self.windows:
            features[f"{FEATURE_PREFIX}rolling_mean_{window}"] = past.rolling(window, min_periods=1).mean()
            features[f"{FEATURE_PREFIX}rolling_std_{window}"] = past.rolling(window, min_periods=2).std()
        generated = pd.DataFrame(features).iloc[len(history):].set_index(ordered.index)

        generated[f"{FEATURE_PREFIX}dayofyear"] = times.dt.dayofyear.astype("float64")
        generated[f"{FEATURE_PREFIX}weekofyear"] = times.dt.isocalendar().week.astype("float64")
        generated[f"{FEATURE_PREFIX}quarter"] = times.dt.quarter.astype("float64")
        generated[f"{FEATURE_PREFIX}is_weekend"] = (times.dt.dayofweek >= 5).astype("float64")

        out = ordered.drop(columns=[self.target]).join(generated)
        return out.loc[frame.index]  # back to the caller's row order

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return self.feature_names_out_

    def _sorted(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(self.time_column, key=parse_datetimes, kind="stable")
