"""Generate the synthetic sample datasets in data/samples/ (seeded, so output is reproducible).

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\make_samples.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from idss.core.logger import get_logger  # noqa: E402  (needs PROJECT_ROOT on sys.path)

logger = get_logger()

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
SEED = 42


def make_customer_churn(rng: np.random.Generator, n: int = 3000) -> pd.DataFrame:
    """Telecom customers; target `churn` (Yes/No). Includes missing values, categoricals and a date column."""
    snapshot = pd.Timestamp("2025-01-01")
    signup = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, n), unit="D")
    tenure_months = ((snapshot - signup).days.to_numpy() / 30.44).astype(int)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    # "No internet", not "None": pandas reads the string "None" back as a missing value.
    internet = rng.choice(["DSL", "Fiber optic", "No internet"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n, p=[0.35, 0.20, 0.22, 0.23]
    )
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    senior_citizen = (rng.random(n) < 0.16).astype(int)

    base_charge = np.select([internet == "DSL", internet == "Fiber optic"], [50.0, 80.0], default=25.0)
    monthly = np.clip(base_charge + rng.normal(0, 10, n) + np.where(contract == "Month-to-month", 5.0, 0.0), 18, 120)
    total = monthly * np.maximum(tenure_months, 1) * rng.normal(1.0, 0.03, n)
    support_calls = rng.poisson(np.where(internet == "Fiber optic", 1.8, 1.0))

    logit = (
        -0.9
        + 1.4 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        - 0.04 * tenure_months
        + 0.02 * (monthly - 60)
        + 0.30 * support_calls
        + 0.50 * (payment == "Electronic check")
        + 0.30 * senior_citizen
    )
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "region": region,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior_citizen,
        "tenure_months": tenure_months,
        "monthly_charges": monthly.round(2),
        "total_charges": total.round(2),
        "support_calls": support_calls,
        "churn": churn,
    })
    df.loc[rng.random(n) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.04, "internet_service"] = np.nan
    df.loc[rng.random(n) < 0.02, "monthly_charges"] = np.nan
    return df


# crop -> (base price INR/quintal, season, base yield t/ha)
_CROPS = {
    "Wheat": (2200, "Rabi", 3.4),
    "Rice": (2100, "Kharif", 2.8),
    "Maize": (1950, "Kharif", 3.0),
    "Cotton": (6600, "Kharif", 0.5),
    "Soybean": (4400, "Kharif", 1.2),
    "Chickpea": (5300, "Rabi", 1.0),
    "Mustard": (5500, "Rabi", 1.4),
    "Groundnut": (6000, "Zaid", 1.8),
}
_STATES = {
    "Punjab": 1.04, "Maharashtra": 1.00, "Uttar Pradesh": 0.97, "Madhya Pradesh": 0.96,
    "Karnataka": 1.02, "Gujarat": 1.03, "Rajasthan": 0.98, "Andhra Pradesh": 1.01,
}
# season -> (rainfall mean, rainfall sd, temperature mean, humidity mean)
_SEASONS = {"Kharif": (850, 220, 29.0, 78.0), "Rabi": (120, 60, 19.0, 55.0), "Zaid": (220, 90, 33.0, 48.0)}


def make_crop_price(rng: np.random.Generator, n: int = 2000) -> pd.DataFrame:
    """Indian crop market prices; target `price` (INR per quintal)."""
    crop = rng.choice(list(_CROPS), n)
    state = rng.choice(list(_STATES), n)
    year = rng.integers(2015, 2025, n)
    base_price = np.array([_CROPS[c][0] for c in crop], dtype=float)
    season = np.array([_CROPS[c][1] for c in crop])
    base_yield = np.array([_CROPS[c][2] for c in crop])

    rain_mean, rain_sd, temp_mean, humidity_mean = (np.array([_SEASONS[s][i] for s in season]) for i in range(4))
    rainfall = np.clip(rng.normal(rain_mean, rain_sd), 0, None)
    temperature = rng.normal(temp_mean, 2.5)
    humidity = np.clip(rng.normal(humidity_mean, 9), 15, 100)
    soil_ph = np.clip(rng.normal(7.0, 0.5, n), 5.0, 9.0)
    fertilizer = np.clip(rng.normal(120, 30, n), 20, None)

    rain_z = (rainfall - rain_mean) / rain_sd
    yield_t = np.clip(base_yield * (1 + 0.15 * rain_z + 0.0015 * (fertilizer - 120)) * rng.normal(1, 0.08, n), 0.05, None)
    state_factor = np.array([_STATES[s] for s in state])
    price = (
        base_price
        * 1.045 ** (year - 2015)
        * state_factor
        * (1 - 0.25 * (yield_t / base_yield - 1))
        * rng.normal(1, 0.05, n)
    )

    df = pd.DataFrame({
        "year": year,
        "state": state,
        "crop": crop,
        "season": season,
        "rainfall_mm": rainfall.round(1),
        "temperature_c": temperature.round(1),
        "humidity_pct": humidity.round(1),
        "soil_ph": soil_ph.round(2),
        "fertilizer_kg_per_ha": fertilizer.round(1),
        "yield_t_per_ha": yield_t.round(2),
        "price": price.round(2),
    })
    df.loc[rng.random(n) < 0.03, "humidity_pct"] = np.nan
    df.loc[rng.random(n) < 0.02, "soil_ph"] = np.nan
    return df


_HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-01-26", "2023-03-08", "2023-08-15", "2023-10-02", "2023-11-12", "2023-12-25",
    "2024-01-01", "2024-01-26", "2024-03-25", "2024-08-15", "2024-10-02", "2024-11-01", "2024-12-25",
])
_DAY_FACTORS = np.array([0.90, 0.92, 0.95, 1.00, 1.12, 1.30, 1.18])  # Monday .. Sunday


def make_store_sales(rng: np.random.Generator, n_days: int = 730) -> pd.DataFrame:
    """Daily sales of one retail store over two years; target `sales`. Trend + weekly/yearly seasonality."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    is_holiday = dates.isin(_HOLIDAYS)
    is_promo = rng.random(n_days) < 0.15
    temperature = 26 + 6 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25) + rng.normal(0, 1.5, n_days)

    sales = (
        12_000
        * _DAY_FACTORS[dates.dayofweek.to_numpy()]
        * (1 + 0.10 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25))
        * (1 + 0.00035 * np.arange(n_days))
        * np.where(is_promo, 1.22, 1.0)
        * np.where(is_holiday, 1.35, 1.0)
        * rng.normal(1, 0.05, n_days)
    )

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "day_of_week": dates.day_name(),
        "is_promo": is_promo.astype(int),
        "is_holiday": is_holiday.astype(int),
        "temperature_c": temperature.round(1),
        "sales": sales.round(2),
    })
    df.loc[rng.random(n_days) < 0.01, "temperature_c"] = np.nan
    return df


_CARRIER_DELAY = {"Carrier A": 0.0, "Carrier B": 0.45, "Carrier C": -0.25, "Carrier D": 0.8}
_DELIVERY_CLASSES = ("On time", "Late", "Very late")
_DELIVERY_SHARES = (0.55, 0.30, 0.15)


def make_delivery_status(rng: np.random.Generator, n: int = 2500) -> pd.DataFrame:
    """Logistics shipments; multiclass target `delivery_status` (On time / Late / Very late)."""
    order_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, n), unit="D")
    warehouse = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Kolkata", "Chennai"], n)
    carrier = rng.choice(list(_CARRIER_DELAY), n, p=[0.35, 0.25, 0.25, 0.15])
    mode = rng.choice(["Standard", "Express", "Economy"], n, p=[0.55, 0.25, 0.20])
    distance = np.clip(rng.lognormal(6.3, 0.75, n), 20, 3000)
    weight = np.clip(rng.gamma(2.0, 4.0, n), 0.2, 60)
    items = rng.poisson(2.5, n) + 1
    dispatch_hours = np.clip(rng.gamma(2.0, 6.0, n), 0.5, 96)
    peak = np.isin(order_date.month, [10, 11, 12]).astype(int)
    weather_alert = (rng.random(n) < 0.10).astype(int)

    score = (
        0.0012 * distance
        + np.select([mode == "Economy", mode == "Express"], [0.9, -1.1], default=0.0)
        + 0.06 * dispatch_hours
        + np.array([_CARRIER_DELAY[c] for c in carrier])
        + 0.8 * weather_alert
        + 0.6 * peak
        + 0.015 * weight
        + rng.normal(0, 0.8, n)
    )
    cuts = np.quantile(score, np.cumsum(_DELIVERY_SHARES)[:-1])
    status = np.asarray(_DELIVERY_CLASSES, dtype=object)[np.searchsorted(cuts, score)]

    df = pd.DataFrame({
        "shipment_id": [f"SHP-{i:05d}" for i in range(1, n + 1)],
        "order_date": order_date.strftime("%Y-%m-%d"),
        "warehouse": warehouse,
        "carrier": carrier,
        "shipping_mode": mode,
        "distance_km": distance.round(1),
        "package_weight_kg": weight.round(2),
        "items_count": items,
        "dispatch_delay_hours": dispatch_hours.round(1),
        "peak_season": peak,
        "weather_alert": weather_alert,
        "delivery_status": status,
    })
    df.loc[rng.random(n) < 0.02, "package_weight_kg"] = np.nan
    df.loc[rng.random(n) < 0.03, "carrier"] = np.nan
    return df


# segment -> (share, visits/month, basket value, online share %, discount usage %, tenure years)
_SEGMENTS = {
    "Budget shoppers": (0.30, 6.0, 900.0, 25.0, 65.0, 3.0),
    "Premium loyalists": (0.25, 9.0, 3200.0, 30.0, 12.0, 7.0),
    "Occasional visitors": (0.25, 1.5, 1500.0, 20.0, 30.0, 1.5),
    "Online-first": (0.20, 4.0, 1800.0, 85.0, 35.0, 2.5),
}


def make_customer_segments(rng: np.random.Generator, n: int = 1600) -> pd.DataFrame:
    """Retail customers drawn from four behavioural segments; `segment` is the ground truth for clustering."""
    names = list(_SEGMENTS)
    segment = rng.choice(names, n, p=[_SEGMENTS[s][0] for s in names])
    visits, basket, online, discount, tenure = (np.array([_SEGMENTS[s][i] for s in segment]) for i in range(1, 6))

    # Wide spreads so neighbouring segments overlap: a good clustering recovers most, not all, of the truth.
    visits = np.clip(visits * rng.normal(1, 0.45, n), 0.2, None)
    basket = np.clip(basket * rng.normal(1, 0.40, n), 100, None)
    online = np.clip(online + rng.normal(0, 18, n), 0, 100)
    discount = np.clip(discount + rng.normal(0, 18, n), 0, 100)
    tenure = np.clip(tenure + rng.normal(0, 2.2, n), 0.1, None)
    annual_spend = visits * 12 * basket * rng.normal(1, 0.08, n)

    return pd.DataFrame({
        "customer_id": [f"RC-{i:05d}" for i in range(1, n + 1)],
        "visits_per_month": visits.round(2),
        "avg_basket_value": basket.round(2),
        "annual_spend": annual_spend.round(2),
        "online_share_pct": online.round(1),
        "discount_usage_pct": discount.round(1),
        "tenure_years": tenure.round(1),
        "segment": segment,
    })


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "customer_churn.csv": make_customer_churn(np.random.default_rng(SEED)),
        "crop_price.csv": make_crop_price(np.random.default_rng(SEED + 1)),
        "store_sales.csv": make_store_sales(np.random.default_rng(SEED + 2)),
        "delivery_status.csv": make_delivery_status(np.random.default_rng(SEED + 3)),
        "customer_segments.csv": make_customer_segments(np.random.default_rng(SEED + 4)),
    }
    for filename, df in datasets.items():
        path = SAMPLES_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Wrote {} ({} rows x {} cols).", path.relative_to(PROJECT_ROOT), *df.shape)


if __name__ == "__main__":
    main()
