"""Strictly past-only agronomic and climate anomaly feature engineering."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.core.config import TemporalSplitConfig
from src.core.contracts import TemporalSplitFrames

NUMERIC_MODEL_FEATURES: tuple[str, ...] = (
    "yield_lag_1",
    "yield_lag_2",
    "yield_roll3_mean",
    "yield_roll3_std",
    "yield_trend_pct",
    "years_since_prev_obs",
    "avg_temp",
    "temp_sq",
    "temp_anomaly_c",
    "rainfall_mm",
    "log_rainfall",
    "rain_anomaly_pct",
    "temp_x_log_rain",
    "hydrothermal_index",
    "pesticide_intensity_log",
    "pesticide_anomaly_pct",
    "year_centered",
)

CATEGORICAL_MODEL_FEATURES: tuple[str, ...] = ("Area", "Item", "Region")

FEATURE_DISPLAY_LABELS: dict[str, str] = {
    "yield_lag_1": "Yield Lag-1 (t-1)",
    "yield_lag_2": "Yield Lag-2 (t-2)",
    "yield_roll3_mean": "Rolling 3-Yr Yield Mean (Past)",
    "yield_roll3_std": "Rolling 3-Yr Yield Volatility",
    "yield_trend_pct": "Short-Term Yield Momentum (%)",
    "years_since_prev_obs": "Inter-Observation Gap (Yrs)",
    "avg_temp": "Annual Mean Temperature (°C)",
    "temp_sq": "Temperature Squared (°C²)",
    "temp_anomaly_c": "Country Temp Anomaly (°C, Train Base)",
    "rainfall_mm": "Annual Climatological Rainfall (mm)",
    "log_rainfall": "Log Annual Rainfall",
    "rain_anomaly_pct": "Crop Rainfall Departure (%)",
    "temp_x_log_rain": "Temp × Log-Rainfall Interaction",
    "hydrothermal_index": "Hydrothermal Aridity Index",
    "pesticide_intensity_log": "Log Pesticide Usage (tonnes)",
    "pesticide_anomaly_pct": "Country Pesticide Departure (%)",
    "year_centered": "Secular Technology Trend (Year)",
    "Area": "Country Fixed Effect (Area)",
    "Item": "Staple Crop Type (Item)",
    "Region": "FAO Macro-Region",
}


@dataclass(frozen=True)
class _TrainBaselines:
    country_temp_mean: pd.Series
    global_temp_mean: float
    crop_rain_mean: pd.Series
    global_rain_mean: float
    country_pest_median: pd.Series
    global_pest_median: float
    crop_yield_median: pd.Series
    global_yield_median: float


def _fit_train_only_baselines(
    panel_df: pd.DataFrame, train_end_year: int
) -> _TrainBaselines:
    train_slice = panel_df.loc[panel_df["Year"] <= train_end_year]
    if train_slice.empty:
        raise ValueError("Training slice is empty; cannot compute historical baselines.")
    return _TrainBaselines(
        country_temp_mean=train_slice.groupby("Area")["avg_temp"].mean(),
        global_temp_mean=float(train_slice["avg_temp"].mean()),
        crop_rain_mean=train_slice.groupby("Item")["rainfall_mm"].mean(),
        global_rain_mean=float(train_slice["rainfall_mm"].mean()),
        country_pest_median=train_slice.groupby("Area")["pesticides_tonnes"].median(),
        global_pest_median=float(train_slice["pesticides_tonnes"].median()),
        crop_yield_median=train_slice.groupby("Item")["yield_hg_ha"].median(),
        global_yield_median=float(train_slice["yield_hg_ha"].median()),
    )


def _compute_past_only_yield_lags(
    df: pd.DataFrame, baselines: _TrainBaselines
) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(["Area", "Item"], sort=False)

    lag1 = grp["yield_hg_ha"].shift(1)
    lag2 = grp["yield_hg_ha"].shift(2)
    roll3_mean = grp["yield_hg_ha"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    roll3_std = grp["yield_hg_ha"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=2).std()
    )
    roll5_baseline = grp["yield_hg_ha"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    year_gap = grp["Year"].diff().fillna(1.0)

    crop_fallback = out["Item"].map(baselines.crop_yield_median).fillna(
        baselines.global_yield_median
    )
    out["has_observed_lag1"] = lag1.notna().astype(int)
    out["yield_lag_1"] = lag1.fillna(crop_fallback)
    out["yield_lag_2"] = lag2.fillna(out["yield_lag_1"])
    out["yield_roll3_mean"] = roll3_mean.fillna(crop_fallback)
    out["yield_roll3_std"] = roll3_std.fillna(0.0)
    out["local_baseline_yield"] = roll5_baseline.fillna(crop_fallback)
    out["yield_trend_pct"] = (
        (out["yield_lag_1"] - out["yield_lag_2"])
        / np.maximum(out["yield_lag_2"], 1.0)
    ).clip(-1.5, 1.5)
    out["years_since_prev_obs"] = year_gap.clip(1.0, 10.0)
    return out


def _compute_climate_and_input_features(
    df: pd.DataFrame, baselines: _TrainBaselines
) -> pd.DataFrame:
    out = df.copy()
    country_temp_base = out["Area"].map(baselines.country_temp_mean).fillna(
        baselines.global_temp_mean
    )
    crop_rain_base = out["Item"].map(baselines.crop_rain_mean).fillna(
        baselines.global_rain_mean
    )
    country_pest_base = out["Area"].map(baselines.country_pest_median).fillna(
        baselines.global_pest_median
    )

    out["temp_sq"] = out["avg_temp"] ** 2
    out["temp_anomaly_c"] = out["avg_temp"] - country_temp_base
    out["log_rainfall"] = np.log1p(out["rainfall_mm"])
    out["rain_anomaly_pct"] = (
        (out["rainfall_mm"] - crop_rain_base) / np.maximum(crop_rain_base, 1.0)
    )
    out["temp_x_log_rain"] = out["avg_temp"] * out["log_rainfall"]
    out["hydrothermal_index"] = out["avg_temp"] / np.maximum(out["log_rainfall"], 1.0)
    out["pesticide_intensity_log"] = np.log1p(out["pesticides_tonnes"])
    out["pesticide_anomaly_pct"] = (
        (out["pesticides_tonnes"] - country_pest_base)
        / np.maximum(country_pest_base, 1.0)
    ).clip(-2.0, 5.0)
    out["year_centered"] = out["Year"].astype(float) - 2000.0
    out["country_crop_fe"] = out["Area"].astype(str) + "__" + out["Item"].astype(str)
    return out


def build_temporal_panel_features(
    panel_df: pd.DataFrame, split_config: TemporalSplitConfig
) -> TemporalSplitFrames:
    """Build strictly past-only panel features and chronological train/cal/test splits.

    All historical baselines (country temperature means, crop rainfall means,
    country pesticide medians, and fallback crop yield medians) are estimated
    exclusively on `Year <= split_config.train_end_year` (1990-2008) to guarantee
    zero future leakage into the calibration (2009-2010) and test (2011-2013) sets.
    """
    ordered = panel_df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)
    baselines = _fit_train_only_baselines(ordered, split_config.train_end_year)
    enriched = _compute_past_only_yield_lags(ordered, baselines)
    enriched = _compute_climate_and_input_features(enriched, baselines)

    train_df = enriched.loc[
        enriched["Year"] <= split_config.train_end_year
    ].reset_index(drop=True)
    cal_df = enriched.loc[
        (enriched["Year"] >= split_config.calibration_start_year)
        & (enriched["Year"] <= split_config.calibration_end_year)
    ].reset_index(drop=True)
    test_df = enriched.loc[
        enriched["Year"] >= split_config.test_start_year
    ].reset_index(drop=True)

    return TemporalSplitFrames(
        full_panel=enriched,
        train_df=train_df,
        calibration_df=cal_df,
        test_df=test_df,
        numeric_features=NUMERIC_MODEL_FEATURES,
        categorical_features=CATEGORICAL_MODEL_FEATURES,
        target_column="yield_hg_ha",
    )
