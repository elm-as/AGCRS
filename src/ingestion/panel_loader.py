"""Load, validate, and deduplicate the FAO/World Bank global crop yield panel."""

from pathlib import Path
import pandas as pd

from src.core.config import TemporalSplitConfig
from src.core.contracts import PanelAuditStats
from src.core.region_taxonomy import resolve_country_region

_REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "Area",
    "Item",
    "Year",
    "hg/ha_yield",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
)


def load_raw_yield_csv(csv_path: Path) -> pd.DataFrame:
    """Read raw `yield_df.csv` and validate required schema and value bounds."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw agricultural yield CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    missing_cols = [c for c in _REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in yield_df.csv: {missing_cols}")

    if raw_df[list(_REQUIRED_RAW_COLUMNS)].isna().any().any():
        raise ValueError("Unexpected missing values found in required columns.")

    if (raw_df["hg/ha_yield"] <= 0).any():
        raise ValueError("Non-positive yield values detected in raw dataset.")

    return raw_df


def _count_varying_groups(grouped: pd.core.groupby.DataFrameGroupBy, col: str) -> int:
    return int((grouped[col].nunique() > 1).sum())


def _find_missing_years(years: pd.Series) -> tuple[int, ...]:
    year_min, year_max = int(years.min()), int(years.max())
    observed = set(int(y) for y in years.unique())
    return tuple(y for y in range(year_min, year_max + 1) if y not in observed)


def _aggregate_unique_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    panel = raw_df.groupby(["Area", "Item", "Year"], as_index=False).agg(
        yield_hg_ha=("hg/ha_yield", "first"),
        rainfall_mm=("average_rain_fall_mm_per_year", "first"),
        pesticides_tonnes=("pesticides_tonnes", "first"),
        avg_temp=("avg_temp", "mean"),
        temp_station_std=("avg_temp", "std"),
        station_count=("avg_temp", "size"),
    )
    panel["temp_station_std"] = panel["temp_station_std"].fillna(0.0)
    panel["Region"] = panel["Area"].map(resolve_country_region)
    return panel.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)


def deduplicate_and_audit_panel(
    raw_df: pd.DataFrame, split_config: TemporalSplitConfig
) -> tuple[pd.DataFrame, PanelAuditStats]:
    """Deduplicate multi-station `(Area, Item, Year)` records and return audit metrics.

    In the raw Kaggle FAO/World Bank merge (`yield_df.csv`), countries with multiple
    weather stations produce up to 22 duplicate rows per `(Area, Item, Year)` sharing
    identical yield, rainfall, and pesticide values while varying only in station
    temperature. Collapsing via mean `avg_temp` eliminates leakage-prone row inflation.
    """
    value_cols = list(_REQUIRED_RAW_COLUMNS)
    exact_full_dups = int(raw_df.duplicated(subset=value_cols).sum())
    grouped = raw_df.groupby(["Area", "Item", "Year"])
    multi_station_temp_groups = _count_varying_groups(grouped, "avg_temp")
    yield_varying_dup_groups = _count_varying_groups(grouped, "hg/ha_yield")
    if yield_varying_dup_groups > 0:
        raise ValueError(
            f"Yield target varies inside {yield_varying_dup_groups} duplicate keys."
        )

    panel = _aggregate_unique_panel(raw_df)
    raw_rows = int(len(raw_df))
    unique_rows = int(len(panel))
    removed_rows = raw_rows - unique_rows
    inflation_pct = 100.0 * removed_rows / raw_rows

    top_dup_series = (
        panel.groupby("Area")["station_count"]
        .max()
        .sort_values(ascending=False)
        .head(8)
    )
    top_duplicated = tuple((str(k), int(v)) for k, v in top_dup_series.items())

    train_mask = panel["Year"] <= split_config.train_end_year
    cal_mask = (panel["Year"] >= split_config.calibration_start_year) & (
        panel["Year"] <= split_config.calibration_end_year
    )
    test_mask = panel["Year"] >= split_config.test_start_year

    audit = PanelAuditStats(
        raw_row_count=raw_rows,
        raw_column_count=int(raw_df.shape[1]),
        unique_panel_rows=unique_rows,
        redundant_rows_removed=removed_rows,
        duplicate_inflation_pct=round(inflation_pct, 2),
        exact_full_duplicate_rows=exact_full_dups,
        multi_station_temp_groups=multi_station_temp_groups,
        yield_varying_dup_groups=yield_varying_dup_groups,
        rainfall_varying_countries=int(
            (panel.groupby("Area")["rainfall_mm"].nunique() > 1).sum()
        ),
        temp_varying_countries=int(
            (panel.groupby("Area")["avg_temp"].nunique() > 1).sum()
        ),
        pesticide_varying_countries=int(
            (panel.groupby("Area")["pesticides_tonnes"].nunique() > 1).sum()
        ),
        n_countries=int(panel["Area"].nunique()),
        n_crops=int(panel["Item"].nunique()),
        n_regions=int(panel["Region"].nunique()),
        year_min=int(panel["Year"].min()),
        year_max=int(panel["Year"].max()),
        missing_years=_find_missing_years(panel["Year"]),
        max_station_duplication_factor=int(panel["station_count"].max()),
        top_duplicated_countries=top_duplicated,
        train_rows=int(train_mask.sum()),
        calibration_rows=int(cal_mask.sum()),
        test_rows=int(test_mask.sum()),
    )
    return panel, audit
