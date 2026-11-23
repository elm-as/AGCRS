"""Panel deduplication audit and strict zero-future-leakage verification tests."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.core.config import TemporalSplitConfig
from src.features.panel_splitter import build_loro_region_splits
from src.features.temporal_feature_builder import build_temporal_panel_features
from src.ingestion.panel_loader import (
    deduplicate_and_audit_panel,
    load_raw_yield_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "yield_df.csv"


def test_panel_deduplication_eliminates_multistation_duplicates() -> None:
    """Raw 28,242 multi-station rows must collapse cleanly to 13,130 unique keys."""
    split_cfg = TemporalSplitConfig()
    raw_df = load_raw_yield_csv(RAW_CSV)
    clean_panel, audit = deduplicate_and_audit_panel(raw_df, split_cfg)

    assert audit.raw_row_count == 28242
    assert audit.unique_panel_rows == 13130
    assert audit.redundant_rows_removed == 15112
    assert audit.yield_varying_dup_groups == 0
    assert audit.multi_station_temp_groups == 3690
    assert audit.n_countries == 101
    assert audit.n_crops == 10
    assert audit.missing_years == (2003,)
    assert not clean_panel.duplicated(subset=["Area", "Item", "Year"]).any()
    assert clean_panel["Region"].isna().sum() == 0


def test_temporal_features_strict_zero_future_leakage() -> None:
    """Perturbing 2011-2013 targets and climate must not alter pre-2011 features."""
    split_cfg = TemporalSplitConfig()
    raw_df = load_raw_yield_csv(RAW_CSV)
    clean_panel, _ = deduplicate_and_audit_panel(raw_df, split_cfg)

    base_frames = build_temporal_panel_features(clean_panel, split_cfg)

    perturbed = clean_panel.copy()
    future_mask = perturbed["Year"] >= split_cfg.test_start_year
    perturbed.loc[future_mask, "yield_hg_ha"] = perturbed.loc[future_mask, "yield_hg_ha"] * 5.0
    perturbed.loc[future_mask, "avg_temp"] = perturbed.loc[future_mask, "avg_temp"] + 15.0
    perturbed.loc[future_mask, "pesticides_tonnes"] = (
        perturbed.loc[future_mask, "pesticides_tonnes"] * 10.0
    )
    perturbed_frames = build_temporal_panel_features(perturbed, split_cfg)

    checked_cols = [
        "yield_lag_1",
        "yield_lag_2",
        "yield_roll3_mean",
        "yield_roll3_std",
        "local_baseline_yield",
        "temp_anomaly_c",
        "rain_anomaly_pct",
        "pesticide_anomaly_pct",
    ]
    for col in checked_cols:
        np.testing.assert_allclose(
            base_frames.train_df[col].to_numpy(),
            perturbed_frames.train_df[col].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            base_frames.calibration_df[col].to_numpy(),
            perturbed_frames.calibration_df[col].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

    # First test year (2011) lags must depend solely on <=2010 realized yields.
    y2011_base = base_frames.test_df.loc[base_frames.test_df["Year"] == 2011]
    y2011_pert = perturbed_frames.test_df.loc[perturbed_frames.test_df["Year"] == 2011]
    for lag_col in ("yield_lag_1", "yield_roll3_mean", "local_baseline_yield"):
        np.testing.assert_allclose(
            y2011_base[lag_col].to_numpy(),
            y2011_pert[lag_col].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )


def test_loro_region_splits_disjoint_and_exhaustive() -> None:
    """Leave-One-Region-Out splits must partition all 101 countries without overlap."""
    split_cfg = TemporalSplitConfig()
    clean_panel, _ = deduplicate_and_audit_panel(load_raw_yield_csv(RAW_CSV), split_cfg)
    splits = build_loro_region_splits(clean_panel)

    assert len(splits) == 6
    seen_test_indices: list[np.ndarray] = []
    for region, tr_idx, te_idx in splits:
        assert len(tr_idx) > 0 and len(te_idx) > 0
        tr_areas = set(clean_panel.iloc[tr_idx]["Area"].unique())
        te_areas = set(clean_panel.iloc[te_idx]["Area"].unique())
        assert tr_areas.isdisjoint(te_areas), f"Country overlap in LORO region {region}"
        seen_test_indices.append(te_idx)

    all_test = np.sort(np.concatenate(seen_test_indices))
    np.testing.assert_array_equal(all_test, np.arange(len(clean_panel)))
