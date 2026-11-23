"""Tests for split-conformal coverage, parametric insurance pricing, and econometrics."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.core.config import (
    ConformalAndInsuranceConfig,
    ModelHyperparamConfig,
    TemporalSplitConfig,
)
from src.evaluation.conformal_intervals import (
    ConformalCalibrationInput,
    compute_adaptive_split_conformal,
    compute_standard_split_conformal,
)
from src.evaluation.parametric_insurance import (
    InsuranceSimulationInput,
    evaluate_parametric_insurance,
)
from src.features.temporal_feature_builder import build_temporal_panel_features
from src.ingestion.panel_loader import (
    deduplicate_and_audit_panel,
    load_raw_yield_csv,
)
from src.modeling.climate_econometrics import run_climate_sensitivity_econometrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "yield_df.csv"


def test_split_conformal_empirical_coverage_and_heteroscedastic_scaling() -> None:
    """Split-conformal 90% intervals achieve ~90% coverage and scale by crop yield."""
    rng = np.random.default_rng(42)
    n_cal, n_test = 800, 1200

    cal_crops = pd.Series(np.where(np.arange(n_cal) % 2 == 0, "Sorghum", "Potatoes"))
    test_crops = pd.Series(np.where(np.arange(n_test) % 2 == 0, "Sorghum", "Potatoes"))
    cal_pred = np.where(cal_crops == "Sorghum", 15_000.0, 200_000.0)
    test_pred = np.where(test_crops == "Sorghum", 15_000.0, 200_000.0)
    cal_vol = 0.12 * cal_pred
    test_vol = 0.12 * test_pred

    cal_actual = np.maximum(100.0, cal_pred + rng.normal(0.0, cal_vol))
    test_actual = np.maximum(100.0, test_pred + rng.normal(0.0, test_vol))

    conf_in = ConformalCalibrationInput(
        cal_actual=cal_actual,
        cal_pred=cal_pred,
        cal_volatility=cal_vol,
        test_actual=test_actual,
        test_pred=test_pred,
        test_volatility=test_vol,
        test_crops=test_crops,
        alpha=0.10,
    )
    std_res = compute_standard_split_conformal(conf_in)
    adapt_res = compute_adaptive_split_conformal(conf_in)

    assert 0.86 <= std_res.empirical_coverage <= 0.94
    assert 0.86 <= adapt_res.empirical_coverage <= 0.94

    # Adaptive intervals must be substantially narrower on low-yield Sorghum than Potatoes.
    sorghum_cov = next(c for c in adapt_res.per_crop_coverage if c.crop == "Sorghum")
    potato_cov = next(c for c in adapt_res.per_crop_coverage if c.crop == "Potatoes")
    assert sorghum_cov.mean_width_hg_ha < potato_cov.mean_width_hg_ha * 0.25


def test_parametric_insurance_actuarial_identities_and_monotonicity() -> None:
    """Pure premium, commercial premium, basis risk, and trigger sweep obey identities."""
    rng = np.random.default_rng(7)
    n_pre, n_test = 400, 200
    pretest_df = pd.DataFrame(
        {
            "Item": rng.choice(["Maize", "Sorghum"], size=n_pre),
            "Region": rng.choice(["Sub-Saharan Africa", "Europe & Central Asia"], size=n_pre),
            "local_baseline_yield": np.full(n_pre, 20_000.0),
            "yield_hg_ha": rng.normal(20_000.0, 4_500.0, size=n_pre).clip(2_000.0),
        }
    )
    test_df = pd.DataFrame(
        {
            "Item": rng.choice(["Maize", "Sorghum"], size=n_test),
            "Region": rng.choice(["Sub-Saharan Africa", "Europe & Central Asia"], size=n_test),
            "local_baseline_yield": np.full(n_test, 20_000.0),
            "yield_hg_ha": rng.normal(19_000.0, 4_800.0, size=n_test).clip(2_000.0),
        }
    )
    test_pred = test_df["yield_hg_ha"].to_numpy() + rng.normal(0.0, 1_500.0, size=n_test)

    cfg = ConformalAndInsuranceConfig()
    summary = evaluate_parametric_insurance(
        InsuranceSimulationInput(
            pretest_df=pretest_df, test_df=test_df, test_predicted_yield=test_pred
        ),
        cfg,
    )
    port = summary.portfolio_at_trigger
    assert port.commercial_premium_rate_pct > port.pure_premium_rate_pct > 0.0
    assert 0.0 <= port.fn_basis_risk_rate <= 1.0
    assert 0.0 <= port.fp_basis_risk_rate <= 1.0
    assert port.fn_basis_risk_rate + port.fp_basis_risk_rate <= 1.0

    # Higher trigger ratios must yield monotonically non-decreasing shortfall & pure premium rates.
    shortfalls = [pt.actual_shortfall_rate for pt in summary.threshold_sweep]
    premiums = [pt.pure_premium_rate_pct for pt in summary.threshold_sweep]
    assert shortfalls == sorted(shortfalls)
    assert premiums == sorted(premiums)


def test_climate_econometrics_covers_all_staple_crops() -> None:
    """Climate shock econometrics returns valid CIs and thermal curves for all 10 crops."""
    split_cfg = TemporalSplitConfig()
    clean_panel, _ = deduplicate_and_audit_panel(load_raw_yield_csv(RAW_CSV), split_cfg)
    frames = build_temporal_panel_features(clean_panel, split_cfg)
    econ = run_climate_sensitivity_econometrics(
        frames, ModelHyperparamConfig.fast_smoke(), ConformalAndInsuranceConfig()
    )
    assert len(econ.crop_impacts) == 10
    assert econ.model_r2_train_log > 0.30
    assert econ.fe_within_r2_train_log > 0.85
    for impact in econ.crop_impacts:
        assert impact.shock_plus_2c_ci_low <= impact.shock_plus_2c_ci_high
        assert impact.shock_drought_ci_low <= impact.shock_drought_ci_high
        assert len(impact.thermal_curve) == 12
