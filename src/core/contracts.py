"""Frozen dataclass contracts defining strict typed interfaces across AGCRS layers."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelAuditStats:
    """Quantitative audit of raw FAO/World Bank CSV vs deduplicated country-crop-year panel."""

    raw_row_count: int
    raw_column_count: int
    unique_panel_rows: int
    redundant_rows_removed: int
    duplicate_inflation_pct: float
    exact_full_duplicate_rows: int
    multi_station_temp_groups: int
    yield_varying_dup_groups: int
    rainfall_varying_countries: int
    temp_varying_countries: int
    pesticide_varying_countries: int
    n_countries: int
    n_crops: int
    n_regions: int
    year_min: int
    year_max: int
    missing_years: tuple[int, ...]
    max_station_duplication_factor: int
    top_duplicated_countries: tuple[tuple[str, int], ...]
    train_rows: int
    calibration_rows: int
    test_rows: int


@dataclass(frozen=True)
class RegressionMetrics:
    """Standard regression performance summary for yield prediction (hg/ha)."""

    rmse: float
    mae: float
    mape_pct: float
    r2: float
    medae: float
    n_samples: int


@dataclass(frozen=True)
class CropPerformanceRow:
    """Out-of-time test accuracy metrics stratified by staple crop."""

    crop: str
    is_african_tropical_staple: bool
    n_test_samples: int
    mean_actual_yield: float
    rmse: float
    mae: float
    mape_pct: float
    r2: float


@dataclass(frozen=True)
class LoroRegionScore:
    """Leave-One-Region-Out spatial generalization score for a macro-region."""

    region: str
    model_name: str
    n_countries: int
    n_samples: int
    rmse: float
    mae: float
    mape_pct: float
    r2: float


@dataclass(frozen=True)
class ModelBenchmarkResult:
    """Evaluation artifact for one forecasting model on temporal OOT and LORO splits."""

    model_name: str
    display_label: str
    calibration_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    per_crop_test_metrics: tuple[CropPerformanceRow, ...]
    loro_scores: tuple[LoroRegionScore, ...]
    loro_mean_r2: float
    loro_mean_rmse: float
    cal_predictions: np.ndarray
    test_predictions: np.ndarray


@dataclass(frozen=True)
class CropConformalCoverage:
    """Empirical split-conformal coverage diagnostics for a single crop."""

    crop: str
    n_test_samples: int
    empirical_coverage: float
    mean_width_hg_ha: float
    relative_width_pct: float


@dataclass(frozen=True)
class ConformalIntervalResult:
    """Distribution-free split-conformal prediction interval summary on 2011-2013 test set."""

    method_name: str
    nominal_coverage: float
    empirical_coverage: float
    conformal_quantile: float
    mean_width_hg_ha: float
    median_width_hg_ha: float
    relative_width_pct: float
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    per_crop_coverage: tuple[CropConformalCoverage, ...]


@dataclass(frozen=True)
class ThermalCurvePoint:
    """Single temperature evaluation point on a crop's econometric response curve."""

    temp_c: float
    relative_yield_pct: float


@dataclass(frozen=True)
class CropClimateShockImpact:
    """Estimated yield elasticity (%) under warming and drought stress scenarios."""

    crop: str
    is_african_tropical_staple: bool
    n_train_observations: int
    mean_temp_c: float
    mean_rainfall_mm: float
    shock_plus_1c_pct: float
    shock_plus_1c_ci_low: float
    shock_plus_1c_ci_high: float
    shock_plus_2c_pct: float
    shock_plus_2c_ci_low: float
    shock_plus_2c_ci_high: float
    shock_drought_minus_20pct: float
    shock_drought_ci_low: float
    shock_drought_ci_high: float
    fe_within_plus_1c_pct: float
    fe_within_plus_2c_pct: float
    thermal_curve: tuple[ThermalCurvePoint, ...]


@dataclass(frozen=True)
class ClimateEconometricsSummary:
    """Aggregate climate sensitivity econometric output across all staple crops."""

    model_r2_train_log: float
    model_r2_test_log: float
    fe_within_r2_train_log: float
    global_mean_plus_1c_pct: float
    global_mean_plus_2c_pct: float
    global_mean_drought_20pct: float
    african_staples_mean_plus_2c_pct: float
    african_staples_mean_drought_20pct: float
    crop_impacts: tuple[CropClimateShockImpact, ...]


@dataclass(frozen=True)
class InsuranceThresholdPoint:
    """Parametric insurance portfolio metrics at a specific baseline trigger ratio."""

    trigger_ratio: float
    actual_shortfall_rate: float
    parametric_trigger_rate: float
    pure_premium_rate_pct: float
    commercial_premium_rate_pct: float
    actual_loss_ratio: float
    parametric_loss_ratio: float
    fn_basis_risk_rate: float
    fp_basis_risk_rate: float
    shortfall_recall: float
    shortfall_precision: float


@dataclass(frozen=True)
class InsuranceSegmentSummary:
    """Parametric crop insurance risk profile for a crop or macro-region segment."""

    segment_type: str
    segment_name: str
    n_policies: int
    actual_shortfall_rate: float
    parametric_trigger_rate: float
    pure_premium_rate_pct: float
    parametric_loss_ratio: float
    fn_basis_risk_rate: float
    fp_basis_risk_rate: float


@dataclass(frozen=True)
class ParametricInsuranceSummary:
    """Complete parametric drought/heatwave crop insurance evaluation on 2011-2013."""

    trigger_ratio: float
    loading_factor: float
    portfolio_at_trigger: InsuranceThresholdPoint
    threshold_sweep: tuple[InsuranceThresholdPoint, ...]
    by_crop: tuple[InsuranceSegmentSummary, ...]
    by_region: tuple[InsuranceSegmentSummary, ...]
    ssa_summary: InsuranceSegmentSummary


@dataclass(frozen=True)
class ShapDriverScore:
    """Aggregated global TreeSHAP feature importance on the 2011-2013 test horizon."""

    feature_name: str
    display_label: str
    mean_abs_shap_hg_ha: float
    share_pct: float
    mean_signed_correlation: float


@dataclass(frozen=True)
class TemporalSplitFrames:
    """Chronologically partitioned feature DataFrames with strictly past-only baselines."""

    full_panel: pd.DataFrame
    train_df: pd.DataFrame
    calibration_df: pd.DataFrame
    test_df: pd.DataFrame
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_column: str


@dataclass(frozen=True)
class AgcrsReportPayload:
    """Master payload handed to visualization, dashboard, and JSON serialization."""

    audit_stats: PanelAuditStats
    best_model_name: str
    model_benchmarks: tuple[ModelBenchmarkResult, ...]
    conformal_standard: ConformalIntervalResult
    conformal_adaptive: ConformalIntervalResult
    climate_econometrics: ClimateEconometricsSummary
    insurance_summary: ParametricInsuranceSummary
    shap_drivers: tuple[ShapDriverScore, ...]
    test_metadata_df: pd.DataFrame
    test_actual_yield: np.ndarray
    best_test_predictions: np.ndarray
