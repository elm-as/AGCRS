"""Distribution-free split-conformal prediction intervals (standard and adaptive)."""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from src.core.contracts import ConformalIntervalResult, CropConformalCoverage


@dataclass(frozen=True)
class ConformalCalibrationInput:
    """Typed bundle of calibration and test arrays for split-conformal inference."""

    cal_actual: np.ndarray
    cal_pred: np.ndarray
    cal_volatility: np.ndarray
    test_actual: np.ndarray
    test_pred: np.ndarray
    test_volatility: np.ndarray
    test_crops: pd.Series
    alpha: float


def _finite_sample_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n_cal = int(scores.size)
    if n_cal == 0:
        raise ValueError("Calibration set cannot be empty for split-conformal quantile.")
    rank_level = math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    clamped = min(max(rank_level, 0.0), 1.0)
    return float(np.quantile(scores, clamped, method="higher"))


def _compute_local_scale(pred: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    # Combine past-only local volatility with prediction magnitude so intervals adapt
    # smoothly across low-yield (Sorghum ~15k hg/ha) and high-yield (Potatoes ~200k).
    pred_floor = 0.08 * np.maximum(np.abs(pred), 1.0)
    return np.maximum(np.maximum(volatility, pred_floor), 500.0)


def _summarize_crop_coverage(
    data: ConformalCalibrationInput, lower: np.ndarray, upper: np.ndarray
) -> tuple[CropConformalCoverage, ...]:
    crops_arr = data.test_crops.to_numpy()
    covered = (data.test_actual >= lower) & (data.test_actual <= upper)
    widths = upper - lower

    rows: list[CropConformalCoverage] = []
    for crop in sorted(data.test_crops.unique().tolist()):
        mask = crops_arr == crop
        mean_actual = max(float(np.mean(data.test_actual[mask])), 1.0)
        mean_w = float(np.mean(widths[mask]))
        rows.append(
            CropConformalCoverage(
                crop=str(crop),
                n_test_samples=int(mask.sum()),
                empirical_coverage=float(np.mean(covered[mask])),
                mean_width_hg_ha=mean_w,
                relative_width_pct=100.0 * mean_w / mean_actual,
            )
        )
    return tuple(rows)


def _assemble_conformal_result(
    method_name: str,
    data: ConformalCalibrationInput,
    bounds_and_q: tuple[np.ndarray, np.ndarray, float],
) -> ConformalIntervalResult:
    lower, upper, quantile = bounds_and_q
    covered = (data.test_actual >= lower) & (data.test_actual <= upper)
    widths = upper - lower
    mean_actual = max(float(np.mean(data.test_actual)), 1.0)
    return ConformalIntervalResult(
        method_name=method_name,
        nominal_coverage=1.0 - data.alpha,
        empirical_coverage=float(np.mean(covered)),
        conformal_quantile=quantile,
        mean_width_hg_ha=float(np.mean(widths)),
        median_width_hg_ha=float(np.median(widths)),
        relative_width_pct=100.0 * float(np.mean(widths)) / mean_actual,
        lower_bounds=lower,
        upper_bounds=upper,
        per_crop_coverage=_summarize_crop_coverage(data, lower, upper),
    )


def compute_standard_split_conformal(
    data: ConformalCalibrationInput,
) -> ConformalIntervalResult:
    """Compute homoscedastic split-conformal intervals calibrated on 2009-2010."""
    cal_scores = np.abs(data.cal_actual - data.cal_pred)
    q_hat = _finite_sample_conformal_quantile(cal_scores, data.alpha)
    lower = np.maximum(0.0, data.test_pred - q_hat)
    upper = data.test_pred + q_hat
    return _assemble_conformal_result(
        "standard_split_conformal", data, (lower, upper, q_hat)
    )


def compute_adaptive_split_conformal(
    data: ConformalCalibrationInput,
) -> ConformalIntervalResult:
    """Compute locally-weighted heteroscedastic split-conformal intervals on 2011-2013."""
    cal_scale = _compute_local_scale(data.cal_pred, data.cal_volatility)
    test_scale = _compute_local_scale(data.test_pred, data.test_volatility)
    cal_scores = np.abs(data.cal_actual - data.cal_pred) / cal_scale
    q_hat = _finite_sample_conformal_quantile(cal_scores, data.alpha)
    half_width = q_hat * test_scale
    lower = np.maximum(0.0, data.test_pred - half_width)
    upper = data.test_pred + half_width
    return _assemble_conformal_result(
        "adaptive_scaled_split_conformal", data, (lower, upper, q_hat)
    )
