"""Pure regression evaluation metrics for agricultural yield forecasting (hg/ha)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.core.contracts import CropPerformanceRow, RegressionMetrics
from src.core.region_taxonomy import is_african_tropical_staple


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> RegressionMetrics:
    """Compute RMSE, MAE, MAPE (%), R², and MedAE without mutating inputs."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal shape.")

    denom = np.maximum(np.abs(actual), 1.0)
    mape_pct = float(np.mean(np.abs(actual - predicted) / denom) * 100.0)
    return RegressionMetrics(
        rmse=float(root_mean_squared_error(actual, predicted)),
        mae=float(mean_absolute_error(actual, predicted)),
        mape_pct=mape_pct,
        r2=float(r2_score(actual, predicted)),
        medae=float(median_absolute_error(actual, predicted)),
        n_samples=int(actual.size),
    )


def compute_per_crop_metrics(
    meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[CropPerformanceRow, ...]:
    """Compute out-of-time regression performance stratified by staple crop."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    crops = meta_df["Item"].to_numpy()

    rows: list[CropPerformanceRow] = []
    for crop in sorted(meta_df["Item"].unique().tolist()):
        mask = crops == crop
        metrics = compute_regression_metrics(actual[mask], predicted[mask])
        rows.append(
            CropPerformanceRow(
                crop=str(crop),
                is_african_tropical_staple=is_african_tropical_staple(str(crop)),
                n_test_samples=metrics.n_samples,
                mean_actual_yield=float(np.mean(actual[mask])),
                rmse=metrics.rmse,
                mae=metrics.mae,
                mape_pct=metrics.mape_pct,
                r2=metrics.r2,
            )
        )
    return tuple(rows)
