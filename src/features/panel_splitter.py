"""Leave-One-Region-Out spatial splits and leak-free feature matrix encoders."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.temporal_feature_builder import (
    CATEGORICAL_MODEL_FEATURES,
    NUMERIC_MODEL_FEATURES,
)

RIDGE_FE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "yield_lag_1",
    "yield_lag_2",
    "yield_roll3_mean",
    "yield_roll3_std",
    "yield_trend_pct",
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


@dataclass(frozen=True)
class FittedDesignBundle:
    """Fitted transformer paired with transformed train and evaluation matrices."""

    transformer: ColumnTransformer
    feature_names: tuple[str, ...]


def build_loro_region_splits(
    panel_df: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return deterministic Leave-One-Region-Out `(region, train_idx, test_idx)` splits."""
    regions = sorted(panel_df["Region"].unique().tolist())
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    for region in regions:
        is_holdout = (panel_df["Region"] == region).to_numpy()
        train_idx = np.flatnonzero(~is_holdout)
        test_idx = np.flatnonzero(is_holdout)
        splits.append((str(region), train_idx, test_idx))
    return splits


def fit_ridge_fe_transformer(
    train_df: pd.DataFrame, include_country_crop_fe: bool = True
) -> FittedDesignBundle:
    """Fit a standardized numeric + fixed-effects one-hot encoder on `train_df` only."""
    cat_cols = (
        ["country_crop_fe", "Item", "Region"]
        if include_country_crop_fe
        else ["Item", "Region"]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(RIDGE_FE_NUMERIC_COLUMNS)),
            (
                "fe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                cat_cols,
            ),
        ],
        sparse_threshold=0.8,
    )
    transformer.fit(train_df)
    names = tuple(str(n) for n in transformer.get_feature_names_out())
    return FittedDesignBundle(transformer=transformer, feature_names=names)


def transform_ridge_matrix(
    bundle: FittedDesignBundle, df: pd.DataFrame
) -> sparse.spmatrix:
    """Transform `df` using a previously fitted Ridge Fixed-Effects transformer."""
    return bundle.transformer.transform(df)


def fit_tree_feature_encoder(train_df: pd.DataFrame) -> ColumnTransformer:
    """Fit an ordinal categorical encoder + numeric passthrough strictly on `train_df`."""
    encoder = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(NUMERIC_MODEL_FEATURES)),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                list(CATEGORICAL_MODEL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    encoder.fit(train_df)
    return encoder


def transform_tree_dataframe(
    encoder: ColumnTransformer, df: pd.DataFrame
) -> pd.DataFrame:
    """Return a labeled DataFrame for LightGBM / HistGB with integer-coded categories."""
    cols = list(NUMERIC_MODEL_FEATURES) + list(CATEGORICAL_MODEL_FEATURES)
    arr = encoder.transform(df)
    out = pd.DataFrame(arr, columns=cols, index=df.index)
    for cat_col in CATEGORICAL_MODEL_FEATURES:
        out[cat_col] = out[cat_col].astype(int)
    return out
