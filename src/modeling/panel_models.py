"""Train and benchmark Ridge Fixed-Effects, LightGBM, and HistGradientBoosting."""

from dataclasses import dataclass
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.core.config import ModelHyperparamConfig
from src.core.contracts import (
    LoroRegionScore,
    ModelBenchmarkResult,
    TemporalSplitFrames,
)
from src.evaluation.regression_metrics import (
    compute_per_crop_metrics,
    compute_regression_metrics,
)
from src.features.panel_splitter import (
    build_loro_region_splits,
    fit_ridge_fe_transformer,
    fit_tree_feature_encoder,
    transform_ridge_matrix,
    transform_tree_dataframe,
)


@dataclass(frozen=True)
class PanelModelingOutput:
    """Bundled benchmark artifacts plus fitted LightGBM model for TreeSHAP."""

    best_model_name: str
    benchmarks: tuple[ModelBenchmarkResult, ...]
    lgbm_model: lgb.LGBMRegressor
    lgbm_test_features: pd.DataFrame


def _fit_best_ridge_fe(
    frames: TemporalSplitFrames, hp: ModelHyperparamConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    design = fit_ridge_fe_transformer(frames.train_df, include_country_crop_fe=True)
    x_tr = transform_ridge_matrix(design, frames.train_df)
    x_cal = transform_ridge_matrix(design, frames.calibration_df)
    x_te = transform_ridge_matrix(design, frames.test_df)
    y_tr = frames.train_df[frames.target_column].to_numpy(dtype=float)
    y_cal = frames.calibration_df[frames.target_column].to_numpy(dtype=float)

    best_alpha = hp.ridge_alphas[0]
    best_rmse = float("inf")
    best_cal_pred = np.zeros_like(y_cal)
    best_model: Ridge | None = None

    for alpha in hp.ridge_alphas:
        model = Ridge(alpha=alpha, random_state=hp.random_seed)
        model.fit(x_tr, y_tr)
        cal_pred = np.maximum(0.0, model.predict(x_cal))
        rmse = compute_regression_metrics(y_cal, cal_pred).rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_cal_pred = cal_pred
            best_model = model

    assert best_model is not None
    test_pred = np.maximum(0.0, best_model.predict(x_te))
    return best_cal_pred, test_pred, best_alpha


def _fit_lightgbm_temporal(
    frames: TemporalSplitFrames, hp: ModelHyperparamConfig
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, tuple[np.ndarray, np.ndarray]]:
    encoder = fit_tree_feature_encoder(frames.train_df)
    x_tr = transform_tree_dataframe(encoder, frames.train_df)
    x_cal = transform_tree_dataframe(encoder, frames.calibration_df)
    x_te = transform_tree_dataframe(encoder, frames.test_df)
    y_tr = frames.train_df[frames.target_column].to_numpy(dtype=float)

    cat_cols = list(frames.categorical_features)
    model = lgb.LGBMRegressor(
        n_estimators=hp.lgbm_n_estimators,
        learning_rate=hp.lgbm_learning_rate,
        num_leaves=hp.lgbm_num_leaves,
        min_child_samples=hp.lgbm_min_child_samples,
        subsample_for_bin=50000,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        random_state=hp.random_seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_tr, y_tr, categorical_feature=cat_cols)
    cal_pred = np.maximum(0.0, model.predict(x_cal))
    test_pred = np.maximum(0.0, model.predict(x_te))
    return model, x_te, (cal_pred, test_pred)


def _fit_hgb_temporal(
    frames: TemporalSplitFrames, hp: ModelHyperparamConfig
) -> tuple[np.ndarray, np.ndarray]:
    encoder = fit_tree_feature_encoder(frames.train_df)
    x_tr = transform_tree_dataframe(encoder, frames.train_df)
    x_cal = transform_tree_dataframe(encoder, frames.calibration_df)
    x_te = transform_tree_dataframe(encoder, frames.test_df)
    y_tr = frames.train_df[frames.target_column].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        max_iter=hp.hgb_max_iter,
        learning_rate=hp.hgb_learning_rate,
        max_leaf_nodes=hp.hgb_max_leaf_nodes,
        l2_regularization=1.0,
        random_state=hp.random_seed,
    )
    model.fit(x_tr, y_tr)
    cal_pred = np.maximum(0.0, model.predict(x_cal))
    test_pred = np.maximum(0.0, model.predict(x_te))
    return cal_pred, test_pred


def _predict_loro_fold(
    model_name: str,
    fold_frames: tuple[pd.DataFrame, pd.DataFrame],
    hp: ModelHyperparamConfig,
) -> np.ndarray:
    train_fold, test_fold = fold_frames
    y_tr = train_fold["yield_hg_ha"].to_numpy(dtype=float)
    if model_name == "ridge_fixed_effects":
        design = fit_ridge_fe_transformer(train_fold, include_country_crop_fe=False)
        model = Ridge(alpha=10.0, random_state=hp.random_seed)
        model.fit(transform_ridge_matrix(design, train_fold), y_tr)
        return np.maximum(0.0, model.predict(transform_ridge_matrix(design, test_fold)))

    encoder = fit_tree_feature_encoder(train_fold)
    x_tr = transform_tree_dataframe(encoder, train_fold)
    x_te = transform_tree_dataframe(encoder, test_fold)
    if model_name == "lightgbm":
        model_lgb = lgb.LGBMRegressor(
            n_estimators=max(hp.lgbm_n_estimators // 2, 60),
            learning_rate=hp.lgbm_learning_rate,
            num_leaves=hp.lgbm_num_leaves,
            min_child_samples=hp.lgbm_min_child_samples,
            colsample_bytree=0.85,
            subsample=0.85,
            subsample_freq=1,
            random_state=hp.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(x_tr, y_tr)
        return np.maximum(0.0, model_lgb.predict(x_te))

    model_hgb = HistGradientBoostingRegressor(
        max_iter=max(hp.hgb_max_iter // 2, 60),
        learning_rate=hp.hgb_learning_rate,
        max_leaf_nodes=hp.hgb_max_leaf_nodes,
        l2_regularization=1.0,
        random_state=hp.random_seed,
    )
    model_hgb.fit(x_tr, y_tr)
    return np.maximum(0.0, model_hgb.predict(x_te))


def _run_loro_audit(
    model_name: str, panel_df: pd.DataFrame, hp: ModelHyperparamConfig
) -> tuple[LoroRegionScore, ...]:
    scores: list[LoroRegionScore] = []
    for region, tr_idx, te_idx in build_loro_region_splits(panel_df):
        train_fold = panel_df.iloc[tr_idx]
        test_fold = panel_df.iloc[te_idx]
        y_te = test_fold["yield_hg_ha"].to_numpy(dtype=float)
        preds = _predict_loro_fold(model_name, (train_fold, test_fold), hp)
        metrics = compute_regression_metrics(y_te, preds)
        scores.append(
            LoroRegionScore(
                region=region,
                model_name=model_name,
                n_countries=int(test_fold["Area"].nunique()),
                n_samples=metrics.n_samples,
                rmse=metrics.rmse,
                mae=metrics.mae,
                mape_pct=metrics.mape_pct,
                r2=metrics.r2,
            )
        )
    return tuple(scores)


def _assemble_benchmark(
    identity: tuple[str, str],
    frames: TemporalSplitFrames,
    preds_and_loro: tuple[np.ndarray, np.ndarray, tuple[LoroRegionScore, ...]],
) -> ModelBenchmarkResult:
    model_name, display_label = identity
    cal_pred, test_pred, loro_scores = preds_and_loro
    y_cal = frames.calibration_df[frames.target_column].to_numpy(dtype=float)
    y_te = frames.test_df[frames.target_column].to_numpy(dtype=float)
    return ModelBenchmarkResult(
        model_name=model_name,
        display_label=display_label,
        calibration_metrics=compute_regression_metrics(y_cal, cal_pred),
        test_metrics=compute_regression_metrics(y_te, test_pred),
        per_crop_test_metrics=compute_per_crop_metrics(frames.test_df, y_te, test_pred),
        loro_scores=loro_scores,
        loro_mean_r2=float(np.mean([s.r2 for s in loro_scores])),
        loro_mean_rmse=float(np.mean([s.rmse for s in loro_scores])),
        cal_predictions=cal_pred,
        test_predictions=test_pred,
    )


def run_panel_model_benchmarks(
    frames: TemporalSplitFrames, hp: ModelHyperparamConfig
) -> PanelModelingOutput:
    """Fit Ridge FE, LightGBM, and HistGB on chronological splits plus LORO audit.

    Model selection uses strictly the 2009-2010 calibration RMSE so the 2011-2013
    test horizon remains an untouched out-of-time evaluation benchmark.
    """
    ridge_cal, ridge_te, best_alpha = _fit_best_ridge_fe(frames, hp)
    lgbm_model, lgbm_x_te, (lgbm_cal, lgbm_te) = _fit_lightgbm_temporal(frames, hp)
    hgb_cal, hgb_te = _fit_hgb_temporal(frames, hp)

    loro_ridge = _run_loro_audit("ridge_fixed_effects", frames.full_panel, hp)
    loro_lgbm = _run_loro_audit("lightgbm", frames.full_panel, hp)
    loro_hgb = _run_loro_audit("hist_gradient_boosting", frames.full_panel, hp)

    benchmarks = (
        _assemble_benchmark(
            ("ridge_fixed_effects", f"Ridge Fixed-Effects (alpha={best_alpha:g})"),
            frames,
            (ridge_cal, ridge_te, loro_ridge),
        ),
        _assemble_benchmark(
            ("lightgbm", "LightGBM Regressor"),
            frames,
            (lgbm_cal, lgbm_te, loro_lgbm),
        ),
        _assemble_benchmark(
            ("hist_gradient_boosting", "HistGradientBoosting"),
            frames,
            (hgb_cal, hgb_te, loro_hgb),
        ),
    )
    best = min(benchmarks, key=lambda b: b.calibration_metrics.rmse)
    return PanelModelingOutput(
        best_model_name=best.model_name,
        benchmarks=benchmarks,
        lgbm_model=lgbm_model,
        lgbm_test_features=lgbm_x_te,
    )
