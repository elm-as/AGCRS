"""Non-linear climate sensitivity econometrics (+1°C, +2°C warming, -20% drought)."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

from src.core.config import ConformalAndInsuranceConfig, ModelHyperparamConfig
from src.core.contracts import (
    ClimateEconometricsSummary,
    CropClimateShockImpact,
    TemporalSplitFrames,
    ThermalCurvePoint,
)
from src.core.region_taxonomy import is_african_tropical_staple

_THERMAL_GRID_C: tuple[float, ...] = tuple(float(t) for t in range(10, 33, 2))


@dataclass(frozen=True)
class _CropEconometricFit:
    model: Ridge
    region_encoder: OneHotEncoder
    temp_mean: float
    log_rain_mean: float


def _build_crop_climate_matrix(
    df: pd.DataFrame,
    region_encoder: OneHotEncoder,
    centers: tuple[float, float],
) -> np.ndarray:
    temp_c = df["avg_temp"].to_numpy(dtype=float) - centers[0]
    log_rain = np.log1p(df["rainfall_mm"].to_numpy(dtype=float)) - centers[1]
    pest = np.log1p(df["pesticides_tonnes"].to_numpy(dtype=float))
    year_c = df["year_centered"].to_numpy(dtype=float)

    num_block = np.column_stack(
        [
            temp_c,
            temp_c**2,
            log_rain,
            log_rain**2,
            temp_c * log_rain,
            pest,
            year_c,
        ]
    )
    reg_block = region_encoder.transform(df[["Region"]])
    return np.hstack([num_block, reg_block])


def _fit_single_crop_model(crop_train_df: pd.DataFrame) -> _CropEconometricFit:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(crop_train_df[["Region"]])
    t_mean = float(crop_train_df["avg_temp"].mean())
    lr_mean = float(np.log1p(crop_train_df["rainfall_mm"]).mean())
    x_mat = _build_crop_climate_matrix(crop_train_df, encoder, (t_mean, lr_mean))
    y_log = np.log1p(crop_train_df["yield_hg_ha"].to_numpy(dtype=float))
    model = Ridge(alpha=2.0)
    model.fit(x_mat, y_log)
    return _CropEconometricFit(
        model=model,
        region_encoder=encoder,
        temp_mean=t_mean,
        log_rain_mean=lr_mean,
    )


def _predict_counterfactual_pct(
    fit: _CropEconometricFit,
    base_df: pd.DataFrame,
    shock: tuple[float, float],
) -> float:
    delta_temp_c, rain_multiplier = shock
    centers = (fit.temp_mean, fit.log_rain_mean)
    x_base = _build_crop_climate_matrix(base_df, fit.region_encoder, centers)
    shocked = base_df.copy()
    shocked["avg_temp"] = shocked["avg_temp"] + delta_temp_c
    shocked["rainfall_mm"] = np.maximum(shocked["rainfall_mm"] * rain_multiplier, 1.0)
    x_shock = _build_crop_climate_matrix(shocked, fit.region_encoder, centers)
    delta_log = fit.model.predict(x_shock) - fit.model.predict(x_base)
    return float(np.mean(np.expm1(delta_log)) * 100.0)


def _bootstrap_crop_shocks(
    crop_df: pd.DataFrame,
    n_replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    countries = sorted(crop_df["Area"].unique().tolist())
    n_countries = len(countries)
    by_country = {c: crop_df.loc[crop_df["Area"] == c] for c in countries}

    boot_1c = np.zeros(n_replicates, dtype=float)
    boot_2c = np.zeros(n_replicates, dtype=float)
    boot_drought = np.zeros(n_replicates, dtype=float)

    for rep in range(n_replicates):
        sampled = rng.choice(countries, size=n_countries, replace=True)
        boot_df = pd.concat([by_country[c] for c in sampled], ignore_index=True)
        fit = _fit_single_crop_model(boot_df)
        boot_1c[rep] = _predict_counterfactual_pct(fit, crop_df, (1.0, 1.0))
        boot_2c[rep] = _predict_counterfactual_pct(fit, crop_df, (2.0, 1.0))
        boot_drought[rep] = _predict_counterfactual_pct(fit, crop_df, (0.0, 0.80))
    return boot_1c, boot_2c, boot_drought


def _build_thermal_curve(
    fit: _CropEconometricFit, crop_df: pd.DataFrame
) -> tuple[ThermalCurvePoint, ...]:
    centers = (fit.temp_mean, fit.log_rain_mean)
    baseline_row = crop_df.iloc[[0]].copy()
    baseline_row["avg_temp"] = fit.temp_mean
    baseline_row["rainfall_mm"] = float(crop_df["rainfall_mm"].mean())
    baseline_row["pesticides_tonnes"] = float(crop_df["pesticides_tonnes"].median())
    baseline_row["year_centered"] = 0.0
    baseline_row["Region"] = str(crop_df["Region"].mode().iloc[0])
    base_log = float(
        fit.model.predict(
            _build_crop_climate_matrix(baseline_row, fit.region_encoder, centers)
        )[0]
    )

    points: list[ThermalCurvePoint] = []
    for temp_val in _THERMAL_GRID_C:
        probe = baseline_row.copy()
        probe["avg_temp"] = temp_val
        pred_log = float(
            fit.model.predict(
                _build_crop_climate_matrix(probe, fit.region_encoder, centers)
            )[0]
        )
        points.append(
            ThermalCurvePoint(
                temp_c=temp_val,
                relative_yield_pct=float(np.expm1(pred_log - base_log) * 100.0),
            )
        )
    return tuple(points)


def _fit_within_country_fe_warming(
    train_df: pd.DataFrame,
) -> tuple[float, dict[str, tuple[float, float]]]:
    """Cross-check pure within-country +1°C/+2°C warming via Country×Crop FE."""
    crops = sorted(train_df["Item"].unique().tolist())
    fe_enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    x_fe = fe_enc.fit_transform(train_df[["country_crop_fe"]])

    t_center = float(train_df["avg_temp"].mean())
    t_val = train_df["avg_temp"].to_numpy(dtype=float) - t_center
    pest = np.log1p(train_df["pesticides_tonnes"].to_numpy(dtype=float))
    yr = train_df["year_centered"].to_numpy(dtype=float)

    crop_cols: list[np.ndarray] = []
    for crop in crops:
        mask = (train_df["Item"] == crop).to_numpy(dtype=float)
        crop_cols.append(mask * t_val)
        crop_cols.append(mask * (t_val**2))

    from scipy import sparse

    x_num = np.column_stack(crop_cols + [pest, yr])
    x_all = sparse.hstack([x_fe, sparse.csr_matrix(x_num)], format="csr")
    y_log = np.log1p(train_df["yield_hg_ha"].to_numpy(dtype=float))

    model = Ridge(alpha=5.0)
    model.fit(x_all, y_log)
    r2_within = float(r2_score(y_log, model.predict(x_all)))

    n_fe = x_fe.shape[1]
    coef = model.coef_
    within_shocks: dict[str, tuple[float, float]] = {}
    for idx, crop in enumerate(crops):
        b1 = float(coef[n_fe + 2 * idx])
        b2 = float(coef[n_fe + 2 * idx + 1])
        sub_t = train_df.loc[train_df["Item"] == crop, "avg_temp"].to_numpy(dtype=float)
        t_c = sub_t - t_center
        d1 = b1 * 1.0 + b2 * ((t_c + 1.0) ** 2 - t_c**2)
        d2 = b1 * 2.0 + b2 * ((t_c + 2.0) ** 2 - t_c**2)
        within_shocks[crop] = (
            float(np.mean(np.expm1(d1)) * 100.0),
            float(np.mean(np.expm1(d2)) * 100.0),
        )
    return r2_within, within_shocks


def run_climate_sensitivity_econometrics(
    frames: TemporalSplitFrames,
    hp: ModelHyperparamConfig,
    risk_cfg: ConformalAndInsuranceConfig,
) -> ClimateEconometricsSummary:
    """Estimate non-linear per-crop warming (+1°C/+2°C) and drought (-20%) elasticities.

    Models are estimated strictly on the 1990-2008 training panel with country-block
    bootstrap 95% confidence intervals and validated out-of-time on 2011-2013 log-yield.
    """
    train_df = frames.train_df
    test_df = frames.test_df
    fe_r2, fe_within_map = _fit_within_country_fe_warming(train_df)

    tr_actual_all: list[np.ndarray] = []
    tr_pred_all: list[np.ndarray] = []
    te_actual_all: list[np.ndarray] = []
    te_pred_all: list[np.ndarray] = []
    impacts: list[CropClimateShockImpact] = []

    drought_mult = 1.0 + risk_cfg.drought_shock_pct
    for idx, crop in enumerate(sorted(train_df["Item"].unique().tolist())):
        c_tr = train_df.loc[train_df["Item"] == crop].reset_index(drop=True)
        c_te = test_df.loc[test_df["Item"] == crop].reset_index(drop=True)
        fit = _fit_single_crop_model(c_tr)
        centers = (fit.temp_mean, fit.log_rain_mean)

        tr_actual_all.append(np.log1p(c_tr["yield_hg_ha"].to_numpy(dtype=float)))
        tr_pred_all.append(
            fit.model.predict(_build_crop_climate_matrix(c_tr, fit.region_encoder, centers))
        )
        if not c_te.empty:
            te_actual_all.append(np.log1p(c_te["yield_hg_ha"].to_numpy(dtype=float)))
            te_pred_all.append(
                fit.model.predict(
                    _build_crop_climate_matrix(c_te, fit.region_encoder, centers)
                )
            )

        s1 = _predict_counterfactual_pct(fit, c_tr, (1.0, 1.0))
        s2 = _predict_counterfactual_pct(fit, c_tr, (2.0, 1.0))
        sd = _predict_counterfactual_pct(fit, c_tr, (0.0, drought_mult))
        b1, b2, bd = _bootstrap_crop_shocks(
            c_tr, hp.bootstrap_replicates, hp.random_seed + idx
        )
        fe_1c, fe_2c = fe_within_map[crop]
        impacts.append(
            CropClimateShockImpact(
                crop=str(crop),
                is_african_tropical_staple=is_african_tropical_staple(str(crop)),
                n_train_observations=int(len(c_tr)),
                mean_temp_c=fit.temp_mean,
                mean_rainfall_mm=float(c_tr["rainfall_mm"].mean()),
                shock_plus_1c_pct=s1,
                shock_plus_1c_ci_low=float(np.percentile(b1, 2.5)),
                shock_plus_1c_ci_high=float(np.percentile(b1, 97.5)),
                shock_plus_2c_pct=s2,
                shock_plus_2c_ci_low=float(np.percentile(b2, 2.5)),
                shock_plus_2c_ci_high=float(np.percentile(b2, 97.5)),
                shock_drought_minus_20pct=sd,
                shock_drought_ci_low=float(np.percentile(bd, 2.5)),
                shock_drought_ci_high=float(np.percentile(bd, 97.5)),
                fe_within_plus_1c_pct=fe_1c,
                fe_within_plus_2c_pct=fe_2c,
                thermal_curve=_build_thermal_curve(fit, c_tr),
            )
        )

    african = [c for c in impacts if c.is_african_tropical_staple]
    return ClimateEconometricsSummary(
        model_r2_train_log=float(
            r2_score(np.concatenate(tr_actual_all), np.concatenate(tr_pred_all))
        ),
        model_r2_test_log=float(
            r2_score(np.concatenate(te_actual_all), np.concatenate(te_pred_all))
        ),
        fe_within_r2_train_log=fe_r2,
        global_mean_plus_1c_pct=float(np.mean([c.shock_plus_1c_pct for c in impacts])),
        global_mean_plus_2c_pct=float(np.mean([c.shock_plus_2c_pct for c in impacts])),
        global_mean_drought_20pct=float(
            np.mean([c.shock_drought_minus_20pct for c in impacts])
        ),
        african_staples_mean_plus_2c_pct=float(
            np.mean([c.shock_plus_2c_pct for c in african])
        ),
        african_staples_mean_drought_20pct=float(
            np.mean([c.shock_drought_minus_20pct for c in african])
        ),
        crop_impacts=tuple(impacts),
    )
