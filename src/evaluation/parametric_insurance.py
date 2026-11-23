"""Parametric drought/heatwave crop insurance pricing and basis risk simulator."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.core.config import ConformalAndInsuranceConfig
from src.core.contracts import (
    InsuranceSegmentSummary,
    InsuranceThresholdPoint,
    ParametricInsuranceSummary,
)


@dataclass(frozen=True)
class InsuranceSimulationInput:
    """Actuarial input frames for leak-free pre-test pricing and 2011-2013 evaluation."""

    pretest_df: pd.DataFrame
    test_df: pd.DataFrame
    test_predicted_yield: np.ndarray


def _compute_payout_ratio(
    yield_values: np.ndarray, baseline_values: np.ndarray, trigger_ratio: float
) -> np.ndarray:
    strike = np.maximum(trigger_ratio * baseline_values, 1.0)
    shortfall = np.maximum(0.0, strike - yield_values)
    return np.clip(shortfall / strike, 0.0, 1.0)


def _evaluate_threshold_point(
    sim_input: InsuranceSimulationInput,
    trigger_ratio: float,
    loading_factor: float,
) -> InsuranceThresholdPoint:
    pre_y = sim_input.pretest_df["yield_hg_ha"].to_numpy(dtype=float)
    pre_base = sim_input.pretest_df["local_baseline_yield"].to_numpy(dtype=float)
    pre_payouts = _compute_payout_ratio(pre_y, pre_base, trigger_ratio)
    pure_premium = max(float(np.mean(pre_payouts)), 1e-6)
    commercial_premium = pure_premium * (1.0 + loading_factor)

    test_y = sim_input.test_df["yield_hg_ha"].to_numpy(dtype=float)
    test_base = sim_input.test_df["local_baseline_yield"].to_numpy(dtype=float)
    test_pred = np.asarray(sim_input.test_predicted_yield, dtype=float)

    actual_payouts = _compute_payout_ratio(test_y, test_base, trigger_ratio)
    param_payouts = _compute_payout_ratio(test_pred, test_base, trigger_ratio)

    actual_event = test_y < (trigger_ratio * test_base)
    param_event = test_pred < (trigger_ratio * test_base)

    tp = int(np.sum(actual_event & param_event))
    fn = int(np.sum(actual_event & (~param_event)))
    fp = int(np.sum((~actual_event) & param_event))
    n_obs = int(test_y.size)

    return InsuranceThresholdPoint(
        trigger_ratio=float(trigger_ratio),
        actual_shortfall_rate=float(np.mean(actual_event)),
        parametric_trigger_rate=float(np.mean(param_event)),
        pure_premium_rate_pct=100.0 * pure_premium,
        commercial_premium_rate_pct=100.0 * commercial_premium,
        actual_loss_ratio=float(np.mean(actual_payouts) / commercial_premium),
        parametric_loss_ratio=float(np.mean(param_payouts) / commercial_premium),
        fn_basis_risk_rate=float(fn / max(n_obs, 1)),
        fp_basis_risk_rate=float(fp / max(n_obs, 1)),
        shortfall_recall=float(tp / max(tp + fn, 1)),
        shortfall_precision=float(tp / max(tp + fp, 1)),
    )


def _evaluate_segments(
    sim_input: InsuranceSimulationInput,
    column_name: str,
    config: ConformalAndInsuranceConfig,
) -> tuple[InsuranceSegmentSummary, ...]:
    trigger_ratio = config.trigger_baseline_ratio
    loading = config.insurer_loading_factor
    global_pre_y = sim_input.pretest_df["yield_hg_ha"].to_numpy(dtype=float)
    global_pre_b = sim_input.pretest_df["local_baseline_yield"].to_numpy(dtype=float)
    global_pure = max(
        float(np.mean(_compute_payout_ratio(global_pre_y, global_pre_b, trigger_ratio))),
        1e-6,
    )

    summaries: list[InsuranceSegmentSummary] = []
    for seg_val in sorted(sim_input.test_df[column_name].unique().tolist()):
        pre_mask = (sim_input.pretest_df[column_name] == seg_val).to_numpy()
        test_mask = (sim_input.test_df[column_name] == seg_val).to_numpy()
        seg_pre = sim_input.pretest_df.loc[pre_mask]
        seg_test = sim_input.test_df.loc[test_mask]
        seg_pred = sim_input.test_predicted_yield[test_mask]

        seg_pure = (
            float(
                np.mean(
                    _compute_payout_ratio(
                        seg_pre["yield_hg_ha"].to_numpy(dtype=float),
                        seg_pre["local_baseline_yield"].to_numpy(dtype=float),
                        trigger_ratio,
                    )
                )
            )
            if len(seg_pre) > 0
            else global_pure
        )
        comm_prem = max(seg_pure, 0.001) * (1.0 + loading)

        test_y = seg_test["yield_hg_ha"].to_numpy(dtype=float)
        test_b = seg_test["local_baseline_yield"].to_numpy(dtype=float)
        param_payouts = _compute_payout_ratio(seg_pred, test_b, trigger_ratio)
        actual_event = test_y < (trigger_ratio * test_b)
        param_event = seg_pred < (trigger_ratio * test_b)

        summaries.append(
            InsuranceSegmentSummary(
                segment_type=column_name,
                segment_name=str(seg_val),
                n_policies=int(test_y.size),
                actual_shortfall_rate=float(np.mean(actual_event)),
                parametric_trigger_rate=float(np.mean(param_event)),
                pure_premium_rate_pct=100.0 * seg_pure,
                parametric_loss_ratio=float(np.mean(param_payouts) / comm_prem),
                fn_basis_risk_rate=float(np.mean(actual_event & (~param_event))),
                fp_basis_risk_rate=float(np.mean((~actual_event) & param_event)),
            )
        )
    return tuple(summaries)


def evaluate_parametric_insurance(
    sim_input: InsuranceSimulationInput, config: ConformalAndInsuranceConfig
) -> ParametricInsuranceSummary:
    """Evaluate parametric drought/heatwave crop insurance pricing and basis risk.

    Pure premiums are actuarially priced strictly on the pre-test 1990-2010 loss
    distribution, then evaluated out-of-time on the 2011-2013 harvest portfolio.
    """
    sweep = tuple(
        _evaluate_threshold_point(sim_input, ratio, config.insurer_loading_factor)
        for ratio in config.trigger_grid
    )
    at_trigger = _evaluate_threshold_point(
        sim_input, config.trigger_baseline_ratio, config.insurer_loading_factor
    )
    by_crop = _evaluate_segments(sim_input, "Item", config)
    by_region = _evaluate_segments(sim_input, "Region", config)
    ssa_summary = next(
        seg for seg in by_region if seg.segment_name == "Sub-Saharan Africa"
    )
    return ParametricInsuranceSummary(
        trigger_ratio=config.trigger_baseline_ratio,
        loading_factor=config.insurer_loading_factor,
        portfolio_at_trigger=at_trigger,
        threshold_sweep=sweep,
        by_crop=by_crop,
        by_region=by_region,
        ssa_summary=ssa_summary,
    )
