"""CLI entrypoint orchestrating the end-to-end AGCRS agricultural climate pipeline."""

import argparse
from dataclasses import replace
from pathlib import Path
import time
import pandas as pd

from src.core.config import ModelHyperparamConfig, PipelineConfig
from src.core.contracts import AgcrsReportPayload
from src.evaluation.conformal_intervals import (
    ConformalCalibrationInput,
    compute_adaptive_split_conformal,
    compute_standard_split_conformal,
)
from src.evaluation.parametric_insurance import (
    InsuranceSimulationInput,
    evaluate_parametric_insurance,
)
from src.evaluation.shap_drivers import compute_tree_shap_drivers
from src.features.temporal_feature_builder import (
    FEATURE_DISPLAY_LABELS,
    build_temporal_panel_features,
)
from src.ingestion.panel_loader import (
    deduplicate_and_audit_panel,
    load_raw_yield_csv,
)
from src.modeling.climate_econometrics import run_climate_sensitivity_econometrics
from src.modeling.panel_models import run_panel_model_benchmarks
from src.visualization.agronomy_dashboard import build_agronomy_dashboard_html
from src.visualization.agronomy_plots import render_all_agronomy_figures
from src.visualization.results_exporter import export_results_json


def run_agcrs_study(config: PipelineConfig) -> AgcrsReportPayload:
    """Execute panel deduplication, OOT/LORO modeling, econometrics, conformal & insurance."""
    start_time = time.perf_counter()

    print("[1/6] Auditing and deduplicating FAO/World Bank crop yield panel...", flush=True)
    raw_df = load_raw_yield_csv(config.paths.raw_csv_path)
    clean_panel, audit_stats = deduplicate_and_audit_panel(raw_df, config.split)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    clean_panel.to_parquet(
        config.paths.processed_dir / "clean_unique_panel.parquet", index=False
    )

    print("[2/6] Building strictly past-only features & chronological splits...", flush=True)
    frames = build_temporal_panel_features(clean_panel, config.split)

    print("[3/6] Benchmarking Ridge FE, LightGBM & HistGB (OOT + LORO)...", flush=True)
    modeling_out = run_panel_model_benchmarks(frames, config.hyperparams)
    best_bench = next(
        b for b in modeling_out.benchmarks if b.model_name == modeling_out.best_model_name
    )

    print("[4/6] Computing split-conformal intervals, climate econometrics & insurance...", flush=True)
    conf_input = ConformalCalibrationInput(
        cal_actual=frames.calibration_df[frames.target_column].to_numpy(dtype=float),
        cal_pred=best_bench.cal_predictions,
        cal_volatility=frames.calibration_df["yield_roll3_std"].to_numpy(dtype=float),
        test_actual=frames.test_df[frames.target_column].to_numpy(dtype=float),
        test_pred=best_bench.test_predictions,
        test_volatility=frames.test_df["yield_roll3_std"].to_numpy(dtype=float),
        test_crops=frames.test_df["Item"],
        alpha=config.risk.conformal_alpha,
    )
    conformal_std = compute_standard_split_conformal(conf_input)
    conformal_adapt = compute_adaptive_split_conformal(conf_input)

    climate_econ = run_climate_sensitivity_econometrics(
        frames, config.hyperparams, config.risk
    )

    pretest_df = pd.concat(
        [frames.train_df, frames.calibration_df], ignore_index=True
    )
    ins_input = InsuranceSimulationInput(
        pretest_df=pretest_df,
        test_df=frames.test_df,
        test_predicted_yield=best_bench.test_predictions,
    )
    insurance_summary = evaluate_parametric_insurance(ins_input, config.risk)

    print("[5/6] Computing TreeSHAP yield drivers & rendering visual artifacts...", flush=True)
    shap_drivers = compute_tree_shap_drivers(
        modeling_out.lgbm_model,
        modeling_out.lgbm_test_features,
        FEATURE_DISPLAY_LABELS,
    )

    payload = AgcrsReportPayload(
        audit_stats=audit_stats,
        best_model_name=modeling_out.best_model_name,
        model_benchmarks=modeling_out.benchmarks,
        conformal_standard=conformal_std,
        conformal_adaptive=conformal_adapt,
        climate_econometrics=climate_econ,
        insurance_summary=insurance_summary,
        shap_drivers=shap_drivers,
        test_metadata_df=frames.test_df[["Area", "Item", "Region", "Year"]],
        test_actual_yield=frames.test_df[frames.target_column].to_numpy(dtype=float),
        best_test_predictions=best_bench.test_predictions,
    )

    render_all_agronomy_figures(payload, config.paths.images_dir)
    build_agronomy_dashboard_html(
        payload, config.paths.artifacts_dir / "dashboard.html"
    )
    export_results_json(payload, config.paths.artifacts_dir / "results.json")

    elapsed = time.perf_counter() - start_time
    print(f"[6/6] AGCRS pipeline completed cleanly in {elapsed:.1f}s.", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AGCRS: Global Agricultural Yield, Climate Shocks & Parametric Insurance"
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Use reduced tree/bootstrap counts for rapid smoke verification.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint building default configuration and running the AGCRS study."""
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    config = PipelineConfig.create_default(project_root)
    if args.fast_smoke:
        config = replace(config, hyperparams=ModelHyperparamConfig.fast_smoke())
    run_agcrs_study(config)


if __name__ == "__main__":
    main()
