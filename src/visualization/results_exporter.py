"""JSON serialization of the complete AGCRS evaluation and risk payload."""

from dataclasses import asdict
import json
from pathlib import Path

from src.core.contracts import (
    AgcrsReportPayload,
    ConformalIntervalResult,
    ModelBenchmarkResult,
)


def _serialize_conformal(res: ConformalIntervalResult) -> dict[str, object]:
    return {
        "method_name": res.method_name,
        "nominal_coverage": round(res.nominal_coverage, 4),
        "empirical_coverage": round(res.empirical_coverage, 4),
        "conformal_quantile": round(res.conformal_quantile, 4),
        "mean_width_hg_ha": round(res.mean_width_hg_ha, 2),
        "median_width_hg_ha": round(res.median_width_hg_ha, 2),
        "relative_width_pct": round(res.relative_width_pct, 2),
        "per_crop_coverage": [asdict(c) for c in res.per_crop_coverage],
    }


def _serialize_benchmark(bench: ModelBenchmarkResult) -> dict[str, object]:
    return {
        "model_name": bench.model_name,
        "display_label": bench.display_label,
        "calibration_metrics": asdict(bench.calibration_metrics),
        "test_metrics": asdict(bench.test_metrics),
        "per_crop_test_metrics": [asdict(c) for c in bench.per_crop_test_metrics],
        "loro_mean_r2": round(bench.loro_mean_r2, 4),
        "loro_mean_rmse": round(bench.loro_mean_rmse, 2),
        "loro_scores": [asdict(s) for s in bench.loro_scores],
    }


def serialize_agcrs_payload(payload: AgcrsReportPayload) -> dict[str, object]:
    """Convert `AgcrsReportPayload` into a deterministic JSON-serializable dictionary."""
    audit_dict = asdict(payload.audit_stats)
    audit_dict["missing_years"] = list(payload.audit_stats.missing_years)
    audit_dict["top_duplicated_countries"] = [
        {"country": name, "max_station_rows": count}
        for name, count in payload.audit_stats.top_duplicated_countries
    ]

    best_bench = next(
        b for b in payload.model_benchmarks if b.model_name == payload.best_model_name
    )

    return {
        "project": "AGCRS — Global Agricultural Yield, Climate Shocks & Parametric Insurance Risk",
        "panel_audit": audit_dict,
        "best_model_name": payload.best_model_name,
        "best_model_test_metrics": asdict(best_bench.test_metrics),
        "model_benchmarks": [
            _serialize_benchmark(b) for b in payload.model_benchmarks
        ],
        "conformal_prediction": {
            "standard": _serialize_conformal(payload.conformal_standard),
            "adaptive": _serialize_conformal(payload.conformal_adaptive),
        },
        "climate_econometrics": asdict(payload.climate_econometrics),
        "parametric_insurance": asdict(payload.insurance_summary),
        "shap_drivers": [asdict(d) for d in payload.shap_drivers],
    }


def export_results_json(payload: AgcrsReportPayload, output_path: Path) -> Path:
    """Write the complete AGCRS metrics and stress-test report to `output_path`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_agcrs_payload(payload)
    output_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
