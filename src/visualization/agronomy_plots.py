"""Publication-grade 300 DPI editorial agronomy & climate risk figures."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.contracts import AgcrsReportPayload

_BG = "#FAF9F5"
_PANEL = "#FFFFFF"
_INK = "#1C2526"
_MUTED = "#5C6B73"
_GRID = "#E2E4DF"
_FOREST = "#1B4332"
_SAGE = "#40916C"
_AMBER = "#B45309"
_OCHRE = "#D97706"
_CRIMSON = "#991B1B"
_SLATE = "#334155"


def _apply_editorial_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": _BG,
            "axes.facecolor": _PANEL,
            "axes.edgecolor": "#CBD2CD",
            "axes.labelcolor": _INK,
            "text.color": _INK,
            "xtick.color": _INK,
            "ytick.color": _INK,
            "grid.color": _GRID,
            "grid.linestyle": "--",
            "grid.alpha": 0.85,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def _clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)


def _plot_temporal_and_conformal(
    payload: AgcrsReportPayload, output_path: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
    fig.patch.set_facecolor(_BG)

    benches = payload.model_benchmarks
    labels = [b.display_label.split(" (")[0] for b in benches]
    oot_rmse = [b.test_metrics.rmse / 1000.0 for b in benches]
    loro_rmse = [b.loro_mean_rmse / 1000.0 for b in benches]
    oot_r2 = [b.test_metrics.r2 for b in benches]

    x = np.arange(len(labels))
    w = 0.34
    bars1 = ax1.bar(x - w / 2, oot_rmse, width=w, color=_FOREST, label="Temporal OOT (2011-2013)", zorder=3)
    bars2 = ax1.bar(x + w / 2, loro_rmse, width=w, color=_AMBER, label="LORO Spatial Holdout", zorder=3)
    for bar, r2_val in zip(bars1, oot_r2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.1f}k\n(R²={r2_val:.3f})", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=_FOREST)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.1f}k", ha="center", va="bottom", fontsize=8.2, color=_AMBER)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("RMSE (×1,000 hg/ha)")
    ax1.set_title("A. Out-of-Time (2011-2013) vs LORO Spatial Benchmark", pad=12)
    ax1.set_ylim(0, max(max(oot_rmse), max(loro_rmse)) * 1.24)
    ax1.legend(frameon=False, loc="upper right", fontsize=8.5)
    _clean_axes(ax1, "y")

    cov_rows = sorted(payload.conformal_adaptive.per_crop_coverage, key=lambda c: c.empirical_coverage)
    crops = [c.crop for c in cov_rows]
    cov_adapt = [c.empirical_coverage * 100.0 for c in cov_rows]
    std_map = {c.crop: c.empirical_coverage * 100.0 for c in payload.conformal_standard.per_crop_coverage}
    cov_std = [std_map[c] for c in crops]
    y_pos = np.arange(len(crops))

    ax2.barh(y_pos + 0.18, cov_adapt, height=0.35, color=_SAGE, label="Adaptive Split-Conformal", zorder=3)
    ax2.barh(y_pos - 0.18, cov_std, height=0.35, color=_SLATE, alpha=0.72, label="Standard Split-Conformal", zorder=3)
    ax2.axvline(90.0, color=_CRIMSON, linestyle="--", linewidth=1.6, label="Nominal 90% Target", zorder=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(crops, fontsize=8.5)
    ax2.set_xlabel("Empirical 2011-2013 Coverage (%)")
    ax2.set_xlim(55, 105)
    ax2.set_ylim(-0.6, len(crops) + 0.9)
    ax2.set_title(f"B. Split-Conformal 90% Coverage (Adaptive: {payload.conformal_adaptive.empirical_coverage:.1%})", pad=12)
    ax2.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.0)
    _clean_axes(ax2, "x")

    fig.suptitle("AGCRS — Temporal Forecasting Benchmark & Distribution-Free Conformal Uncertainty", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_climate_shock_stress_test(
    payload: AgcrsReportPayload, output_path: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.9), dpi=300)
    fig.patch.set_facecolor(_BG)

    impacts = sorted(
        payload.climate_econometrics.crop_impacts,
        key=lambda c: c.shock_plus_2c_pct,
    )
    y_pos = np.arange(len(impacts))
    crop_labels = [
        f"{c.crop} *" if c.is_african_tropical_staple else c.crop for c in impacts
    ]
    s2 = [c.shock_plus_2c_pct for c in impacts]
    sd = [c.shock_drought_minus_20pct for c in impacts]

    ax1.barh(
        y_pos + 0.18, s2, height=0.35, color=_CRIMSON, label="+2°C Warming Shock", zorder=3
    )
    ax1.barh(
        y_pos - 0.18, sd, height=0.35, color=_OCHRE, label="-20% Drought Shock", zorder=3
    )
    ax1.axvline(0.0, color=_INK, linewidth=1.1, zorder=4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(crop_labels, fontsize=8.5)
    ax1.set_xlabel("Estimated Yield Elasticity (% Change)")
    ax1.set_title("A. Per-Crop Climate Shock Elasticity (* African/Tropical Staple)", pad=12)
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="x", zorder=0)

    featured_crops = ["Maize", "Cassava", "Sorghum", "Rice, paddy", "Wheat"]
    palette = [_CRIMSON, _FOREST, _AMBER, _SAGE, _SLATE]
    impact_map = {c.crop: c for c in impacts}
    for crop_name, color in zip(featured_crops, palette):
        if crop_name not in impact_map:
            continue
        curve = impact_map[crop_name].thermal_curve
        temps = [pt.temp_c for pt in curve]
        rel = [pt.relative_yield_pct for pt in curve]
        ax2.plot(temps, rel, marker="o", markersize=4.5, linewidth=2.1, color=color, label=crop_name)

    ax2.axhline(0.0, color=_MUTED, linestyle="--", linewidth=1.1)
    ax2.set_xlabel("Annual Mean Temperature (°C)")
    ax2.set_ylabel("Yield Departure vs Crop Mean Temp (%)")
    ax2.set_title("B. Non-Linear Thermal Response Curves (Semi-Log Quadratic FE)", pad=12)
    ax2.legend(frameon=False, loc="best", fontsize=8.5)
    _clean_axes(ax2)

    fig.suptitle(
        "AGCRS — Econometric Climate Sensitivity: Warming (+1°C/+2°C) & Drought (-20%) Shocks",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_shap_and_dedup_audit(
    payload: AgcrsReportPayload, output_path: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
    fig.patch.set_facecolor(_BG)

    top_drivers = list(payload.shap_drivers[:12])[::-1]
    labels = [d.display_label for d in top_drivers]
    vals = [d.mean_abs_shap_hg_ha / 1000.0 for d in top_drivers]
    colors = [
        _FOREST if d.mean_signed_correlation >= 0 else _CRIMSON for d in top_drivers
    ]
    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, vals, color=colors, height=0.65, zorder=3)
    for idx, (val, drv) in enumerate(zip(vals, top_drivers)):
        ax1.text(
            val + max(vals) * 0.015,
            idx,
            f"{val:.2f}k ({drv.share_pct:.1f}%)",
            va="center",
            fontsize=8.0,
            color=_INK,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.4)
    ax1.set_xlabel("Mean |SHAP| Impact (×1,000 hg/ha)")
    ax1.set_xlim(0, max(vals) * 1.28)
    ax1.set_title("A. LightGBM TreeSHAP Yield Drivers (Green: +, Red: -)", pad=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="x", zorder=0)

    audit = payload.audit_stats
    top_dup = list(audit.top_duplicated_countries)[::-1]
    c_names = [c for c, _ in top_dup]
    c_mult = [m for _, m in top_dup]
    y2 = np.arange(len(c_names))
    ax2.barh(y2, c_mult, color=_AMBER, height=0.62, zorder=3)
    for idx, m in enumerate(c_mult):
        ax2.text(m + 0.35, idx, f"{m}× rows/crop-yr", va="center", fontsize=8.4, fontweight="bold")
    ax2.axvline(1.0, color=_FOREST, linestyle="--", linewidth=1.4, label="Clean 1× Panel Key")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(c_names, fontsize=8.8)
    ax2.set_xlabel("Max Multi-Station Duplication Factor in Raw yield_df.csv")
    ax2.set_xlim(0, max(c_mult) * 1.32)
    ax2.set_title(
        f"B. Panel Integrity Audit ({audit.raw_row_count:,} → {audit.unique_panel_rows:,} Rows, -{audit.duplicate_inflation_pct:.1f}%)",
        pad=12,
    )
    ax2.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="x", zorder=0)

    fig.suptitle(
        "AGCRS — Explainable Yield Drivers (TreeSHAP) & Multi-Station Panel Deduplication Audit",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_parametric_insurance_risk(
    payload: AgcrsReportPayload, output_path: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
    fig.patch.set_facecolor(_BG)

    regions = sorted(
        payload.insurance_summary.by_region,
        key=lambda r: r.actual_shortfall_rate,
        reverse=True,
    )
    reg_labels = [r.segment_name.replace(" & ", "\n& ") for r in regions]
    x = np.arange(len(regions))
    w = 0.27
    pure_p = [r.pure_premium_rate_pct for r in regions]
    actual_sf = [r.actual_shortfall_rate * 100.0 for r in regions]
    fn_br = [r.fn_basis_risk_rate * 100.0 for r in regions]

    ax1.bar(x - w, actual_sf, width=w, color=_CRIMSON, label="Actual Shortfall <80% (%)", zorder=3)
    ax1.bar(x, pure_p, width=w, color=_FOREST, label="Pre-Test Pure Premium (%)", zorder=3)
    ax1.bar(x + w, fn_br, width=w, color=_OCHRE, label="FN Basis Risk (Unpaid Loss %)", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_labels, fontsize=8.0)
    ax1.set_ylabel("Rate (% of Policies / Sum Insured)")
    ax1.set_title("A. Regional Parametric Crop Insurance Risk (<80% Trigger)", pad=12)
    ax1.legend(frameon=False, loc="upper right", fontsize=8.2)
    _clean_axes(ax1)

    sweep = payload.insurance_summary.threshold_sweep
    ratios = [int(round(p.trigger_ratio * 100)) for p in sweep]
    ax2.plot(
        ratios,
        [p.pure_premium_rate_pct for p in sweep],
        marker="o",
        linewidth=2.2,
        color=_FOREST,
        label="Pure Premium Rate (%)",
    )
    ax2.plot(
        ratios,
        [p.fn_basis_risk_rate * 100.0 for p in sweep],
        marker="s",
        linewidth=2.2,
        color=_CRIMSON,
        label="FN Basis Risk (Farmer Unpaid %)",
    )
    ax2.plot(
        ratios,
        [p.fp_basis_risk_rate * 100.0 for p in sweep],
        marker="^",
        linewidth=2.0,
        linestyle="--",
        color=_AMBER,
        label="FP Basis Risk (Insurer Leakage %)",
    )
    ax2.axvline(80, color=_SLATE, linestyle=":", linewidth=1.5, label="Selected 80% Baseline Strike")
    ax2.set_xticks(ratios)
    ax2.set_xticklabels([f"{r}%" for r in ratios])
    ax2.set_xlabel("Parametric Yield Trigger (% of Trailing Local Baseline)")
    ax2.set_ylabel("Portfolio Rate (%)")
    ax2.set_title("B. Actuarial Pricing & Basis Risk Frontier (2011-2013)", pad=12)
    ax2.legend(frameon=False, loc="upper left", fontsize=8.3)
    _clean_axes(ax2)

    fig.suptitle(
        "AGCRS — Parametric Drought/Heatwave Crop Insurance Pricing & Basis Risk Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_all_agronomy_figures(
    payload: AgcrsReportPayload, images_dir: Path
) -> tuple[Path, ...]:
    """Render all four 300 DPI publication-grade AGCRS figures into `images_dir`."""
    images_dir.mkdir(parents=True, exist_ok=True)
    _apply_editorial_style()

    paths = (
        images_dir / "agcrs_temporal_benchmark_conformal.png",
        images_dir / "agcrs_climate_shock_stress_test.png",
        images_dir / "agcrs_shap_yield_drivers.png",
        images_dir / "agcrs_parametric_insurance_risk.png",
    )
    _plot_temporal_and_conformal(payload, paths[0])
    _plot_climate_shock_stress_test(payload, paths[1])
    _plot_shap_and_dedup_audit(payload, paths[2])
    _plot_parametric_insurance_risk(payload, paths[3])
    return paths
