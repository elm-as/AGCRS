"""Standalone editorial agronomy & climate risk HTML5 + Plotly dashboard."""

import json
from pathlib import Path

from src.core.contracts import AgcrsReportPayload
from src.visualization.results_exporter import serialize_agcrs_payload

_DASHBOARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #FAF9F5; color: #1C2526; line-height: 1.45; padding: 20px;
}
.container { max-width: 1380px; margin: 0 auto; }
.header {
  background: #FFFFFF; border: 1px solid #D6D8D2; border-left: 5px solid #1B4332;
  border-radius: 6px; padding: 18px 22px; margin-bottom: 16px;
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
}
.header h1 { font-size: 1.32rem; font-weight: 700; color: #1C2526; }
.header p { font-size: 0.88rem; color: #4B5B5E; margin-top: 3px; }
.meta-strip { display: flex; gap: 8px; flex-wrap: wrap; }
.meta-tag {
  font-size: 0.78rem; padding: 4px 10px; border-radius: 4px;
  border: 1px solid #CBD2CD; background: #F4F5F0; color: #1B4332; font-weight: 600;
}
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
  gap: 12px; margin-bottom: 16px;
}
.kpi-card {
  background: #FFFFFF; border: 1px solid #D6D8D2; border-radius: 6px; padding: 14px 16px;
}
.kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: #5C6B73; font-weight: 600; }
.kpi-val {
  font-size: 1.48rem; font-weight: 700; color: #1C2526; margin: 4px 0;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
  font-family: "JetBrains Mono", "Consolas", ui-monospace, monospace;
}
.kpi-sub { font-size: 0.78rem; color: #5C6B73; }
.accent-forest { color: #1B4332; }
.accent-crimson { color: #991B1B; }
.accent-amber { color: #B45309; }
.chart-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
  gap: 14px; margin-bottom: 16px;
}
.panel {
  background: #FFFFFF; border: 1px solid #D6D8D2; border-radius: 6px; padding: 14px 16px;
}
.panel-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px; gap: 10px; flex-wrap: wrap;
}
.panel-title { font-size: 0.95rem; font-weight: 700; color: #1C2526; }
select {
  border: 1px solid #CBD2CD; border-radius: 4px; padding: 4px 8px;
  font-size: 0.82rem; background: #FFFFFF; color: #1C2526;
}
.table-wrap { overflow-x: auto; margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
th, td { padding: 7px 10px; border-bottom: 1px solid #E6E8E3; text-align: left; }
th { background: #F4F5F0; color: #1C2526; font-weight: 600; border-top: 1px solid #D6D8D2; }
.num, td.num, th.num {
  text-align: right; font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
  font-family: "JetBrains Mono", "Consolas", ui-monospace, monospace;
}
.badge-staple {
  background: #ECFDF5; color: #1B4332; border: 1px solid #A7F3D0;
  padding: 2px 6px; border-radius: 4px; font-size: 0.73rem; font-weight: 600;
}
"""

_DASHBOARD_JS = """
const DATA = __AGCRS_JSON__;
const layoutBase = {
  template: "plotly_white", paper_bgcolor: "#FFFFFF", plot_bgcolor: "#FFFFFF",
  margin: { l: 62, r: 22, t: 24, b: 56 }, font: { family: "Inter, sans-serif", color: "#1C2526", size: 11 }
};

function renderModelBenchmark() {
  const metric = document.getElementById("bench-metric").value;
  const labels = DATA.model_benchmarks.map(b => b.display_label);
  const ootVals = DATA.model_benchmarks.map(b => b.test_metrics[metric]);
  const loroKey = metric === "r2" ? "loro_mean_r2" : "loro_mean_rmse";
  const traces = [{ x: labels, y: ootVals, name: "Temporal OOT (2011-2013)", type: "bar", marker: { color: "#1B4332" } }];
  if (metric === "rmse" || metric === "r2") {
    traces.push({ x: labels, y: DATA.model_benchmarks.map(b => b[loroKey]), name: "LORO Spatial Holdout", type: "bar", marker: { color: "#B45309" } });
  }
  Plotly.newPlot("plot-benchmark", traces, {
    ...layoutBase, barmode: "group", yaxis: { title: metric.toUpperCase(), gridcolor: "#E6E8E3" },
    legend: { orientation: "h", y: 1.13 }
  }, { responsive: true });
}

function renderClimateShocks() {
  const scenario = document.getElementById("shock-scenario").value;
  const rows = [...DATA.climate_econometrics.crop_impacts].sort((a, b) => a[scenario] - b[scenario]);
  const colors = rows.map(r => r.is_african_tropical_staple ? "#991B1B" : "#40916C");
  const traces = [{
    y: rows.map(r => r.is_african_tropical_staple ? r.crop + " *" : r.crop),
    x: rows.map(r => r[scenario]),
    type: "bar", orientation: "h", marker: { color: colors }
  }];
  Plotly.newPlot("plot-shocks", traces, {
    ...layoutBase, margin: { l: 135, r: 24, t: 20, b: 48 },
    xaxis: { title: "Estimated Yield Elasticity (% Change)", gridcolor: "#E6E8E3", zeroline: true, zerolinecolor: "#1C2526" }
  }, { responsive: true });
}

function renderConformal() {
  const adapt = DATA.conformal_prediction.adaptive.per_crop_coverage;
  const stdMap = Object.fromEntries(DATA.conformal_prediction.standard.per_crop_coverage.map(c => [c.crop, c.empirical_coverage * 100]));
  const traces = [
    { x: adapt.map(c => c.crop), y: adapt.map(c => c.empirical_coverage * 100), name: "Adaptive Split-Conformal", type: "bar", marker: { color: "#40916C" } },
    { x: adapt.map(c => c.crop), y: adapt.map(c => stdMap[c.crop]), name: "Standard Split-Conformal", type: "bar", marker: { color: "#334155" } }
  ];
  Plotly.newPlot("plot-conformal", traces, {
    ...layoutBase, barmode: "group", yaxis: { title: "Empirical Coverage (%)", range: [50, 105], gridcolor: "#E6E8E3" },
    shapes: [{ type: "line", x0: -0.5, x1: adapt.length - 0.5, y0: 90, y1: 90, line: { color: "#991B1B", dash: "dash", width: 1.6 } }],
    legend: { orientation: "h", y: 1.13 }
  }, { responsive: true });
}

function renderInsurance() {
  const view = document.getElementById("ins-segment").value;
  const segs = view === "region" ? DATA.parametric_insurance.by_region : DATA.parametric_insurance.by_crop;
  const traces = [
    { x: segs.map(s => s.segment_name), y: segs.map(s => s.actual_shortfall_rate * 100), name: "Actual Shortfall <80% (%)", type: "bar", marker: { color: "#991B1B" } },
    { x: segs.map(s => s.segment_name), y: segs.map(s => s.pure_premium_rate_pct), name: "Pre-Test Pure Premium (%)", type: "bar", marker: { color: "#1B4332" } },
    { x: segs.map(s => s.segment_name), y: segs.map(s => s.fn_basis_risk_rate * 100), name: "FN Basis Risk (%)", type: "bar", marker: { color: "#D97706" } }
  ];
  Plotly.newPlot("plot-insurance", traces, {
    ...layoutBase, barmode: "group", yaxis: { title: "Rate (%)", gridcolor: "#E6E8E3" },
    legend: { orientation: "h", y: 1.13 }
  }, { responsive: true });
}

function populateTables() {
  document.getElementById("model-table-body").innerHTML = DATA.model_benchmarks.map(b => `<tr>
    <td>${b.display_label}</td>
    <td class="num">${b.calibration_metrics.rmse.toLocaleString(undefined, {maximumFractionDigits: 1})}</td>
    <td class="num">${b.test_metrics.rmse.toLocaleString(undefined, {maximumFractionDigits: 1})}</td>
    <td class="num">${b.test_metrics.mae.toLocaleString(undefined, {maximumFractionDigits: 1})}</td>
    <td class="num">${b.test_metrics.mape_pct.toFixed(2)}%</td>
    <td class="num">${b.test_metrics.r2.toFixed(4)}</td>
    <td class="num">${b.loro_mean_r2.toFixed(4)}</td>
  </tr>`).join("");

  document.getElementById("climate-table-body").innerHTML = DATA.climate_econometrics.crop_impacts.map(c => `<tr>
    <td>${c.crop} ${c.is_african_tropical_staple ? '<span class="badge-staple">African/Tropical</span>' : ''}</td>
    <td class="num">${c.mean_temp_c.toFixed(1)}°C</td>
    <td class="num">${c.shock_plus_1c_pct.toFixed(2)}%</td>
    <td class="num">${c.shock_plus_2c_pct.toFixed(2)}%</td>
    <td class="num">${c.shock_drought_minus_20pct.toFixed(2)}%</td>
    <td class="num">${c.fe_within_plus_2c_pct.toFixed(2)}%</td>
  </tr>`).join("");
}

renderModelBenchmark(); renderClimateShocks(); renderConformal(); renderInsurance(); populateTables();
document.getElementById("bench-metric").addEventListener("change", renderModelBenchmark);
document.getElementById("shock-scenario").addEventListener("change", renderClimateShocks);
document.getElementById("ins-segment").addEventListener("change", renderInsurance);
"""


def _extract_dashboard_kpis(serialized: dict[str, object]) -> dict[str, str]:
    audit = serialized["panel_audit"]
    best_m = serialized["best_model_test_metrics"]
    conf_a = serialized["conformal_prediction"]["adaptive"]
    econ = serialized["climate_econometrics"]
    ins = serialized["parametric_insurance"]["portfolio_at_trigger"]
    ssa = serialized["parametric_insurance"]["ssa_summary"]
    return {
        "unique_rows": f"{audit['unique_panel_rows']:,}",
        "raw_rows": f"{audit['raw_row_count']:,}",
        "dup_pct": f"-{audit['duplicate_inflation_pct']:.1f}%",
        "best_model": str(serialized["best_model_name"]),
        "best_r2": f"{best_m['r2']:.4f}",
        "best_rmse": f"{best_m['rmse']:,.0f}",
        "best_mape": f"{best_m['mape_pct']:.2f}%",
        "conf_cov": f"{conf_a['empirical_coverage'] * 100:.2f}%",
        "conf_width": f"{conf_a['relative_width_pct']:.1f}%",
        "af_2c": f"{econ['african_staples_mean_plus_2c_pct']:+.2f}%",
        "af_drought": f"{econ['african_staples_mean_drought_20pct']:+.2f}%",
        "pure_prem": f"{ins['pure_premium_rate_pct']:.2f}%",
        "ssa_shortfall": f"{ssa['actual_shortfall_rate'] * 100:.2f}%",
    }


def build_agronomy_dashboard_html(
    payload: AgcrsReportPayload, dashboard_path: Path
) -> Path:
    """Generate a standalone light editorial HTML5 + Plotly agronomy dashboard."""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_agcrs_payload(payload)
    kpis = _extract_dashboard_kpis(serialized)
    json_blob = json.dumps(serialized)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AGCRS - Global Agricultural Yield, Climate Shocks &amp; Parametric Insurance</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body>
<div class="container">
  <header class="header">
    <div>
      <h1>AGCRS - Global Agricultural Yield, Climate Shocks &amp; Parametric Insurance Risk</h1>
      <p>FAO/World Bank 101-Country Staple Panel (1990-2013) | Strict Out-of-Time Split, Split-Conformal Intervals &amp; Climate Econometrics</p>
    </div>
    <div class="meta-strip">
      <span class="meta-tag">Best: {kpis['best_model']}</span>
      <span class="meta-tag">Train 1990-2008 | Cal 2009-2010 | Test 2011-2013</span>
      <span class="meta-tag">ELMAS.md Compliant</span>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Deduplicated Panel</div><div class="kpi-val accent-forest">{kpis['unique_rows']}</div><div class="kpi-sub">Cleaned from {kpis['raw_rows']} raw ({kpis['dup_pct']} dup)</div></div>
    <div class="kpi-card"><div class="kpi-label">2011-2013 OOT R²</div><div class="kpi-val accent-forest">{kpis['best_r2']}</div><div class="kpi-sub">RMSE: {kpis['best_rmse']} hg/ha | MAPE: {kpis['best_mape']}</div></div>
    <div class="kpi-card"><div class="kpi-label">Conformal 90% Coverage</div><div class="kpi-val accent-forest">{kpis['conf_cov']}</div><div class="kpi-sub">Adaptive rel. width: {kpis['conf_width']} of mean yield</div></div>
    <div class="kpi-card"><div class="kpi-label">African Staples (+2°C)</div><div class="kpi-val accent-crimson">{kpis['af_2c']}</div><div class="kpi-sub">Drought (-20% rain): {kpis['af_drought']}</div></div>
    <div class="kpi-card"><div class="kpi-label">Parametric Pure Premium</div><div class="kpi-val accent-amber">{kpis['pure_prem']}</div><div class="kpi-sub">Strike &lt;80% local baseline | SSA shortfall: {kpis['ssa_shortfall']}</div></div>
  </section>

  <section class="chart-grid">
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">1. Temporal OOT vs LORO Spatial Benchmark</span>
        <select id="bench-metric">
          <option value="rmse">RMSE (hg/ha)</option><option value="r2">R²</option>
          <option value="mae">MAE (hg/ha)</option><option value="mape_pct">MAPE (%)</option>
        </select>
      </div>
      <div id="plot-benchmark" style="height:350px;"></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">2. Climate Shock Elasticity by Staple Crop (* African/Tropical)</span>
        <select id="shock-scenario">
          <option value="shock_plus_2c_pct">+2°C Warming</option>
          <option value="shock_plus_1c_pct">+1°C Warming</option>
          <option value="shock_drought_minus_20pct">-20% Drought</option>
          <option value="fe_within_plus_2c_pct">Within-Country FE (+2°C)</option>
        </select>
      </div>
      <div id="plot-shocks" style="height:350px;"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">3. Split-Conformal 90% Empirical Coverage by Crop</span></div>
      <div id="plot-conformal" style="height:350px;"></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">4. Parametric Crop Insurance Risk &amp; Basis Risk (&lt;80% Trigger)</span>
        <select id="ins-segment">
          <option value="region">By FAO Macro-Region</option>
          <option value="crop">By Staple Crop</option>
        </select>
      </div>
      <div id="plot-insurance" style="height:350px;"></div>
    </div>
  </section>

  <section class="panel" style="margin-bottom:14px;">
    <div class="panel-head"><span class="panel-title">Forecasting Model Matrix (Temporal OOT 2011-2013 &amp; LORO Spatial Audit)</span></div>
    <div class="table-wrap">
      <table><thead><tr><th>Model</th><th class="num">Cal RMSE</th><th class="num">Test RMSE</th><th class="num">Test MAE</th><th class="num">MAPE</th><th class="num">Test R²</th><th class="num">LORO Mean R²</th></tr></thead><tbody id="model-table-body"></tbody></table>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><span class="panel-title">Econometric Climate Sensitivity Table (1990-2008 Semi-Log Quadratic FE)</span></div>
    <div class="table-wrap">
      <table><thead><tr><th>Staple Crop</th><th class="num">Mean Temp</th><th class="num">+1°C Shock</th><th class="num">+2°C Shock</th><th class="num">-20% Drought</th><th class="num">Within-FE +2°C</th></tr></thead><tbody id="climate-table-body"></tbody></table>
    </div>
  </section>
</div>
<script>{_DASHBOARD_JS.replace("__AGCRS_JSON__", json_blob)}</script>
</body>
</html>"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
