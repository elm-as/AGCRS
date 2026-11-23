"""TreeSHAP yield driver attribution on the 2011-2013 out-of-time test horizon."""

import numpy as np
import pandas as pd
import shap

from src.core.contracts import ShapDriverScore


def compute_tree_shap_drivers(
    tree_model: object,
    x_test: pd.DataFrame,
    feature_labels: dict[str, str],
) -> tuple[ShapDriverScore, ...]:
    """Compute global mean |SHAP| feature importance and directional correlation."""
    explainer = shap.TreeExplainer(tree_model)
    shap_values = np.asarray(explainer.shap_values(x_test), dtype=float)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total_abs = max(float(np.sum(mean_abs)), 1e-9)

    drivers: list[ShapDriverScore] = []
    for idx, col in enumerate(x_test.columns):
        feat_vals = x_test[col].to_numpy(dtype=float)
        sv = shap_values[:, idx]
        std_f = float(np.std(feat_vals))
        std_s = float(np.std(sv))
        corr = (
            float(np.corrcoef(feat_vals, sv)[0, 1])
            if std_f > 1e-12 and std_s > 1e-12
            else 0.0
        )
        drivers.append(
            ShapDriverScore(
                feature_name=str(col),
                display_label=feature_labels.get(str(col), str(col)),
                mean_abs_shap_hg_ha=float(mean_abs[idx]),
                share_pct=100.0 * float(mean_abs[idx]) / total_abs,
                mean_signed_correlation=corr,
            )
        )

    drivers.sort(key=lambda d: d.mean_abs_shap_hg_ha, reverse=True)
    return tuple(drivers)
