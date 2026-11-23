"""Typed immutable configuration for the AGCRS agricultural climate risk pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations for raw data, processed panels, artifacts, and figures."""

    project_root: Path
    raw_csv_path: Path
    processed_dir: Path
    artifacts_dir: Path
    images_dir: Path


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Strict chronological train, conformal-calibration, and test horizon bounds."""

    train_start_year: int = 1990
    train_end_year: int = 2008
    calibration_start_year: int = 2009
    calibration_end_year: int = 2010
    test_start_year: int = 2011
    test_end_year: int = 2013
    random_seed: int = 42


@dataclass(frozen=True)
class ModelHyperparamConfig:
    """Hyperparameters for Ridge FE, LightGBM, and HistGradientBoosting."""

    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0, 200.0)
    lgbm_n_estimators: int = 450
    lgbm_learning_rate: float = 0.04
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 20
    hgb_max_iter: int = 350
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 31
    bootstrap_replicates: int = 150
    random_seed: int = 42

    @classmethod
    def fast_smoke(cls) -> "ModelHyperparamConfig":
        """Return lightweight hyperparameters for rapid smoke verification."""
        return cls(
            ridge_alphas=(1.0, 10.0),
            lgbm_n_estimators=80,
            lgbm_learning_rate=0.08,
            lgbm_num_leaves=15,
            lgbm_min_child_samples=20,
            hgb_max_iter=80,
            hgb_learning_rate=0.08,
            hgb_max_leaf_nodes=15,
            bootstrap_replicates=30,
            random_seed=42,
        )


@dataclass(frozen=True)
class ConformalAndInsuranceConfig:
    """Configuration for split-conformal coverage and parametric crop insurance."""

    conformal_alpha: float = 0.10
    trigger_baseline_ratio: float = 0.80
    trigger_grid: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90)
    insurer_loading_factor: float = 0.15
    warming_shocks_c: tuple[float, ...] = (1.0, 2.0)
    drought_shock_pct: float = -0.20


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level immutable configuration bundle for AGCRS execution."""

    paths: PathConfig
    split: TemporalSplitConfig
    hyperparams: ModelHyperparamConfig
    risk: ConformalAndInsuranceConfig

    @classmethod
    def create_default(cls, project_root: Path) -> "PipelineConfig":
        """Construct default AGCRS configuration rooted at `project_root`."""
        paths = PathConfig(
            project_root=project_root,
            raw_csv_path=project_root / "data" / "raw" / "yield_df.csv",
            processed_dir=project_root / "data" / "processed",
            artifacts_dir=project_root / "artifacts",
            images_dir=project_root / "images",
        )
        return cls(
            paths=paths,
            split=TemporalSplitConfig(),
            hyperparams=ModelHyperparamConfig(),
            risk=ConformalAndInsuranceConfig(),
        )
