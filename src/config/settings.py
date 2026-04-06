"""
Centralized project configuration.

Single source of truth for all names, paths, and parameters.
Supports Databricks notebook widget overrides for top-level runtime values.
"""
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    catalog="gnn_hls_graphsage",
    schema="gnn_hls_graphsage_db",
    volume="gnn_data",
    experiment_name="/Shared/gnn-patient-recommendations",
    registered_model_name="patient_recommendations_gnn",
    app_name="gnn-patient-recommendations",
    workflow_name="gnn-patient-recommendations-workflow",
    serving_endpoint_name="gnn-patient-rec-agent",
    synthetic_scale=1000,
    random_seed=42,
    rebuild_synthetic=True,
    retrain_model=True,
    register_artifacts=True,
)


@dataclass
class ProjectConfig:
    """Immutable-ish runtime configuration for the entire project."""

    # --- Unity Catalog namespace ---
    catalog: str = _DEFAULTS["catalog"]
    schema: str = _DEFAULTS["schema"]
    volume: str = _DEFAULTS["volume"]

    # --- MLflow ---
    experiment_name: str = _DEFAULTS["experiment_name"]
    registered_model_name: str = _DEFAULTS["registered_model_name"]

    # --- Databricks resources ---
    app_name: str = _DEFAULTS["app_name"]
    workflow_name: str = _DEFAULTS["workflow_name"]
    serving_endpoint_name: str = _DEFAULTS["serving_endpoint_name"]

    # --- Data generation ---
    synthetic_scale: int = _DEFAULTS["synthetic_scale"]
    random_seed: int = _DEFAULTS["random_seed"]
    rebuild_synthetic: bool = _DEFAULTS["rebuild_synthetic"]

    # --- Model training ---
    retrain_model: bool = _DEFAULTS["retrain_model"]
    register_artifacts: bool = _DEFAULTS["register_artifacts"]

    # --- Graph / GraphSAGE hyper-parameters ---
    num_node_features: int = 32
    num_hidden: int = 64
    num_out: int = 32
    num_classes: int = 2
    aggregator_type: str = "mean"
    num_epochs: int = 500
    batch_size: int = 64
    lr: float = 0.001
    l2_regularisation: float = 5e-4
    momentum: float = 0.05
    optimiser: str = "Adam"
    loss: str = "binary_cross_entropy"
    test_p: float = 0.10
    valid_p: float = 0.20
    num_negative_samples: int = 10
    num_workers: int = 0
    device: str = "cpu"
    neighbor_sample_sizes: list = field(default_factory=lambda: [8, 8])

    # --- Derived helpers (computed, not stored) ---
    @property
    def uc_prefix(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def uc_model(self) -> str:
        return f"{self.uc_prefix}.{self.registered_model_name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    def table(self, name: str) -> str:
        """Return fully-qualified UC table name."""
        return f"{self.uc_prefix}.{name}"

    def display_summary(self) -> dict:
        """Return a dict of key config values for notebook display."""
        return {
            "catalog": self.catalog,
            "schema": self.schema,
            "uc_model": self.uc_model,
            "experiment": self.experiment_name,
            "synthetic_scale": self.synthetic_scale,
            "rebuild_synthetic": self.rebuild_synthetic,
            "retrain_model": self.retrain_model,
            "device": self.device,
            "num_epochs": self.num_epochs,
        }


def load_config(overrides: Optional[dict] = None) -> ProjectConfig:
    """
    Build a ProjectConfig, optionally applying overrides from notebook widgets.

    Usage in a notebook:
        from src.config.settings import load_config
        cfg = load_config({
            "catalog": dbutils.widgets.get("catalog"),
            "synthetic_scale": int(dbutils.widgets.get("synthetic_scale")),
        })
    """
    cfg = ProjectConfig()
    if overrides:
        for k, v in overrides.items():
            if v is not None and v != "" and hasattr(cfg, k):
                expected_type = type(getattr(cfg, k))
                if expected_type is bool:
                    v = str(v).lower() in ("true", "1", "yes")
                else:
                    v = expected_type(v)
                setattr(cfg, k, v)
    return cfg
