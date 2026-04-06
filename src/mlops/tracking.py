"""
MLflow experiment tracking utilities.

Provides setup, run management, and structured logging for
GraphSAGE training runs.
"""
import mlflow
from src.config.settings import ProjectConfig


def setup_experiment(cfg: ProjectConfig) -> str:
    """
    Configure MLflow for Unity Catalog model registry and set experiment.

    Returns the experiment ID.
    """
    mlflow.set_registry_uri("databricks-uc")
    experiment = mlflow.set_experiment(cfg.experiment_name)
    return experiment.experiment_id


def log_config_as_params(cfg: ProjectConfig) -> None:
    """Log key configuration values as MLflow parameters."""
    mlflow.log_params({
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "synthetic_scale": cfg.synthetic_scale,
        "num_node_features": cfg.num_node_features,
        "num_hidden": cfg.num_hidden,
        "num_out": cfg.num_out,
        "aggregator_type": cfg.aggregator_type,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "l2_regularisation": cfg.l2_regularisation,
        "optimiser": cfg.optimiser,
        "loss": cfg.loss,
        "test_p": cfg.test_p,
        "valid_p": cfg.valid_p,
        "num_negative_samples": cfg.num_negative_samples,
        "neighbor_sample_sizes": str(cfg.neighbor_sample_sizes),
        "device": cfg.device,
    })


def log_graph_metadata(metadata: dict) -> None:
    """Log graph construction metadata as MLflow params and metrics."""
    mlflow.log_params({
        "n_patients": metadata["n_patients"],
        "n_care_sites": metadata["n_care_sites"],
    })
    mlflow.log_metric("n_edges", metadata["n_edges"])


def log_training_step(step: int, loss: float, auc: float) -> None:
    """Log a single training step's metrics."""
    mlflow.log_metric("train_loss", loss, step=step)
    mlflow.log_metric("train_auc", auc, step=step)


def log_evaluation_metrics(split: str, auc: float, ap: float) -> None:
    """Log evaluation metrics for a data split."""
    mlflow.log_metric(f"{split}_auc", auc)
    mlflow.log_metric(f"{split}_ap", ap)
