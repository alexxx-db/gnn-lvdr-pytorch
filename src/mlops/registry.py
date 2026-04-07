"""
MLflow model registry integration for Unity Catalog.

Handles model packaging, registration, alias management,
and loading for inference.
"""
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import pandas as pd
import numpy as np
import dgl
import torch

from src.graph.graphsage import Model
from src.config.settings import ProjectConfig


class GNNPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for the GraphSAGE model.

    Accepts a DataFrame with src_id and dst_id columns,
    constructs a DGL graph, and returns link prediction scores.

    IMPORTANT: This wrapper constructs a fresh graph from the input edges and
    assigns random node features because the pyfunc interface has no access to
    the original feature tables. Scores from this wrapper reflect learned graph
    *structure* patterns but lack real clinical feature signal. For production
    inference with real features, use src.graph.inference.predict_patient_care_site_scores()
    which operates on pre-computed embeddings from real feature vectors.

    Preserved from legacy GNNWrapper with modernized interface.
    """

    def __init__(self, model: Model, cfg: ProjectConfig):
        self.model = model
        self.num_node_features = cfg.num_node_features
        self.batch_size = cfg.batch_size

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            source_id = np.array(model_input["src_id"].values, dtype=np.int64)
            dest_id = np.array(model_input["dst_id"].values, dtype=np.int64)
            g = dgl.graph((source_id, dest_id))
        else:
            raise ValueError("Expected a pandas DataFrame with src_id and dst_id columns")

        # Random features — structural scoring only; see docstring
        g.ndata["feature"] = torch.randn(
            g.num_nodes(), self.num_node_features
        )

        self.model.eval()
        with torch.no_grad():
            scores = self.model.get_embeddings(
                g=g, x=g.ndata["feature"],
                batch_size=self.batch_size,
                device="cpu",
                provide_prediction=True,
            )
        return scores.cpu().detach().numpy().squeeze()


def register_model(
    model: Model,
    cfg: ProjectConfig,
    run_id: str,
    sample_input: pd.DataFrame = None,
) -> str:
    """
    Log and register the GNN model in Unity Catalog.

    Returns the registered model version.
    """
    wrapper = GNNPyfuncWrapper(model, cfg)

    signature = None
    if sample_input is not None:
        from mlflow.models.signature import infer_signature
        sample_output = wrapper.predict(None, sample_input)
        signature = infer_signature(sample_input, sample_output)

    mlflow.pyfunc.log_model(
        artifact_path="gnn_model",
        python_model=wrapper,
        signature=signature,
    )

    if cfg.register_artifacts:
        result = mlflow.register_model(
            f"runs:/{run_id}/gnn_model",
            cfg.uc_model,
        )
        return result.version
    return None


def set_champion_alias(cfg: ProjectConfig, version: str) -> None:
    """Set the 'champion' alias on a model version."""
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(cfg.uc_model, "champion", version)


def load_champion_model(cfg: ProjectConfig):
    """Load the champion model from Unity Catalog."""
    return mlflow.pyfunc.load_model(f"models:/{cfg.uc_model}@champion")
