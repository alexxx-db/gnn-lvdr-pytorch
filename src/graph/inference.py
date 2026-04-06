"""
GraphSAGE inference pipeline.

Generates full-graph embeddings and edge predictions for
patient→care_site recommendations.
"""
import torch
import numpy as np
import pandas as pd
from typing import Optional

from src.graph.graphsage import Model
from src.config.settings import ProjectConfig


def generate_embeddings(
    model: Model,
    graph,
    cfg: ProjectConfig,
) -> torch.Tensor:
    """
    Generate node embeddings for the full graph using the trained model.

    Returns a tensor of shape (num_nodes, out_features).
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(
            g=graph,
            x=graph.ndata["feature"],
            batch_size=cfg.batch_size,
            device=cfg.device,
        )
    return embeddings


def predict_patient_care_site_scores(
    model: Model,
    embeddings: torch.Tensor,
    metadata: dict,
    patient_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute link prediction scores for patient→care_site pairs.

    For each patient, score all care sites using the MLP predictor
    on the concatenated embeddings.

    Args:
        model: Trained Model instance
        embeddings: Full-graph node embeddings
        metadata: Graph metadata from build_homogeneous_graph
        patient_ids: Subset of patients to score (None = all)

    Returns:
        DataFrame with columns: person_id, care_site_id, score
    """
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    pid_to_idx = metadata["pid_to_idx"]
    sid_to_idx = metadata["sid_to_idx"]
    care_site_ids = metadata["care_site_ids"]

    if patient_ids is None:
        patient_ids = metadata["patient_ids"]

    results = []
    with torch.no_grad():
        for pid in patient_ids:
            if pid not in pid_to_idx:
                continue
            p_idx = pid_to_idx[pid]
            p_emb = embeddings[p_idx].unsqueeze(0)  # (1, dim)

            for sid in care_site_ids:
                s_idx = sid_to_idx[sid]
                s_emb = embeddings[s_idx].unsqueeze(0)  # (1, dim)

                concat = torch.cat([p_emb, s_emb], dim=1)  # (1, 2*dim)
                score = sigmoid(model.pred.W(concat)).item()
                results.append({
                    "person_id": pid,
                    "care_site_id": sid,
                    "score": round(score, 4),
                })

    return pd.DataFrame(results)
