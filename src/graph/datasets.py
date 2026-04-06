"""
DGL graph construction from OMOP edge and feature tables.

Converts Spark DataFrames into DGL graphs with proper node features,
edge splits, and data loaders for GraphSAGE training.
"""
import numpy as np
import dgl
import torch
from typing import Dict, Tuple
from pyspark.sql import DataFrame
from src.config.settings import ProjectConfig


def build_homogeneous_graph(
    edges_df: DataFrame,
    patient_features: np.ndarray,
    patient_ids: list[int],
    care_site_features: np.ndarray,
    care_site_ids: list[int],
    target_edge_type: str = "visited",
    num_node_features: int = 32,
) -> Tuple[dgl.DGLGraph, dict]:
    """
    Build a homogeneous DGL graph from the patient→care_site edges.

    For GraphSAGE link prediction, we project the bipartite
    patient-care_site relationship into a single node space where:
      - nodes 0..N_patients-1 are patients
      - nodes N_patients..N_patients+N_sites-1 are care sites

    Returns: (graph, metadata_dict)
    """
    # Filter to target edge type and collect
    target_edges = (
        edges_df
        .filter(edges_df.edge_type == target_edge_type)
        .select("src_id", "dst_id")
        .toPandas()
    )

    # Build ID maps
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    n_patients = len(patient_ids)
    sid_to_idx = {sid: i + n_patients for i, sid in enumerate(care_site_ids)}

    # Map edges to contiguous indices
    src_indices = []
    dst_indices = []
    for _, row in target_edges.iterrows():
        src = pid_to_idx.get(int(row["src_id"]))
        dst = sid_to_idx.get(int(row["dst_id"]))
        if src is not None and dst is not None:
            src_indices.append(src)
            dst_indices.append(dst)

    src_t = torch.tensor(src_indices, dtype=torch.int64)
    dst_t = torch.tensor(dst_indices, dtype=torch.int64)

    n_total = n_patients + len(care_site_ids)
    g = dgl.graph((src_t, dst_t), num_nodes=n_total)

    # Assign node features
    from src.graph.features import pad_or_project_features, features_to_tensor
    pf = pad_or_project_features(patient_features, num_node_features)
    sf = pad_or_project_features(care_site_features, num_node_features)
    all_features = np.vstack([pf, sf])
    g.ndata["feature"] = features_to_tensor(all_features)

    # Node type mask (useful for downstream analysis)
    node_type = torch.zeros(n_total, dtype=torch.long)
    node_type[n_patients:] = 1
    g.ndata["node_type"] = node_type

    metadata = {
        "n_patients": n_patients,
        "n_care_sites": len(care_site_ids),
        "n_edges": len(src_indices),
        "patient_ids": patient_ids,
        "care_site_ids": care_site_ids,
        "pid_to_idx": pid_to_idx,
        "sid_to_idx": sid_to_idx,
    }

    return g, metadata


def split_graph(
    graph: dgl.DGLGraph,
    cfg: ProjectConfig,
) -> Dict[str, dgl.DGLGraph]:
    """
    Split graph edges into train/validation/test partitions.

    Preserves the legacy 70/20/10 split strategy.
    """
    src_ids, dst_ids = graph.edges()
    n = len(src_ids)
    test_size = int(cfg.test_p * n)
    valid_size = int(cfg.valid_p * n)
    train_size = n - valid_size - test_size

    # Shuffle edges for randomness
    perm = torch.randperm(n)
    src_ids = src_ids[perm]
    dst_ids = dst_ids[perm]

    splits = {}
    boundaries = {
        "training": (0, train_size),
        "validation": (train_size, train_size + valid_size),
        "testing": (train_size + valid_size, n),
    }

    num_nodes = graph.num_nodes()
    features = graph.ndata["feature"]

    for name, (start, end) in boundaries.items():
        g = dgl.graph(
            (src_ids[start:end], dst_ids[start:end]),
            num_nodes=num_nodes,
        )
        g.ndata["feature"] = features
        if "node_type" in graph.ndata:
            g.ndata["node_type"] = graph.ndata["node_type"]
        splits[name] = g

    return splits


def create_edge_dataloaders(
    graph_partitions: Dict[str, dgl.DGLGraph],
    cfg: ProjectConfig,
) -> Dict[str, dgl.dataloading.DataLoader]:
    """
    Create DGL edge DataLoaders for each graph partition.

    Uses MultiLayerNeighborSampler for message-passing blocks
    and Uniform negative sampler for link prediction training.
    """
    sampler = dgl.dataloading.MultiLayerNeighborSampler(cfg.neighbor_sample_sizes)
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(cfg.num_negative_samples)

    data_loaders = {}
    for split_name, g in graph_partitions.items():
        data_loaders[split_name] = dgl.dataloading.DataLoader(
            g,
            g.edges(form="eid"),
            sampler,
            device=cfg.device,
            negative_sampler=negative_sampler,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
        )
    return data_loaders
