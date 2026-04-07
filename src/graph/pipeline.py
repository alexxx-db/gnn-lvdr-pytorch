"""
Graph pipeline orchestration.

Consolidates the repeated graph-rebuild boilerplate used across notebooks
05–08 into a single reusable function.
"""
import torch
from pyspark.sql import SparkSession
from src.config.settings import ProjectConfig
from src.config import naming
from src.graph.features import encode_patient_features, encode_care_site_features
from src.graph.datasets import build_homogeneous_graph, split_graph, create_edge_dataloaders


def rebuild_graph(spark: SparkSession, cfg: ProjectConfig) -> tuple:
    """
    Rebuild the full graph from Delta tables.

    Returns (graph, metadata, patient_ids, care_site_ids).
    Called by notebooks that need the graph in their execution context.
    """
    edges_df = spark.table(cfg.table(naming.GRAPH_EDGES))
    patient_features, patient_ids = encode_patient_features(
        spark.table(cfg.table(naming.PATIENT_FEATURES))
    )
    care_site_features, care_site_ids = encode_care_site_features(
        spark.table(cfg.table(naming.CARE_SITE_FEATURES))
    )

    graph, metadata = build_homogeneous_graph(
        edges_df=edges_df,
        patient_features=patient_features,
        patient_ids=patient_ids,
        care_site_features=care_site_features,
        care_site_ids=care_site_ids,
        target_edge_type="visited",
        num_node_features=cfg.num_node_features,
    )

    return graph, metadata


def rebuild_graph_with_splits(spark: SparkSession, cfg: ProjectConfig) -> tuple:
    """
    Rebuild graph, split it, and create data loaders.

    Returns (graph, metadata, graph_partitions, data_loaders).
    Sets the random seed for reproducible splits.
    """
    torch.manual_seed(cfg.random_seed)

    graph, metadata = rebuild_graph(spark, cfg)
    graph_partitions = split_graph(graph, cfg)
    data_loaders = create_edge_dataloaders(graph_partitions, cfg)

    return graph, metadata, graph_partitions, data_loaders
