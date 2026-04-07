# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Evaluation and MLflow
# MAGIC
# MAGIC **Purpose:** Evaluate the trained GraphSAGE model on validation and test sets,
# MAGIC generate t-SNE embedding visualization, register model in Unity Catalog.
# MAGIC
# MAGIC **Inputs:** Graph edges and features, cfg
# MAGIC **Outputs:** Evaluation metrics, t-SNE artifact, registered model with champion alias
# MAGIC **Upstream:** 06_graphsage_training
# MAGIC **Downstream:** 08_batch_recommendations

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

if not cfg.retrain_model:
    print("Skipping evaluation (retrain_model=False)")
    dbutils.notebook.exit("SKIPPED")

# COMMAND ----------

# DBTITLE 1,Rebuild graph with deterministic split (same seed as 06)
import torch
from src.graph.pipeline import rebuild_graph_with_splits

graph, metadata, graph_partitions, data_loaders = rebuild_graph_with_splits(spark, cfg)

# COMMAND ----------

# DBTITLE 1,Train, evaluate, and register within a single MLflow run
from src.graph.graphsage import Model
from src.mlops.tracking import setup_experiment, log_config_as_params, log_graph_metadata
from src.mlops.evaluation import train_model, evaluate_model
from src.mlops.registry import register_model, set_champion_alias
from src.graph.inference import generate_embeddings
import mlflow
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

experiment_id = setup_experiment(cfg)
torch.manual_seed(cfg.random_seed)

with mlflow.start_run(run_name="GraphSAGE-PatientRec-Eval") as run:
    mlflow.set_tag("link-prediction", "GraphSAGE")
    log_config_as_params(cfg)
    log_graph_metadata(metadata)

    # Train (same seed → same model as 06)
    model = Model(
        in_features=cfg.num_node_features,
        hidden_features=cfg.num_hidden,
        out_features=cfg.num_out,
        num_classes=cfg.num_classes,
        aggregator_type=cfg.aggregator_type,
    )
    trained_model = train_model(model, data_loaders["training"], cfg)

    # Evaluate on validation and test splits
    val_auc, val_ap = evaluate_model(trained_model, data_loaders["validation"], "validation")
    test_auc, test_ap = evaluate_model(trained_model, data_loaders["testing"], "test")

    print(f"Validation AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
    print(f"Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

    # Generate embeddings and t-SNE visualization
    embeddings = generate_embeddings(trained_model, graph_partitions["training"], cfg)
    n_sample = min(500, embeddings.shape[0])
    indices = np.random.choice(embeddings.shape[0], n_sample, replace=False)
    tsne = TSNE(n_components=2, perplexity=min(30, n_sample - 1), method="barnes_hut")
    tsne_result = tsne.fit_transform(embeddings[indices].numpy())

    fig, ax = plt.subplots(figsize=(10, 7))
    node_types = graph_partitions["training"].ndata["node_type"][indices].numpy()
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
               c=node_types, cmap="Set1", alpha=0.6, s=10)
    ax.set_title("t-SNE of GraphSAGE Embeddings (blue=patient, red=care_site)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    mlflow.log_figure(fig, "visualizations/tsne_embeddings.png")
    plt.close()

    # Register model
    import pandas as pd
    sample_input = pd.DataFrame({"src_id": [1, 2], "dst_id": [3, 4]})
    version = register_model(trained_model, cfg, run.info.run_id, sample_input)

    if version:
        set_champion_alias(cfg, version)
        print(f"Model registered: {cfg.uc_model} v{version} (champion)")

    run_id = run.info.run_id
    print(f"MLflow run_id: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `08_batch_recommendations`
