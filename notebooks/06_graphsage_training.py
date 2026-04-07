# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — GraphSAGE Training
# MAGIC
# MAGIC **Purpose:** Train the GraphSAGE link prediction model on the constructed graph.
# MAGIC Splits edges, creates data loaders, trains the model, and logs to MLflow.
# MAGIC
# MAGIC **Inputs:** Graph edges and features from 05, cfg
# MAGIC **Outputs:** Trained model logged to MLflow, graph partitions
# MAGIC **Upstream:** 05_graph_construction
# MAGIC **Downstream:** 07_evaluation_and_mlflow

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

if not cfg.retrain_model:
    print("Skipping training (retrain_model=False)")
    dbutils.notebook.exit("SKIPPED")

# COMMAND ----------

# DBTITLE 1,Rebuild graph and create data loaders (deterministic split)
from src.graph.pipeline import rebuild_graph_with_splits

graph, metadata, graph_partitions, data_loaders = rebuild_graph_with_splits(spark, cfg)

print(f"Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
for name, g in graph_partitions.items():
    print(f"  {name}: {g.num_edges()} edges")

# COMMAND ----------

# DBTITLE 1,Create model and start MLflow training run
import torch
from src.graph.graphsage import Model
from src.mlops.tracking import setup_experiment, log_config_as_params, log_graph_metadata
from src.mlops.evaluation import train_model
import mlflow

experiment_id = setup_experiment(cfg)
torch.manual_seed(cfg.random_seed)

with mlflow.start_run(run_name="GraphSAGE-PatientRec") as run:
    mlflow.set_tag("link-prediction", "GraphSAGE")
    mlflow.set_tag("data_scale", cfg.synthetic_scale)
    log_config_as_params(cfg)
    log_graph_metadata(metadata)

    model = Model(
        in_features=cfg.num_node_features,
        hidden_features=cfg.num_hidden,
        out_features=cfg.num_out,
        num_classes=cfg.num_classes,
        aggregator_type=cfg.aggregator_type,
    )

    print(f"Model architecture:\n{model}")
    trained_model = train_model(model, data_loaders["training"], cfg)

    run_id = run.info.run_id
    print(f"\nTraining complete. MLflow run_id: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `07_evaluation_and_mlflow`
