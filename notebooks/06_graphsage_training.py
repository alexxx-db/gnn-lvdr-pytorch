# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — GraphSAGE Training
# MAGIC
# MAGIC **Purpose:** Train the GraphSAGE link prediction model on the constructed graph.
# MAGIC Splits edges, creates data loaders, trains the model, and logs to MLflow.
# MAGIC
# MAGIC **Inputs:** DGL graph from 05, cfg
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

# DBTITLE 1,Rebuild graph (needed in this notebook's execution context)
from src.graph.edges import build_all_edges
from src.graph.features import encode_patient_features, encode_care_site_features
from src.graph.datasets import build_homogeneous_graph, split_graph, create_edge_dataloaders
from src.config import naming

edges_df = spark.table(cfg.table(naming.GRAPH_EDGES))
patient_features, patient_ids = encode_patient_features(
    spark.table(cfg.table(naming.PATIENT_FEATURES)))
care_site_features, care_site_ids = encode_care_site_features(
    spark.table(cfg.table(naming.CARE_SITE_FEATURES)))

graph, metadata = build_homogeneous_graph(
    edges_df=edges_df,
    patient_features=patient_features,
    patient_ids=patient_ids,
    care_site_features=care_site_features,
    care_site_ids=care_site_ids,
    target_edge_type="visited",
    num_node_features=cfg.num_node_features,
)

print(f"Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")

# COMMAND ----------

# DBTITLE 1,Split graph and create data loaders
graph_partitions = split_graph(graph, cfg)
data_loaders = create_edge_dataloaders(graph_partitions, cfg)

for name, g in graph_partitions.items():
    print(f"  {name}: {g.num_edges()} edges")

# COMMAND ----------

# DBTITLE 1,Create model and start MLflow training run
from src.graph.graphsage import Model
from src.mlops.tracking import setup_experiment, log_config_as_params, log_graph_metadata
from src.mlops.evaluation import train_model
import mlflow

experiment_id = setup_experiment(cfg)

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
