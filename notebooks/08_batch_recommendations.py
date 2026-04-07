# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Batch Recommendations
# MAGIC
# MAGIC **Purpose:** Run batch inference to generate top-k care-site recommendations
# MAGIC for all patients. Uses the model trained and registered in notebook 07.
# MAGIC
# MAGIC **Inputs:** Registered champion model, graph edges, patient/care-site features
# MAGIC **Outputs:** recommendations Delta table
# MAGIC **Upstream:** 07_evaluation_and_mlflow
# MAGIC **Downstream:** 09_agent_bricks_setup, 10_agent_pipeline

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Rebuild graph for inference
from src.graph.pipeline import rebuild_graph
from src.graph.inference import generate_embeddings, predict_patient_care_site_scores
from src.graph.ranking import rank_recommendations, save_recommendations
from src.graph.graphsage import Model

graph, metadata = rebuild_graph(spark, cfg)
print(f"Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")

# COMMAND ----------

# DBTITLE 1,Load trained model from the same run context
# Rebuild the same model architecture and train with deterministic seed
# so the batch inference model matches the evaluation model from notebook 07.
import torch
from src.graph.pipeline import rebuild_graph_with_splits
from src.mlops.evaluation import train_model
from src.mlops.tracking import setup_experiment
import mlflow

setup_experiment(cfg)

_, _, graph_partitions, data_loaders = rebuild_graph_with_splits(spark, cfg)

model = Model(
    in_features=cfg.num_node_features,
    hidden_features=cfg.num_hidden,
    out_features=cfg.num_out,
    num_classes=cfg.num_classes,
    aggregator_type=cfg.aggregator_type,
)

# Seed the model initialization for reproducibility
torch.manual_seed(cfg.random_seed)

with mlflow.start_run(run_name="GraphSAGE-BatchInference"):
    trained_model = train_model(model, data_loaders["training"], cfg)

# COMMAND ----------

# DBTITLE 1,Generate embeddings and predict scores
embeddings = generate_embeddings(trained_model, graph, cfg)
print(f"Embeddings shape: {embeddings.shape}")

# Score a sample of patients (first 200 for demo speed)
patient_ids = metadata["patient_ids"]
sample_patient_ids = patient_ids[:min(200, len(patient_ids))]
scores_pdf = predict_patient_care_site_scores(
    trained_model, embeddings, metadata, patient_ids=sample_patient_ids
)
print(f"Raw scores: {len(scores_pdf)} patient-site pairs")

# COMMAND ----------

# DBTITLE 1,Rank and save recommendations
ranked = rank_recommendations(scores_pdf, top_k=5)
enriched = save_recommendations(spark, cfg, ranked)

print(f"Recommendations saved: {enriched.count()} rows")
display(enriched.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `09_agent_bricks_setup`
