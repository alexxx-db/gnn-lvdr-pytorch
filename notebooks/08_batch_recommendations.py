# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Batch Recommendations
# MAGIC
# MAGIC **Purpose:** Run batch inference to generate top-k care-site recommendations
# MAGIC for all patients. Saves enriched recommendations as a Delta table.
# MAGIC
# MAGIC **Inputs:** Trained model, graph, patient/care-site features
# MAGIC **Outputs:** recommendations Delta table
# MAGIC **Upstream:** 07_evaluation_and_mlflow
# MAGIC **Downstream:** 09_agent_bricks_setup, 10_agent_pipeline

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Rebuild graph and model for inference
from src.graph.features import encode_patient_features, encode_care_site_features
from src.graph.datasets import build_homogeneous_graph
from src.graph.graphsage import Model
from src.graph.inference import generate_embeddings, predict_patient_care_site_scores
from src.graph.ranking import rank_recommendations, save_recommendations
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

# COMMAND ----------

# DBTITLE 1,Train a fresh model or load champion (using fresh train for reproducibility)
from src.mlops.evaluation import train_model
from src.graph.datasets import split_graph, create_edge_dataloaders

graph_partitions = split_graph(graph, cfg)
data_loaders = create_edge_dataloaders(graph_partitions, cfg)

model = Model(
    in_features=cfg.num_node_features,
    hidden_features=cfg.num_hidden,
    out_features=cfg.num_out,
    num_classes=cfg.num_classes,
    aggregator_type=cfg.aggregator_type,
)

import mlflow
from src.mlops.tracking import setup_experiment
setup_experiment(cfg)

with mlflow.start_run(run_name="GraphSAGE-BatchInference"):
    trained_model = train_model(model, data_loaders["training"], cfg)

# COMMAND ----------

# DBTITLE 1,Generate embeddings and predict scores
embeddings = generate_embeddings(trained_model, graph, cfg)
print(f"Embeddings shape: {embeddings.shape}")

# Score a sample of patients (first 200 for demo speed)
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
