# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Graph Construction
# MAGIC
# MAGIC **Purpose:** Build the patient-care_site graph from curated OMOP data.
# MAGIC Extracts edges (visited, diagnosed_with, treated_by, works_at, near),
# MAGIC constructs node features, and creates the DGL graph for training.
# MAGIC
# MAGIC **Inputs:** Gold tables, patient_features, care_site_features, nearest_sites
# MAGIC **Outputs:** graph_edges Delta table, DGL graph artifacts saved to volume
# MAGIC **Upstream:** 04_curate_omop_gold
# MAGIC **Downstream:** 06_graphsage_training

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Extract and unify all edge types
from src.graph.edges import build_all_edges
from src.config import naming

edges_df = build_all_edges(spark, cfg)
edges_df.write.format("delta").mode("overwrite").saveAsTable(cfg.table(naming.GRAPH_EDGES))

print(f"Total edges: {edges_df.count():,}")
display(edges_df.groupBy("edge_type").count().orderBy("edge_type"))

# COMMAND ----------

# DBTITLE 1,Encode node features
from src.graph.features import encode_patient_features, encode_care_site_features

patient_feat_df = spark.table(cfg.table(naming.PATIENT_FEATURES))
care_site_feat_df = spark.table(cfg.table(naming.CARE_SITE_FEATURES))

patient_features, patient_ids = encode_patient_features(patient_feat_df)
care_site_features, care_site_ids = encode_care_site_features(care_site_feat_df)

print(f"Patient features: {patient_features.shape}")
print(f"Care site features: {care_site_features.shape}")

# COMMAND ----------

# DBTITLE 1,Build DGL graph
from src.graph.datasets import build_homogeneous_graph
import pickle

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
print(f"Node feature shape: {graph.ndata['feature'].shape}")

# COMMAND ----------

# DBTITLE 1,Save graph metadata to volume
import json

metadata_serializable = {
    "n_patients": metadata["n_patients"],
    "n_care_sites": metadata["n_care_sites"],
    "n_edges": metadata["n_edges"],
    "patient_ids": metadata["patient_ids"],
    "care_site_ids": metadata["care_site_ids"],
}

metadata_path = f"{cfg.volume_path}/graph_metadata.json"
dbutils.fs.put(metadata_path, json.dumps(metadata_serializable), overwrite=True)
print(f"Metadata saved to {metadata_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `06_graphsage_training`
