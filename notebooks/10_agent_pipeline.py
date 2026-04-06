# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Agent Pipeline
# MAGIC
# MAGIC **Purpose:** Run the agentic recommendation explanation pipeline.
# MAGIC Generates natural-language explanations for top recommendations
# MAGIC and saves them to a Delta table with MLflow tracing.
# MAGIC
# MAGIC **Inputs:** Recommendations table, gold tables
# MAGIC **Outputs:** recommendations_explained Delta table
# MAGIC **Upstream:** 09_agent_bricks_setup
# MAGIC **Downstream:** 11_databricks_app_prep

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Enable MLflow tracing
import mlflow
from src.mlops.tracking import setup_experiment

setup_experiment(cfg)
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# DBTITLE 1,Run batch explanations for a sample of patients
from src.agentic.orchestration import batch_explain
from src.config import naming

# Get a sample of patients who have recommendations
sample_patients = (
    spark.table(cfg.table(naming.RECOMMENDATIONS))
    .select("person_id")
    .distinct()
    .limit(50)
    .toPandas()["person_id"]
    .tolist()
)

print(f"Generating explanations for {len(sample_patients)} patients...")

with mlflow.start_run(run_name="AgentPipeline-Explanations"):
    mlflow.set_tag("pipeline", "agent-explanations")
    mlflow.log_param("n_patients", len(sample_patients))

    explanations = batch_explain(spark, cfg, sample_patients, top_k=3)
    mlflow.log_metric("n_explanations", len(explanations))

print(f"Generated {len(explanations)} explanations")

# COMMAND ----------

# DBTITLE 1,Save explanations to Delta
from src.agentic.explainability import build_explanation_table

explained_df = build_explanation_table(spark, cfg, explanations)
print(f"Explained recommendations: {explained_df.count()} rows")
display(explained_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Preview a single explanation
if explanations:
    print(explanations[0]["explanation_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `11_databricks_app_prep`
