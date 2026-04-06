# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Validation and Smoke Tests
# MAGIC
# MAGIC **Purpose:** End-to-end validation that all project outputs exist and
# MAGIC pass basic quality checks. Run this after the full pipeline.
# MAGIC
# MAGIC **Inputs:** All project outputs
# MAGIC **Outputs:** Validation report
# MAGIC **Upstream:** All previous notebooks
# MAGIC **Downstream:** None (final notebook)

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Check all expected Delta tables
from src.config import naming

all_tables = (
    naming.all_raw_tables()
    + naming.all_gold_tables()
    + [naming.GRAPH_EDGES, naming.RECOMMENDATIONS, naming.RECOMMENDATIONS_EXPLAINED]
    + ["patient_locations", "care_site_locations", "nearest_sites"]
)

print("Delta Table Validation")
print("=" * 70)
failures = []
for t in all_tables:
    fq = cfg.table(t)
    try:
        count = spark.table(fq).count()
        status = "OK" if count > 0 else "EMPTY"
        print(f"  [{status:5s}] {fq}: {count:,} rows")
        if count == 0:
            failures.append(fq)
    except Exception as e:
        print(f"  [FAIL ] {fq}: {e}")
        failures.append(fq)

# COMMAND ----------

# DBTITLE 1,Check MLflow experiment exists
import mlflow

experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
if experiment:
    print(f"[OK] MLflow experiment: {cfg.experiment_name} (id={experiment.experiment_id})")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
    print(f"     Recent runs: {len(runs)}")
    if not runs.empty:
        display(runs[["run_id", "status", "start_time", "tags.mlflow.runName"]].head())
else:
    print(f"[FAIL] MLflow experiment not found: {cfg.experiment_name}")
    failures.append("mlflow_experiment")

# COMMAND ----------

# DBTITLE 1,Check model registry
try:
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.uc_model}'")
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        print(f"[OK] Registered model: {cfg.uc_model}")
        print(f"     Latest version: {latest.version}")
        try:
            champion = client.get_model_version_by_alias(cfg.uc_model, "champion")
            print(f"     Champion alias: v{champion.version}")
        except Exception:
            print("     Champion alias: not set")
    else:
        print(f"[WARN] No model versions found for {cfg.uc_model}")
except Exception as e:
    print(f"[WARN] Model registry check: {e}")

# COMMAND ----------

# DBTITLE 1,Recommendation quality checks
recs = spark.table(cfg.table(naming.RECOMMENDATIONS))

rec_count = recs.count()
patient_count = recs.select("person_id").distinct().count()
avg_score = recs.selectExpr("avg(score)").first()[0]

print(f"\nRecommendation Quality")
print(f"  Total recommendations: {rec_count:,}")
print(f"  Unique patients: {patient_count:,}")
print(f"  Average score: {avg_score:.4f}" if avg_score else "  Average score: N/A")

assert rec_count > 0, "No recommendations generated!"
assert patient_count > 0, "No patients with recommendations!"

# COMMAND ----------

# DBTITLE 1,Summary
print("\n" + "=" * 70)
if failures:
    print(f"VALIDATION COMPLETE: {len(failures)} issue(s)")
    for f in failures:
        print(f"  - {f}")
else:
    print("VALIDATION COMPLETE: All checks passed!")
print("=" * 70)
