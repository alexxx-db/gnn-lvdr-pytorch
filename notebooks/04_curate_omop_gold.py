# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Curate OMOP Gold Tables
# MAGIC
# MAGIC **Purpose:** Transform raw OMOP tables into curated gold layer with
# MAGIC derived fields (age, age_bucket) and build patient/care-site feature tables.
# MAGIC
# MAGIC **Inputs:** Raw OMOP tables, patient_locations, care_site_locations
# MAGIC **Outputs:** Gold tables + patient_features + care_site_features
# MAGIC **Upstream:** 02_synthetic_omop_generation, 03_synthetic_geospatial_generation
# MAGIC **Downstream:** 05_graph_construction

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Curate gold tables
from src.omop.transforms import curate_all_gold

gold_tables = curate_all_gold(spark, cfg)

for name, df in gold_tables.items():
    fq_name = cfg.table(name)
    df.write.format("delta").mode("overwrite").saveAsTable(fq_name)
    count = spark.table(fq_name).count()
    print(f"  {fq_name}: {count:,} rows")

# COMMAND ----------

# DBTITLE 1,Build patient feature table
from src.omop.transforms import build_patient_features, build_care_site_features
from src.config import naming

patient_features = build_patient_features(spark, cfg)
patient_features.write.format("delta").mode("overwrite").saveAsTable(cfg.table(naming.PATIENT_FEATURES))
print(f"Patient features: {patient_features.count():,} rows, {len(patient_features.columns)} columns")

# COMMAND ----------

# DBTITLE 1,Build care site feature table
care_site_features = build_care_site_features(spark, cfg)
care_site_features.write.format("delta").mode("overwrite").saveAsTable(cfg.table(naming.CARE_SITE_FEATURES))
print(f"Care site features: {care_site_features.count():,} rows, {len(care_site_features.columns)} columns")

# COMMAND ----------

# DBTITLE 1,Validate gold tables
from src.omop.validators import validate_gold_tables, print_results

results = validate_gold_tables(spark, cfg)
all_passed = print_results(results)
assert all_passed, "Gold table validation failed!"

# COMMAND ----------

# DBTITLE 1,Preview patient features
display(patient_features.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `05_graph_construction`
