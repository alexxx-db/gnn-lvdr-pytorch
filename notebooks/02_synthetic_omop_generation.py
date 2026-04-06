# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Synthetic OMOP Data Generation
# MAGIC
# MAGIC **Purpose:** Generate synthetic OMOP-aligned clinical data using dbldatagen.
# MAGIC Creates person, care_site, provider, location, visit_occurrence,
# MAGIC condition_occurrence, drug_exposure, and procedure_occurrence tables.
# MAGIC
# MAGIC **Inputs:** `cfg` from 00_project_config
# MAGIC **Outputs:** 8 raw Delta tables in `{catalog}.{schema}`
# MAGIC **Upstream:** 01_workspace_setup
# MAGIC **Downstream:** 03_synthetic_geospatial_generation, 04_curate_omop_gold

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Check if rebuild is requested
if not cfg.rebuild_synthetic:
    print("Skipping synthetic data generation (rebuild_synthetic=False)")
    dbutils.notebook.exit("SKIPPED")

# COMMAND ----------

# DBTITLE 1,Generate all OMOP tables
from src.data_generation.omop_generator import generate_all_omop

tables = generate_all_omop(spark, cfg)

for name, df in tables.items():
    fq_name = cfg.table(name)
    df.write.format("delta").mode("overwrite").saveAsTable(fq_name)
    count = spark.table(fq_name).count()
    print(f"  {fq_name}: {count:,} rows")

# COMMAND ----------

# DBTITLE 1,Validate: all raw tables exist and have data
from src.omop.validators import validate_raw_tables, print_results

results = validate_raw_tables(spark, cfg)
all_passed = print_results(results)
assert all_passed, "Raw table validation failed!"
print("\nAll raw OMOP tables generated and validated.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `03_synthetic_geospatial_generation`
