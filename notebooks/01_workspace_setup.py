# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Workspace Setup
# MAGIC
# MAGIC **Purpose:** Create the Unity Catalog catalog, schema, and volume if they
# MAGIC do not already exist. Idempotent — safe to re-run.
# MAGIC
# MAGIC **Inputs:** `cfg` from 00_project_config
# MAGIC **Outputs:** UC catalog, schema, volume
# MAGIC **Upstream:** 00_project_config
# MAGIC **Downstream:** 02_synthetic_omop_generation

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Create Unity Catalog resources
spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{cfg.volume}")

# COMMAND ----------

# DBTITLE 1,Set active namespace
spark.sql(f"USE CATALOG {cfg.catalog}")
spark.sql(f"USE SCHEMA {cfg.schema}")

# COMMAND ----------

# DBTITLE 1,Validate setup
print(f"Catalog:  {cfg.catalog}")
print(f"Schema:   {cfg.schema}")
print(f"Volume:   {cfg.volume_path}")
print(f"Tables will be written to: {cfg.uc_prefix}.*")

# Verify access
tables = spark.sql(f"SHOW TABLES IN {cfg.uc_prefix}").count()
print(f"Existing tables in schema: {tables}")
print("Workspace setup complete.")
