# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — Databricks App Preparation
# MAGIC
# MAGIC **Purpose:** Prepare and optionally deploy the Gradio-based Databricks App
# MAGIC for interactive recommendation exploration.
# MAGIC
# MAGIC **Inputs:** Recommendations, recommendations_explained tables
# MAGIC **Outputs:** App deployment (if Databricks Apps is available)
# MAGIC **Upstream:** 10_agent_pipeline
# MAGIC **Downstream:** 12_validation_and_smoke_tests

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Verify app data dependencies
from src.config import naming

for table in [naming.RECOMMENDATIONS, naming.RECOMMENDATIONS_EXPLAINED, naming.PATIENT_FEATURES]:
    fq = cfg.table(table)
    count = spark.table(fq).count()
    print(f"  {fq}: {count:,} rows")

# COMMAND ----------

# DBTITLE 1,App configuration summary
print("Databricks App Configuration")
print("=" * 50)
print(f"  App name:     {cfg.app_name}")
print(f"  Catalog:      {cfg.catalog}")
print(f"  Schema:       {cfg.schema}")
print(f"  Entry point:  src/app/gradio_app.py")
print(f"  Config file:  resources/app/app.yaml")
print()
print("To deploy the app manually:")
print(f"  1. Navigate to Databricks Workspace > Apps")
print(f"  2. Create a new app named '{cfg.app_name}'")
print(f"  3. Point source to this repo's src/app/gradio_app.py")
print(f"  4. Set environment variables:")
print(f"     APP_CATALOG={cfg.catalog}")
print(f"     APP_SCHEMA={cfg.schema}")
print(f"     DATABRICKS_WAREHOUSE_ID=<your-warehouse-id>")

# COMMAND ----------

# DBTITLE 1,Attempt automated app deployment (requires Databricks Apps support)
try:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    print(f"Workspace URL: {w.config.host}")
    print("Databricks SDK available. App can be deployed via API.")
    print("See resources/app/app.yaml for the app configuration.")
except Exception as e:
    print(f"Note: Automated deployment requires Databricks SDK. ({e})")
    print("Deploy the app manually using the instructions above.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `12_validation_and_smoke_tests`
