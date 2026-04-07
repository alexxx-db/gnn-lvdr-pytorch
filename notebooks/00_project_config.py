# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Project Configuration
# MAGIC
# MAGIC **Purpose:** Initialize and display the project configuration. All downstream
# MAGIC notebooks import from this shared config rather than defining their own constants.
# MAGIC
# MAGIC **Inputs:** Widget overrides (optional) — passed by bundle job parameters or manually
# MAGIC **Outputs:** `cfg` object available for `%run` consumers
# MAGIC **Upstream:** None (first notebook)
# MAGIC **Downstream:** All other notebooks

# COMMAND ----------

# DBTITLE 1,Install project dependencies
# pip install is fast when packages are already present (no-op).
# For job clusters, prefer installing via cluster libraries/init scripts instead.
# MAGIC %pip install dgl dbldatagen --quiet --disable-pip-version-check

# COMMAND ----------

# DBTITLE 1,Create notebook widgets for runtime overrides
dbutils.widgets.text("catalog", "gnn_hls_graphsage", "Catalog Name")
dbutils.widgets.text("schema", "gnn_hls_graphsage_db", "Schema Name")
dbutils.widgets.text("synthetic_scale", "1000", "Synthetic Data Scale")
dbutils.widgets.dropdown("rebuild_synthetic", "true", ["true", "false"], "Rebuild Synthetic Data")
dbutils.widgets.dropdown("retrain_model", "true", ["true", "false"], "Retrain Model")
dbutils.widgets.text("random_seed", "42", "Random Seed")

# COMMAND ----------

# DBTITLE 1,Configure sys.path for src/ imports
import sys
import os

# Works for both Git folder execution and bundle-deployed notebooks.
# The notebook is in notebooks/, so the repo root is one level up.
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace" + _nb_path.rsplit("/", 2)[0] if "/notebooks/" in _nb_path else "/Workspace" + _nb_path.rsplit("/", 1)[0]

# Also handle bundle deployment where files are synced to workspace root_path
for candidate in [_repo_root, os.getcwd(), "/Workspace" + _nb_path.rsplit("/", 1)[0] + "/.."]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

# COMMAND ----------

# DBTITLE 1,Build configuration object
from src.config.settings import load_config

cfg = load_config({
    "catalog": dbutils.widgets.get("catalog"),
    "schema": dbutils.widgets.get("schema"),
    "synthetic_scale": dbutils.widgets.get("synthetic_scale"),
    "rebuild_synthetic": dbutils.widgets.get("rebuild_synthetic"),
    "retrain_model": dbutils.widgets.get("retrain_model"),
    "random_seed": dbutils.widgets.get("random_seed"),
})

# COMMAND ----------

# DBTITLE 1,Display resolved configuration
print("=" * 60)
print("GNN Patient Recommendations — Runtime Configuration")
print("=" * 60)
for k, v in cfg.display_summary().items():
    print(f"  {k:25s} = {v}")
print("=" * 60)
