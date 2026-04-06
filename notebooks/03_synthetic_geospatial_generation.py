# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Synthetic Geospatial Generation
# MAGIC
# MAGIC **Purpose:** Generate synthetic lat/lon coordinates for patients and care sites,
# MAGIC compute pairwise distances, and identify nearest-site relationships.
# MAGIC
# MAGIC **Inputs:** `cfg`, person_raw, care_site_raw
# MAGIC **Outputs:** patient_locations, care_site_locations, nearest_sites tables
# MAGIC **Upstream:** 02_synthetic_omop_generation
# MAGIC **Downstream:** 04_curate_omop_gold, 05_graph_construction

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

if not cfg.rebuild_synthetic:
    print("Skipping geospatial generation (rebuild_synthetic=False)")
    dbutils.notebook.exit("SKIPPED")

# COMMAND ----------

# DBTITLE 1,Generate patient and care site coordinates
from src.data_generation.geospatial_generator import (
    generate_patient_locations,
    generate_care_site_locations,
    compute_patient_site_distances,
    compute_nearest_sites,
)

patient_locs = generate_patient_locations(spark, cfg)
site_locs = generate_care_site_locations(spark, cfg)

patient_locs.write.format("delta").mode("overwrite").saveAsTable(cfg.table("patient_locations"))
site_locs.write.format("delta").mode("overwrite").saveAsTable(cfg.table("care_site_locations"))

print(f"Patient locations: {patient_locs.count():,} rows")
print(f"Care site locations: {site_locs.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Compute patient-site distances and nearest sites
distances = compute_patient_site_distances(patient_locs, site_locs)
nearest = compute_nearest_sites(distances, top_k=5)

nearest.write.format("delta").mode("overwrite").saveAsTable(cfg.table("nearest_sites"))
print(f"Nearest sites: {nearest.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Preview distance distribution
display(
    nearest
    .groupBy("distance_band")
    .count()
    .orderBy("distance_band")
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `04_curate_omop_gold`
