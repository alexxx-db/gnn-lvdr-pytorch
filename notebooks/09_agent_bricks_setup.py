# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Agent Bricks Setup
# MAGIC
# MAGIC **Purpose:** Configure the Agent Bricks-based care-navigation recommendation
# MAGIC explainer. Validates that required tables exist and previews agent tools.
# MAGIC
# MAGIC **Inputs:** Recommendations table, gold tables
# MAGIC **Outputs:** Validated agent readiness
# MAGIC **Upstream:** 08_batch_recommendations
# MAGIC **Downstream:** 10_agent_pipeline

# COMMAND ----------

# MAGIC %run ./00_project_config

# COMMAND ----------

# DBTITLE 1,Verify required tables for agent tools
from src.config import naming

required_tables = [
    naming.PATIENT_FEATURES,
    naming.CARE_SITE_FEATURES,
    naming.CONDITION_GOLD,
    naming.RECOMMENDATIONS,
]

print("Checking agent data dependencies:")
for t in required_tables:
    fq = cfg.table(t)
    try:
        count = spark.table(fq).count()
        print(f"  [OK] {fq}: {count:,} rows")
    except Exception as e:
        print(f"  [MISSING] {fq}: {e}")
        raise RuntimeError(f"Required table {fq} not found. Run upstream notebooks first.")

# COMMAND ----------

# DBTITLE 1,Preview agent configuration
from src.agentic.agent_config import AGENT_NAME, SYSTEM_PROMPT, AGENT_DESCRIPTION

print(f"Agent: {AGENT_NAME}")
print(f"Description: {AGENT_DESCRIPTION}")
print(f"\nSystem prompt preview:\n{SYSTEM_PROMPT[:300]}...")

# COMMAND ----------

# DBTITLE 1,Test agent tools with a sample patient
from src.agentic.tools import get_patient_summary, get_top_recommendations

sample_patient_id = spark.table(cfg.table(naming.PATIENT_FEATURES)).first()["person_id"]
print(f"\nTesting tools with patient {sample_patient_id}:")

summary = get_patient_summary(spark, cfg, sample_patient_id)
print(f"\nPatient summary: {summary}")

recs = get_top_recommendations(spark, cfg, sample_patient_id, top_k=3)
print(f"\nTop recommendations: {recs}")

# COMMAND ----------

print("Agent setup validated. Ready for pipeline execution.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `10_agent_pipeline`
