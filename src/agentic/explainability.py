"""
Explainability utilities for recommendation outputs.

Generates structured explanation artifacts for logging and display.
"""
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from src.config.settings import ProjectConfig
from src.config import naming


def build_explanation_table(
    spark: SparkSession,
    cfg: ProjectConfig,
    explanations: list[dict],
) -> DataFrame:
    """
    Convert a list of explanation dicts into a Delta table.

    Each row contains: person_id, care_site_id, score, rank,
    explanation_text, and key factors.
    """
    rows = []
    for exp in explanations:
        rows.append({
            "person_id": exp["person_id"],
            "care_site_id": exp["care_site_id"],
            "score": exp.get("recommendation", {}).get("score"),
            "rank": exp.get("recommendation", {}).get("rank"),
            "explanation_text": exp.get("explanation_text", ""),
            "factors": " | ".join(exp.get("explanation_factors", [])),
        })

    pdf = pd.DataFrame(rows)
    sdf = spark.createDataFrame(pdf)

    fq = cfg.table(naming.RECOMMENDATIONS_EXPLAINED)
    sdf.write.format("delta").mode("overwrite").saveAsTable(fq)
    return sdf
