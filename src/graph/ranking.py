"""
Recommendation ranking and output formatting.

Takes raw GNN scores and produces top-k ranked recommendations
with supporting metadata for explainability.
"""
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.config.settings import ProjectConfig
from src.config import naming


def rank_recommendations(
    scores_pdf: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Rank care sites for each patient by GNN score.

    Returns top-k recommendations per patient.
    """
    ranked = (
        scores_pdf
        .sort_values(["person_id", "score"], ascending=[True, False])
        .groupby("person_id")
        .head(top_k)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.groupby("person_id").cumcount() + 1
    return ranked


def enrich_recommendations(
    spark: SparkSession,
    cfg: ProjectConfig,
    recommendations_sdf: DataFrame,
) -> DataFrame:
    """
    Enrich recommendations with care site names, types, and patient context.

    Joins recommendation scores with care_site_gold and person_gold tables.
    """
    care_sites = spark.table(cfg.table(naming.CARE_SITE_GOLD)).select(
        "care_site_id", "care_site_name", "care_site_type",
    )
    persons = spark.table(cfg.table(naming.PERSON_GOLD)).select(
        "person_id", "gender_source_value", "age_bucket",
    )

    enriched = (
        recommendations_sdf
        .join(care_sites, "care_site_id", "left")
        .join(persons, "person_id", "left")
    )

    # Try to add distance info if nearest_sites table exists
    try:
        nearest = spark.table(cfg.table("nearest_sites")).select(
            "person_id", "care_site_id", "distance_miles",
        )
        enriched = enriched.join(nearest, ["person_id", "care_site_id"], "left")
    except Exception:
        enriched = enriched.withColumn("distance_miles", F.lit(None).cast("float"))

    return enriched


def save_recommendations(
    spark: SparkSession,
    cfg: ProjectConfig,
    recommendations_pdf: pd.DataFrame,
) -> DataFrame:
    """
    Save ranked recommendations as a Delta table.

    Returns the Spark DataFrame that was written.
    """
    sdf = spark.createDataFrame(recommendations_pdf)
    enriched = enrich_recommendations(spark, cfg, sdf)
    fq = cfg.table(naming.RECOMMENDATIONS)
    enriched.write.format("delta").mode("overwrite").saveAsTable(fq)
    return enriched
