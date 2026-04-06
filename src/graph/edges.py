"""
Edge extraction from curated OMOP tables.

Builds edge lists for each edge type, with deduplication and
optional edge weights (visit count, distance inverse, etc.).
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from src.config.settings import ProjectConfig
from src.config import naming


def extract_visited_edges(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Patient → visited → care_site edges from visit_gold."""
    visits = spark.table(cfg.table(naming.VISIT_GOLD))
    return (
        visits
        .groupBy("person_id", "care_site_id")
        .agg(F.count("visit_occurrence_id").alias("visit_count"))
        .withColumn("edge_type", F.lit("visited"))
        .select(
            F.col("person_id").alias("src_id"),
            F.col("care_site_id").alias("dst_id"),
            "edge_type",
            F.col("visit_count").cast("float").alias("weight"),
        )
    )


def extract_diagnosed_edges(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Patient → diagnosed_with → condition edges from condition_gold."""
    conditions = spark.table(cfg.table(naming.CONDITION_GOLD))
    return (
        conditions
        .groupBy("person_id", "condition_concept_id")
        .agg(F.count("condition_occurrence_id").alias("dx_count"))
        .withColumn("edge_type", F.lit("diagnosed_with"))
        .select(
            F.col("person_id").alias("src_id"),
            F.col("condition_concept_id").alias("dst_id"),
            "edge_type",
            F.col("dx_count").cast("float").alias("weight"),
        )
    )


def extract_treated_by_edges(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Patient → treated_by → provider edges from visit_gold."""
    visits = spark.table(cfg.table(naming.VISIT_GOLD))
    return (
        visits
        .filter(F.col("provider_id").isNotNull())
        .groupBy("person_id", "provider_id")
        .agg(F.count("visit_occurrence_id").alias("encounter_count"))
        .withColumn("edge_type", F.lit("treated_by"))
        .select(
            F.col("person_id").alias("src_id"),
            F.col("provider_id").alias("dst_id"),
            "edge_type",
            F.col("encounter_count").cast("float").alias("weight"),
        )
    )


def extract_works_at_edges(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Provider → works_at → care_site edges from provider_gold."""
    providers = spark.table(cfg.table(naming.PROVIDER_GOLD))
    return (
        providers
        .filter(F.col("care_site_id").isNotNull())
        .select(
            F.col("provider_id").alias("src_id"),
            F.col("care_site_id").alias("dst_id"),
        )
        .distinct()
        .withColumn("edge_type", F.lit("works_at"))
        .withColumn("weight", F.lit(1.0))
    )


def extract_near_edges(spark: SparkSession, cfg: ProjectConfig,
                       nearest_sites_table: str = "nearest_sites") -> DataFrame:
    """Patient → near → care_site edges from geospatial nearest-site computation."""
    nearest = spark.table(cfg.table(nearest_sites_table))
    return (
        nearest
        .select(
            F.col("person_id").alias("src_id"),
            F.col("care_site_id").alias("dst_id"),
            F.lit("near").alias("edge_type"),
            (F.lit(1.0) / (F.col("distance_miles") + F.lit(1.0))).alias("weight"),
        )
    )


def build_all_edges(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """
    Combine all edge types into a single unified edge table.

    Returns DataFrame with: src_id, dst_id, edge_type, weight.
    """
    edges = [
        extract_visited_edges(spark, cfg),
        extract_diagnosed_edges(spark, cfg),
        extract_treated_by_edges(spark, cfg),
        extract_works_at_edges(spark, cfg),
    ]
    # Only include near edges if the table exists
    try:
        near = extract_near_edges(spark, cfg)
        edges.append(near)
    except Exception:
        pass

    from functools import reduce
    return reduce(DataFrame.unionByName, edges)
