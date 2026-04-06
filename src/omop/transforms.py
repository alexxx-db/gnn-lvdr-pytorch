"""
OMOP data curation transforms: raw → gold layer.

Selects relevant columns, applies type coercions, deduplicates,
and computes derived fields needed by the graph and feature layers.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from src.config.settings import ProjectConfig
from src.config import naming


def curate_person(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Raw person → gold person with age and derived fields."""
    raw = spark.table(cfg.table(naming.PERSON_RAW))
    return (
        raw
        .select(
            "person_id", "gender_source_value", "year_of_birth",
            "race_source_value", "ethnicity_source_value",
            "location_id", "provider_id", "care_site_id",
        )
        .withColumn("age", F.lit(2024) - F.col("year_of_birth"))
        .withColumn("age_bucket",
                    F.when(F.col("age") < 30, "18-29")
                    .when(F.col("age") < 45, "30-44")
                    .when(F.col("age") < 60, "45-59")
                    .when(F.col("age") < 75, "60-74")
                    .otherwise("75+"))
        .dropDuplicates(["person_id"])
    )


def curate_care_site(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Raw care_site → gold care_site."""
    raw = spark.table(cfg.table(naming.CARE_SITE_RAW))
    return (
        raw
        .select("care_site_id", "care_site_name", "care_site_type", "location_id")
        .dropDuplicates(["care_site_id"])
    )


def curate_provider(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Raw provider → gold provider."""
    raw = spark.table(cfg.table(naming.PROVIDER_RAW))
    return (
        raw
        .select("provider_id", "provider_name", "specialty_source_value", "care_site_id")
        .dropDuplicates(["provider_id"])
    )


def curate_visits(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Raw visit_occurrence → gold visits."""
    raw = spark.table(cfg.table(naming.VISIT_OCCURRENCE_RAW))
    return (
        raw
        .select(
            "visit_occurrence_id", "person_id", "visit_concept_id",
            "visit_start_date", "visit_end_date",
            "care_site_id", "provider_id",
        )
        .dropDuplicates(["visit_occurrence_id"])
    )


def curate_conditions(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Raw condition_occurrence → gold conditions."""
    raw = spark.table(cfg.table(naming.CONDITION_OCCURRENCE_RAW))
    return (
        raw
        .select(
            "condition_occurrence_id", "person_id",
            "condition_concept_id", "condition_start_date",
            "condition_source_value",
        )
        .dropDuplicates(["condition_occurrence_id"])
    )


def build_patient_features(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """
    Build patient feature table for graph node features.

    Aggregates visit counts, condition counts, distinct care sites visited,
    and demographic info into a single row per patient.
    """
    person = spark.table(cfg.table(naming.PERSON_GOLD))
    visits = spark.table(cfg.table(naming.VISIT_GOLD))
    conditions = spark.table(cfg.table(naming.CONDITION_GOLD))

    visit_agg = (
        visits.groupBy("person_id")
        .agg(
            F.count("visit_occurrence_id").alias("visit_count"),
            F.countDistinct("care_site_id").alias("distinct_care_sites"),
            F.countDistinct("provider_id").alias("distinct_providers"),
        )
    )

    cond_agg = (
        conditions.groupBy("person_id")
        .agg(
            F.count("condition_occurrence_id").alias("condition_count"),
            F.countDistinct("condition_concept_id").alias("distinct_conditions"),
        )
    )

    return (
        person
        .join(visit_agg, "person_id", "left")
        .join(cond_agg, "person_id", "left")
        .fillna(0, subset=[
            "visit_count", "distinct_care_sites", "distinct_providers",
            "condition_count", "distinct_conditions",
        ])
    )


def build_care_site_features(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """
    Build care site feature table for graph node features.

    Aggregates patient volume, provider count, and visit patterns per site.
    """
    care_site = spark.table(cfg.table(naming.CARE_SITE_GOLD))
    visits = spark.table(cfg.table(naming.VISIT_GOLD))

    site_agg = (
        visits.groupBy("care_site_id")
        .agg(
            F.countDistinct("person_id").alias("patient_volume"),
            F.countDistinct("provider_id").alias("provider_count"),
            F.count("visit_occurrence_id").alias("total_visits"),
        )
    )

    return (
        care_site
        .join(site_agg, "care_site_id", "left")
        .fillna(0, subset=["patient_volume", "provider_count", "total_visits"])
    )


def curate_all_gold(spark: SparkSession, cfg: ProjectConfig) -> dict[str, DataFrame]:
    """Run all curation transforms. Returns {table_name: DataFrame}."""
    return {
        naming.PERSON_GOLD: curate_person(spark, cfg),
        naming.CARE_SITE_GOLD: curate_care_site(spark, cfg),
        naming.PROVIDER_GOLD: curate_provider(spark, cfg),
        naming.VISIT_GOLD: curate_visits(spark, cfg),
        naming.CONDITION_GOLD: curate_conditions(spark, cfg),
    }
