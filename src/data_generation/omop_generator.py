"""
Synthetic OMOP-aligned data generation using dbldatagen.

Generates a practical subset of OMOP CDM tables with coherent relationships:
person, care_site, provider, location, condition_occurrence, visit_occurrence,
drug_exposure, procedure_occurrence.

All data is synthetic and contains no real PHI.
"""
import dbldatagen as dg
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType,
    DateType, FloatType, LongType,
)
from src.config.settings import ProjectConfig


# ---------------------------------------------------------------------------
# Reference value pools
# ---------------------------------------------------------------------------
CONDITION_CONCEPTS = [
    (201826, "Type 2 diabetes mellitus"),
    (316139, "Heart failure"),
    (317009, "Asthma"),
    (4229881, "Acute myocardial infarction"),
    (255573, "Chronic obstructive lung disease"),
    (4024552, "Malignant neoplasm of breast"),
    (40481902, "Major depressive disorder"),
    (4185932, "Chronic kidney disease"),
    (320128, "Essential hypertension"),
    (4116142, "Osteoarthritis"),
]

DRUG_CONCEPTS = [
    (1503297, "Metformin"),
    (1308216, "Lisinopril"),
    (19078461, "Atorvastatin"),
    (1124300, "Amlodipine"),
    (1332419, "Metoprolol"),
    (1154343, "Albuterol"),
    (40163924, "Sertraline"),
    (19049105, "Omeprazole"),
    (1177480, "Furosemide"),
    (1112807, "Aspirin"),
]

PROCEDURE_CONCEPTS = [
    (2213473, "Office visit - established patient"),
    (2313905, "Complete blood count"),
    (2212389, "Comprehensive metabolic panel"),
    (2313842, "Lipid panel"),
    (2414397, "Chest X-ray"),
    (2514576, "Echocardiography"),
    (2313853, "Hemoglobin A1c"),
    (2212272, "Urinalysis"),
    (2414398, "Electrocardiogram"),
    (2313901, "Thyroid stimulating hormone"),
]

GENDER_VALUES = ["MALE", "FEMALE"]
RACE_VALUES = ["White", "Black or African American", "Asian", "Other"]
ETHNICITY_VALUES = ["Not Hispanic or Latino", "Hispanic or Latino"]

CARE_SITE_TYPES = [
    "Primary Care Clinic", "Urgent Care Center", "Hospital",
    "Specialty Clinic", "Community Health Center",
    "Rehabilitation Center", "Mental Health Clinic",
]

PROVIDER_SPECIALTIES = [
    "Internal Medicine", "Family Medicine", "Cardiology",
    "Endocrinology", "Pulmonology", "Psychiatry",
    "Orthopedics", "Nephrology", "Oncology", "General Practice",
]


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def generate_locations(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic location records for patients, care sites, providers."""
    n_locations = cfg.synthetic_scale * 3  # patients + care sites + providers
    spec = (
        dg.DataGenerator(spark, name="locations", rows=n_locations, seedColumnName="location_id")
        .withColumn("address_1", "string", template=r"\\d\\d\\d\\d Synthetic St")
        .withColumn("city", "string", values=["Springfield", "Riverside", "Fairview", "Madison",
                                               "Georgetown", "Clinton", "Greenville", "Bristol",
                                               "Oakland", "Burlington"],
                    random=True)
        .withColumn("state", "string", values=["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
                    random=True)
        .withColumn("zip", "string", template=r"\\d\\d\\d\\d\\d")
        .withColumn("county", "string", values=["County A", "County B", "County C", "County D"],
                    random=True)
    )
    return spec.build()


def generate_care_sites(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic care site records."""
    n_sites = max(20, cfg.synthetic_scale // 10)
    spec = (
        dg.DataGenerator(spark, name="care_sites", rows=n_sites, seedColumnName="care_site_id")
        .withColumn("care_site_name", "string",
                    template=r"Clinic \\w\\w\\w-\\d\\d\\d")
        .withColumn("place_of_service_concept_id", "int",
                    values=[8717, 8756, 8940, 8883, 8976], random=True)
        .withColumn("care_site_type", "string",
                    values=CARE_SITE_TYPES, random=True)
        .withColumn("location_id", "int",
                    minValue=cfg.synthetic_scale + 1,
                    maxValue=cfg.synthetic_scale + n_sites)
    )
    return spec.build()


def generate_providers(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic provider records."""
    n_providers = max(30, cfg.synthetic_scale // 5)
    n_sites = max(20, cfg.synthetic_scale // 10)
    spec = (
        dg.DataGenerator(spark, name="providers", rows=n_providers, seedColumnName="provider_id")
        .withColumn("provider_name", "string",
                    template=r"Dr. \\w\\w\\w\\w\\w\\w")
        .withColumn("npi", "string", template=r"\\d\\d\\d\\d\\d\\d\\d\\d\\d\\d")
        .withColumn("specialty_concept_id", "int",
                    values=[38004446, 38004448, 38004449, 38004450, 38004451,
                            38004452, 38004453, 38004454, 38004455, 38004456],
                    random=True)
        .withColumn("specialty_source_value", "string",
                    values=PROVIDER_SPECIALTIES, random=True)
        .withColumn("care_site_id", "int", minValue=1, maxValue=n_sites, random=True)
    )
    return spec.build()


def generate_persons(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic person records."""
    spec = (
        dg.DataGenerator(spark, name="persons", rows=cfg.synthetic_scale,
                         seedColumnName="person_id")
        .withColumn("gender_source_value", "string",
                    values=GENDER_VALUES, random=True, weights=[0.48, 0.52])
        .withColumn("year_of_birth", "int", minValue=1940, maxValue=2005, random=True)
        .withColumn("month_of_birth", "int", minValue=1, maxValue=12, random=True)
        .withColumn("day_of_birth", "int", minValue=1, maxValue=28, random=True)
        .withColumn("race_source_value", "string",
                    values=RACE_VALUES, random=True, weights=[0.55, 0.20, 0.10, 0.15])
        .withColumn("ethnicity_source_value", "string",
                    values=ETHNICITY_VALUES, random=True, weights=[0.80, 0.20])
        .withColumn("location_id", "int", minValue=1, maxValue=cfg.synthetic_scale)
        .withColumn("provider_id", "int",
                    minValue=1, maxValue=max(30, cfg.synthetic_scale // 5),
                    random=True)
        .withColumn("care_site_id", "int",
                    minValue=1, maxValue=max(20, cfg.synthetic_scale // 10),
                    random=True)
    )
    return spec.build()


def generate_visit_occurrences(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic visit records linking patients to care sites."""
    n_visits = cfg.synthetic_scale * 5
    n_sites = max(20, cfg.synthetic_scale // 10)
    n_providers = max(30, cfg.synthetic_scale // 5)
    spec = (
        dg.DataGenerator(spark, name="visits", rows=n_visits,
                         seedColumnName="visit_occurrence_id")
        .withColumn("person_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale, random=True)
        .withColumn("visit_concept_id", "int",
                    values=[9201, 9202, 9203, 262], random=True,
                    weights=[0.15, 0.60, 0.10, 0.15])
        .withColumn("visit_start_date", "date",
                    begin="2020-01-01", end="2024-12-31", random=True)
        .withColumn("visit_end_date", "date",
                    begin="2020-01-01", end="2024-12-31", random=True)
        .withColumn("visit_type_concept_id", "int", values=[44818517], random=True)
        .withColumn("care_site_id", "int",
                    minValue=1, maxValue=n_sites, random=True)
        .withColumn("provider_id", "int",
                    minValue=1, maxValue=n_providers, random=True)
    )
    df = spec.build()
    # Ensure visit_end_date >= visit_start_date
    df = df.withColumn("visit_end_date",
                       F.when(F.col("visit_end_date") < F.col("visit_start_date"),
                              F.col("visit_start_date"))
                       .otherwise(F.col("visit_end_date")))
    return df


def generate_condition_occurrences(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic condition occurrence records."""
    n_conditions = cfg.synthetic_scale * 3
    concept_ids = [c[0] for c in CONDITION_CONCEPTS]
    concept_names = [c[1] for c in CONDITION_CONCEPTS]
    spec = (
        dg.DataGenerator(spark, name="conditions", rows=n_conditions,
                         seedColumnName="condition_occurrence_id")
        .withColumn("person_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale, random=True)
        .withColumn("condition_concept_id", "int",
                    values=concept_ids, random=True)
        .withColumn("condition_start_date", "date",
                    begin="2020-01-01", end="2024-12-31", random=True)
        .withColumn("condition_type_concept_id", "int", values=[32817], random=True)
        .withColumn("condition_source_value", "string",
                    values=concept_names, random=True)
        .withColumn("visit_occurrence_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale * 5, random=True)
    )
    return spec.build()


def generate_drug_exposures(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic drug exposure records."""
    n_drugs = cfg.synthetic_scale * 2
    concept_ids = [d[0] for d in DRUG_CONCEPTS]
    concept_names = [d[1] for d in DRUG_CONCEPTS]
    spec = (
        dg.DataGenerator(spark, name="drugs", rows=n_drugs,
                         seedColumnName="drug_exposure_id")
        .withColumn("person_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale, random=True)
        .withColumn("drug_concept_id", "int",
                    values=concept_ids, random=True)
        .withColumn("drug_exposure_start_date", "date",
                    begin="2020-01-01", end="2024-12-31", random=True)
        .withColumn("drug_type_concept_id", "int", values=[38000177], random=True)
        .withColumn("drug_source_value", "string",
                    values=concept_names, random=True)
        .withColumn("visit_occurrence_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale * 5, random=True)
    )
    return spec.build()


def generate_procedure_occurrences(spark: SparkSession, cfg: ProjectConfig) -> DataFrame:
    """Generate synthetic procedure occurrence records."""
    n_procedures = cfg.synthetic_scale * 2
    concept_ids = [p[0] for p in PROCEDURE_CONCEPTS]
    concept_names = [p[1] for p in PROCEDURE_CONCEPTS]
    spec = (
        dg.DataGenerator(spark, name="procedures", rows=n_procedures,
                         seedColumnName="procedure_occurrence_id")
        .withColumn("person_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale, random=True)
        .withColumn("procedure_concept_id", "int",
                    values=concept_ids, random=True)
        .withColumn("procedure_date", "date",
                    begin="2020-01-01", end="2024-12-31", random=True)
        .withColumn("procedure_type_concept_id", "int", values=[38000275], random=True)
        .withColumn("procedure_source_value", "string",
                    values=concept_names, random=True)
        .withColumn("visit_occurrence_id", "int",
                    minValue=1, maxValue=cfg.synthetic_scale * 5, random=True)
    )
    return spec.build()


def generate_all_omop(spark: SparkSession, cfg: ProjectConfig) -> dict[str, DataFrame]:
    """Generate all OMOP tables. Returns {table_name: DataFrame}."""
    return {
        "location_raw": generate_locations(spark, cfg),
        "care_site_raw": generate_care_sites(spark, cfg),
        "provider_raw": generate_providers(spark, cfg),
        "person_raw": generate_persons(spark, cfg),
        "visit_occurrence_raw": generate_visit_occurrences(spark, cfg),
        "condition_occurrence_raw": generate_condition_occurrences(spark, cfg),
        "drug_exposure_raw": generate_drug_exposures(spark, cfg),
        "procedure_occurrence_raw": generate_procedure_occurrences(spark, cfg),
    }
