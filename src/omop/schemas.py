"""
OMOP CDM schema definitions for the practical subset used in this project.

These schemas define the expected structure for curated gold tables.
They are used for validation and documentation, not enforcement at write time.
"""
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType,
    DateType, FloatType, LongType,
)

PERSON_SCHEMA = StructType([
    StructField("person_id", IntegerType(), False),
    StructField("gender_source_value", StringType(), True),
    StructField("year_of_birth", IntegerType(), True),
    StructField("race_source_value", StringType(), True),
    StructField("ethnicity_source_value", StringType(), True),
    StructField("location_id", IntegerType(), True),
    StructField("provider_id", IntegerType(), True),
    StructField("care_site_id", IntegerType(), True),
])

CARE_SITE_SCHEMA = StructType([
    StructField("care_site_id", IntegerType(), False),
    StructField("care_site_name", StringType(), True),
    StructField("care_site_type", StringType(), True),
    StructField("location_id", IntegerType(), True),
])

PROVIDER_SCHEMA = StructType([
    StructField("provider_id", IntegerType(), False),
    StructField("provider_name", StringType(), True),
    StructField("specialty_source_value", StringType(), True),
    StructField("care_site_id", IntegerType(), True),
])

VISIT_SCHEMA = StructType([
    StructField("visit_occurrence_id", IntegerType(), False),
    StructField("person_id", IntegerType(), False),
    StructField("visit_concept_id", IntegerType(), True),
    StructField("visit_start_date", DateType(), True),
    StructField("visit_end_date", DateType(), True),
    StructField("care_site_id", IntegerType(), True),
    StructField("provider_id", IntegerType(), True),
])

CONDITION_SCHEMA = StructType([
    StructField("condition_occurrence_id", IntegerType(), False),
    StructField("person_id", IntegerType(), False),
    StructField("condition_concept_id", IntegerType(), True),
    StructField("condition_start_date", DateType(), True),
    StructField("condition_source_value", StringType(), True),
])
