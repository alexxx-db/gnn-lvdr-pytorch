"""
Data quality validators for OMOP tables.

Simple row-count, null-check, and referential integrity assertions
used after data generation and curation steps.
"""
from pyspark.sql import SparkSession, DataFrame
from src.config.settings import ProjectConfig
from src.config import naming


class ValidationResult:
    def __init__(self, table: str, check: str, passed: bool, detail: str = ""):
        self.table = table
        self.check = check
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.table}.{self.check}: {self.detail}"


def check_not_empty(spark: SparkSession, cfg: ProjectConfig,
                    table_name: str) -> ValidationResult:
    """Assert table has > 0 rows."""
    fq = cfg.table(table_name)
    try:
        count = spark.table(fq).count()
        return ValidationResult(table_name, "not_empty", count > 0,
                                f"{count} rows")
    except Exception as e:
        return ValidationResult(table_name, "not_empty", False, str(e))


def check_no_null_pk(spark: SparkSession, cfg: ProjectConfig,
                     table_name: str, pk_col: str) -> ValidationResult:
    """Assert primary key column has no nulls."""
    fq = cfg.table(table_name)
    try:
        df = spark.table(fq)
        null_count = df.filter(df[pk_col].isNull()).count()
        return ValidationResult(table_name, f"no_null_{pk_col}",
                                null_count == 0, f"{null_count} nulls")
    except Exception as e:
        return ValidationResult(table_name, f"no_null_{pk_col}", False, str(e))


def check_referential_integrity(
    spark: SparkSession, cfg: ProjectConfig,
    child_table: str, child_col: str,
    parent_table: str, parent_col: str,
) -> ValidationResult:
    """Assert all values in child_col exist in parent_col."""
    try:
        child = spark.table(cfg.table(child_table)).select(child_col).distinct()
        parent = spark.table(cfg.table(parent_table)).select(parent_col).distinct()
        orphans = child.subtract(parent).count()
        return ValidationResult(
            child_table, f"fk_{child_col}→{parent_table}.{parent_col}",
            orphans == 0, f"{orphans} orphans")
    except Exception as e:
        return ValidationResult(child_table, f"fk_{child_col}", False, str(e))


def validate_raw_tables(spark: SparkSession, cfg: ProjectConfig) -> list[ValidationResult]:
    """Run all raw-layer validations."""
    results = []
    pk_map = {
        naming.PERSON_RAW: "person_id",
        naming.CARE_SITE_RAW: "care_site_id",
        naming.PROVIDER_RAW: "provider_id",
        naming.LOCATION_RAW: "location_id",
        naming.VISIT_OCCURRENCE_RAW: "visit_occurrence_id",
        naming.CONDITION_OCCURRENCE_RAW: "condition_occurrence_id",
        naming.DRUG_EXPOSURE_RAW: "drug_exposure_id",
        naming.PROCEDURE_OCCURRENCE_RAW: "procedure_occurrence_id",
    }
    for table, pk in pk_map.items():
        results.append(check_not_empty(spark, cfg, table))
        results.append(check_no_null_pk(spark, cfg, table, pk))
    return results


def validate_gold_tables(spark: SparkSession, cfg: ProjectConfig) -> list[ValidationResult]:
    """Run all gold-layer validations."""
    results = []
    for table in naming.all_gold_tables():
        results.append(check_not_empty(spark, cfg, table))

    # Referential integrity checks
    results.append(check_referential_integrity(
        spark, cfg, naming.VISIT_GOLD, "person_id",
        naming.PERSON_GOLD, "person_id"))
    results.append(check_referential_integrity(
        spark, cfg, naming.CONDITION_GOLD, "person_id",
        naming.PERSON_GOLD, "person_id"))
    return results


def print_results(results: list[ValidationResult]) -> bool:
    """Print validation results and return True if all passed."""
    all_passed = True
    for r in results:
        print(r)
        if not r.passed:
            all_passed = False
    return all_passed
