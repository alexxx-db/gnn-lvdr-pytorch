"""
Agent tools for the care-navigation recommendation explainer.

Each tool is a function the agent can invoke to retrieve structured data
for grounding its explanations. Tools query Unity Catalog tables.
"""
from pyspark.sql import SparkSession
from src.config.settings import ProjectConfig
from src.config import naming
from src.mlops.tracing import traced


@traced("tool:get_patient_summary")
def get_patient_summary(spark: SparkSession, cfg: ProjectConfig,
                        person_id: int) -> dict:
    """
    Retrieve a summary of a patient's profile and clinical history.

    Returns demographics, condition count, visit count, and top conditions.
    """
    person = (
        spark.table(cfg.table(naming.PATIENT_FEATURES))
        .filter(f"person_id = {person_id}")
        .first()
    )
    if person is None:
        return {"error": f"Patient {person_id} not found"}

    conditions = (
        spark.table(cfg.table(naming.CONDITION_GOLD))
        .filter(f"person_id = {person_id}")
        .select("condition_source_value")
        .distinct()
        .limit(10)
        .toPandas()
    )

    return {
        "person_id": person_id,
        "gender": person["gender_source_value"],
        "age_bucket": person["age_bucket"],
        "visit_count": person["visit_count"],
        "condition_count": person["condition_count"],
        "distinct_care_sites": person["distinct_care_sites"],
        "top_conditions": conditions["condition_source_value"].tolist(),
    }


@traced("tool:get_care_site_profile")
def get_care_site_profile(spark: SparkSession, cfg: ProjectConfig,
                          care_site_id: int) -> dict:
    """
    Retrieve a care site's profile including type, volume, and location.
    """
    site = (
        spark.table(cfg.table(naming.CARE_SITE_FEATURES))
        .filter(f"care_site_id = {care_site_id}")
        .first()
    )
    if site is None:
        return {"error": f"Care site {care_site_id} not found"}

    return {
        "care_site_id": care_site_id,
        "care_site_name": site["care_site_name"],
        "care_site_type": site["care_site_type"],
        "patient_volume": site["patient_volume"],
        "provider_count": site["provider_count"],
        "total_visits": site["total_visits"],
    }


@traced("tool:get_top_recommendations")
def get_top_recommendations(spark: SparkSession, cfg: ProjectConfig,
                            person_id: int, top_k: int = 5) -> list[dict]:
    """
    Retrieve the top-k recommended care sites for a patient.
    """
    recs = (
        spark.table(cfg.table(naming.RECOMMENDATIONS))
        .filter(f"person_id = {person_id}")
        .orderBy("rank")
        .limit(top_k)
        .toPandas()
    )
    if recs.empty:
        return [{"error": f"No recommendations found for patient {person_id}"}]

    return recs.to_dict(orient="records")


@traced("tool:get_distance_context")
def get_distance_context(spark: SparkSession, cfg: ProjectConfig,
                         person_id: int, care_site_id: int) -> dict:
    """
    Retrieve geospatial distance between a patient and care site.
    """
    try:
        row = (
            spark.table(cfg.table("nearest_sites"))
            .filter(f"person_id = {person_id} AND care_site_id = {care_site_id}")
            .first()
        )
        if row is None:
            return {"distance_miles": None, "note": "Pair not in nearest-sites table"}
        return {
            "person_id": person_id,
            "care_site_id": care_site_id,
            "distance_miles": round(row["distance_miles"], 1),
            "distance_band": row["distance_band"],
            "site_rank": row["site_rank"],
        }
    except Exception:
        return {"distance_miles": None, "note": "Geospatial data not available"}


@traced("tool:compare_recommendations")
def compare_recommendations(spark: SparkSession, cfg: ProjectConfig,
                            person_id: int, site_ids: list[int]) -> list[dict]:
    """
    Compare specific care sites for a patient: scores, distance, type.
    """
    results = []
    for sid in site_ids:
        rec_row = (
            spark.table(cfg.table(naming.RECOMMENDATIONS))
            .filter(f"person_id = {person_id} AND care_site_id = {sid}")
            .first()
        )
        site_info = get_care_site_profile(spark, cfg, sid)
        dist_info = get_distance_context(spark, cfg, person_id, sid)

        results.append({
            "care_site_id": sid,
            "score": rec_row["score"] if rec_row else None,
            "rank": rec_row["rank"] if rec_row else None,
            "care_site_name": site_info.get("care_site_name"),
            "care_site_type": site_info.get("care_site_type"),
            "distance_miles": dist_info.get("distance_miles"),
        })
    return results
