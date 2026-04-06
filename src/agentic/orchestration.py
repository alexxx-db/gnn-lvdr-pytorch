"""
Agent orchestration for recommendation explanation.

Coordinates tool calls and LLM reasoning to produce
natural-language explanations of GNN recommendations.
"""
import json
from pyspark.sql import SparkSession
from src.config.settings import ProjectConfig
from src.agentic import agent_config
from src.agentic.tools import (
    get_patient_summary,
    get_care_site_profile,
    get_top_recommendations,
    get_distance_context,
)
from src.mlops.tracing import traced


@traced("explain_recommendation")
def explain_recommendation(
    spark: SparkSession,
    cfg: ProjectConfig,
    person_id: int,
    care_site_id: int,
) -> dict:
    """
    Build a structured explanation for why a care site was recommended.

    Gathers patient context, care site profile, recommendation score,
    and geospatial distance, then assembles an explanation payload.
    """
    patient = get_patient_summary(spark, cfg, person_id)
    site = get_care_site_profile(spark, cfg, care_site_id)
    distance = get_distance_context(spark, cfg, person_id, care_site_id)
    recs = get_top_recommendations(spark, cfg, person_id, top_k=10)

    # Find this site's rank and score
    this_rec = next(
        (r for r in recs if r.get("care_site_id") == care_site_id), None
    )

    factors = []
    if this_rec and this_rec.get("score"):
        factors.append(
            f"GNN link prediction score: {this_rec['score']:.3f} "
            f"(ranked #{this_rec.get('rank', '?')} of recommended sites)"
        )

    if distance.get("distance_miles") is not None:
        factors.append(
            f"Distance: {distance['distance_miles']:.1f} miles "
            f"({distance.get('distance_band', 'unknown band')})"
        )

    if site.get("care_site_type"):
        factors.append(f"Facility type: {site['care_site_type']}")

    if site.get("patient_volume"):
        factors.append(f"Patient volume: {site['patient_volume']} patients served")

    if patient.get("top_conditions"):
        conditions_str = ", ".join(patient["top_conditions"][:5])
        factors.append(f"Patient conditions: {conditions_str}")

    return {
        "person_id": person_id,
        "care_site_id": care_site_id,
        "patient_summary": patient,
        "care_site_profile": site,
        "distance": distance,
        "recommendation": this_rec,
        "explanation_factors": factors,
        "explanation_text": _format_explanation(patient, site, this_rec, distance, factors),
    }


def _format_explanation(patient, site, rec, distance, factors) -> str:
    """Format a human-readable explanation string."""
    lines = []
    lines.append(
        f"Patient {patient.get('person_id')} "
        f"({patient.get('gender', '?')}, {patient.get('age_bucket', '?')}) "
        f"was recommended {site.get('care_site_name', f'Site {site.get(\"care_site_id\")}')}."
    )
    lines.append("")
    lines.append("Key factors:")
    for f in factors:
        lines.append(f"  - {f}")

    if rec and rec.get("score"):
        lines.append("")
        lines.append(
            f"The GNN model learned this connection from patterns in visit history, "
            f"condition profiles, and care-site utilization across "
            f"{patient.get('visit_count', '?')} visits to "
            f"{patient.get('distinct_care_sites_visited', '?')} distinct sites."
        )

    return "\n".join(lines)


@traced("batch_explain")
def batch_explain(
    spark: SparkSession,
    cfg: ProjectConfig,
    person_ids: list[int],
    top_k: int = 3,
) -> list[dict]:
    """
    Generate explanations for the top-k recommendations of multiple patients.
    """
    all_explanations = []
    for pid in person_ids:
        recs = get_top_recommendations(spark, cfg, pid, top_k=top_k)
        for rec in recs:
            if "error" not in rec:
                explanation = explain_recommendation(
                    spark, cfg, pid, rec["care_site_id"]
                )
                all_explanations.append(explanation)
    return all_explanations
