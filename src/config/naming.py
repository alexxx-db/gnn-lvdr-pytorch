"""
Centralized table and resource naming conventions.

All UC table names flow through here so nothing is scattered across notebooks.
"""

# ---------------------------------------------------------------------------
# Delta table names (relative — call cfg.table(NAME) for fully qualified)
# ---------------------------------------------------------------------------

# Raw synthetic layer
PERSON_RAW = "person_raw"
CARE_SITE_RAW = "care_site_raw"
PROVIDER_RAW = "provider_raw"
LOCATION_RAW = "location_raw"
CONDITION_OCCURRENCE_RAW = "condition_occurrence_raw"
VISIT_OCCURRENCE_RAW = "visit_occurrence_raw"
DRUG_EXPOSURE_RAW = "drug_exposure_raw"
PROCEDURE_OCCURRENCE_RAW = "procedure_occurrence_raw"

# Curated gold layer
PERSON_GOLD = "person_gold"
CARE_SITE_GOLD = "care_site_gold"
PROVIDER_GOLD = "provider_gold"
VISIT_GOLD = "visit_gold"
CONDITION_GOLD = "condition_gold"
PATIENT_FEATURES = "patient_features"
CARE_SITE_FEATURES = "care_site_features"

# Graph layer
GRAPH_EDGES = "graph_edges"
GRAPH_NODES = "graph_nodes"

# Recommendation outputs
RECOMMENDATIONS = "recommendations"
RECOMMENDATIONS_EXPLAINED = "recommendations_explained"


def all_raw_tables():
    return [
        PERSON_RAW, CARE_SITE_RAW, PROVIDER_RAW, LOCATION_RAW,
        CONDITION_OCCURRENCE_RAW, VISIT_OCCURRENCE_RAW,
        DRUG_EXPOSURE_RAW, PROCEDURE_OCCURRENCE_RAW,
    ]


def all_gold_tables():
    return [
        PERSON_GOLD, CARE_SITE_GOLD, PROVIDER_GOLD,
        VISIT_GOLD, CONDITION_GOLD,
        PATIENT_FEATURES, CARE_SITE_FEATURES,
    ]
