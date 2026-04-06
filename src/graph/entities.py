"""
Graph entity definitions — explicit node and edge type semantics.

Node types:
  - patient (person_id): healthcare consumer
  - care_site (care_site_id): clinic/hospital/facility
  - condition (condition_concept_id): diagnosed condition
  - provider (provider_id): healthcare provider

Edge types:
  - patient → visited → care_site: patient visited a care site
  - patient → diagnosed_with → condition: patient has a condition
  - patient → treated_by → provider: patient saw a provider
  - provider → works_at → care_site: provider affiliated with a site
  - patient → near → care_site: geospatial proximity (from nearest-site computation)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeType:
    name: str
    id_column: str


@dataclass(frozen=True)
class EdgeType:
    name: str
    src_type: NodeType
    dst_type: NodeType
    src_column: str
    dst_column: str


# Node types
PATIENT = NodeType("patient", "person_id")
CARE_SITE = NodeType("care_site", "care_site_id")
CONDITION = NodeType("condition", "condition_concept_id")
PROVIDER = NodeType("provider", "provider_id")

# Edge types
VISITED = EdgeType("visited", PATIENT, CARE_SITE, "person_id", "care_site_id")
DIAGNOSED_WITH = EdgeType("diagnosed_with", PATIENT, CONDITION, "person_id", "condition_concept_id")
TREATED_BY = EdgeType("treated_by", PATIENT, PROVIDER, "person_id", "provider_id")
WORKS_AT = EdgeType("works_at", PROVIDER, CARE_SITE, "provider_id", "care_site_id")
NEAR = EdgeType("near", PATIENT, CARE_SITE, "person_id", "care_site_id")

ALL_EDGE_TYPES = [VISITED, DIAGNOSED_WITH, TREATED_BY, WORKS_AT, NEAR]
