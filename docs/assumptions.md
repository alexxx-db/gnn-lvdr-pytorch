# Assumptions and Simplifications

## Data Assumptions

1. **All data is synthetic.** No real PHI is used anywhere in this project. Generated via dbldatagen with configurable scale.

2. **OMOP CDM subset.** We implement a minimal viable OMOP subset (person, care_site, provider, location, visit_occurrence, condition_occurrence, drug_exposure, procedure_occurrence). Full OMOP compliance is not the goal.

3. **Random node features are replaced with real features.** Unlike the legacy project which used random 15-dim features, we now derive features from clinical aggregates (visit count, condition count, etc.) and pad/project to the target dimension.

4. **Geospatial data is US-centric and clustered.** Synthetic coordinates cluster around 8 regional centers. Distances are haversine approximations. This is sufficient for demonstrating distance-based recommendation ranking.

5. **Edge relationships are derived from synthetic visit patterns.** Since the data is generated, the graph structure reflects statistical properties of the generators, not real healthcare network dynamics.

## Model Assumptions

1. **Homogeneous graph projection.** The heterogeneous OMOP relationships are projected into a homogeneous graph for GraphSAGE training. The primary prediction target is patient→care_site edges (visited relationship).

2. **GraphSAGE architecture preserved.** The 2-layer SAGEConv encoder with MLPPredictor is kept from the legacy implementation. This is a proven architecture for link prediction at this scale.

3. **No temporal dynamics.** The graph treats all edges equally regardless of visit date. A production system should incorporate temporal weighting.

4. **Score interpretation.** GNN scores represent learned structural similarity, not clinical appropriateness. They indicate "patients like you visit care sites like this."

## Platform Assumptions

1. **Databricks Runtime 15.x+ with ML libraries.**
2. **Unity Catalog enabled.** All tables and models use 3-level namespace.
3. **Single-node training.** GraphSAGE runs on the driver node (CPU). For larger graphs, distributed training with DGL's partition-based approach would be needed.
4. **Databricks Apps support.** The Gradio app requires Databricks Apps (preview feature). If unavailable, the app can run locally.

## Simplifications

- Drug exposure and procedure occurrence tables are generated but not currently used as graph edges. They can be added as additional edge types.
- The agent pipeline uses a structured explanation builder rather than an LLM-based conversational agent. This is more deterministic and auditable.
- HyperOpt tuning from the legacy notebooks is not included in the automated workflow. Add it as an optional notebook if needed.
