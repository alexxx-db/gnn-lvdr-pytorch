# Architecture

## System Overview

This project implements a GNN-based patient-to-care-site recommendation engine on Databricks, with an agentic explainability layer and a Gradio-based frontend.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Databricks Workspace                                               │
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │ Synthetic     │───►│ OMOP Gold     │───►│ Graph Construction   │  │
│  │ Data Gen      │    │ Curation      │    │ (DGL)                │  │
│  │ (dbldatagen)  │    │               │    │                      │  │
│  └──────────────┘    └───────────────┘    └──────────┬───────────┘  │
│                                                      │              │
│                                                      ▼              │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │ Databricks   │◄───│ Agent         │◄───│ GraphSAGE Training   │  │
│  │ App (Gradio) │    │ Explainer     │    │ + MLflow Tracking    │  │
│  │              │    │               │    │                      │  │
│  └──────────────┘    └───────────────┘    └──────────┬───────────┘  │
│                                                      │              │
│                                                      ▼              │
│                                           ┌──────────────────────┐  │
│                                           │ Batch Inference      │  │
│                                           │ + Recommendations    │  │
│                                           │ (Delta Tables)       │  │
│                                           └──────────────────────┘  │
│                                                                     │
│  Unity Catalog: {catalog}.{schema}.*                                │
│  MLflow: Experiment tracking, Model registry, Tracing               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Synthetic Generation** (notebooks 02-03): dbldatagen creates OMOP-aligned clinical tables + geospatial data
2. **Curation** (notebook 04): Raw → Gold transforms, patient/care-site feature engineering
3. **Graph Construction** (notebook 05): OMOP edges → DGL homogeneous graph with real features
4. **Training** (notebooks 06-07): GraphSAGE link prediction, MLflow tracking, UC model registry
5. **Inference** (notebook 08): Batch scoring, top-k ranking, enriched recommendations
6. **Agent Pipeline** (notebooks 09-10): Care-navigation explanations with MLflow tracing
7. **App** (notebook 11): Gradio frontend for interactive exploration

## Graph Schema

### Node Types
- **Patient** (person_id): Healthcare consumer with clinical features
- **Care Site** (care_site_id): Clinic/hospital with utilization features

### Edge Types (extracted from OMOP data)
- `patient → visited → care_site`: From visit_occurrence
- `patient → diagnosed_with → condition`: From condition_occurrence
- `patient → treated_by → provider`: From visit_occurrence
- `provider → works_at → care_site`: From provider table
- `patient → near → care_site`: From geospatial distance computation

### Recommendation Task
**Patient-to-care-site link prediction**: Given a patient's clinical history, conditions, and geography, predict which care sites they are most likely to benefit from visiting.

## Key Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Graph framework | DGL + PyTorch | Mature GraphSAGE support, mini-batch training |
| GNN architecture | GraphSAGE (2-layer SAGEConv) | Inductive, scalable, preserved from legacy |
| Link predictor | MLPPredictor (concat + linear) | Learns edge scores from embeddings |
| Synthetic data | dbldatagen | Spark-native, scalable, deterministic |
| Experiment tracking | MLflow | Databricks-native, UC registry |
| Agent | Agent Bricks pattern | Databricks-native, tool-calling |
| Frontend | Gradio | Databricks Apps compatible |
