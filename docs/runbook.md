# Runbook

## Quick Start

### Option 1: Run notebooks sequentially
1. Import/sync this repo into a Databricks Git folder
2. Open `notebooks/00_project_config.py` and adjust widgets if needed
3. Run notebooks 01 through 12 in order

### Option 2: Create a workflow via RUNME
1. Import/sync this repo into a Databricks Git folder
2. Open `notebooks/RUNME.py`
3. Set the `catalog`, `schema`, and `cluster_id` widgets
4. Run the notebook — it creates a Databricks job
5. Navigate to the job URL and click "Run Now"

## Configuration

All config is centralized in `src/config/settings.py`. Override at runtime via notebook widgets:

| Widget | Default | Description |
|--------|---------|-------------|
| `catalog` | `gnn_hls_graphsage` | Unity Catalog catalog name |
| `schema` | `gnn_hls_graphsage_db` | Schema name |
| `synthetic_scale` | `1000` | Number of synthetic patients |
| `rebuild_synthetic` | `true` | Regenerate synthetic data |
| `retrain_model` | `true` | Retrain the GNN model |
| `random_seed` | `42` | Reproducibility seed |

## Scaling

- `synthetic_scale=1000`: ~5 min end-to-end, good for demos
- `synthetic_scale=10000`: ~15-30 min, more realistic graph density
- `synthetic_scale=100000`: ~1-2 hours, requires larger cluster

## Troubleshooting

### "Table not found" errors
Run upstream notebooks first. The pipeline has sequential dependencies.

### DGL installation issues
The `00_project_config` notebook runs `%pip install dgl`. If this fails on your cluster, install DGL via cluster init scripts or libraries tab.

### MLflow registry errors
Ensure the catalog and schema exist and you have CREATE MODEL permissions in Unity Catalog.

### Databricks App deployment
Databricks Apps requires a compatible workspace. If unavailable, the Gradio app can be tested locally with `python src/app/gradio_app.py`.

## Artifacts Produced

| Artifact | Location | Description |
|----------|----------|-------------|
| Raw OMOP tables | `{catalog}.{schema}.*_raw` | Synthetic clinical data |
| Gold tables | `{catalog}.{schema}.*_gold` | Curated OMOP data |
| Feature tables | `{catalog}.{schema}.patient_features` | Graph-ready features |
| Graph edges | `{catalog}.{schema}.graph_edges` | Unified edge table |
| Recommendations | `{catalog}.{schema}.recommendations` | Top-k ranked recommendations |
| Explanations | `{catalog}.{schema}.recommendations_explained` | Agent-generated explanations |
| MLflow model | `{catalog}.{schema}.patient_recommendations_gnn` | Registered pyfunc model |
| MLflow experiment | `/Shared/gnn-patient-recommendations` | Training runs and metrics |
