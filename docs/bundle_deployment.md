# Bundle Deployment Guide

## Prerequisites

1. [Databricks CLI v0.200+](https://docs.databricks.com/dev-tools/cli/install.html) installed
2. Authentication configured (`databricks auth login` or `.databrickscfg`)
3. Access to a Unity Catalog-enabled workspace

## Quick Start

```bash
# Clone the repository
git clone https://github.com/alexxx-db/gnn-lvdr-pytorch.git
cd gnn-lvdr-pytorch

# Validate the bundle configuration
databricks bundle validate -t dev

# Deploy to dev environment
databricks bundle deploy -t dev

# Run the end-to-end pipeline
databricks bundle run -t dev gnn_patient_recommendations
```

## Targets

| Target | Catalog | Scale | Mode | Use Case |
|--------|---------|-------|------|----------|
| `dev` | `gnn_hls_graphsage_dev` | 500 patients | development | Fast iteration, testing |
| `staging` | `gnn_hls_graphsage_staging` | 5,000 patients | default | Pre-production validation |
| `prod` | `gnn_hls_graphsage` | 10,000 patients | production | Production deployment |

## Bundle Resources

The bundle deploys:

| Resource | Type | Description |
|----------|------|-------------|
| `gnn_patient_recommendations` | Job | 12-task pipeline from data gen to validation |
| `gnn_patient_rec_app` | App | Gradio frontend for recommendation exploration |
| `gnn_experiment` | Experiment | MLflow experiment for training runs |
| `gnn_model` | Registered Model | UC model for GraphSAGE pyfunc |
| `gnn_schema` | Schema | UC schema for all Delta tables |
| `gnn_volume` | Volume | UC volume for graph metadata and artifacts |

## Common Commands

```bash
# Validate before deploying
databricks bundle validate -t dev

# Deploy all resources to dev
databricks bundle deploy -t dev

# Run the pipeline job
databricks bundle run -t dev gnn_patient_recommendations

# Deploy to staging with a different data scale
databricks bundle deploy -t staging --var="synthetic_scale=2000"

# Deploy to production
databricks bundle deploy -t prod

# Run specific job task only (if supported)
databricks bundle run -t dev gnn_patient_recommendations --params catalog=my_catalog

# Destroy dev resources (cleanup)
databricks bundle destroy -t dev
```

## Variable Overrides

Override any variable at deploy or run time:

```bash
# Custom catalog for a specific run
databricks bundle deploy -t dev --var="catalog=my_dev_catalog"

# Skip data regeneration
databricks bundle deploy -t dev --var="rebuild_synthetic=false"

# Use a specific compute type
databricks bundle deploy -t dev --var="node_type_id=m5.xlarge"
```

## Environment Promotion

```
dev ──deploy──► staging ──deploy──► prod
 │                │                  │
 ├─ small data    ├─ medium data     ├─ full data
 ├─ rebuild all   ├─ rebuild all     ├─ skip rebuild
 └─ dev catalog   └─ staging catalog └─ prod catalog
```

Each target deploys to its own catalog, so there is no data or model collision between environments.

## File Structure

```
databricks.yml              Root bundle manifest
bundle/
  variables.yml             All parameterized names and defaults
  targets.yml               dev/staging/prod target overrides
  includes/
    jobs.yml                12-task pipeline job definition
    apps.yml                Gradio Databricks App resource
    experiments.yml         MLflow experiment resource
    models.yml              UC registered model resource
    schemas.yml             UC schema and volume resources
```

## Alternative: Interactive Setup

If you prefer not to use the CLI, run `notebooks/RUNME.py` inside Databricks
to create the workflow job interactively. The RUNME approach creates the same
pipeline but without bundle-managed lifecycle (validate/deploy/destroy).
