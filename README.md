# GNN Patient Recommendations with Databricks

Graph Neural Network-based patient-to-care-site recommendations using GraphSAGE, synthetic OMOP clinical data, geospatial intelligence, MLflow observability, and an Agentic AI explainability layer — deployed as a Databricks Asset Bundle.

## Architecture

<img src="https://github.com/alexxx-db/gnn-lvdr-pytorch/blob/main/images/architecture_including_ml.png?raw=True" width="100%" alt="architecture">

## Quick Start

### Deploy with Databricks Asset Bundles (recommended)

```bash
# Install Databricks CLI (v0.200+)
# Configure auth: databricks auth login

# Validate the bundle
databricks bundle validate -t dev

# Deploy to dev target
databricks bundle deploy -t dev

# Run the end-to-end pipeline
databricks bundle run -t dev gnn_patient_recommendations
```

### Alternative: Interactive execution

1. Import this repo into a Databricks Git folder
2. Run notebooks `01` through `12` sequentially, or run `RUNME.py` to create a job

## What This Project Does

1. **Generates synthetic OMOP-aligned healthcare data** — patients, care sites, providers, visits, conditions, drugs, procedures, and geospatial coordinates
2. **Curates gold-layer feature tables** — patient clinical summaries and care-site utilization profiles
3. **Constructs a patient-care-site graph** — from visit, condition, treatment, and proximity edges
4. **Trains a GraphSAGE link prediction model** — two-layer SAGEConv encoder with MLP edge predictor
5. **Generates top-k recommendations** — ranked care sites per patient with enriched metadata
6. **Explains recommendations via an agent pipeline** — structured natural-language explanations grounded in clinical and geospatial data
7. **Provides a Gradio-based Databricks App** — interactive patient selection, recommendation viewing, and explanation

## Targets

| Target | Catalog | Scale | Use Case |
|--------|---------|-------|----------|
| `dev` | `gnn_hls_graphsage_dev` | 500 patients | Fast iteration |
| `staging` | `gnn_hls_graphsage_staging` | 5,000 patients | Pre-production |
| `prod` | `gnn_hls_graphsage` | 10,000 patients | Production |

## Repository Structure

```
databricks.yml          Root bundle configuration
bundle/                 Bundle variables, targets, and resource definitions
  variables.yml         All parameterized names
  targets.yml           dev / staging / prod overrides
  includes/             Resource definitions (jobs, apps, experiments, models, schemas)
notebooks/              Orchestration notebooks (00-12 + RUNME)
src/
  config/               Centralized configuration and UC naming
  data_generation/      Synthetic OMOP + geospatial generators
  omop/                 Schema definitions, transforms, validators
  graph/                Graph construction, GraphSAGE model, inference, ranking
  mlops/                MLflow tracking, evaluation, registry, tracing
  agentic/              Agent config, tools, orchestration, explainability
  app/                  Gradio Databricks App frontend
docs/                   Architecture, assumptions, runbook, bundle deployment guide
tests/                  Unit tests
resources/              App config, prompts, sample configs
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Deployment | Databricks Asset Bundles |
| Graph Neural Network | DGL + PyTorch (GraphSAGE) |
| Data Generation | dbldatagen |
| Data Platform | Databricks + Unity Catalog + Delta Lake |
| Experiment Tracking | MLflow |
| Agent Explainability | Agent Bricks pattern |
| Frontend | Gradio + Databricks Apps |

## Bundle Commands

```bash
databricks bundle validate -t dev          # Validate config
databricks bundle deploy -t dev            # Deploy resources
databricks bundle run -t dev gnn_patient_recommendations  # Run pipeline
databricks bundle destroy -t dev           # Cleanup
```

See [docs/bundle_deployment.md](docs/bundle_deployment.md) for full deployment guide.

## Configuration

All parameters are centralized in `bundle/variables.yml` (for deployment) and `src/config/settings.py` (for runtime). Override at deploy time:

```bash
databricks bundle deploy -t dev --var="synthetic_scale=2000"
```

Or via notebook widgets when running interactively.

## License

See [LICENSE](LICENSE).
