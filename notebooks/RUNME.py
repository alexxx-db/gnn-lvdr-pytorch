# Databricks notebook source
# MAGIC %md
# MAGIC # RUNME — Project Onboarding & Workflow Bootstrap
# MAGIC
# MAGIC **Purpose:** Onboard new users, explain the project structure, and optionally
# MAGIC create a workflow job for users who prefer interactive setup over CLI bundles.
# MAGIC
# MAGIC ## Deployment Options
# MAGIC
# MAGIC ### Option 1: Databricks Asset Bundles (recommended)
# MAGIC
# MAGIC From your local machine:
# MAGIC ```bash
# MAGIC # Validate the bundle
# MAGIC databricks bundle validate -t dev
# MAGIC
# MAGIC # Deploy to dev
# MAGIC databricks bundle deploy -t dev
# MAGIC
# MAGIC # Run the pipeline
# MAGIC databricks bundle run -t dev gnn_patient_recommendations
# MAGIC
# MAGIC # Promote to staging
# MAGIC databricks bundle deploy -t staging
# MAGIC databricks bundle run -t staging gnn_patient_recommendations
# MAGIC ```
# MAGIC
# MAGIC ### Option 2: Interactive notebook execution
# MAGIC Run notebooks 01 through 12 sequentially from a Databricks Git folder.
# MAGIC
# MAGIC ### Option 3: This notebook (RUNME)
# MAGIC Run this notebook to create a Databricks job programmatically.
# MAGIC This is useful if you cannot use the CLI or want a quick demo setup.
# MAGIC
# MAGIC ## Pipeline DAG
# MAGIC
# MAGIC ```
# MAGIC 01_workspace_setup
# MAGIC   └─► 02_synthetic_omop_generation
# MAGIC        └─► 03_synthetic_geospatial_generation
# MAGIC             └─► 04_curate_omop_gold
# MAGIC                  └─► 05_graph_construction
# MAGIC                       └─► 06_graphsage_training
# MAGIC                            └─► 07_evaluation_and_mlflow
# MAGIC                                 └─► 08_batch_recommendations
# MAGIC                                      └─► 09_agent_bricks_setup
# MAGIC                                           └─► 10_agent_pipeline
# MAGIC                                                └─► 11_databricks_app_prep
# MAGIC                                                     └─► 12_validation_and_smoke_tests
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Configuration
dbutils.widgets.text("catalog", "gnn_hls_graphsage", "Catalog Name")
dbutils.widgets.text("schema", "gnn_hls_graphsage_db", "Schema Name")
dbutils.widgets.text("synthetic_scale", "1000", "Synthetic Data Scale")
dbutils.widgets.text("job_name", "gnn-patient-recommendations-interactive", "Job Name")
dbutils.widgets.text("cluster_id", "", "Existing Cluster ID (leave blank for new)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
synthetic_scale = dbutils.widgets.get("synthetic_scale")
job_name = dbutils.widgets.get("job_name")
cluster_id = dbutils.widgets.get("cluster_id")

# COMMAND ----------

# DBTITLE 1,Determine notebook base path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = notebook_path.rsplit("/", 1)[0]
print(f"Notebook base path: {base_path}")

# COMMAND ----------

# DBTITLE 1,Define the workflow tasks
NOTEBOOK_SEQUENCE = [
    "01_workspace_setup",
    "02_synthetic_omop_generation",
    "03_synthetic_geospatial_generation",
    "04_curate_omop_gold",
    "05_graph_construction",
    "06_graphsage_training",
    "07_evaluation_and_mlflow",
    "08_batch_recommendations",
    "09_agent_bricks_setup",
    "10_agent_pipeline",
    "11_databricks_app_prep",
    "12_validation_and_smoke_tests",
]

base_parameters = {
    "catalog": catalog,
    "schema": schema,
    "synthetic_scale": synthetic_scale,
    "rebuild_synthetic": "true",
    "retrain_model": "true",
    "random_seed": "42",
}

# COMMAND ----------

# DBTITLE 1,Build and create the workflow job
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, TaskDependency, JobCluster,
    ClusterSpec, RuntimeEngine,
)

w = WorkspaceClient()

tasks = []
for i, nb_name in enumerate(NOTEBOOK_SEQUENCE):
    task = Task(
        task_key=nb_name,
        notebook_task=NotebookTask(
            notebook_path=f"{base_path}/{nb_name}",
            base_parameters=base_parameters,
        ),
        depends_on=[TaskDependency(task_key=NOTEBOOK_SEQUENCE[i - 1])] if i > 0 else None,
    )
    if cluster_id:
        task.existing_cluster_id = cluster_id
    else:
        task.job_cluster_key = "gnn_cluster"
    tasks.append(task)

print(f"Defined {len(tasks)} tasks")

job_clusters = []
if not cluster_id:
    job_clusters = [
        JobCluster(
            job_cluster_key="gnn_cluster",
            new_cluster=ClusterSpec(
                spark_version="15.4.x-cpu-ml-scala2.12",
                num_workers=0,
                node_type_id="i3.xlarge",
                runtime_engine=RuntimeEngine.STANDARD,
                spark_conf={"spark.master": "local[*]"},
            ),
        )
    ]

existing_jobs = list(w.jobs.list(name=job_name))
if existing_jobs:
    job_id = existing_jobs[0].job_id
    w.jobs.reset(job_id=job_id, new_settings={
        "name": job_name, "tasks": tasks,
        "job_clusters": job_clusters if job_clusters else None,
    })
    print(f"Updated existing job: {job_name} (ID: {job_id})")
else:
    created = w.jobs.create(
        name=job_name, tasks=tasks,
        job_clusters=job_clusters if job_clusters else None,
    )
    job_id = created.job_id
    print(f"Created new job: {job_name} (ID: {job_id})")

# COMMAND ----------

# DBTITLE 1,Display job URL
host = w.config.host.rstrip("/")
print(f"\nWorkflow created successfully!")
print(f"Job URL: {host}/#job/{job_id}")
print(f"\nTo run: click 'Run Now' in the job page")
print(f"\nNote: For repeatable deployments, prefer Databricks Asset Bundles:")
print(f"  databricks bundle deploy -t dev")
print(f"  databricks bundle run -t dev gnn_patient_recommendations")
