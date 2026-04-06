# Databricks notebook source
# MAGIC %md
# MAGIC # RUNME — Workflow Bootstrap
# MAGIC
# MAGIC **Purpose:** Create a Databricks workflow/job that runs all project notebooks
# MAGIC in the correct dependency order. Run this notebook once to set up the workflow.
# MAGIC
# MAGIC ## What this creates
# MAGIC
# MAGIC A multi-task Databricks job with the following DAG:
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
dbutils.widgets.text("job_name", "gnn-patient-recommendations-workflow", "Job Name")
dbutils.widgets.text("cluster_id", "", "Existing Cluster ID (leave blank for new)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
synthetic_scale = dbutils.widgets.get("synthetic_scale")
job_name = dbutils.widgets.get("job_name")
cluster_id = dbutils.widgets.get("cluster_id")

# COMMAND ----------

# DBTITLE 1,Determine notebook base path
import os

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

# Shared widget parameters passed to every task
base_parameters = {
    "catalog": catalog,
    "schema": schema,
    "synthetic_scale": synthetic_scale,
    "rebuild_synthetic": "true",
    "retrain_model": "true",
    "random_seed": "42",
}

# COMMAND ----------

# DBTITLE 1,Build task definitions
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, TaskDependency, JobCluster,
    ClusterSpec, AutoScale, RuntimeEngine,
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

    # Use existing cluster if provided, otherwise use job cluster
    if cluster_id:
        task.existing_cluster_id = cluster_id
    else:
        task.job_cluster_key = "gnn_cluster"

    tasks.append(task)

print(f"Defined {len(tasks)} tasks:")
for t in tasks:
    deps = [d.task_key for d in t.depends_on] if t.depends_on else ["(none)"]
    print(f"  {t.task_key} ← {', '.join(deps)}")

# COMMAND ----------

# DBTITLE 1,Create or update the workflow job
from databricks.sdk.service.jobs import CreateJob

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

# Check if job already exists
existing_jobs = list(w.jobs.list(name=job_name))
if existing_jobs:
    job_id = existing_jobs[0].job_id
    w.jobs.reset(
        job_id=job_id,
        new_settings={
            "name": job_name,
            "tasks": tasks,
            "job_clusters": job_clusters if job_clusters else None,
        },
    )
    print(f"Updated existing job: {job_name} (ID: {job_id})")
else:
    created = w.jobs.create(
        name=job_name,
        tasks=tasks,
        job_clusters=job_clusters if job_clusters else None,
    )
    job_id = created.job_id
    print(f"Created new job: {job_name} (ID: {job_id})")

# COMMAND ----------

# DBTITLE 1,Display job URL
host = w.config.host.rstrip("/")
print(f"\nWorkflow created successfully!")
print(f"Job URL: {host}/#job/{job_id}")
print(f"\nTo run: click 'Run Now' in the job page, or:")
print(f"  w.jobs.run_now(job_id={job_id})")
