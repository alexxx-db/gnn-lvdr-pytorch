"""
Helper functions for the Gradio Databricks App.

Provides data access wrappers that work within the Databricks App
execution context using the Databricks SDK.
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import pandas as pd


def get_sql_client():
    """Get a Databricks SQL statement execution client."""
    return WorkspaceClient()


def execute_sql(query: str, warehouse_id: str) -> pd.DataFrame:
    """Execute a SQL query via Databricks SQL Statement Execution API."""
    w = get_sql_client()
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
    )
    if response.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL query failed: {response.status.error}")

    columns = [col.name for col in response.manifest.schema.columns]
    rows = []
    if response.result and response.result.data_array:
        rows = response.result.data_array
    return pd.DataFrame(rows, columns=columns)


def get_patient_list(catalog: str, schema: str, warehouse_id: str,
                     limit: int = 100) -> pd.DataFrame:
    """Fetch a sample of patients for the dropdown selector."""
    query = f"""
    SELECT person_id, gender_source_value, age_bucket, visit_count
    FROM {catalog}.{schema}.patient_features
    ORDER BY person_id
    LIMIT {limit}
    """
    return execute_sql(query, warehouse_id)


def get_recommendations_for_patient(
    catalog: str, schema: str, warehouse_id: str,
    person_id: int, top_k: int = 5,
) -> pd.DataFrame:
    """Fetch top-k recommendations for a specific patient."""
    query = f"""
    SELECT person_id, care_site_id, care_site_name, care_site_type,
           score, rank, distance_miles
    FROM {catalog}.{schema}.recommendations
    WHERE person_id = {int(person_id)}
    ORDER BY rank
    LIMIT {int(top_k)}
    """
    return execute_sql(query, warehouse_id)


def get_explanation(
    catalog: str, schema: str, warehouse_id: str,
    person_id: int, care_site_id: int,
) -> str:
    """Fetch the explanation text for a specific recommendation."""
    query = f"""
    SELECT explanation_text
    FROM {catalog}.{schema}.recommendations_explained
    WHERE person_id = {int(person_id)} AND care_site_id = {int(care_site_id)}
    LIMIT 1
    """
    df = execute_sql(query, warehouse_id)
    if df.empty:
        return "No explanation available for this recommendation."
    return df.iloc[0]["explanation_text"]
