"""
Gradio frontend for the GNN Patient Recommendations Databricks App.

Provides:
1. Patient selector dropdown
2. Top-k recommendations table
3. Recommendation explanation viewer
4. Model metadata display
"""
import os
import gradio as gr
import pandas as pd
from src.app.app_helpers import (
    get_patient_list,
    get_recommendations_for_patient,
    get_explanation,
)

# Configuration from environment (set by Databricks App deployment)
CATALOG = os.getenv("APP_CATALOG", "gnn_hls_graphsage")
SCHEMA = os.getenv("APP_SCHEMA", "gnn_hls_graphsage_db")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")


def load_patients():
    """Load patient list for the dropdown."""
    try:
        df = get_patient_list(CATALOG, SCHEMA, WAREHOUSE_ID)
        choices = [
            f"{row['person_id']} ({row['gender_source_value']}, {row['age_bucket']})"
            for _, row in df.iterrows()
        ]
        return gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        return gr.update(choices=[f"Error: {e}"])


def get_recommendations(patient_str: str, top_k: int):
    """Fetch and display recommendations for selected patient."""
    if not patient_str:
        return pd.DataFrame(), "Select a patient first."

    person_id = int(patient_str.split(" ")[0])
    try:
        df = get_recommendations_for_patient(
            CATALOG, SCHEMA, WAREHOUSE_ID, person_id, top_k
        )
        if df.empty:
            return pd.DataFrame(), "No recommendations found."
        return df, f"Showing top {len(df)} recommendations for patient {person_id}"
    except Exception as e:
        return pd.DataFrame(), f"Error: {e}"


def explain_rec(patient_str: str, care_site_id: str):
    """Get natural-language explanation for a recommendation."""
    if not patient_str or not care_site_id:
        return "Select a patient and care site first."

    person_id = int(patient_str.split(" ")[0])
    try:
        return get_explanation(
            CATALOG, SCHEMA, WAREHOUSE_ID,
            person_id, int(care_site_id),
        )
    except Exception as e:
        return f"Error: {e}"


def create_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="GNN Patient Recommendations",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# GNN Patient Care-Site Recommendations")
        gr.Markdown(
            "Explore GraphSAGE-based patient-to-care-site recommendations "
            "with explainability. All data is synthetic."
        )

        with gr.Row():
            with gr.Column(scale=1):
                patient_dd = gr.Dropdown(
                    label="Select Patient",
                    choices=[],
                    interactive=True,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Number of Recommendations",
                )
                refresh_btn = gr.Button("Load Patients", variant="secondary")
                rec_btn = gr.Button("Get Recommendations", variant="primary")

            with gr.Column(scale=2):
                status_text = gr.Textbox(label="Status", interactive=False)
                rec_table = gr.DataFrame(
                    label="Recommendations",
                    interactive=False,
                )

        gr.Markdown("---")
        gr.Markdown("### Recommendation Explanation")

        with gr.Row():
            care_site_input = gr.Textbox(
                label="Care Site ID (from table above)",
                placeholder="Enter care_site_id",
            )
            explain_btn = gr.Button("Explain", variant="primary")

        explanation_box = gr.Textbox(
            label="Explanation",
            lines=10,
            interactive=False,
        )

        gr.Markdown(
            "---\n*Built with GraphSAGE, MLflow, and Databricks Agent Bricks. "
            "All patient data is synthetic.*"
        )

        # Event handlers
        refresh_btn.click(fn=load_patients, outputs=[patient_dd])
        rec_btn.click(
            fn=get_recommendations,
            inputs=[patient_dd, top_k_slider],
            outputs=[rec_table, status_text],
        )
        explain_btn.click(
            fn=explain_rec,
            inputs=[patient_dd, care_site_input],
            outputs=[explanation_box],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=8080)
