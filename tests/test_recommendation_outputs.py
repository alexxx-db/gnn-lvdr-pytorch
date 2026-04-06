"""Tests for recommendation ranking logic."""
import pandas as pd
from src.graph.ranking import rank_recommendations


def test_rank_recommendations_top_k():
    scores = pd.DataFrame({
        "person_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "care_site_id": [10, 20, 30, 40, 50, 10, 20, 30],
        "score": [0.9, 0.7, 0.8, 0.5, 0.6, 0.3, 0.9, 0.1],
    })
    ranked = rank_recommendations(scores, top_k=3)

    # Each patient should have at most 3 recommendations
    for pid in ranked["person_id"].unique():
        assert len(ranked[ranked["person_id"] == pid]) <= 3

    # Recommendations should be in descending score order per patient
    for pid in ranked["person_id"].unique():
        patient_recs = ranked[ranked["person_id"] == pid]
        scores_list = patient_recs["score"].tolist()
        assert scores_list == sorted(scores_list, reverse=True)


def test_rank_recommendations_has_rank_column():
    scores = pd.DataFrame({
        "person_id": [1, 1, 1],
        "care_site_id": [10, 20, 30],
        "score": [0.9, 0.7, 0.8],
    })
    ranked = rank_recommendations(scores, top_k=5)
    assert "rank" in ranked.columns
    assert ranked["rank"].tolist() == [1, 2, 3]
