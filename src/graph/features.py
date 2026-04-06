"""
Node feature engineering for the graph.

Converts patient and care-site feature tables into numeric tensors
suitable for GraphSAGE input.
"""
import numpy as np
import torch
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_patient_features(patient_features_df: DataFrame) -> tuple[np.ndarray, list[int]]:
    """
    Convert patient feature DataFrame to a numeric numpy array.

    Returns (features_array, person_ids) where features_array[i]
    corresponds to person_ids[i].
    """
    pdf = (
        patient_features_df
        .select(
            "person_id",
            "age",
            "visit_count",
            "distinct_care_sites",
            "distinct_providers",
            "condition_count",
            "distinct_conditions",
        )
        .orderBy("person_id")
        .toPandas()
    )
    person_ids = pdf["person_id"].tolist()
    numeric_cols = [c for c in pdf.columns if c != "person_id"]
    features = pdf[numeric_cols].values.astype(np.float32)

    # Standardize
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, person_ids


def encode_care_site_features(care_site_features_df: DataFrame) -> tuple[np.ndarray, list[int]]:
    """
    Convert care site feature DataFrame to a numeric numpy array.

    Returns (features_array, care_site_ids).
    """
    pdf = (
        care_site_features_df
        .select(
            "care_site_id",
            "patient_volume",
            "provider_count",
            "total_visits",
        )
        .orderBy("care_site_id")
        .toPandas()
    )
    site_ids = pdf["care_site_id"].tolist()
    numeric_cols = [c for c in pdf.columns if c != "care_site_id"]
    features = pdf[numeric_cols].values.astype(np.float32)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, site_ids


def pad_or_project_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Ensure feature vectors have exactly target_dim dimensions.

    If features have fewer dims, pad with zeros.
    If more, truncate (or project via random projection for large gaps).
    """
    n, d = features.shape
    if d == target_dim:
        return features
    elif d < target_dim:
        padding = np.zeros((n, target_dim - d), dtype=np.float32)
        return np.hstack([features, padding])
    else:
        return features[:, :target_dim]


def features_to_tensor(features: np.ndarray) -> torch.Tensor:
    """Convert numpy feature array to a PyTorch tensor."""
    return torch.from_numpy(features).float()
