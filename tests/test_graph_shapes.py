"""Tests for graph construction shapes and invariants."""
import numpy as np
import torch
from src.graph.features import pad_or_project_features, features_to_tensor


def test_pad_features():
    features = np.random.randn(10, 5).astype(np.float32)
    padded = pad_or_project_features(features, 32)
    assert padded.shape == (10, 32)
    # Original features preserved
    np.testing.assert_array_equal(padded[:, :5], features)
    # Padding is zeros
    np.testing.assert_array_equal(padded[:, 5:], 0)


def test_truncate_features():
    features = np.random.randn(10, 64).astype(np.float32)
    truncated = pad_or_project_features(features, 32)
    assert truncated.shape == (10, 32)


def test_exact_features():
    features = np.random.randn(10, 32).astype(np.float32)
    result = pad_or_project_features(features, 32)
    np.testing.assert_array_equal(result, features)


def test_features_to_tensor():
    features = np.random.randn(10, 32).astype(np.float32)
    tensor = features_to_tensor(features)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (10, 32)
    assert tensor.dtype == torch.float32
