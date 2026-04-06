"""Tests for configuration module."""
from src.config.settings import ProjectConfig, load_config


def test_default_config():
    cfg = ProjectConfig()
    assert cfg.catalog == "gnn_hls_graphsage"
    assert cfg.schema == "gnn_hls_graphsage_db"
    assert cfg.uc_prefix == "gnn_hls_graphsage.gnn_hls_graphsage_db"
    assert cfg.num_node_features == 32
    assert cfg.test_p + cfg.valid_p < 1.0


def test_load_config_with_overrides():
    cfg = load_config({"catalog": "my_catalog", "synthetic_scale": "5000"})
    assert cfg.catalog == "my_catalog"
    assert cfg.synthetic_scale == 5000
    assert cfg.schema == "gnn_hls_graphsage_db"  # unchanged


def test_load_config_bool_override():
    cfg = load_config({"rebuild_synthetic": "false"})
    assert cfg.rebuild_synthetic is False


def test_table_name():
    cfg = ProjectConfig()
    assert cfg.table("my_table") == "gnn_hls_graphsage.gnn_hls_graphsage_db.my_table"


def test_uc_model():
    cfg = ProjectConfig()
    assert cfg.uc_model == "gnn_hls_graphsage.gnn_hls_graphsage_db.patient_recommendations_gnn"


def test_volume_path():
    cfg = ProjectConfig()
    assert cfg.volume_path == "/Volumes/gnn_hls_graphsage/gnn_hls_graphsage_db/gnn_data"
