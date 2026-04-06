"""Tests for the validation result class."""
from src.omop.validators import ValidationResult


def test_validation_result_pass():
    r = ValidationResult("person_raw", "not_empty", True, "100 rows")
    assert r.passed
    assert "PASS" in repr(r)


def test_validation_result_fail():
    r = ValidationResult("person_raw", "not_empty", False, "0 rows")
    assert not r.passed
    assert "FAIL" in repr(r)
