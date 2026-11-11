"""SDD-SCANSURE-AI-002 §4 / T5: the registry says at start which mode it runs in."""
import logging
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from models import model_registry  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_registry():
    model_registry.reset_cache()
    yield
    model_registry.reset_cache()


def test_default_mode_logs_counts_and_does_not_warn(monkeypatch, caplog):
    monkeypatch.delenv("AI_AUTHORIZED_ONLY", raising=False)
    with caplog.at_level(logging.INFO, logger="ai_inference.model_registry"):
        reg = model_registry.registry()
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("AI registry loaded" in r.getMessage() and "authorized_only=False" in r.getMessage() for r in infos)
    assert any(f"{len(reg.enabled)} enabled" in r.getMessage() for r in infos)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_authorized_only_with_no_t1_warns_loudly(monkeypatch, caplog):
    """No shipped manifest carries a current authorisation (decision D3: no T1 vendor yet)."""
    monkeypatch.setenv("AI_AUTHORIZED_ONLY", "true")
    with caplog.at_level(logging.INFO, logger="ai_inference.model_registry"):
        model_registry.registry()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "AI_AUTHORIZED_ONLY=true" in warnings[0] and "descriptive summary" in warnings[0]
    assert any("authorized_only=True, 0 with current authorization" in r.getMessage() for r in caplog.records)


def test_mode_is_logged_once_per_load(monkeypatch, caplog):
    monkeypatch.setenv("AI_AUTHORIZED_ONLY", "true")
    with caplog.at_level(logging.INFO, logger="ai_inference.model_registry"):
        model_registry.registry()
        model_registry.registry()
    assert len([r for r in caplog.records if "AI registry loaded" in r.getMessage()]) == 1
