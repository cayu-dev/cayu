"""Canonical exports for the knowledge package migration."""

import importlib

import pytest

import cayu

_MODULES = (
    ("curator", "LearningSignal"),
    ("enrichment", "KnowledgeEnrichmentQueueConfig"),
    ("governance", "KnowledgeActivationPolicyError"),
    ("maintenance", "KnowledgeMaintenanceSignalKind"),
    ("maintenance_governance", "KnowledgeMaintenanceGovernanceDisposition"),
    ("maintenance_persistence", "KnowledgeMaintenanceProposalPublicationOutcome"),
    ("maintenance_planning", "KnowledgeMaintenancePlanner"),
    ("semantic_watch", "KnowledgeSemanticWatchConfig"),
)


@pytest.mark.parametrize(("module_name", "symbol"), _MODULES)
def test_root_exports_share_canonical_identity(module_name, symbol):
    canonical = importlib.import_module(f"cayu.knowledge.{module_name}")
    assert getattr(cayu, symbol) is getattr(canonical, symbol)
