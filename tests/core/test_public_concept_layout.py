"""Public imports must preserve runtime identity and execute real sessions."""

from __future__ import annotations

import asyncio
import importlib
import inspect

import pytest
from examples.public_concepts.app import run_example

from cayu.events import EventType


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("cayu.agents", "AgentSpec"),
        ("cayu.messages", "Message"),
        ("cayu.events", "Event"),
        ("cayu.applications", "CayuApp"),
    ],
)
def test_public_definitions_have_canonical_identity(module_name, symbol):
    canonical = importlib.import_module(module_name)
    cls = getattr(canonical, symbol)
    assert cls is getattr(importlib.import_module("cayu"), symbol)
    assert inspect.getmodule(cls) is canonical


@pytest.mark.parametrize(
    "module_name",
    [
        "sessions",
        "tasks",
        "context",
        "approvals",
        "budgets",
        "snapshots",
        "snapshots.bundles",
        "delivery.git",
        "delivery.github",
        "exceptions",
        "tools",
        "workflows",
    ],
)
def test_public_packages_export_resolvable_objects(module_name):
    module = importlib.import_module(f"cayu.{module_name}")
    assert module.__all__
    assert len(module.__all__) == len(set(module.__all__))
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_example_completes_a_real_session_through_public_imports():
    events = asyncio.run(run_example())
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert any(event.type == EventType.MODEL_TEXT_DELTA for event in events)
    assert not any(event.type == EventType.SESSION_FAILED for event in events)
