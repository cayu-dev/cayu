"""Portable-text contracts for public verification evidence."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from cayu import ToolEffect
from cayu.testing import (
    ProviderCredentialIsolationVerification,
    ToolEffectVerification,
    ToolEffectVerificationStatus,
)


def _tool_effect_verification(**overrides) -> ToolEffectVerification:
    values = {
        "status": ToolEffectVerificationStatus.CONSISTENT,
        "agent_name": "reviewer",
        "tool_name": "inspect",
        "declared_effect": ToolEffect.NONE,
        "observed_mutation": False,
        "execution_succeeded": True,
        "result_is_error": False,
        "timeout_seconds": 1.0,
        "workspace_max_entries": 10,
        "workspace_max_files": 10,
        "workspace_max_file_bytes": 1024,
        "workspace_max_total_bytes": 4096,
        "unobserved_systems": ("network",),
        "limitations": ("Workspace-only evidence.",),
    }
    values.update(overrides)
    return ToolEffectVerification(**values)


def _provider_verification(**overrides) -> ProviderCredentialIsolationVerification:
    values = {
        "status": "environment_minimized",
        "adapter": "openai",
        "scope": "local_environment",
        "canary_labels": ("api_key",),
        "positive_controls": ("environment",),
    }
    values.update(overrides)
    return ProviderCredentialIsolationVerification(**values)


@pytest.mark.parametrize("bad_text", ["bad\x00text", "bad\ud800text"])
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda value: _tool_effect_verification(agent_name=value), id="agent-name"),
        pytest.param(lambda value: _tool_effect_verification(tool_name=value), id="tool-name"),
        pytest.param(
            lambda value: _tool_effect_verification(
                status=ToolEffectVerificationStatus.MISMATCH,
                created_paths=(value,),
                observed_mutation=True,
            ),
            id="created-path",
        ),
        pytest.param(
            lambda value: _tool_effect_verification(
                status=ToolEffectVerificationStatus.MISMATCH,
                updated_paths=(value,),
                observed_mutation=True,
            ),
            id="updated-path",
        ),
        pytest.param(
            lambda value: _tool_effect_verification(
                status=ToolEffectVerificationStatus.MISMATCH,
                deleted_paths=(value,),
                observed_mutation=True,
            ),
            id="deleted-path",
        ),
        pytest.param(
            lambda value: _tool_effect_verification(
                status=ToolEffectVerificationStatus.EXECUTION_FAILED,
                execution_succeeded=False,
                result_is_error=None,
                exception_type=value,
            ),
            id="exception-type",
        ),
        pytest.param(
            lambda value: _tool_effect_verification(unobserved_systems=(value,)),
            id="unobserved-system",
        ),
        pytest.param(
            lambda value: _tool_effect_verification(limitations=(value,)), id="limitation"
        ),
        pytest.param(lambda value: _provider_verification(adapter=value), id="adapter"),
        pytest.param(
            lambda value: _provider_verification(canary_labels=(value,)),
            id="canary-label",
        ),
        pytest.param(
            lambda value: _provider_verification(positive_controls=(value,)),
            id="positive-control",
        ),
        pytest.param(
            lambda value: _provider_verification(
                status="verified",
                scope="isolated_guest",
                auth_search_labels=(value,),
            ),
            id="auth-search-label",
        ),
    ],
)
def test_verification_evidence_rejects_nonportable_text(
    factory: Callable[[str], object],
    bad_text: str,
) -> None:
    with pytest.raises(ValidationError, match="NUL|surrogate"):
        factory(bad_text)


def test_verification_evidence_preserves_ordinary_unicode() -> None:
    tool = _tool_effect_verification(
        agent_name="réviseur-🤖",
        created_paths=(),
        limitations=("Vérification bornée — 東京",),
    )
    provider = _provider_verification(
        adapter="fournisseur-é",
        canary_labels=("clé-東京",),
        positive_controls=("contrôle-🤖",),
    )

    assert tool.agent_name == "réviseur-🤖"
    assert tool.limitations == ("Vérification bornée — 東京",)
    assert provider.canary_labels == ("clé-東京",)


@pytest.mark.parametrize("limitation", ["", "   "])
def test_tool_effect_verification_preserves_existing_blank_limitation_semantics(
    limitation: str,
) -> None:
    evidence = _tool_effect_verification(limitations=(limitation,))

    assert evidence.limitations == (limitation,)
