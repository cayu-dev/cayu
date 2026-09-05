"""Closed diagnostics cannot become an exception-content persistence channel."""

import pytest
from pydantic import ValidationError

from cayu.evals import EvalRunFailureDiagnostic, EvalRunFailureReason


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "secret-provider-response"},
        {"reason": "execution_failed", "traceback": "secret"},
        {"reason": "execution_failed", "provider_protocol_reason": "unspecified"},
        {"reason": "provider_protocol_failed", "provider_protocol_reason": "secret-token"},
        {"reason": "provider_protocol_failed", "provider_protocol_reason": "x" * 129},
    ],
)
def test_failure_diagnostics_reject_untrusted_fields(payload):
    with pytest.raises(ValidationError):
        EvalRunFailureDiagnostic.model_validate(payload)


def test_protocol_exception_projection_uses_catalogue_not_message():
    from cayu.providers.openai import OpenAIProtocolError
    from cayu.server.evals_worker import _execution_diagnostic

    diagnostic = _execution_diagnostic(
        ExceptionGroup(
            "secret",
            [
                OpenAIProtocolError("secret", reason_code="secret-value"),
            ],
        )
    )
    assert diagnostic.reason is EvalRunFailureReason.PROVIDER_PROTOCOL_FAILED
    assert diagnostic.provider_protocol_reason == "unspecified"
    assert "secret" not in diagnostic.model_dump_json()
