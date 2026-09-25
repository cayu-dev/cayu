"""Native inert-session preflight shared by creation and retained preparation.

The result contains resolved in-process runtime registrations. It is not durable
authority, is not a public projection, and must never be serialized as a plan.
Only its bounded content identities may cross a durable preparation boundary.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.runtime import _session_request_boundary as request_boundary
from cayu.sessions.context_views import json_commitment

if TYPE_CHECKING:
    from cayu.runtime._session_engine import _PreparedInitialRun


@dataclass(frozen=True, slots=True)
class ParticipantCreationMaterial:
    initial_run: "_PreparedInitialRun"
    profile_json: str
    initial_input_json: str
    historical_definition_json: str


async def prepare_participant_creation_material(
    app, creation, *, resource_owner=None, attachments=()
):
    prepared = await app._session_engine._prepare_initial_run(
        app._with_application_run_defaults(creation.request),
        admit_session=False,
    )
    if prepared is None:
        raise RuntimeError("Participant session preparation did not produce a profile.")
    if attachments:
        environment = prepared.registered_environment
        if resource_owner is None or environment is None or environment.factory is not None:
            raise PermissionError(
                "Recipient attachments require a qualified static artifact environment."
            )
        # Creation calls this inside the retained resource-owner fence. Inspect
        # the resolved environment, never a caller-selected artifact store.
        await resource_owner._validate_context_artifacts(
            attachments,
            artifact_store=environment.environment.artifact_store,
            environment_name=environment.spec.name,
        )
    profile_json = canonical_bounded_durable_json_bytes(
        prepared.execution_profile.model_dump(mode="json"),
        "execution_profile",
        max_bytes=256 * 1024,
        max_nodes=8192,
        max_nesting=64,
    ).decode("utf-8")
    initial_input_json = canonical_bounded_durable_json_bytes(
        [message.model_dump(mode="json") for message in prepared.request.messages],
        "initial_input",
        max_bytes=8 * 1024 * 1024,
        max_nodes=8192,
        max_nesting=64,
    ).decode("utf-8")
    # Validate literal secrets before JSON escaping and preserve the resolved
    # definition, rather than re-reading a potentially replaced registration.
    for field_name, value in (
        ("agent_name", prepared.registered_agent.spec.name),
        ("agent_system_prompt", prepared.registered_agent.spec.system_prompt),
        ("rendered_system_prompt", prepared.rendered_system_prompt),
    ):
        request_boundary.require_secret_free_session_authority(
            value, field_name=field_name, redactor=app._secret_redactor
        )
    historical_definition_json = canonical_bounded_durable_json_bytes(
        {
            "historical_only": True,
            "agent_name": prepared.registered_agent.spec.name,
            "agent_system_prompt": prepared.registered_agent.spec.system_prompt,
            "rendered_system_prompt": prepared.rendered_system_prompt,
            "agent_definition_commitment": json_commitment(
                canonical_bounded_durable_json_bytes(
                    prepared.registered_agent.spec.model_dump(mode="json"),
                    "resolved agent definition",
                    max_bytes=256 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                ).decode("utf-8")
            ),
            "execution_profile_commitment": json_commitment(profile_json, "execution_profile"),
        },
        "historical definition",
        max_bytes=256 * 1024,
        max_nodes=8192,
        max_nesting=64,
    ).decode("utf-8")
    request_boundary.require_secret_free_session_authority(
        historical_definition_json,
        field_name="historical_definition",
        redactor=app._secret_redactor,
    )
    return ParticipantCreationMaterial(
        prepared, profile_json, initial_input_json, historical_definition_json
    )
