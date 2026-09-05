"""Pixel authority survives artifact readback before provider image projection."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation

from cayu import (
    BrowserVisualPolicy,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LocalArtifactStore,
    Message,
)
from cayu.artifacts import ArtifactScope, file_attachment
from cayu.runtime import Session
from cayu.runtime._model_step_executor import _FileAttachmentUnavailable, _resolved_file_attachments


@pytest.mark.parametrize(
    "alteration",
    [None, "missing_policy", "model_denied", "wrong_store", "changed_pixels", "wrong_digest"],
)
def test_visual_image_projection_authenticates_bytes_and_policy(
    tmp_path: Path, alteration: str | None
) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="visual")
        app = CayuApp(enable_logging=False)
        app.register_environment(
            Environment(EnvironmentSpec(name="browser"), artifact_store=store), default=True
        )
        session = Session(
            id="visual_projection",
            agent_name="assistant",
            provider_name="fake",
            model="fake",
            causal_budget_id="visual_projection",
            invocation=fixture_session_invocation("visual_projection"),
        )
        pixels = b"pixels"
        digest = hashlib.sha256(pixels).hexdigest()
        policy = BrowserVisualPolicy(
            artifact_store_id="other" if alteration == "wrong_store" else store.id,
            allowed_origins=("https://visual.browser.test",),
            retention="application_managed",
            publish_to_model=alteration != "model_denied",
        )
        metadata = {"content_sha256": digest}
        if alteration != "missing_policy":
            metadata["visual_publication"] = policy.model_dump(mode="json")
        artifact = await store.put_bytes(
            b"mutate" if alteration == "changed_pixels" else pixels,
            filename="visual.png",
            content_type="image/png",
            scope=ArtifactScope.SESSION,
            session_id=session.id,
            agent_name=session.agent_name,
            environment_name="browser",
            metadata=metadata,
        )
        attachment = file_attachment(
            artifact_id=artifact.id,
            kind="image",
            filename="visual.png",
            content_type="image/png",
            size_bytes=len(pixels),
            metadata={
                "browser_visual_screenshot_sha256": "0" * 64
                if alteration == "wrong_digest"
                else digest
            },
        )
        messages = [
            Message.tool_call(tool_call_id="capture", tool_name="browser_session"),
            Message.tool_result(
                tool_call_id="capture",
                tool_name="browser_session",
                content="visual evidence",
                artifacts=[attachment],
            ),
        ]

        async def resolve():
            return await _resolved_file_attachments(
                messages=messages,
                session=session,
                registered_environment=app._environments["browser"],
                max_file_attachment_bytes=100,
                max_total_file_attachment_bytes=100,
                max_file_attachments_per_request=1,
            )

        if alteration is not None:
            with pytest.raises(_FileAttachmentUnavailable):
                await resolve()
        else:
            resolved, missing = await resolve()
            assert not missing
            assert resolved[artifact.id]["content_sha256"] == digest

    asyncio.run(scenario())
