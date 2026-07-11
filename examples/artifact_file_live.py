"""Verified live provider contract for file-backed artifacts."""

from __future__ import annotations

import asyncio
import io
import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from _live_checks import require, require_positive_model_usage, require_successful_terminal
from cayu import (
    AgentSpec,
    AnthropicProvider,
    ArtifactScope,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ListArtifactsTool,
    LocalArtifactStore,
    Message,
    OpenAIProvider,
    ReadFileTool,
    RunRequest,
    file_attachment_from_payload,
)

TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00"
    b"\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def main() -> None:
    if not _has_file_reader_dependencies():
        raise SystemExit("Install optional file readers first: uv sync --extra dev --extra files")

    try:
        provider_name, model = _provider_config()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    session_id = f"demo_{provider_name}_artifact_file"
    artifact_kind = _artifact_kind()
    filename, content_type, content, prompt_description = _artifact_fixture(artifact_kind)
    with TemporaryDirectory(prefix="cayu-artifact-file-live-") as temporary_directory:
        artifact_store = LocalArtifactStore(
            Path(temporary_directory) / "artifacts",
            store_id="artifact-file-demo",
        )
        artifact = await artifact_store.put_bytes(
            content,
            filename=filename,
            content_type=content_type,
            scope=ArtifactScope.SESSION,
            session_id=session_id,
            agent_name="assistant",
            environment_name="local-dev",
            metadata={"example": "artifact_file_live", "artifact_kind": artifact_kind},
        )

        print("artifact_store_root", artifact_store.root)
        print("artifact_id", artifact.id)
        print("artifact_kind", artifact_kind)
        print("provider", provider_name)
        print("model", model)

        app = CayuApp(enable_logging=False)
        if provider_name == "openai":
            app.register_provider(OpenAIProvider(), default=True)
        else:
            app.register_provider(AnthropicProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local-dev", metadata={"kind": "local"}),
                artifact_store=artifact_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="assistant",
                model=model,
                system_prompt=(
                    "You are testing Cayu artifact file attachments. Use the read_file tool "
                    "with the provided artifact_id before answering. Keep the final answer short."
                ),
            ),
            tools=[ReadFileTool(), ListArtifactsTool()],
        )

        request = RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[
                Message.text(
                    "user",
                    (
                        f"Inspect artifact_id {artifact.id} with read_file. "
                        f"After the tool result is returned, describe {prompt_description}."
                    ),
                )
            ],
        )
        events: list[Event] = []
        async for event in app.run(request):
            events.append(event)
            print(
                event.type,
                event.environment_name or "-",
                event.tool_name or "-",
                event.payload,
            )

        _validate_runtime_events(events, artifact_id=artifact.id)
        print("status ok")


def _validate_runtime_events(events: list[Event], *, artifact_id: str) -> None:
    require_successful_terminal(events)

    read_events = [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED and event.tool_name == "read_file"
    ]
    require(bool(read_events), "read_file did not complete")
    attachments = [
        attachment
        for event in read_events
        for item in event.payload.get("result", {}).get("artifacts", [])
        if (attachment := file_attachment_from_payload(item)) is not None
    ]
    require(
        any(attachment.artifact_id == artifact_id for attachment in attachments),
        "read_file result did not include a canonical file attachment for "
        f"expected artifact {artifact_id!r}: {attachments!r}",
    )

    require_positive_model_usage(events)


def _provider_config() -> tuple[str, str]:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
    if requested in {None, ""}:
        if os.environ.get("OPENAI_API_KEY"):
            requested = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            requested = "anthropic"
        else:
            raise RuntimeError("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this live example.")
    if requested == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
        return "openai", os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.5")
    if requested == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")
        return "anthropic", os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    raise RuntimeError("CAYU_PROVIDER must be openai or anthropic.")


def _artifact_kind() -> str:
    kind = os.environ.get("CAYU_ARTIFACT_KIND", "image").strip().lower()
    if kind not in {"image", "pdf"}:
        raise RuntimeError("CAYU_ARTIFACT_KIND must be image or pdf.")
    return kind


def _artifact_fixture(kind: str) -> tuple[str, str, bytes, str]:
    if kind == "image":
        return "red-dot.png", "image/png", TINY_PNG_BYTES, "what the image contains"
    if kind == "pdf":
        return "blank-page.pdf", "application/pdf", _tiny_pdf_bytes(), "what the PDF contains"
    raise RuntimeError("CAYU_ARTIFACT_KIND must be image or pdf.")


def _tiny_pdf_bytes() -> bytes:
    pypdf = import_module("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _has_file_reader_dependencies() -> bool:
    try:
        import_module("PIL.Image")
        import_module("pypdf")
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    asyncio.run(main())
