"""Prompt context contracts."""

from __future__ import annotations

import asyncio
import threading

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_attachment_support import _app_with_artifact_store, _valid_png_bytes
from tests.runtime.runtime_environment_support import RecordingEnvironmentFactory
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

import cayu.applications as runtime_app_module
import cayu.runtime._model_step_executor as model_step_executor_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION, file_attachment
from cayu.artifacts.local import LocalArtifactStore
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.context.base import strip_old_file_attachments, trim_context_messages, trim_context_turns
from cayu.environments.base import Environment, EnvironmentSpec, WorkspaceInstructionsConfig
from cayu.events import EventType
from cayu.messages import FilePart, Message, TextPart
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.workspaces.local import LocalWorkspace


def test_trim_context_messages_preserves_complete_tool_rounds():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_1",
            tool_name="echo",
            arguments={"text": "value"},
        ),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="echo",
            content="value",
        ),
        Message.text("assistant", "done"),
    ]

    trimmed = trim_context_messages(messages, max_messages=2)

    assert [message.role for message in trimmed] == ["system", "assistant"]
    assert trimmed[0].content[0].text == "You are careful."
    assert trimmed[1].content[0].text == "done"


def test_trim_context_turns_keeps_last_user_turns_with_tool_rounds():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old one"),
        Message.text("assistant", "old answer"),
        Message.text("user", "old two"),
        Message.tool_call(
            tool_call_id="call_old",
            tool_name="echo",
            arguments={"text": "old"},
        ),
        Message.tool_result(
            tool_call_id="call_old",
            tool_name="echo",
            content="old",
        ),
        Message.text("assistant", "old tool answer"),
        Message.text("user", "current one"),
        Message.tool_call(
            tool_call_id="call_current",
            tool_name="echo",
            arguments={"text": "current"},
        ),
        Message.tool_result(
            tool_call_id="call_current",
            tool_name="echo",
            content="current",
        ),
        Message.text("assistant", "current answer"),
    ]

    trimmed = trim_context_turns(messages, max_user_turns=1)

    assert [message.role for message in trimmed] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert trimmed[0].content[0].text == "You are careful."
    assert trimmed[1].content[0].text == "current one"
    assert trimmed[2].content[0].tool_call_id == "call_current"
    assert trimmed[3].content[0].tool_call_id == "call_current"
    assert trimmed[4].content[0].text == "current answer"


def test_trim_context_turns_keeps_all_messages_when_there_are_no_user_turns():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("assistant", "hello"),
    ]

    trimmed = trim_context_turns(messages, max_user_turns=1)

    assert [message.content[0].text for message in trimmed] == [
        "You are careful.",
        "hello",
    ]


def test_trim_context_turns_can_drop_system_messages_when_requested():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
        Message.text("assistant", "current answer"),
    ]

    trimmed = trim_context_turns(
        messages,
        max_user_turns=1,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user", "assistant"]
    assert [message.content[0].text for message in trimmed] == [
        "current",
        "current answer",
    ]


def test_trim_context_turns_can_drop_system_messages_without_trimming_turns():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "current"),
        Message.text("assistant", "current answer"),
    ]

    trimmed = trim_context_turns(
        messages,
        max_user_turns=10,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user", "assistant"]
    assert [message.content[0].text for message in trimmed] == [
        "current",
        "current answer",
    ]


def test_trim_context_messages_can_drop_system_messages_without_trimming():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "current"),
    ]

    trimmed = trim_context_messages(
        messages,
        max_messages=10,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user"]
    assert trimmed[0].content[0].text == "current"


def test_trim_context_messages_uses_limit_after_dropping_system_messages():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("system", "Use concise answers."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    trimmed = trim_context_messages(
        messages,
        max_messages=3,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user", "assistant", "user"]
    assert [message.content[0].text for message in trimmed] == [
        "old",
        "old answer",
        "current",
    ]


def test_strip_old_file_attachments_preserves_latest_attachment_result_only():
    old_attachment = file_attachment(
        artifact_id="art_old",
        kind="image",
        filename="old.png",
        content_type="image/png",
        size_bytes=3,
    )
    current_attachment = file_attachment(
        artifact_id="art_current",
        kind="image",
        filename="current.png",
        content_type="image/png",
        size_bytes=7,
    )
    messages = [
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_old",
            tool_name="read_file",
            arguments={"artifact_id": "art_old"},
        ),
        Message.tool_result(
            tool_call_id="call_old",
            tool_name="read_file",
            content="Attached image artifact art_old: old.png.",
            artifacts=[old_attachment],
        ),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
        Message.tool_call(
            tool_call_id="call_current",
            tool_name="read_file",
            arguments={"artifact_id": "art_current"},
        ),
        Message.tool_result(
            tool_call_id="call_current",
            tool_name="read_file",
            content="Attached image artifact art_current: current.png.",
            artifacts=[current_attachment],
        ),
    ]

    projected = strip_old_file_attachments(messages, max_attachment_results=1)

    old_result = projected[2].content[0]
    current_result = projected[6].content[0]
    assert old_result.artifacts == []
    assert "omitted from this provider request" in old_result.content
    assert old_result.structured == {
        "cayu_file_attachments_stripped": [
            {
                "artifact_id": "art_old",
                "filename": "old.png",
                "content_type": "image/png",
                "size_bytes": 3,
                "kind": "image",
            }
        ]
    }
    assert current_result.artifacts == [current_attachment]
    assert messages[2].content[0].artifacts == [old_attachment]

    strip_all = strip_old_file_attachments(messages, max_attachment_results=0)
    assert strip_all[2].content[0].artifacts == []
    assert strip_all[6].content[0].artifacts == []


def _valid_jpeg_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _valid_pdf_bytes() -> bytes:
    import io

    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_attach_file_saves_reference_and_persists_bytes(tmp_path):
    app, store = _app_with_artifact_store(tmp_path)
    png = _valid_png_bytes()

    part = asyncio.run(
        app.attach_file(png, filename="pic.png", kind="image", session_id="sess_attach")
    )

    assert isinstance(part, FilePart)
    attachment = part.attachment
    assert attachment["kind"] == "image"
    assert attachment["filename"] == "pic.png"
    assert attachment["content_type"] == "image/png"
    assert attachment["size_bytes"] == len(png)

    read = asyncio.run(store.read_bytes(attachment["artifact_id"]))
    assert read.content == png


def test_attach_file_uses_factory_registered_store_without_materializing_factory(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="factory-artifacts")
    spec = EnvironmentSpec(name="factory-env")
    factory = RecordingEnvironmentFactory(Environment(spec, artifact_store=store))
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(spec, factory, artifact_store=store, default=True)
    png = _valid_png_bytes()

    part = asyncio.run(
        app.attach_file(png, filename="pic.png", kind="image", session_id="sess_attach")
    )

    read = asyncio.run(store.read_bytes(part.attachment["artifact_id"]))
    assert read.content == png
    assert factory.requests == []


def test_attach_file_runs_byte_validation_off_event_loop(tmp_path, monkeypatch):
    app, _ = _app_with_artifact_store(tmp_path)
    validation_threads: list[int] = []

    def record_validation(**_kwargs: object) -> None:
        validation_threads.append(threading.get_ident())

    monkeypatch.setattr(
        runtime_app_module,
        "validate_file_attachment_bytes",
        record_validation,
    )

    async def attach() -> tuple[int, FilePart]:
        event_loop_thread = threading.get_ident()
        part = await app.attach_file(
            b"image bytes",
            filename="pic.png",
            kind="image",
            session_id="sess_attach",
        )
        return event_loop_thread, part

    event_loop_thread, part = asyncio.run(attach())

    assert isinstance(part, FilePart)
    assert validation_threads
    assert validation_threads[0] != event_loop_thread


def test_attach_file_infers_content_type_from_filename(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    part = asyncio.run(
        app.attach_file(
            _valid_pdf_bytes(), filename="invoice.pdf", kind="document", session_id="sess_attach"
        )
    )

    assert part.attachment["content_type"] == "application/pdf"


def test_attach_file_rejects_bytes_over_limit(tmp_path):
    app, _ = _app_with_artifact_store(
        tmp_path, config=CayuConfig(tool_execution=ToolExecutionConfig(max_file_attachment_bytes=4))
    )

    with pytest.raises(ValueError, match="prompt attachment byte limit"):
        asyncio.run(app.attach_file(b"hello", filename="pic.png", kind="image"))


def test_attach_file_rejects_kind_content_type_mismatch(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError):
        asyncio.run(app.attach_file(_valid_pdf_bytes(), filename="invoice.pdf", kind="image"))


def test_attach_file_rejects_invalid_image_bytes(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(
            app.attach_file(b"not an image", filename="pic.png", kind="image", session_id="s")
        )


def test_attach_file_rejects_truncated_jpeg_that_passes_header_verification(tmp_path):
    import io

    from PIL import Image

    app, _ = _app_with_artifact_store(tmp_path)
    truncated = _valid_jpeg_bytes()[:-1]
    with Image.open(io.BytesIO(truncated)) as image:
        image.verify()

    with pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(
            app.attach_file(
                truncated,
                filename="truncated.jpg",
                kind="image",
                session_id="s",
            )
        )


def test_attach_file_rejects_image_decompression_bomb_warning(tmp_path, monkeypatch):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (15, 10), "white").save(buffer, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(
            app.attach_file(
                buffer.getvalue(),
                filename="oversized.png",
                kind="image",
                session_id="s",
            )
        )


def test_attach_file_rejects_image_over_decoded_size_limit(tmp_path):
    import io

    from PIL import Image

    from cayu import CayuConfig
    from cayu.configuration import ToolExecutionConfig

    buffer = io.BytesIO()
    Image.new("RGB", (15, 10), "white").save(buffer, format="PNG")
    app, _ = _app_with_artifact_store(
        tmp_path, config=CayuConfig(tool_execution=ToolExecutionConfig(image_max_frame_bytes=512))
    )

    with pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(
            app.attach_file(
                buffer.getvalue(),
                filename="oversized.png",
                kind="image",
                session_id="s",
            )
        )


def test_attach_file_rejects_image_bytes_content_type_mismatch(tmp_path):
    # JPEG bytes declared as image/png (via the .png filename) must be rejected before storing.
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="image/jpeg but the declared content type is image/png"):
        asyncio.run(
            app.attach_file(_valid_jpeg_bytes(), filename="wrong.png", kind="image", session_id="s")
        )

    # The correctly-labeled JPEG (via .jpg) is accepted.
    part = asyncio.run(
        app.attach_file(_valid_jpeg_bytes(), filename="ok.jpg", kind="image", session_id="s")
    )
    assert part.attachment["content_type"] == "image/jpeg"


def test_attach_file_requires_artifact_store(tmp_path):
    app = CayuApp(enable_logging=False)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(RuntimeError, match="environment registration with an artifact store"):
        asyncio.run(app.attach_file(_valid_png_bytes(), filename="pic.png", kind="image"))


def _user_image_part(artifact_id: str, filename: str) -> FilePart:
    return FilePart(
        attachment=file_attachment(
            artifact_id=artifact_id,
            kind="image",
            filename=filename,
            content_type="image/png",
            size_bytes=5,
        )
    )


def test_strip_old_prompt_files_keeps_latest_attach_turn():
    # Turn 1 attaches a file; turn 2 attaches another. The earlier prompt file is projected to a
    # text note; the latest attach turn keeps its FilePart. Durable transcript stays intact.
    messages = [
        Message(role="user", content=[TextPart(text="turn1"), _user_image_part("art_a", "a.png")]),
        Message.text("assistant", "ok"),
        Message(role="user", content=[TextPart(text="turn2"), _user_image_part("art_c", "c.png")]),
    ]

    projected = strip_old_file_attachments(messages)

    assert [type(part).__name__ for part in projected[0].content] == ["TextPart", "TextPart"]
    note = projected[0].content[1].text
    assert "omitted from this provider request" in note
    assert "artifact_id=art_a" in note
    assert [type(part).__name__ for part in projected[2].content] == ["TextPart", "FilePart"]
    # Durable transcript input is untouched (projection only).
    assert type(messages[0].content[1]) is FilePart


def test_strip_old_prompt_files_keeps_multi_file_prompt_intact_on_its_turn():
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="describe both"),
                _user_image_part("art_a", "a.png"),
                _user_image_part("art_b", "b.png"),
            ],
        )
    ]

    projected = strip_old_file_attachments(messages)

    kept = [type(part).__name__ for part in projected[0].content]
    assert kept.count("FilePart") == 2


def test_strip_old_prompt_files_projects_after_a_later_text_turn():
    # Once the attach turn has been answered and a new (text-only) user turn begins, the earlier
    # prompt file is projected to a note so its bytes are not re-sent on every later request.
    messages = [
        Message(role="user", content=[TextPart(text="turn1"), _user_image_part("art_a", "a.png")]),
        Message.text("assistant", "ok"),
        Message.text("user", "plain follow-up"),
    ]

    projected = strip_old_file_attachments(messages)

    assert not any(type(part) is FilePart for part in projected[0].content)
    note = next(
        part.text
        for part in projected[0].content
        if type(part) is TextPart and "omitted" in part.text
    )
    assert "art_a" in note


def test_strip_old_prompt_files_keeps_file_live_through_its_tool_loop():
    # A prompt file must survive its OWN run's tool loop: an assistant/tool response after the file
    # with no NEWER user turn is still the current attach turn, so the file stays provider-resolvable.
    messages = [
        Message(
            role="user", content=[TextPart(text="summarize"), _user_image_part("art_a", "a.png")]
        ),
        Message.tool_call(tool_call_id="call_1", tool_name="read_file"),
        Message.tool_result(tool_call_id="call_1", tool_name="read_file", content="page text"),
    ]

    projected = strip_old_file_attachments(messages)

    assert any(type(part) is FilePart for part in projected[0].content)


def test_strip_old_file_attachments_strips_old_tool_and_prompt_files_together():
    # Both stripping mechanisms fire in one call: an over-budget tool-result attachment AND an
    # earlier prompt file. Each keeps only its latest occurrence; the two are independent.
    old_tool = file_attachment(
        artifact_id="tool_old",
        kind="image",
        filename="told.png",
        content_type="image/png",
        size_bytes=3,
    )
    new_tool = file_attachment(
        artifact_id="tool_new",
        kind="image",
        filename="tnew.png",
        content_type="image/png",
        size_bytes=7,
    )
    messages = [
        Message(
            role="user", content=[TextPart(text="turn1"), _user_image_part("file_old", "old.png")]
        ),
        Message.tool_call(tool_call_id="call_old", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="call_old", tool_name="read_file", content="old", artifacts=[old_tool]
        ),
        Message(
            role="user", content=[TextPart(text="turn2"), _user_image_part("file_new", "new.png")]
        ),
        Message.tool_call(tool_call_id="call_new", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="call_new", tool_name="read_file", content="new", artifacts=[new_tool]
        ),
    ]

    projected = strip_old_file_attachments(messages, max_attachment_results=1)

    # Old prompt file -> note; latest prompt file kept.
    assert not any(type(part) is FilePart for part in projected[0].content)
    assert any(type(part) is FilePart for part in projected[3].content)
    # Old tool-result attachment stripped; latest tool-result attachment kept.
    assert projected[2].content[0].artifacts == []
    assert projected[5].content[0].artifacts == [new_tool]


def test_strip_old_prompt_files_keeps_all_files_from_the_same_turn():
    # Two file-bearing user messages in ONE turn (no assistant response between them) must both
    # reach the model; only a prior turn's files (separated by an assistant response) are stripped.
    same_turn = strip_old_file_attachments(
        [
            Message(role="user", content=[_user_image_part("art_a", "a.png")]),
            Message(role="user", content=[_user_image_part("art_b", "b.png")]),
        ]
    )
    assert any(type(part) is FilePart for part in same_turn[0].content)
    assert any(type(part) is FilePart for part in same_turn[1].content)

    across_turns = strip_old_file_attachments(
        [
            Message(role="user", content=[_user_image_part("art_a", "a.png")]),
            Message.text("assistant", "ok"),
            Message(role="user", content=[_user_image_part("art_b", "b.png")]),
        ]
    )
    assert not any(type(part) is FilePart for part in across_turns[0].content)
    assert any(type(part) is FilePart for part in across_turns[2].content)


def test_attach_file_rejects_inference_for_encoded_suffix(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="encoding"):
        asyncio.run(app.attach_file(b"gzipped", filename="invoice.pdf.gz", kind="document"))


def test_noteify_unresolvable_prompt_files_replaces_only_targeted_parts():
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="look"),
                _user_image_part("art_bad", "bad.png"),
                _user_image_part("art_ok", "ok.png"),
            ],
        )
    ]

    projected = model_step_executor_module.noteify_unresolvable_prompt_files(
        messages,
        {"art_bad"},
    )

    kept = projected[0].content
    assert [type(part).__name__ for part in kept] == ["TextPart", "FilePart", "TextPart"]
    assert kept[1].attachment["artifact_id"] == "art_ok"
    assert "could not be resolved" in kept[2].text
    assert "art_bad" in kept[2].text


def test_run_fails_open_on_unresolvable_prompt_file(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )

    async def flow() -> FilePart:
        # Attach under one session but run under another -> the live prompt file is unresolvable.
        part = await app.attach_file(
            _valid_png_bytes(), filename="pic.png", kind="image", session_id="sess_other"
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_run",
                messages=[Message(role="user", content=[TextPart(text="describe"), part])],
            )
        ):
            pass
        return part

    part = asyncio.run(flow())
    request = provider.requests[-1]

    # The run reached the provider (did not brick) with the file projected to a note, no bytes.
    assert part.attachment["artifact_id"] not in request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    user_message = next(message for message in request.messages if message.role == "user")
    assert not any(type(part_) is FilePart for part_ in user_message.content)
    assert any(
        type(part_) is TextPart and "could not be resolved" in part_.text
        for part_ in user_message.content
    )


def test_multi_turn_resume_ships_prompt_file_only_on_its_attach_turn(tmp_path):
    # End-to-end run -> resume through the real runtime: turn 1 attaches a file, turn 2 resumes and
    # attaches another. The turn-2 model request must carry the new file's bytes only; the old prompt
    # file must be projected to a text note and never re-resolved to bytes.
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("turn1"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("turn2"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )

    async def flow() -> tuple[FilePart, FilePart]:
        sid = "sess_multiturn_strip"
        old_part = await app.attach_file(
            _valid_png_bytes(), filename="old.png", kind="image", session_id=sid
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=sid,
                messages=[Message(role="user", content=[TextPart(text="turn1"), old_part])],
            )
        ):
            pass
        new_part = await app.attach_file(
            _valid_png_bytes(), filename="new.png", kind="image", session_id=sid
        )
        async for _ in app.resume(
            ResumeRequest(
                session_id=sid,
                messages=[Message(role="user", content=[TextPart(text="turn2"), new_part])],
            )
        ):
            pass
        return old_part, new_part

    old_part, new_part = asyncio.run(flow())
    old_id = old_part.attachment["artifact_id"]
    new_id = new_part.attachment["artifact_id"]

    turn2_request = provider.requests[-1]

    # Only the latest attach turn's file is resolved to bytes for the provider.
    resolved = turn2_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert set(resolved) == {new_id}

    user_messages = [message for message in turn2_request.messages if message.role == "user"]
    old_turn, new_turn = user_messages[0], user_messages[1]
    # Old prompt file projected to a text note; new prompt file kept as a FilePart.
    assert not any(type(part) is FilePart for part in old_turn.content)
    assert any(type(part) is FilePart for part in new_turn.content)
    old_note = " ".join(part.text for part in old_turn.content if type(part) is TextPart)
    assert "omitted from this provider request" in old_note
    assert old_id in old_note


def test_cayu_app_sends_agent_system_prompt_as_first_message():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        )
    )
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_system_prompt",
        messages=[Message.text("user", "hi")],
    )

    events = asyncio.run(collect_events(app, request))

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "user",
    ]
    assert provider.requests[0].messages[0].content[0].text == "You are careful."
    assert provider.requests[0].messages[1].content[0].text == "hi"
    assert [message.role for message in request.messages] == ["user"]
    transcript = asyncio.run(app.session_store.load_transcript("sess_system_prompt"))
    assert [message.role for message in transcript] == [
        "system",
        "user",
        "assistant",
    ]


def test_cayu_app_renders_explicit_workspace_instructions_with_agent_prompt():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace_instructions="Run tests with uv run pytest.",
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_workspace_instructions",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "user",
    ]
    system_text = provider.requests[0].messages[0].content[0].text
    assert "[Agent instructions]\nYou are careful." in system_text
    assert "[Workspace instructions]" in system_text
    assert "Source: explicit" in system_text
    assert "Run tests with uv run pytest." in system_text
    assert provider.requests[0].messages[1].content[0].text == "hi"
    transcript = asyncio.run(app.session_store.load_transcript("sess_workspace_instructions"))
    assert [message.role for message in transcript] == ["system", "user", "assistant"]
    assert transcript[0].content[0].text == system_text


def test_cayu_app_loads_workspace_instructions_from_configured_file(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Use pnpm and do not edit generated files.\n")
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="repo"),
            workspace=LocalWorkspace(tmp_path, workspace_id="repo"),
            workspace_instructions=WorkspaceInstructionsConfig(paths=("CLAUDE.md",)),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_workspace_instruction_file",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "user",
    ]
    system_text = provider.requests[0].messages[0].content[0].text
    assert "Source: CLAUDE.md" in system_text
    assert "Use pnpm and do not edit generated files." in system_text


def test_cayu_app_merges_workspace_instruction_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use uv for Python commands.\n")
    (tmp_path / ".cayu").mkdir()
    (tmp_path / ".cayu" / "AGENTS.md").write_text("Run focused tests before broad tests.\n")
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="repo"),
            workspace=LocalWorkspace(tmp_path, workspace_id="repo"),
            workspace_instructions=WorkspaceInstructionsConfig(mode="merge"),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_merged_workspace_instruction_files",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    system_text = provider.requests[0].messages[0].content[0].text
    assert "Source: AGENTS.md, .cayu/AGENTS.md" in system_text
    assert "Source: AGENTS.md\nUse uv for Python commands." in system_text
    assert "Source: .cayu/AGENTS.md\nRun focused tests before broad tests." in system_text


def test_workspace_instruction_file_config_rejects_paths_outside_workspace():
    with pytest.raises(ValidationError, match="must stay inside the workspace"):
        WorkspaceInstructionsConfig(paths=("../AGENTS.md",))


def test_workspace_instruction_failure_rejects_before_profile_admission(tmp_path):
    (tmp_path / "AGENTS.md").write_text("too long")
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="repo"),
            workspace=LocalWorkspace(tmp_path, workspace_id="repo"),
            workspace_instructions=WorkspaceInstructionsConfig(
                paths=("AGENTS.md",),
                max_bytes=3,
            ),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="Workspace instructions file exceeds 3 bytes"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_oversized_workspace_instructions",
                    messages=[Message.text("user", "hi")],
                ),
            )
        )

    session = asyncio.run(app.session_store.load("sess_oversized_workspace_instructions"))
    assert session is None
    assert provider.requests == []


@pytest.mark.parametrize("system_prompt", [None, "", "   "])
def test_cayu_app_does_not_send_blank_agent_system_prompt(system_prompt):
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt=system_prompt,
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_blank_system_prompt_{system_prompt!r}",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == ["user"]
