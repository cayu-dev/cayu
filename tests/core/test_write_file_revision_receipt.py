"""Successful write receipts remain usable at the model message boundary."""

import asyncio

from cayu import LocalWorkspace, Message, ToolContext, WriteFileTool
from cayu.providers import ModelRequest, build_openai_payload


def test_write_file_model_visible_revisions_support_guarded_overwrites(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="receipt-workspace")
    ctx = ToolContext(session_id="receipt-session", workspace=workspace)
    tool = WriteFileTool()
    messages = [Message.text("user", "Update notes.txt.")]

    def write_and_render(content, *, expected_revision=None):
        args = {
            "path": "notes.txt",
            "content": content,
            "mode": "create" if expected_revision is None else "overwrite",
        }
        if expected_revision is not None:
            args["expected_revision"] = expected_revision
        result = asyncio.run(tool.run(ctx, args))
        assert not result.is_error
        call_id = f"call_{len(messages)}"
        messages.extend(
            [
                Message.tool_call(tool_call_id=call_id, tool_name="write_file", arguments=args),
                Message.tool_result(
                    tool_call_id=call_id,
                    tool_name="write_file",
                    content=result.content,
                    structured=result.structured,
                ),
            ]
        )
        # Round-trip the durable message before rendering the provider request.
        restored = [Message.model_validate_json(message.model_dump_json()) for message in messages]
        payload = build_openai_payload(ModelRequest(model="gpt-test", messages=restored))
        output = payload["input"][-1]
        assert output["type"] == "function_call_output"
        assert output["call_id"] == call_id
        receipt = output["output"]
        assert receipt.startswith(f"Wrote {len(content.encode('utf-8'))} bytes to notes.txt.\n")
        revision = receipt.split("\nRevision: ", 1)[1]
        assert revision == result.structured["revision"]
        assert revision == asyncio.run(workspace.read_bytes("notes.txt")).revision
        assert content not in receipt
        return revision

    created_revision = write_and_render("first private body")
    overwritten_revision = write_and_render(
        "second private body", expected_revision=created_revision
    )
    assert overwritten_revision != created_revision
    stale = asyncio.run(
        tool.run(
            ctx,
            {
                "path": "notes.txt",
                "content": "stale private body",
                "mode": "overwrite",
                "expected_revision": created_revision,
            },
        )
    )
    assert stale.is_error
    assert stale.structured["actual_revision"] == overwritten_revision
    assert (tmp_path / "notes.txt").read_text() == "second private body"
    write_and_render("third private body", expected_revision=overwritten_revision)
