"""Generic visual-source controls through the native image tool boundary."""

import asyncio
from hashlib import sha256
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from cayu import (
    CayuApp,
    CayuConfig,
    Environment,
    EnvironmentSpec,
    LocalArtifactStore,
    ToolExecutionConfig,
)
from cayu.tools.base import ToolContext
from cayu.tools.files import ReadFileTool


def image_bytes(size=(1280, 19884)):
    with Image.new("RGB", size, "white") as image:
        draw = ImageDraw.Draw(image)
        for y, color in [(0, "red"), (size[1] // 2, "green"), (size[1] - 20, "blue")]:
            draw.rectangle((0, y, 19, y + 19), fill=color)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def test_long_image_regions_preserve_source_and_pixels(tmp_path):
    async def run():
        content = image_bytes()
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            content, filename="long.png", content_type="image/png", session_id="s"
        )
        ctx = ToolContext(session_id="s", artifact_store=store)
        denied = await ReadFileTool().run(ctx, {"artifact_id": source.id})
        assert denied.is_error
        assert denied.structured["image_decode_limits"]["max_frame_bytes"] == 64 * 1024 * 1024
        assert "ToolExecutionConfig.image_max_frame_bytes" in denied.content
        limits = {
            "max_frame_bytes": 128 * 1024 * 1024,
            "max_total_bytes": 256 * 1024 * 1024,
            "max_frames": 1024,
        }
        ctx = ctx.model_copy(update={"image_decode_limits": limits})
        for top, pixel in [(0, (255, 0, 0)), (9942, (0, 128, 0)), (19864, (0, 0, 255))]:
            result = await ReadFileTool().run(
                ctx, {"artifact_id": source.id, "image_region": [0, top, 20, top + 20]}
            )
            assert not result.is_error, result.content
            assert result.structured["source_sha256"] == sha256(content).hexdigest()
            assert sha256(content).hexdigest() in result.content
            assert result.structured["source_dimensions"] == {"width": 1280, "height": 19884}
            assert result.structured["scale"] == 1
            assert result.structured["whole_source_delivered"] is False
            assert result.structured["semantic_inspection"] == "not_established_by_tool"
            derived = await store.read_bytes(result.structured["attachment_artifact_id"])
            assert sha256(derived.content).hexdigest() == result.structured["attachment_sha256"]
            with Image.open(BytesIO(derived.content)) as image:
                assert image.size == (20, 20)
                assert image.getpixel((10, 10)) == pixel
        assert (await store.read_bytes(source.id)).content == content

    asyncio.run(run())


@pytest.mark.parametrize(
    "region", [[-1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 1], [False, 0, 1, 1], [0, 0, 21, 20]]
)
def test_invalid_regions_fail_without_image_delivery(tmp_path, region):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            image_bytes((20, 20)), filename="small.png", content_type="image/png", session_id="s"
        )
        result = await ReadFileTool().run(
            ToolContext(session_id="s", artifact_store=store),
            {"artifact_id": source.id, "image_region": region},
        )
        assert result.is_error
        assert not result.artifacts

    asyncio.run(run())


def test_region_attachment_limit_never_silently_downsamples(tmp_path):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            image_bytes((20, 20)), filename="small.png", content_type="image/png", session_id="s"
        )
        result = await ReadFileTool().run(
            ToolContext(session_id="s", artifact_store=store),
            {"artifact_id": source.id, "image_region": [0, 0, 20, 20], "max_attachment_bytes": 1},
        )
        assert result.is_error
        assert result.structured["error"] == "image_region_attachment_too_large"
        assert result.structured["region_delivered"] is False
        assert result.structured["whole_source_delivered"] is False
        assert result.structured["scale"] == 1
        assert not result.artifacts

    asyncio.run(run())


def test_app_attachment_uses_host_image_policy(tmp_path):
    async def run():
        content = image_bytes()

        def app(limit):
            result = CayuApp(
                enable_logging=False,
                config=CayuConfig(tool_execution=ToolExecutionConfig(image_max_frame_bytes=limit)),
            )
            result.register_environment(
                Environment(
                    EnvironmentSpec(name="local"),
                    artifact_store=LocalArtifactStore(tmp_path / str(limit)),
                ),
                default=True,
            )
            return result

        with pytest.raises(ValueError, match="101806080 > 67108864"):
            await app(64 * 1024 * 1024).attach_file(
                content, filename="long.png", kind="image", session_id="s"
            )
        part = await app(128 * 1024 * 1024).attach_file(
            content, filename="long.png", kind="image", session_id="s"
        )
        assert part

    asyncio.run(run())


def test_native_runtime_delivers_configured_region_with_source_identity(tmp_path):
    from cayu import (
        AgentSpec,
        Message,
        ModelStreamEvent,
        RunRequest,
        ScriptedModelProvider,
        ToolResultPart,
    )

    async def run():
        content = image_bytes()
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            content, filename="long.png", content_type="image/png", session_id="native"
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="read_file",
                        id="region",
                        arguments={"artifact_id": source.id, "image_region": [0, 19864, 20, 19884]},
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.text_delta("Received the requested region."),
                    ModelStreamEvent.completed(),
                ],
            ]
        )
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(image_max_frame_bytes=128 * 1024 * 1024)
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), artifact_store=store), default=True
        )
        app.register_agent(AgentSpec(name="reader", model="scripted"), tools=[ReadFileTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="reader",
                    session_id="native",
                    messages=[Message.text("user", "Inspect the requested bottom region.")],
                )
            )
        ]
        results = [
            event.payload["result"] for event in events if str(event.type) == "tool.call.completed"
        ]
        assert len(results) == 1
        result = results[0]
        assert not result["is_error"], result
        assert result["structured"]["image_decode_limits"]["max_frame_bytes"] == 128 * 1024 * 1024
        assert result["structured"]["source_sha256"] == sha256(content).hexdigest()
        assert result["structured"]["source_region"] == [0, 19864, 20, 19884]
        assert len(provider.requests) == 2
        assert any(
            isinstance(part, ToolResultPart) and sha256(content).hexdigest() in part.content
            for message in provider.requests[-1].messages
            for part in message.content
        )
        assert any(str(event.type) == "session.completed" for event in events)

    asyncio.run(run())


def test_region_uses_original_workspace_snapshot_after_source_changes(tmp_path):
    from cayu import LocalWorkspace

    async def run():
        content = image_bytes((20, 20))
        (tmp_path / "page.png").write_bytes(content)
        store = LocalArtifactStore(tmp_path / "artifacts")
        ctx = ToolContext(session_id="s", artifact_store=store, workspace=LocalWorkspace(tmp_path))
        first = await ReadFileTool().run(ctx, {"path": "page.png", "image_region": [0, 0, 10, 10]})
        assert not first.is_error, first.content
        (tmp_path / "page.png").write_bytes(b"changed source is not the original image")
        second = await ReadFileTool().run(
            ctx,
            {
                "artifact_id": first.structured["source_artifact_id"],
                "image_region": [10, 10, 20, 20],
            },
        )
        assert not second.is_error, second.content
        assert (
            second.structured["source_sha256"]
            == first.structured["source_sha256"]
            == sha256(content).hexdigest()
        )
        assert second.structured["source_region"] == [10, 10, 20, 20]

    asyncio.run(run())


@pytest.mark.parametrize(
    "limits, message",
    [
        ({"max_frame_bytes": 399}, "decoded size"),
        ({"max_total_bytes": 1199}, "aggregate decoded size"),
        ({"max_frames": 2}, "frame count"),
    ],
)
def test_region_cannot_bypass_source_animation_policy(tmp_path, limits, message):
    async def run():
        frames = [Image.new("RGB", (10, 10), color) for color in ["red", "green", "blue"]]
        output = BytesIO()
        try:
            frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:])
        finally:
            for frame in frames:
                frame.close()
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            output.getvalue(), filename="frames.gif", content_type="image/gif", session_id="s"
        )
        ctx = ToolContext(session_id="s", artifact_store=store, image_decode_limits=limits)
        result = await ReadFileTool().run(
            ctx, {"artifact_id": source.id, "image_region": [0, 0, 1, 1]}
        )
        assert result.is_error
        assert message in result.content
        assert result.artifacts == []
        assert result.structured["source_sha256"] == sha256(output.getvalue()).hexdigest()

    asyncio.run(run())


def test_effective_image_policy_is_discoverable_and_changes_manifest_identity():
    default = CayuApp(enable_logging=False).describe()
    configured = CayuApp(
        enable_logging=False,
        config=CayuConfig(
            tool_execution=ToolExecutionConfig(image_max_frame_bytes=128 * 1024 * 1024)
        ),
    ).describe()
    assert default.fingerprint != configured.fingerprint
    assert (
        configured.runtime.configuration.values["tool_execution"]["image_max_frame_bytes"]
        == 128 * 1024 * 1024
    )
    provenance = next(
        row
        for row in configured.runtime.configuration.provenance
        if row.path == "tool_execution.image_max_frame_bytes"
    )
    assert provenance.owner == "cayu.configuration.ToolExecutionConfig"


def test_region_coordinates_do_not_inherit_exif_rotation(tmp_path):
    async def run():
        with Image.new("RGB", (30, 20), "red") as image:
            exif = Image.Exif()
            exif[274] = 6
            buffer = BytesIO()
            image.save(buffer, format="PNG", exif=exif)
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            buffer.getvalue(), filename="oriented.png", content_type="image/png", session_id="s"
        )
        result = await ReadFileTool().run(
            ToolContext(session_id="s", artifact_store=store),
            {"artifact_id": source.id, "image_region": [0, 0, 15, 10]},
        )
        assert not result.is_error, result.content
        derivative = await store.read_bytes(result.structured["attachment_artifact_id"])
        with Image.open(BytesIO(derivative.content)) as image:
            assert image.size == (15, 10)
            assert image.getexif().get(274) is None
        assert result.structured["source_dimensions"] == {"width": 30, "height": 20}

    asyncio.run(run())


def test_cmyk_jpeg_region_preserves_pixels_and_source_identity(tmp_path):
    async def run():
        with Image.new("CMYK", (30, 20), (0, 255, 255, 0)) as image:
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
        content = buffer.getvalue()
        store = LocalArtifactStore(tmp_path / "artifacts")
        source = await store.put_bytes(
            content, filename="cmyk.jpg", content_type="image/jpeg", session_id="s"
        )
        ctx = ToolContext(session_id="s", artifact_store=store)
        original = await ReadFileTool().run(ctx, {"artifact_id": source.id})
        assert not original.is_error, original.content
        result = await ReadFileTool().run(
            ctx, {"artifact_id": source.id, "image_region": [5, 3, 15, 13]}
        )
        assert not result.is_error, result.content
        derivative = await store.read_bytes(result.structured["attachment_artifact_id"])
        with Image.open(BytesIO(derivative.content)) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (10, 10)
            assert image.getpixel((5, 5)) == (255, 0, 0)
        assert result.structured["source_artifact_id"] == source.id
        assert result.structured["source_sha256"] == sha256(content).hexdigest()
        assert result.structured["source_dimensions"] == {"width": 30, "height": 20}
        assert result.structured["source_region"] == [5, 3, 15, 13]
        assert result.structured["scale"] == 1
        assert (await store.read_bytes(source.id)).content == content

    asyncio.run(run())
