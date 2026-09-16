"""Original-resolution image regions with source-bound coordinate provenance."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from io import BytesIO

from cayu.artifacts.attachments import FileAttachmentKind, file_attachment
from cayu.artifacts.base import ArtifactScope
from cayu.tools.base import ToolResult


def image_information(content):
    from PIL import Image

    with Image.open(BytesIO(content)) as image:
        return {"width": image.width, "height": image.height}, getattr(image, "n_frames", 1)


def _crop(content, box):
    from PIL import Image

    with Image.open(BytesIO(content)) as source:
        width, height = source.size
        if box[2] > width or box[3] > height:
            raise ValueError(f"image_region exceeds source dimensions {width} x {height}.")
        source.seek(0)
        with source.crop(box) as region:
            # Coordinates describe encoded source pixels. Do not carry an EXIF
            # orientation that would rotate the derived crop a second time.
            region.info.pop("exif", None)
            output = BytesIO()
            if region.mode == "CMYK":
                # JPEG supports CMYK, but PNG requires an RGB conversion.
                with region.convert("RGB") as rgb_region:
                    rgb_region.save(output, format="PNG")
            else:
                region.save(output, format="PNG")
            return output.getvalue(), width, height, getattr(source, "n_frames", 1)


async def read_image_region(request, policy):
    from cayu.tools.files import (
        MAX_IMAGE_SOURCE_BYTES,
        _detect_image_content_type,
        _read_artifact_store,
    )

    source = await _read_artifact_store(
        request.artifact_store,
        request.artifact.id,
        max_bytes=MAX_IMAGE_SOURCE_BYTES,
        missing_as_result=request._missing_as_result,
    )
    structured = {
        **request.structured,
        "image_decode_limits": policy.as_dict() if policy else None,
        "source_artifact_id": request.artifact.id,
        "semantic_inspection": "not_established_by_tool",
        "region_delivered": False,
        "whole_source_delivered": False,
        "uninspected_scope": "No image pixels delivered.",
    }
    if source.truncated:
        return ToolResult(
            content="Image source exceeds the native encoded-source limit; no region was decoded.",
            structured={
                **structured,
                "error": "image_source_too_large",
                "max_source_bytes": MAX_IMAGE_SOURCE_BYTES,
            },
            is_error=True,
        )
    structured["source_sha256"] = sha256(source.content).hexdigest()
    content_type, error = await asyncio.to_thread(
        _detect_image_content_type, source.content, policy
    )
    if error or content_type != request.artifact.content_type:
        return ToolResult(
            content=(
                f"Image region could not be inspected: {error or 'content type mismatch'}. "
                "A crop still requires decoding the source. The host can configure "
                "ToolExecutionConfig.image_max_frame_bytes, image_max_total_bytes and "
                "image_max_frames; changing attachment bytes does not change that policy."
            ),
            structured={**structured, "error": "image_source_validation_failed"},
            is_error=True,
        )
    box = request.options.image_region
    try:
        content, width, height, frames = await asyncio.to_thread(_crop, source.content, box)
    except Exception as exc:
        return ToolResult(
            content=f"Image region could not be inspected: {exc}",
            structured={**structured, "error": "image_region_failed"},
            is_error=True,
        )
    structured.update(
        {
            "source_dimensions": {"width": width, "height": height},
            "source_frame_count": frames,
            "frame_index": 0,
            "source_region": list(box),
            "coordinate_system": "original pixels, top-left origin, right/bottom exclusive",
            "scale": 1,
        }
    )
    if len(content) > request.options.max_attachment_bytes:
        return ToolResult(
            content="Region exceeds attachment bytes. Request a smaller image_region; no downsampling was applied.",
            structured={
                **structured,
                "error": "image_region_attachment_too_large",
                "region_bytes": len(content),
                "max_attachment_bytes": request.options.max_attachment_bytes,
            },
            is_error=True,
        )
    digest = sha256(content).hexdigest()
    provenance = {
        key: structured[key]
        for key in (
            "source_artifact_id",
            "source_sha256",
            "source_dimensions",
            "source_region",
            "frame_index",
            "coordinate_system",
            "scale",
        )
    }
    artifact = await request.artifact_store.put_bytes(
        content,
        filename="image-region.png",
        content_type="image/png",
        scope=ArtifactScope.SESSION,
        session_id=request.ctx.session_id,
        agent_name=request.ctx.agent_name,
        environment_name=request.ctx.environment_name,
        metadata={**provenance, "operation": "image_region", "content_hash": digest},
    )
    attachment = file_attachment(
        artifact_id=artifact.id,
        kind=FileAttachmentKind.IMAGE,
        filename=artifact.filename,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        metadata=provenance,
    )
    return ToolResult(
        content=(
            f"Attached original-resolution region {list(box)} of {width} x {height}, frame 0. "
            f"Source artifact: {request.artifact.id}. Repeat read_file on that artifact "
            "with another image_region to inspect other pixels. Only this region of frame 0 is attached. "
            f"Source SHA-256: {structured['source_sha256']}. Region SHA-256: {digest}."
        ),
        structured={
            **structured,
            "region_delivered": True,
            "whole_source_delivered": box == (0, 0, width, height) and frames == 1,
            "uninspected_scope": "Pixels outside source_region and all other frames are not delivered.",
            "attachment_artifact_id": artifact.id,
            "attachment_sha256": digest,
            "attachment_bytes": artifact.size_bytes,
        },
        artifacts=[attachment],
    )
