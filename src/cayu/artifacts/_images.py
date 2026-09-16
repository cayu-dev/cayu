from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from threading import Lock
from typing import Any

# Bound each decoded raster independently from the raw attachment/source byte caps.
MAX_IMAGE_DECODED_BYTES = 64 * 1024 * 1024
# Bound the cumulative raster work for animated images as well as each frame.
MAX_IMAGE_TOTAL_DECODED_BYTES = 256 * 1024 * 1024
MAX_IMAGE_FRAMES = 1024
_CONSERVATIVE_IMAGE_BYTES_PER_PIXEL = 4
_IMAGE_DECODE_LOCK = Lock()


@dataclass(frozen=True)
class ImageDecodePolicy:
    """Host-owned raster work limits, independent of encoded attachment size.

    Pillow's own decompression-bomb protection remains enabled. A policy admits
    resource use; successful validation does not establish semantic inspection.
    """

    max_frame_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_frames: int = 1024

    def __post_init__(self) -> None:
        for name in ("max_frame_bytes", "max_total_bytes", "max_frames"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    def as_dict(self) -> dict[str, int]:
        return {
            "max_frame_bytes": self.max_frame_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_frames": self.max_frames,
        }


def effective_image_decode_policy(policy: ImageDecodePolicy | None = None) -> ImageDecodePolicy:
    return (
        policy
        if policy is not None
        else ImageDecodePolicy(
            max_frame_bytes=MAX_IMAGE_DECODED_BYTES,
            max_total_bytes=MAX_IMAGE_TOTAL_DECODED_BYTES,
            max_frames=MAX_IMAGE_FRAMES,
        )
    )


def decode_verified_image_format(
    image_module: Any, content: bytes, *, policy: ImageDecodePolicy | None = None
) -> str | None:
    """Return the Pillow format after bounded verification and a full raster decode."""
    effective_policy = effective_image_decode_policy(policy)
    # warnings.catch_warnings() mutates process-global filters on some supported
    # Python builds. Serialize this bounded worker-thread work so one validation
    # cannot restore the warning filters underneath another.
    with _IMAGE_DECODE_LOCK:
        return _decode_verified_image_format(image_module, content, effective_policy)


def _decode_verified_image_format(
    image_module: Any, content: bytes, policy: ImageDecodePolicy
) -> str | None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_module.DecompressionBombWarning)
        with image_module.open(BytesIO(content)) as image:
            detected_format = image.format
            _validate_decoded_image_size(image, policy)
            image.verify()
        with image_module.open(BytesIO(content)) as image:
            total_decoded_bytes = 0
            for frame_index in range(policy.max_frames):
                try:
                    image.seek(frame_index)
                except EOFError:
                    break
                total_decoded_bytes += _validate_decoded_image_size(image, policy)
                if total_decoded_bytes > policy.max_total_bytes:
                    raise ValueError(
                        "Image aggregate decoded size exceeds the safety limit: "
                        f"{total_decoded_bytes:,} > {policy.max_total_bytes:,} bytes."
                    )
                image.load()
            else:
                try:
                    image.seek(policy.max_frames)
                except EOFError:
                    pass
                else:
                    raise ValueError(
                        "Image frame count exceeds the safety limit: "
                        f"more than {policy.max_frames} frames."
                    )
    return detected_format


def _validate_decoded_image_size(image: Any, policy: ImageDecodePolicy) -> int:
    width, height = image.size
    decoded_bytes = width * height * _CONSERVATIVE_IMAGE_BYTES_PER_PIXEL
    if decoded_bytes > policy.max_frame_bytes:
        raise ValueError(
            "Image decoded size exceeds the safety limit: "
            f"{decoded_bytes} > {policy.max_frame_bytes} bytes."
        )
    return decoded_bytes
