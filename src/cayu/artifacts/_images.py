from __future__ import annotations

import warnings
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


def decode_verified_image_format(image_module: Any, content: bytes) -> str | None:
    """Return the Pillow format after bounded verification and a full raster decode."""
    # warnings.catch_warnings() mutates process-global filters on some supported
    # Python builds. Serialize this bounded worker-thread work so one validation
    # cannot restore the warning filters underneath another.
    with _IMAGE_DECODE_LOCK:
        return _decode_verified_image_format(image_module, content)


def _decode_verified_image_format(image_module: Any, content: bytes) -> str | None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_module.DecompressionBombWarning)
        with image_module.open(BytesIO(content)) as image:
            detected_format = image.format
            _validate_decoded_image_size(image)
            image.verify()
        with image_module.open(BytesIO(content)) as image:
            total_decoded_bytes = 0
            for frame_index in range(MAX_IMAGE_FRAMES):
                try:
                    image.seek(frame_index)
                except EOFError:
                    break
                total_decoded_bytes += _validate_decoded_image_size(image)
                if total_decoded_bytes > MAX_IMAGE_TOTAL_DECODED_BYTES:
                    raise ValueError(
                        "Image aggregate decoded size exceeds the safety limit: "
                        f"{total_decoded_bytes:,} > {MAX_IMAGE_TOTAL_DECODED_BYTES:,} bytes."
                    )
                image.load()
            else:
                try:
                    image.seek(MAX_IMAGE_FRAMES)
                except EOFError:
                    pass
                else:
                    raise ValueError(
                        "Image frame count exceeds the safety limit: "
                        f"more than {MAX_IMAGE_FRAMES} frames."
                    )
    return detected_format


def _validate_decoded_image_size(image: Any) -> int:
    width, height = image.size
    decoded_bytes = width * height * _CONSERVATIVE_IMAGE_BYTES_PER_PIXEL
    if decoded_bytes > MAX_IMAGE_DECODED_BYTES:
        raise ValueError(
            "Image decoded size exceeds the safety limit: "
            f"{decoded_bytes} > {MAX_IMAGE_DECODED_BYTES} bytes."
        )
    return decoded_bytes
