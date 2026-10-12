"""Attachment support."""

from __future__ import annotations

from cayu.applications import CayuApp
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec


def _app_with_artifact_store(tmp_path, **app_kwargs) -> tuple[CayuApp, LocalArtifactStore]:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    app = CayuApp(enable_logging=False, **app_kwargs)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    return app, store


def _valid_png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()
