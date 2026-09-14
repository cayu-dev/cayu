"""Optional authenticated server over the same self-hosted attention fixture."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from examples.human_attention.app import build

from cayu.server import BasicAuth, ServerConfig, create_server


def create_demo_server():
    app, store, _inbox = build(Path(os.environ["ATTENTION_STATE"]), phase="serve")

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield
        finally:
            await store.close()

    return create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(
                username=os.environ["ATTENTION_SERVER_USERNAME"],
                password=os.environ["ATTENTION_SERVER_PASSWORD"],
            ),
        ),
        fastapi_options={"lifespan": lifespan},
    )
