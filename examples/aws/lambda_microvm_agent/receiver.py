"""Trusted private ECS receiver used by the integrated egress example."""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Cayu private action receiver", docs_url=None, redoc_url=None)


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=128)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/actions", status_code=202)
async def create_action(
    request: ActionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    expected = os.environ.get("INTERNAL_SERVICE_TOKEN")
    presented = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not hmac.compare_digest(expected, presented):
        raise HTTPException(status_code=401, detail="invalid service credential")
    return {"accepted": True, "action": request.action}
