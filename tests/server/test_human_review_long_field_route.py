"""The protected route that reads one long review field whole."""

from __future__ import annotations

import asyncio
import html

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.core.test_human_review import pause
from tests.core.test_human_review_long_fields import BODY, KEY, _app

from cayu.approvals import ToolArgumentReview
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.storage import SQLiteSessionStore

ROUTE = "/api/sessions/review-session/human-review/long-field"


def test_an_approver_reads_a_long_field_through_the_protected_route(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    policy = ToolArgumentReview(
        {"side_effect": ("value",)}, binding_key=KEY, attest_listed_arguments=True
    )
    app, tool = _app(store, policy, BODY)
    asyncio.run(pause(app))

    def auth(request):
        return AuthContext(subject=request.headers.get("x-subject", "operator"))

    with TestClient(create_server(app, config=ServerConfig.protected(auth))) as client:
        view = client.get(
            "/api/sessions/review-session/human-review", params={"purpose": "approvals"}
        ).json()
        assert view["status"] == "permitted"
        [long_field] = view["long_fields"]
        body = {"review_reference": view["reference"], "key": long_field["key"]}

        response = client.post(ROUTE, json=body)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        field = response.json()
        assert field["status"] == "permitted"
        assert html.unescape(field["text"]) == BODY
        assert field["characters"] == len(BODY)

        # The reference must name the caller.
        assert client.post(ROUTE, json=body, headers={"x-subject": "guest"}).status_code == 403
        # A reference to content that is no longer current must be inspected again.
        stale = {**body, "review_reference": {**view["reference"], "content_tag": "0" * 64}}
        assert client.post(ROUTE, json=stale).status_code == 409
        # A key the view doesn't list opens nothing.
        unlisted = client.post(ROUTE, json={**body, "key": "1.2"})
        assert unlisted.status_code == 200
        assert unlisted.json()["status"] == "unavailable"
        assert unlisted.json()["text"] is None
    assert tool.calls == []
    asyncio.run(store.close())
