from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.approvals import ToolArgumentReview
from cayu.server import OpenAccess, mount_cayu

KEY = b"test-fixture-only-binding-key-32-bytes"


def test_the_server_points_the_dashboard_at_the_policys_purpose() -> None:
    def injected(server) -> dict:
        page = TestClient(server).get("/cayu/").text
        marker = "window.__CAYU_DASHBOARD_CONFIG__="
        return json.loads(page.split(marker, 1)[1].split(";</script>", 1)[0])

    policy = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY, purpose="filing")
    server = FastAPI()
    mount_cayu(server, CayuApp(human_review_policy=policy), access=OpenAccess())
    assert injected(server)["humanReviewPurpose"] == "filing"

    configured = FastAPI()
    mount_cayu(
        configured,
        CayuApp(human_review_policy=policy),
        access=OpenAccess(),
        dashboard_config={"humanReviewPurpose": "inbox"},
    )
    assert injected(configured)["humanReviewPurpose"] == "inbox"

    plain = FastAPI()
    mount_cayu(plain, CayuApp(), access=OpenAccess())
    assert "humanReviewPurpose" not in injected(plain)
