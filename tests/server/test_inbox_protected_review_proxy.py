"""A member decides an approval under protected review, signed in through the identity proxy.

What an Agent's own app does with the Inbox part and ProtectedReviewPart: read the purpose
from /api/app, inspect the review as the signed-in person, and settle with its reference.
"""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("jwt")

from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token
from tests.server.test_server_inbox import _app, _pause, _wait_for_status

from cayu.approvals import ToolArgumentReview
from cayu.runtime.public_authority import PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server


def _keyring() -> PublicAuthorityAliasKeyring:
    key = base64.urlsafe_b64encode(bytes([41]) * 32).decode().rstrip("=")
    return PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})


def test_a_member_settles_with_their_review_reference_and_no_one_else_can() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    alice = _as(_token(key, role="member", sub="alice"))
    bob = _as(_token(key, role="member", sub="bob"))
    # The review key comes from the app's alias keyring, as in a deployed Agent.
    app = _app(
        human_review_policy=ToolArgumentReview({"record": ("note",)}, purpose="inbox"),
        public_authority_alias_keyring=_keyring(),
    )
    server = create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    with TestClient(server) as client:
        info = client.get("/api/app", headers=alice).json()
        assert (info["role"], info["full_access"]) == ("member", False)
        assert info["human_review_required"] is True
        assert info["human_review_purpose"] == "inbox"

        _pause(client, app, "reviewed", "approver")
        [item] = client.get("/api/inbox", headers=alice).json()["items"]
        settle = f"/api/inbox/{item['id']}/settle"

        unreviewed = client.post(settle, json={"decision": "approve"}, headers=alice)
        assert unreviewed.status_code == 403, unreviewed.text

        review = client.get(
            "/api/sessions/reviewed/human-review",
            params={"purpose": info["human_review_purpose"]},
            headers=alice,
        )
        assert review.status_code == 200, review.text
        view = review.json()
        assert view["status"] == "permitted"
        assert [field["text"] for field in view["fields"]] == ["proposed"]
        reference = view["reference"]

        stolen = client.post(
            settle, json={"decision": "approve", "review_reference": reference}, headers=bob
        )
        assert stolen.status_code == 403, stolen.text

        approved = client.post(
            settle, json={"decision": "approve", "review_reference": reference}, headers=alice
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["item"]["settlement"]["decided_by"]["subject"] == "alice"
        _wait_for_status(client, app, "reviewed")
