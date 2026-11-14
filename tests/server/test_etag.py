"""`cayu.server.etag`: a read route's unchanged answer costs a 304 with no body."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel

from cayu.server.etag import if_none_match_matches, json_body_etag, tagged_json


class Run(BaseModel):
    id: str
    status: str


class Overview(BaseModel):
    as_of: str
    runs: list[Run]


def _client(state: dict[str, Any]) -> TestClient:
    def set_cookie(response: Response) -> None:
        response.set_cookie("seen", "yes")

    routes = APIRouter()

    @routes.get("/runs")
    async def runs(request: Request, response: Response) -> list[Run]:
        return tagged_json(request, response, state["runs"])

    @routes.get("/overview", dependencies=[Depends(set_cookie)])
    async def overview(request: Request, response: Response) -> Overview:
        view = Overview(as_of=str(state["as_of"]), runs=state["runs"])
        return tagged_json(
            request,
            response,
            view,
            tagged=view.model_dump(mode="json", exclude={"as_of"}),
            cache_control="private, no-store",
        )

    server = FastAPI()
    server.include_router(routes)
    return TestClient(server)


def test_an_unchanged_answer_is_a_304_and_a_changed_one_a_new_tag() -> None:
    state: dict[str, Any] = {"runs": [Run(id="march", status="running")], "as_of": 1}
    with _client(state) as client:
        first = client.get("/runs")
        assert first.status_code == 200
        assert first.json() == [{"id": "march", "status": "running"}]
        tag = first.headers["etag"]
        assert tag.startswith('W/"') and first.headers["cache-control"] == "private, no-cache"

        again = client.get("/runs", headers={"if-none-match": tag})
        assert again.status_code == 304 and again.content == b""
        assert again.headers["etag"] == tag
        assert again.headers["cache-control"] == "private, no-cache"

        state["runs"] = [Run(id="march", status="completed")]
        changed = client.get("/runs", headers={"if-none-match": tag})
    assert changed.status_code == 200 and changed.headers["etag"] != tag
    assert changed.json() == [{"id": "march", "status": "completed"}]


def test_a_field_left_out_of_the_tag_doesnt_change_it_and_the_304_keeps_route_headers() -> None:
    state: dict[str, Any] = {"runs": [], "as_of": 1}
    with _client(state) as client:
        first = client.get("/overview")
        tag = first.headers["etag"]
        state["as_of"] = 2
        again = client.get("/overview", headers={"if-none-match": tag})
    assert again.status_code == 304 and again.content == b""
    assert again.headers["cache-control"] == "private, no-store"
    # What the route's dependencies set on its response stays on the 304.
    assert "seen=yes" in again.headers["set-cookie"]


def test_the_tag_depends_on_the_answer_not_on_key_order_or_models() -> None:
    assert json_body_etag({"a": 1, "b": [1, 2]}) == json_body_etag({"b": [1, 2], "a": 1})
    assert json_body_etag({"a": 1}) != json_body_etag({"a": 2})
    assert json_body_etag([Run(id="r", status="done")]) == json_body_etag(
        [{"id": "r", "status": "done"}]
    )


def test_if_none_match_compares_weakly_and_takes_a_list_or_a_star() -> None:
    tag = 'W/"abc"'
    assert if_none_match_matches('W/"abc"', tag)
    assert if_none_match_matches('"abc"', tag)
    assert if_none_match_matches('"other", W/"abc"', tag)
    assert if_none_match_matches("*", tag)
    assert if_none_match_matches('"abc"', '"abc"')
    assert not if_none_match_matches('"other"', tag)
    assert not if_none_match_matches(None, tag)
    assert not if_none_match_matches("", tag)
