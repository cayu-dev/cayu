"""Global public artifact search, ordering and producing-session display."""

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app

from cayu import RunRequest
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionIdentity


@pytest.fixture
def inventory(tmp_path):
    app = _app()
    stores = [LocalArtifactStore(tmp_path / name, store_id=name) for name in ("one", "two")]
    for store in stores:
        app.register_environment(Environment(EnvironmentSpec(name=store.id), artifact_store=store))

    async def seed():
        await app.session_store.create(
            RunRequest(
                session_id="producer", agent_name="writer", title="Quarterly report", messages=[]
            ),
            identity=SessionIdentity(provider_name="configured", model="configured"),
        )
        saved = []
        for index in range(32):
            store = stores[index % 2]
            saved.append(
                await store.put_bytes(
                    b"synthetic",
                    filename=f"file-{index:02}.txt",
                    session_id="producer",
                    agent_name="writer",
                    environment_name=store.id,
                )
            )
        return saved

    saved = asyncio.run(seed())
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        yield client, app, stores, saved


def test_search_matches_beyond_the_loaded_page_and_filters_before_paging(inventory):
    client, _, _, saved = inventory
    response = client.get("/api/artifacts", params={"q": "FiLe-00", "limit": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["id"] for row in body["artifacts"]] == [saved[0].id]
    assert body["total_count"] == 1
    assert body["next_offset"] is None


def test_artifact_sort_is_global_before_offset(inventory):
    client, _, _, _ = inventory
    response = client.get(
        "/api/artifacts", params={"order_by": "filename_asc", "offset": 2, "limit": 2}
    )
    assert response.status_code == 200, response.text
    assert [row["filename"] for row in response.json()["artifacts"]] == [
        "file-02.txt",
        "file-03.txt",
    ]


def test_artifact_rows_include_producing_title_and_search_it(inventory):
    client, _, _, _ = inventory
    response = client.get("/api/artifacts", params={"q": "quarterly", "limit": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] == 32
    assert all(row["session_title"] == "Quarterly report" for row in body["artifacts"])
    assert client.get("/api/artifacts", params={"q": "no-such-title"}).json()["total_count"] == 0


@pytest.mark.parametrize(
    "order",
    [
        "created_at_asc",
        "created_at_desc",
        "filename_asc",
        "filename_desc",
        "size_bytes_asc",
        "size_bytes_desc",
    ],
)
def test_orders_are_complete_stable_across_pages(inventory, order):
    client, _, _, _ = inventory
    complete = client.get("/api/artifacts", params={"order_by": order, "limit": 100}).json()
    pages = [
        client.get("/api/artifacts", params={"order_by": order, "limit": 7, "offset": start}).json()
        for start in range(0, 32, 7)
    ]
    assert [row for page in pages for row in page["artifacts"]] == complete["artifacts"]
    assert all(page["total_count"] == 32 for page in pages)
    rows = complete["artifacts"]

    def key(row):
        primary = (
            row["filename"].casefold()
            if order.startswith("filename")
            else row["size_bytes"]
            if order.startswith("size_bytes")
            else row["created_at"]
        )
        return primary, row["artifact_store_id"], row["id"]

    assert rows == sorted(rows, key=key, reverse=order.endswith("desc"))


def test_filters_and_query_validation_precede_inventory_reads(inventory, monkeypatch):
    client, _, stores, _ = inventory
    result = client.get(
        "/api/artifacts",
        params={
            "q": "quarterly",
            "environment_name": "two",
            "artifact_store_id": "two",
            "order_by": "filename_asc",
            "limit": 1,
        },
    ).json()
    assert result["total_count"] == 16
    assert result["artifacts"][0]["filename"] == "file-01.txt"

    async def forbidden(*args, **kwargs):
        raise AssertionError("invalid query must not dispatch storage")

    for store in stores:
        monkeypatch.setattr(store, "scan_metadata", forbidden)
    for params in ({"q": "x" * 257}, {"order_by": "anything"}, {"offset": 10001}, {"limit": 501}):
        assert client.get("/api/artifacts", params=params).status_code == 422
    assert (
        client.get("/api/artifacts", params={"artifact_store_id": "missing", "q": "x"}).status_code
        == 404
    )


def test_mixed_store_search_is_explicit_and_preflighted(inventory, monkeypatch):
    client, _, stores, _ = inventory
    monkeypatch.setattr(stores[1], "supports_metadata_scan", False)
    capability = client.get("/api/contract").json()["capabilities"]["surfaces"]["artifact_search"]
    assert capability["read"]["enabled"] is False
    original = stores[0].scan_metadata

    async def forbidden(*args, **kwargs):
        raise AssertionError("all selected stores must be checked before any dispatch")

    monkeypatch.setattr(stores[0], "scan_metadata", forbidden)
    assert client.get("/api/artifacts", params={"q": "file"}).status_code == 501
    monkeypatch.setattr(stores[0], "scan_metadata", original)
    assert (
        client.get("/api/artifacts", params={"artifact_store_id": "one", "q": "file"}).json()[
            "total_count"
        ]
        == 16
    )
    # The documented basic list remains available with an opaque extension.
    assert client.get("/api/artifacts").json()["total_count"] == 32
    monkeypatch.setattr(stores[0], "supports_metadata_scan", False)
    assert (
        client.get("/api/contract").json()["capabilities"]["surfaces"]["artifact_search"]["read"][
            "enabled"
        ]
        is False
    )


@pytest.mark.parametrize("failure", ["partial", "receipt", "duplicate", "bound", "title"])
def test_incomplete_inventory_never_publishes_partial_success(inventory, monkeypatch, failure):
    from cayu.server import _artifact_query

    client, app, stores, saved = inventory
    if failure == "bound":
        monkeypatch.setattr(_artifact_query, "ARTIFACT_QUERY_SCAN_MAX", 20)
    elif failure == "title":

        async def broken(query):
            raise OSError("private diagnostic must not escape")

        monkeypatch.setattr(app.session_store, "read_session_titles", broken)
    else:

        async def broken(consume, **kwargs):
            await consume((saved[1],))
            if failure == "partial":
                raise OSError("private diagnostic must not escape")
            if failure == "duplicate":
                await consume((saved[1],))
            return True  # bool must not satisfy an integer completion receipt

        monkeypatch.setattr(stores[1], "scan_metadata", broken)
    response = client.get("/api/artifacts", params={"q": "file", "limit": 1})
    assert response.status_code in (500, 503), response.text
    assert "artifacts" not in response.json()
    assert "private diagnostic" not in response.text


def test_titles_are_batched_and_secret_text_does_not_affect_matches(inventory, monkeypatch):
    from cayu.vaults.redaction import SecretRedactor

    client, app, _, _ = inventory
    original = app.session_store.read_session_titles
    calls = []

    async def track(query):
        calls.append(query.session_ids)
        return await original(query)

    monkeypatch.setattr(app.session_store, "read_session_titles", track)
    # Redaction is applied before both display and search, not after selection.
    monkeypatch.setattr(app, "_secret_redactor", SecretRedactor(["Quarterly"]))
    response = client.get("/api/artifacts", params={"q": "Quarterly"})
    assert response.status_code == 200
    assert response.json()["total_count"] == 0
    assert calls == [("producer",)]
    shown = client.get("/api/artifacts", params={"q": "report"}).json()
    assert shown["total_count"] == 32
    assert "Quarterly" not in str(shown)


@pytest.mark.parametrize("malformed", ["title", "id", "missing"])
def test_malformed_title_projection_is_not_rendered_or_logged(
    inventory, monkeypatch, caplog, capsys, malformed
):
    from cayu.sessions.display import SessionTitleRow, SessionTitleSnapshot

    client, app, _, _ = inventory
    canary = "PRIVATE-TITLE-CANARY"

    class Unsafe:
        def __str__(self):
            return canary

        def __repr__(self):
            return canary

    async def invalid_titles(query):
        row = SessionTitleRow(session_id=query.session_ids[0], title="valid")
        if malformed == "title":
            row.title = Unsafe()
        elif malformed == "id":
            row.session_id = "different-producer"
        return SessionTitleSnapshot.model_construct(
            sessions=() if malformed == "missing" else (row,)
        )

    monkeypatch.setattr(app.session_store, "read_session_titles", invalid_titles)
    response = client.get("/api/artifacts", params={"q": "title"})
    assert response.status_code == 500
    assert canary not in response.text + caplog.text + str(capsys.readouterr())


def test_opaque_default_listing_retains_unknown_count(inventory, monkeypatch):
    from cayu.artifacts.base import ArtifactListResult

    client, _, stores, _ = inventory
    original = stores[1].list

    async def opaque(**kwargs):
        page = await original(**kwargs)
        return ArtifactListResult(artifacts=page.artifacts, total_count=None, truncated=True)

    monkeypatch.setattr(stores[1], "supports_metadata_scan", False)
    monkeypatch.setattr(stores[1], "list", opaque)
    response = client.get("/api/artifacts", params={"limit": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] is None and body["truncated"] is True
    assert len(body["artifacts"]) == 2
    assert body["next_offset"] == 2


def test_file_bytes_and_custom_metadata_are_not_search_fields(inventory):
    client, _, stores, _ = inventory
    asyncio.run(
        stores[0].put_bytes(
            b"content-needle",
            filename="ordinary.txt",
            session_id="producer",
            metadata={"note": "metadata-needle"},
        )
    )
    for query in ("content-needle", "metadata-needle"):
        result = client.get("/api/artifacts", params={"q": query})
        assert result.status_code == 200
        assert result.json()["total_count"] == 0


def test_basic_listing_does_not_scan_the_global_inventory(inventory, monkeypatch):
    from cayu.server import _artifact_query

    client, _, stores, _ = inventory
    monkeypatch.setattr(_artifact_query, "ARTIFACT_QUERY_SCAN_MAX", 1)

    async def forbidden(*args, **kwargs):
        raise AssertionError("basic listing must not scan the global inventory")

    for store in stores:
        monkeypatch.setattr(store, "scan_metadata", forbidden)
    response = client.get("/api/artifacts", params={"limit": 2, "environment_name": "two"})
    assert response.status_code == 200
    assert len(response.json()["artifacts"]) == 2
    assert response.json()["total_count"] == 16
