from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from cayu.evals import BrowserAcceptanceFixtureV1
from cayu.evals import browser_acceptance_fixture as fixture_module


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        del req, fp, code, msg, headers, newurl
        return None


def _fetch(fixture: BrowserAcceptanceFixtureV1, path: str):
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(f"{fixture.upstream_origin}{path}", timeout=2)
    except urllib.error.HTTPError as response:
        return response


def test_browser_acceptance_fixture_routes_both_logical_hosts_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fixture_module, "_fixture_address", lambda: "127.0.0.1")
    fixture = BrowserAcceptanceFixtureV1()

    with fixture:
        docs = _fetch(fixture, "/basic")
        static = _fetch(fixture, "/frame-controls")
        form = _fetch(fixture, "/forms")
        history_start = _fetch(fixture, "/history-start")
        history_next = _fetch(fixture, "/history-next")
        hover = _fetch(fixture, "/hover")
        reload_page = _fetch(fixture, "/reload")
        scroll = _fetch(fixture, "/scroll")
        upload = _fetch(fixture, "/upload")
        detached = _fetch(fixture, "/detached")
        challenge = _fetch(fixture, "/challenge")
        popup = _fetch(fixture, "/popup")
        effect = _fetch(fixture, "/effect/form-saved")

        assert fixture.upstream_origin.startswith("http://127.0.0.1:")
        assert fixture.upstream_routes == {
            "docs.browser.test": fixture.upstream_origin,
            "static.browser.test": fixture.upstream_origin,
        }
        assert docs.status == 200
        assert b"Browser acceptance fixture" in docs.read()
        assert static.status == 200
        assert b"Frame value" in static.read()
        assert b"required" in form.read()
        assert b"Back destination" in history_start.read()
        assert b"Forward destination" in history_next.read()
        assert b"Hover target" in hover.read()
        assert b"Reload confirmed" in reload_page.read()
        assert b"after-scroll" in scroll.read()
        assert b'type="file"' in upload.read()
        assert b"getElementById('target').remove()" in detached.read()
        assert challenge.headers["x-cayu-access-block"] == "bot_challenge"
        assert b"window.open('https://docs.browser.test/basic')" in popup.read()
        assert effect.status == 204
        assert fixture.request_counts() == {
            "/basic": 1,
            "/challenge": 1,
            "/detached": 1,
            "/effect/form-saved": 1,
            "/forms": 1,
            "/frame-controls": 1,
            "/popup": 1,
            "/history-next": 1,
            "/history-start": 1,
            "/hover": 1,
            "/reload": 1,
            "/scroll": 1,
            "/upload": 1,
        }

    with pytest.raises(RuntimeError, match="not running"):
        _ = fixture.upstream_origin


def test_browser_acceptance_fixture_preserves_redirect_and_bounded_artifact_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fixture_module, "_fixture_address", lambda: "127.0.0.1")
    with BrowserAcceptanceFixtureV1() as fixture:
        redirect = _fetch(fixture, "/redirect")
        denied = _fetch(fixture, "/redirect-denied")
        artifact = _fetch(fixture, "/download/report.txt")

    assert redirect.status == 302
    assert redirect.headers["location"] == "/basic"
    assert denied.status == 302
    assert denied.headers["location"] == "https://blocked.browser.test/private"
    assert artifact.status == 200
    assert artifact.headers["content-disposition"] == 'attachment; filename="report.txt"'
    assert artifact.read() == b"bounded browser acceptance download\n"
