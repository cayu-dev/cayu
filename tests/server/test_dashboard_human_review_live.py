"""Real React review clicks against authenticated Runtime resolution routes."""

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient
from tests.core.test_human_review import ReviewPolicy, make_app, pause

from cayu import SQLiteSessionStore
from cayu.server import AuthContext, ServerConfig, create_server

pytestmark = pytest.mark.skipif(
    not os.environ.get("CAYU_BROWSER_DASHBOARD_URL"),
    reason="Requires a local dashboard Vite server.",
)

HTML = """<div id="root"></div><script type="module">
import RefreshRuntime from '/@react-refresh';
RefreshRuntime.injectIntoGlobalHook(window);
window.$RefreshReg$ = () => {}; window.$RefreshSig$ = () => type => type;
window.__vite_plugin_react_preamble_installed__ = true;
window.__CAYU_DASHBOARD_CONFIG__ = {apiBaseUrl:'/api'};
const React = (await import('/node_modules/.vite/deps/react.js')).default;
const {createRoot} = (await import('/node_modules/.vite/deps/react-dom_client.js')).default;
const {ProtectedHumanReview} = await import('/src/components/dashboard/protected-human-review.tsx');
const mutations = await import('/src/lib/mutation-browser.ts');
createRoot(document.getElementById('root')).render(React.createElement(ProtectedHumanReview, {
  sessionId:'review-session', purpose:'delivery', kind: KIND, disabled:false,
  unavailableReason:null, onDecision:async body => {
    const result = await ('input_id' in body ? mutations.executeResolveUserInputMutation(body)
      : mutations.executeResolveToolApprovalMutation(body));
    if (result.phase !== 'terminal') throw new Error('Decision was rejected or unconfirmed.');
  }
}));
</script>"""


@pytest.mark.parametrize("decision", ["answer", "approve", "deny"])
@pytest.mark.parametrize("fault", [None, "stale", "principal", "tenant"])
def test_protected_review_clicks(tmp_path, decision, fault):
    from playwright.async_api import async_playwright, expect

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ReviewPolicy()
        app, provider, tool = make_app(store, policy, approval=decision != "answer")
        await pause(app)
        provenance = {"subject": "operator", "tenant": "tenant-a"}

        def auth(request):
            return AuthContext(**provenance)

        bodies = []
        views = []
        statuses = []
        root = os.environ["CAYU_BROWSER_DASHBOARD_URL"].rstrip("/")
        with TestClient(create_server(app, config=ServerConfig.protected(auth))) as client:

            async def api(route):
                req = route.request
                path = req.url.split(root, 1)[1]
                body = req.post_data
                if path.endswith("/resolve"):
                    bodies.append(json.loads(body))
                response = client.request(
                    req.method,
                    path,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        **{
                            key: value
                            for key, value in req.headers.items()
                            if key.startswith("x-cayu-")
                        },
                    },
                )
                if "human-review?" in path and response.status_code == 200:
                    views.append(response.json())
                if path.endswith("/resolve"):
                    statuses.append(response.status_code)
                await route.fulfill(
                    status=response.status_code,
                    body=response.content,
                    headers={
                        "content-type": response.headers.get("content-type", "application/json")
                    },
                )

            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                try:
                    page = await browser.new_page()
                    await page.route(f"{root}/api/**", api)
                    await page.route(
                        f"{root}/review-test",
                        lambda route: route.fulfill(
                            content_type="text/html",
                            body=HTML.replace(
                                "KIND",
                                json.dumps(
                                    "user_input" if decision == "answer" else "tool_approval"
                                ),
                            ),
                        ),
                    )
                    await page.goto(f"{root}/review-test")
                    button = page.get_by_role(
                        "button",
                        name={"answer": "Submit Answer", "approve": "Approve", "deny": "Deny"}[
                            decision
                        ],
                        exact=True,
                    )
                    await expect(button).to_have_count(0)
                    await page.get_by_role("button", name="Refresh review").click()
                    await expect(button).to_be_enabled() if decision != "answer" else await expect(
                        page.get_by_label("Answer")
                    ).to_be_enabled()
                    view = views[-1]
                    pending_response = client.get("/api/pending-actions?session_id=review-session")
                    assert pending_response.status_code == 200
                    pending = pending_response.json()
                    assert len(pending["actions"]) == 1
                    # Protected interaction IDs must never be equated with projection aliases.
                    assert (
                        pending["actions"][0]["input_id" if decision == "answer" else "approval_id"]
                        != view["interaction_id"]
                    )
                    await expect(
                        page.get_by_text("Whole-round scope:", exact=False)
                    ).to_be_visible()
                    if decision == "answer":
                        await page.get_by_label("Answer", exact=True).fill("morning")
                    if fault == "stale":
                        policy.version = "delivery-v2"
                    elif fault == "principal":
                        provenance["subject"] = "another-operator"
                    elif fault == "tenant":
                        provenance["tenant"] = "tenant-b"
                    await button.click()
                    if fault:
                        await expect(page.get_by_role("alert")).to_contain_text(
                            "Refresh the review"
                        )
                        assert (
                            len(views) == 1
                        )  # Never inspect and resubmit changed content automatically.
                        assert len(provider.requests) == 1
                        await expect(button).to_have_count(0)
                    else:
                        await expect(button).to_have_count(0)
                        assert statuses == [200]
                        assert len(provider.requests) == 2
                    assert len(bodies) == 1
                    body = bodies[0]
                    assert body["review_reference"] == view["reference"]
                    assert (
                        body["input_id" if decision == "answer" else "approval_id"]
                        == view["interaction_id"]
                    )
                    if decision != "answer":
                        assert body["tool_round_id"] == view["tool_round_id"]
                        assert body["tool_call_id"] == view["tool_call_id"]
                    if decision != "answer":
                        assert tool.calls == (
                            [{"value": "morning"}] if not fault and decision == "approve" else []
                        )
                    if fault in {"principal", "tenant"}:
                        assert statuses == [403]
                finally:
                    await browser.close()
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["redacted", "unavailable", "missing", "denied"])
def test_withheld_and_missing_review_clicks(status):
    from playwright.async_api import async_playwright, expect

    async def scenario():
        root = os.environ["CAYU_BROWSER_DASHBOARD_URL"].rstrip("/")
        requests = []
        view = {
            "session_id": "review-session",
            "kind": "tool_approval",
            "interaction_id": "protected-id",
            "tool_round_id": "round",
            "tool_call_id": "call",
            "status": status if status in {"redacted", "unavailable"} else "unavailable",
            "guidance": "Review content is withheld; contact the application owner or deny the action.",
            "reference": None
            if status == "missing"
            else {
                "content_tag": "opaque",
                "policy_version": "v1",
                "context": {"recipient": "operator", "tenant": "tenant-a", "purpose": "delivery"},
            },
            "calls": [{"tool_call_id": "call", "tool_name": "side_effect", "on_grant": "withheld"}],
            "fields": [{"label": "Withheld", "text": "Do not display this field."}],
        }
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
            )
            try:
                page = await browser.new_page()
                await page.route(
                    f"{root}/api/**",
                    lambda route: route.fulfill(
                        status=403 if status == "denied" else 200, json=view
                    ),
                )
                html = HTML.replace("KIND", '"tool_approval"')
                start = html.index("    const result = await")
                end = html.index("\n  }", start)
                html = html[:start] + "    window.decisions.push(body);" + html[end:]
                html = html.replace("const mutations =", "window.decisions = []; const mutations =")
                await page.route(
                    f"{root}/review-test",
                    lambda route: route.fulfill(content_type="text/html", body=html),
                )
                await page.goto(f"{root}/review-test")
                await page.get_by_role("button", name="Refresh review").click()
                if status == "denied":
                    await expect(page.get_by_role("alert")).to_contain_text("denied")
                else:
                    await expect(
                        page.get_by_role("button", name="Approve", exact=True)
                    ).to_be_disabled()
                    deny = page.get_by_role("button", name="Deny", exact=True)
                    await expect(page.get_by_text("Do not display this field.")).to_have_count(0)
                    if status == "missing":
                        await expect(deny).to_be_disabled()
                    else:
                        await deny.click()
                        await expect(deny).to_have_count(0)
                        requests = await page.evaluate("window.decisions")
                        assert requests[0]["review_reference"] == view["reference"]
                        assert requests[0]["decision"] == "deny"
            finally:
                await browser.close()

    asyncio.run(scenario())
