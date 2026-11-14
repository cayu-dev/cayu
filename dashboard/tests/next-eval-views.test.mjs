import assert from "node:assert/strict"
import test from "node:test"
import { passing, passRate, percent } from "../src/next/app/eval-model.ts"

test("flaky variability overlaps settled outcomes and cannot inflate passing totals", () => {
  const counts = { total: 4, passed: 2, failed: 1, error: 1, unavailable: 0, flaky: 3 }
  assert.equal(passing(counts), 2)
  assert.equal(passRate(counts), "2/4")
  assert.equal(percent(counts), 50)
})

const model = await import("../src/next/app/eval-view-model.ts")
test("selector identity includes target, source and exact revision without treating a run id as authority", () => {
  assert.deepEqual(JSON.parse(atob(model.nativeRunSelector("run-1"))), ["run", "run-1"])
  const suite = { source: "corpus", target_key: "main", revision: "sha256:123", suite_id: "suite" }
  assert.deepEqual(JSON.parse(atob(model.suiteSelector(suite))), ["suite", suite])
  assert.notEqual(
    model.suiteSelector(suite),
    model.suiteSelector({ ...suite, target_key: "other" }),
  )
})
test("unavailable comparisons and missing actors never fabricate a verdict or attribution", () => {
  assert.equal(model.comparisonLabel({ state: "incompatible", verdict: null }), "Incompatible runs")
  assert.equal(model.runOrigin({ origin: "sdk", actor: null }), "SDK")
  assert.equal(
    model.checkLabel({ kind: "final_output_contains", label: "Output includes: reconciled" }),
    "Output includes: reconciled",
  )
})
test("route filters accept only the server's closed predicate names", () => {
  for (const filter of ["regressed", "improved", "failing", "flaky"])
    assert.equal(model.evalCaseFilter(filter), filter)
  for (const unknown of [undefined, null, "all", "future", 1, true, {}])
    assert.equal(model.evalCaseFilter(unknown), undefined)
})

test("suite-view read permission is positive and does not grant writes or old detail reads", async () => {
  const { applyViewerAccess } = await import("../src/lib/dashboard-access.ts")
  const member = { name: "Agent", runtime_version: "0.13", role: "member", full_access: false }
  const op = { enabled: true }
  assert.equal(
    applyViewerAccess(op, { kind: "surface", surface: "eval_suite_views" }, member).enabled,
    true,
  )
  for (const identity of [null, undefined, {}, { ...member, full_access: "true" }]) {
    assert.equal(
      applyViewerAccess(op, { kind: "surface", surface: "eval_suite_views" }, identity).enabled,
      false,
    )
  }
  for (const requirement of [
    { kind: "surface", surface: "evals" },
    { kind: "surface", surface: "eval_suite_views", operation: "mutate" },
  ]) {
    assert.equal(applyViewerAccess(op, requirement, member).enabled, false)
  }
})

test("view HTTP entrances preserve opaque selectors, independent cursors and real read cancellation", async () => {
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
  const api = await import("../src/next/api/eval-views.ts")
  const originalFetch = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init })
    return new Response(
      JSON.stringify({ items: [], next_cursor: null, has_more: false, total: null }),
    )
  }
  const controller = new AbortController()
  try {
    await api.fetchSuiteViews({ target_key: "a/b", cursor: "next+page" }, controller.signal)
    await api.fetchSuiteView(
      "opaque/id",
      { cases_cursor: "cases", runs_cursor: "runs" },
      controller.signal,
    )
    await api.fetchRunView(
      "opaque/run",
      { compared_with: "other+id", cases_cursor: "cases", case_filter: "regressed" },
      controller.signal,
    )
    assert.equal(
      calls[0].url,
      "/api/evals/suite-views?target_key=a%2Fb&cursor=next%2Bpage&limit=25&max_result_bytes=8388608",
    )
    assert.equal(
      calls[1].url,
      "/api/evals/suite-views/opaque%2Fid?cases_cursor=cases&runs_cursor=runs&cases_limit=25&runs_limit=20&max_result_bytes=8388608",
    )
    assert.equal(
      calls[2].url,
      "/api/evals/run-views/opaque%2Frun?compared_with=other%2Bid&cases_cursor=cases&case_filter=regressed&cases_limit=25&max_result_bytes=8388608",
    )
    for (const call of calls) {
      assert.equal(call.init.signal, controller.signal)
      assert.equal(call.init.credentials, "same-origin")
    }
    globalThis.fetch = (_url, init) =>
      new Promise((_resolve, reject) =>
        init.signal.addEventListener("abort", () => reject(init.signal.reason), { once: true }),
      )
    const pending = api.fetchRunView("run", {}, controller.signal)
    controller.abort(new DOMException("Navigation", "AbortError"))
    await assert.rejects(pending, { name: "AbortError" })
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("an oversized view is requested once and names the CLI command that reads it", async () => {
  const api = await import("../src/next/api/eval-views.ts")
  const originalFetch = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url) => {
    calls.push(new URL(url, "http://local").searchParams)
    return Response.json({ detail: "Too large" }, { status: 413 })
  }
  try {
    for (const [request, command] of [
      [() => api.fetchSuiteViews({ target_key: "target" }), "cayu eval suites list"],
      [() => api.fetchSuiteView("suite", { runs_cursor: "runs" }), "cayu eval suites show SUITE"],
      [() => api.fetchRunView("run", { case_filter: "flaky" }), "cayu eval runs show RUN_ID"],
    ]) {
      calls.length = 0
      await assert.rejects(request(), (error) => {
        assert.equal(error.status, 413)
        assert.match(error.message, /too large to show here/)
        assert.ok(error.message.includes(command))
        return true
      })
      assert.equal(calls.length, 1)
      assert.equal(calls[0].get("max_result_bytes"), "8388608")
    }
    calls.length = 0
    globalThis.fetch = async () => {
      calls.push(1)
      return new Response("{}", { status: 422 })
    }
    await assert.rejects(api.fetchRunView("run", {}), { status: 422 })
    assert.equal(calls.length, 1)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("a live run page polls one small status row", async () => {
  const api = await import("../src/next/api/eval-views.ts")
  const originalFetch = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url) => {
    calls.push(url)
    return Response.json({ run_id: "live", status: "running", version: "sha256:1" })
  }
  try {
    const status = await api.fetchRunViewStatus("opaque/run")
    assert.equal(status.version, "sha256:1")
    assert.equal(calls.length, 1)
    assert.match(calls[0], /\/evals\/run-views\/opaque%2Frun\/status$/)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("specialist labels and baseline attribution keep opaque identities out of display", () => {
  assert.equal(
    model.evalTargetLabel({ target_key: "eval.opaque", label: "Reporter", agent_name: "reporter" }),
    "Reporter",
  )
  assert.equal(
    model.evalTargetLabel({ target_key: "eval.opaque", label: " ", agent_name: "reporter" }),
    "reporter",
  )
  assert.equal(
    model.evalTargetLabel({ target_key: "eval.opaque", label: null, agent_name: null }),
    "Specialist unavailable",
  )
  assert.equal(model.baselineApprovalLabel("cayu:trusted-local-development"), "Approved locally")
  assert.equal(model.baselineApprovalLabel("Ada"), "Approved by Ada")
  assert.equal(model.baselineApprovalLabel("cayu:other-actor"), "Approved by cayu:other-actor")
})

test("target-less corpus links resolve exact identity independently of catalog presentation", async () => {
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/nested/api" } }
  const api = await import("../src/next/api/eval-views.ts")
  const originalFetch = globalThis.fetch
  const identity = {
    source: "corpus",
    target_key: "writer",
    revision: "sha256:abc",
    suite_id: "one",
  }
  const calls = []
  globalThis.fetch = async (url) => {
    calls.push(url)
    return new Response(
      JSON.stringify(url.includes("resolve-corpus") ? identity : { items: [], has_more: false }),
    )
  }
  try {
    assert.equal(
      await api.resolveLegacyEvalSuite(identity.revision, "one", undefined),
      model.suiteSelector(identity),
    )
    assert.match(calls[0], /resolve-corpus\?corpus=sha256%3Aabc&suite=one/)
    assert.equal(calls.length, 1)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("queued-message setup documents the required authentication dependency", async () => {
  const { readFile } = await import("node:fs/promises")
  const text = await readFile(
    new URL("../src/next/routes/session/session-page.tsx", import.meta.url),
    "utf8",
  )
  assert.match(text, /cayu serve --auth module:auth_dependency/)
  assert.match(text, /session_message_access_policy/)
})
