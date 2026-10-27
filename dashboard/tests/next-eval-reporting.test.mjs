import assert from "node:assert/strict"
import test from "node:test"
import * as model from "../src/next/app/eval-view-model.ts"

test("reported origin keeps authenticated publisher distinct from submitted branch and tested commit", () => {
  const run = {
    source: "reported",
    publisher_id: "ci-publisher",
    origin: {
      kind: "ci",
      branch: "feature",
      commit: "a".repeat(40),
      pr_head_commit: "b".repeat(40),
      pull_request: 42,
    },
  }
  assert.equal(model.isReportedRun(run), true)
  assert.equal(model.runOrigin(run), "Reported CI · ci-publisher · feature · aaaaaaa · PR #42")
  for (const value of [null, {}, { source: "future" }, { source: true }])
    assert.equal(model.isReportedRun(value), false)
  assert.equal(model.runOrigin({ origin: "sdk", actor: "operator" }), "SDK · operator")
})

test("reported detail dispatch requires positive server source and never decodes selectors", () => {
  const reported = { run: { source: "reported" }, results: { items: [] } }
  assert.equal(model.isReportedRunDetail(reported), true)
  assert.equal(model.isReportedRunDetail(undefined), false)
  assert.equal(model.isReportedRunDetail({ run: { native_run_id: null } }), false)
  assert.equal(model.isReportedSuiteDetail({ suite: { source: "reported" } }), true)
  assert.equal(model.isReportedSuiteDetail({ suite: { source: "corpus" } }), false)
})

test("report download uses the authenticated API owner with the exact opaque run selector", async () => {
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
  const { downloadEvalReport } = await import("../src/lib/api.ts")
  const original = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init })
    return new Response('{"report_id":"r"}', {
      headers: { "content-disposition": 'attachment; filename="eval-report.json"' },
    })
  }
  const controller = new AbortController()
  try {
    const file = await downloadEvalReport("opaque/report+key", controller.signal)
    assert.equal(file.filename, "eval-report.json")
    assert.equal(await file.blob.text(), '{"report_id":"r"}')
    assert.equal(calls[0].url, "/api/evals/reports/opaque%2Freport%2Bkey/download")
    assert.equal(calls[0].init.signal, controller.signal)
  } finally {
    globalThis.fetch = original
  }
})
