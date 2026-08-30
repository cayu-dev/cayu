import assert from "node:assert/strict"
import test from "node:test"

globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }

const {
  buildMemoryExperimentReport,
  compareEvalResults,
  compareEvalRuns,
  createEvalRun,
  downloadCatalogEvalResultHtml,
  downloadCatalogEvalResultJson,
  downloadEvalResultJson,
  downloadEvalScenario,
  downloadMemoryExperimentReportHtml,
  downloadEvalAuthoredSuite,
  fetchEvalAuthoredSuite,
  fetchEvalAuthoredSuites,
  fetchEvalCases,
  fetchEvalCorpora,
  fetchEvalRuns,
  fetchEvalScenario,
  fetchEvalScenarios,
  fetchEvalResultDetail,
  fetchEvalResults,
  fetchEvalTargets,
  importEvalCorpus,
  launchEvalScenario,
  launchEvalAuthoredSuiteRun,
  materializeEvalScenarioArtifact,
  previewCapturedEvaluation,
  previewEvalScenario,
  previewEvalAuthoredSuite,
  previewEvalAuthoredSuiteRun,
  saveCapturedEvaluation,
  saveEvalScenario,
  saveEvalAuthoredSuite,
  selectEvalBaseline,
  submitEvalScenarioApproval,
} = await import("../src/lib/api.ts")
const { preflightEvalCorpusFile } = await import("../src/lib/evals-dashboard.ts")

test("eval API adapters encode identities, forward cancellation, and preserve launch idempotency", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  const controller = new AbortController()
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    if (calls.length === 1) {
      return new Response(JSON.stringify({ items: [], has_more: false, next_cursor: null }), {
        headers: { "content-type": "application/json" },
      })
    }
    if (calls.length === 2) {
      return new Response(
        JSON.stringify({
          status: "queued",
          spec: { run_id: "eval-1" },
        }),
        { status: 202, headers: { "content-type": "application/json" } },
      )
    }
    return new Response(
      JSON.stringify({
        comparison: {
          compatibility: { comparable: true },
          regressions: [],
        },
        baseline: { spec: { run_id: "eval/base" } },
        current: { spec: { run_id: "eval/current" } },
      }),
      { headers: { "content-type": "application/json" } },
    )
  }

  try {
    await fetchEvalCases(
      "sha256:corpus/value",
      "suite/value",
      { cursor: "opaque + cursor", limit: 25 },
      controller.signal,
    )
    await createEvalRun(
      { corpus_revision: "sha256:corpus", suite_id: "suite-1", max_concurrency: 2 },
      "dashboard-idempotency-key",
      controller.signal,
    )
    await compareEvalRuns("eval/base", "eval/current", controller.signal)
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(
    calls[0].input,
    "/api/evals/corpora/sha256%3Acorpus%2Fvalue/suites/suite%2Fvalue/cases?cursor=opaque+%2B+cursor&limit=25",
  )
  assert.equal(calls[0].init.signal, controller.signal)
  assert.equal(calls[0].init.credentials, "same-origin")
  assert.equal(calls[1].init.method, "POST")
  assert.equal(
    new Headers(calls[1].init.headers).get("Idempotency-Key"),
    "dashboard-idempotency-key",
  )
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    corpus_revision: "sha256:corpus",
    suite_id: "suite-1",
    max_concurrency: 2,
  })
  assert.equal(calls[2].input, "/api/evals/comparisons")
  assert.deepEqual(JSON.parse(calls[2].init.body), {
    baseline_run_id: "eval/base",
    current_run_id: "eval/current",
  })
})

test("eval downloads reject unsafe server filenames and use a sanitized fallback", async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    new Response("{}", {
      headers: {
        "content-type": "application/json",
        "content-disposition": 'attachment; filename="../../unsafe.json"',
      },
    })
  try {
    const file = await downloadEvalResultJson("eval/unsafe")
    assert.equal(file.filename, "eval-unsafe.eval-result.json")
    assert.equal(await file.blob.text(), "{}")
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("captured and fresh result adapters compare and download by immutable revision", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  const baseline = `sha256:${"a".repeat(64)}`
  const current = `sha256:${"b".repeat(64)}`
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    if (String(input).endsWith("result-comparisons")) {
      return new Response(
        JSON.stringify({
          baseline: { revision: baseline, origin: "captured_session" },
          current: { revision: current, origin: "fresh_execution" },
          comparison: { compatibility: { comparable: true }, regressions: [] },
        }),
        { headers: { "content-type": "application/json" } },
      )
    }
    return new Response("report", {
      headers: { "content-disposition": 'attachment; filename="unsafe/result"' },
    })
  }

  try {
    await compareEvalResults(baseline, current, 0.125)
    const jsonReport = await downloadCatalogEvalResultJson(baseline)
    const htmlReport = await downloadCatalogEvalResultHtml(current)
    assert.equal(jsonReport.filename, `sha256-${"a".repeat(64)}.eval-result.json`)
    assert.equal(htmlReport.filename, `sha256-${"b".repeat(64)}.eval-report.html`)
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(calls[0].input, "/api/evals/result-comparisons")
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    baseline_result_revision: baseline,
    current_result_revision: current,
    score_tolerance: 0.125,
  })
  assert.equal(calls[1].input, `/api/evals/results/${encodeURIComponent(baseline)}/report.json`)
  assert.equal(calls[2].input, `/api/evals/results/${encodeURIComponent(current)}/report.html`)
})

test("memory report adapters preserve the exact campaign request for JSON and HTML", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  const controller = new AbortController()
  const request = {
    schema_version: 1,
    experiment_id: "memory-campaign",
    cases: [{ case_id: "case", case_revision: `sha256:${"a".repeat(64)}` }],
    repetitions: 1,
    baseline_variant_id: "baseline",
    variants: [{ variant_id: "baseline" }, { variant_id: "candidate" }],
  }
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    if (String(input).endsWith("report.html")) {
      return new Response("<html></html>", { headers: { "content-type": "text/html" } })
    }
    return new Response(
      JSON.stringify({
        schema_version: 1,
        experiment_id: "memory-campaign",
        revision: `sha256:${"b".repeat(64)}`,
      }),
      { headers: { "content-type": "application/json" } },
    )
  }

  try {
    await buildMemoryExperimentReport(request, controller.signal)
    const html = await downloadMemoryExperimentReportHtml(request, controller.signal)
    assert.equal(html.filename, "cayu-memory-experiment-report.html")
    assert.equal(await html.blob.text(), "<html></html>")
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.deepEqual(
    calls.map((call) => call.input),
    ["/api/evals/memory-reports", "/api/evals/memory-reports/report.html"],
  )
  for (const call of calls) {
    assert.equal(call.init.method, "POST")
    assert.equal(call.init.signal, controller.signal)
    assert.deepEqual(JSON.parse(call.init.body), request)
  }
})

test("eval catalog adapters select only server-published target keys", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  globalThis.fetch = async (input) => {
    calls.push(String(input))
    return new Response(JSON.stringify({ items: [], has_more: false, next_cursor: null }), {
      headers: { "content-type": "application/json" },
    })
  }
  try {
    await fetchEvalTargets()
    await fetchEvalCorpora({ target_key: "eval.target-one", limit: 25 })
    await fetchEvalRuns({ target_key: "eval.target-two", status: "completed" })
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.deepEqual(calls, [
    "/api/evals/targets",
    "/api/evals/corpora?target_key=eval.target-one&limit=25",
    "/api/evals/runs?target_key=eval.target-two&status=completed",
  ])
})

test("eval corpus imports preserve the selected file bytes for strict server validation", async () => {
  const originalFetch = globalThis.fetch
  const prefix = new TextEncoder().encode(
    '{"schema_version":1,"schema_version":4,"revision":"sha256:corpus",' +
      '"target_key":"target","evidence_policy":{},"description":"',
  )
  const suffix = new TextEncoder().encode('","suites":[],"cases":[]}')
  const raw = new Uint8Array(prefix.length + 1 + suffix.length)
  raw.set(prefix)
  raw[prefix.length] = 0xff
  raw.set(suffix, prefix.length + 1)
  const corpus = new Blob([raw])
  let requestBody
  let requestHeaders

  globalThis.fetch = async (_input, init) => {
    requestBody = init.body
    requestHeaders = new Headers(init.headers)
    return new Response(JSON.stringify({ revision: `sha256:${"a".repeat(64)}` }), {
      status: 201,
      headers: { "content-type": "application/json" },
    })
  }

  try {
    await preflightEvalCorpusFile(corpus)
    await importEvalCorpus(corpus)
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(requestHeaders.get("content-type"), "application/json")
  assert.equal(requestBody, corpus)
  assert.deepEqual(new Uint8Array(await requestBody.arrayBuffer()), raw)
})

test("captured evaluation adapters preserve revisions, target scope, and baseline CAS", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  const revision = `sha256:${"a".repeat(64)}`
  const controller = new AbortController()
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    return new Response(JSON.stringify({ items: [], record: {}, baseline: {}, mutation: {} }), {
      status: String(input).endsWith("/save") ? 201 : 200,
      headers: { "content-type": "application/json" },
    })
  }
  const candidate = { revision }
  try {
    await previewCapturedEvaluation("session/one", undefined, controller.signal)
    await saveCapturedEvaluation(
      "session/one",
      { candidate, expected_candidate_revision: revision },
      controller.signal,
    )
    await fetchEvalResults(
      { target_key: "eval.target", origin: "captured_session", cursor: "next", limit: 10 },
      controller.signal,
    )
    await fetchEvalResultDetail(revision, controller.signal)
    await selectEvalBaseline(
      revision,
      { result_revision: revision, expected_generation: 2, operation_id: revision },
      controller.signal,
    )
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(calls[0].input, "/api/evals/sessions/session%2Fone/evaluation/preview")
  assert.deepEqual(JSON.parse(calls[0].init.body), { draft: null })
  assert.equal(calls[1].input, "/api/evals/sessions/session%2Fone/evaluation/save")
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    candidate,
    expected_candidate_revision: revision,
  })
  assert.equal(
    calls[2].input,
    "/api/evals/results?target_key=eval.target&origin=captured_session&cursor=next&limit=10",
  )
  assert.equal(calls[3].input, `/api/evals/results/${encodeURIComponent(revision)}`)
  assert.equal(calls[4].input, `/api/evals/results/${encodeURIComponent(revision)}/baseline`)
  assert.deepEqual(JSON.parse(calls[4].init.body), {
    result_revision: revision,
    expected_generation: 2,
    operation_id: revision,
  })
  assert.equal(calls[4].init.signal, controller.signal)
})

test("scenario adapters preserve reviewed revisions, bindings, target scope, and cancellation", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  const controller = new AbortController()
  const revision = `sha256:${"c".repeat(64)}`
  const scenario = {
    schema_version: 2,
    revision,
    id: "production-flow",
    target_key: "eval.target",
    name: "Production flow",
    events: [
      {
        kind: "initial",
        sequence: 0,
        id: "initial",
        input: { messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }] },
      },
    ],
  }
  const draft = { ...scenario }
  delete draft.schema_version
  delete draft.revision
  const settings = {
    environment_name: "files",
    trials: 1,
    max_concurrency: 1,
    timeout_seconds: 300,
  }
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    if (String(input).endsWith("/download")) {
      return new Response("{}", {
        headers: { "content-disposition": 'attachment; filename="production.scenario.json"' },
      })
    }
    return new Response(JSON.stringify({ items: [], scenario, preflight: { ready: true } }), {
      status: String(input) === "/api/evals/scenarios" && init?.method === "POST" ? 201 : 200,
      headers: { "content-type": "application/json" },
    })
  }

  try {
    await previewEvalScenario({ draft, settings }, controller.signal)
    await saveEvalScenario(
      { expected_scenario_revision: revision, scenario, settings },
      controller.signal,
    )
    await materializeEvalScenarioArtifact(
      "invoice/file",
      { expected_scenario_revision: revision, scenario, settings },
      controller.signal,
    )
    await fetchEvalScenarios(
      { target_key: "eval.target", scenario_id: "production-flow", cursor: "next", limit: 10 },
      controller.signal,
    )
    await fetchEvalScenario(revision, controller.signal)
    const downloaded = await downloadEvalScenario(revision, controller.signal)
    assert.equal(downloaded.filename, "production.scenario.json")
    await launchEvalScenario(
      revision,
      { expected_binding_revision: revision, settings },
      "scenario-idempotency-key",
      controller.signal,
    )
    await submitEvalScenarioApproval(
      "run/scenario",
      {
        expected_progress_revision: revision,
        trial_number: 1,
        event_id: "approval-1",
        decision: "approve",
      },
      controller.signal,
    )
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(calls[0].input, "/api/evals/scenarios/preview")
  assert.deepEqual(JSON.parse(calls[0].init.body), { draft, settings })
  assert.equal(calls[1].input, "/api/evals/scenarios")
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    expected_scenario_revision: revision,
    scenario,
    settings,
  })
  assert.equal(calls[2].input, "/api/evals/scenarios/artifacts/invoice%2Ffile/materialize")
  assert.equal(
    calls[3].input,
    "/api/evals/scenarios?target_key=eval.target&scenario_id=production-flow&cursor=next&limit=10",
  )
  assert.equal(calls[4].input, `/api/evals/scenarios/${encodeURIComponent(revision)}`)
  assert.equal(calls[5].input, `/api/evals/scenarios/${encodeURIComponent(revision)}/download`)
  assert.equal(calls[6].input, `/api/evals/scenarios/${encodeURIComponent(revision)}/runs`)
  assert.equal(
    new Headers(calls[6].init.headers).get("Idempotency-Key"),
    "scenario-idempotency-key",
  )
  assert.deepEqual(JSON.parse(calls[6].init.body), {
    expected_binding_revision: revision,
    settings,
  })
  assert.equal(calls[7].input, "/api/evals/runs/run%2Fscenario/scenario-approval")
  assert.deepEqual(JSON.parse(calls[7].init.body), {
    expected_progress_revision: revision,
    trial_number: 1,
    event_id: "approval-1",
    decision: "approve",
  })
  assert.ok(calls.every((call) => call.init.signal === controller.signal))
})

test("authored suite adapters preserve immutable selection and launch idempotency", async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  const controller = new AbortController()
  const revision = `sha256:${"d".repeat(64)}`
  const draft = { id: "suite", target_key: "eval.target", name: "Suite", cases: [] }
  const suite = { revision }
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    if (String(input).endsWith("/download")) {
      return new Response("{}", {
        headers: { "content-disposition": 'attachment; filename="suite.eval-suite.json"' },
      })
    }
    return new Response(JSON.stringify({ items: [], suite, selection: {}, runs: [] }), {
      status: String(input).endsWith("/runs") ? 202 : 200,
      headers: { "content-type": "application/json" },
    })
  }

  try {
    await previewEvalAuthoredSuite({ draft }, controller.signal)
    await saveEvalAuthoredSuite({ expected_suite_revision: revision, suite }, controller.signal)
    await fetchEvalAuthoredSuites(
      { target_key: "eval.target", suite_id: "suite/one", limit: 10 },
      controller.signal,
    )
    await fetchEvalAuthoredSuite(revision, controller.signal)
    const download = await downloadEvalAuthoredSuite(revision, controller.signal)
    assert.equal(download.filename, "suite.eval-suite.json")
    await previewEvalAuthoredSuiteRun(revision, { case_ids: ["case-one"] }, controller.signal)
    await launchEvalAuthoredSuiteRun(
      revision,
      {
        case_ids: ["case-one"],
        expected_exposure_revision: revision,
        expected_execution_profiles: [
          {
            case_ids: ["case-one"],
            execution_profile_revision: revision,
          },
        ],
      },
      "authored-suite-idempotency-key",
      controller.signal,
    )
  } finally {
    globalThis.fetch = originalFetch
  }

  assert.equal(calls[0].input, "/api/evals/suites/preview")
  assert.deepEqual(JSON.parse(calls[0].init.body), { draft })
  assert.equal(calls[1].input, "/api/evals/suites")
  assert.deepEqual(JSON.parse(calls[1].init.body), {
    expected_suite_revision: revision,
    suite,
  })
  assert.equal(
    calls[2].input,
    "/api/evals/suites?target_key=eval.target&suite_id=suite%2Fone&limit=10",
  )
  assert.equal(calls[3].input, `/api/evals/suites/${encodeURIComponent(revision)}`)
  assert.equal(calls[4].input, `/api/evals/suites/${encodeURIComponent(revision)}/download`)
  assert.equal(calls[5].input, `/api/evals/suites/${encodeURIComponent(revision)}/runs/preview`)
  assert.deepEqual(JSON.parse(calls[5].init.body), { case_ids: ["case-one"] })
  assert.equal(calls[6].input, `/api/evals/suites/${encodeURIComponent(revision)}/runs`)
  assert.deepEqual(JSON.parse(calls[6].init.body), {
    case_ids: ["case-one"],
    expected_exposure_revision: revision,
    expected_execution_profiles: [
      {
        case_ids: ["case-one"],
        execution_profile_revision: revision,
      },
    ],
  })
  assert.equal(
    new Headers(calls[6].init.headers).get("Idempotency-Key"),
    "authored-suite-idempotency-key",
  )
  assert.ok(calls.every((call) => call.init.signal === controller.signal))
})
