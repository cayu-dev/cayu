import assert from "node:assert/strict"
import test from "node:test"

globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }

const { compareEvalRuns, createEvalRun, downloadEvalResultJson, fetchEvalCases, importEvalCorpus } =
  await import("../src/lib/api.ts")
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
        compatibility: { comparable: true },
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

test("eval corpus imports preserve the selected file bytes for strict server validation", async () => {
  const originalFetch = globalThis.fetch
  const prefix = new TextEncoder().encode(
    '{"schema_version":2,"schema_version":1,"revision":"sha256:corpus",' +
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
