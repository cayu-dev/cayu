import assert from "node:assert/strict"
import test from "node:test"
import { QueryClient, QueryObserver } from "@tanstack/react-query"

import {
  authoredSuiteEvalLaunchRequestIdentity,
  capturedEvalLaunchRequestIdentity,
  createEvalIdempotencyKey,
  EVAL_RESULT_QUERY_RETENTION,
  EVAL_TARGET_QUERY_KEY,
  EVAL_TARGET_STALE_TIME_MS,
  EvalLaunchIdempotencyRegistry,
  evalCancellationNotice,
  evalComparisonReasonText,
  evalLaunchFailureIsDefinitive,
  evalLaunchNotice,
  evalLaunchRequestIdentity,
  evalRunCanCancel,
  evalRunHasResult,
  evalRunIsActive,
  evalTargetCatalogMayBeStale,
  evalTrialCostSummary,
  MAX_EVAL_CORPUS_FILE_BYTES,
  preflightEvalCorpusFile,
  retryEvalQuery,
  scenarioEvalLaunchRequestIdentity,
  shortEvalIdentity,
} from "../src/lib/evals-dashboard.ts"

const evalLaunchRegistryKey = (scope) =>
  `cayu.eval-launch-idempotency.v1:${encodeURIComponent(scope)}`

class MemoryStorage {
  items = new Map()

  getItem(key) {
    return this.items.get(key) ?? null
  }

  setItem(key, value) {
    this.items.set(key, value)
  }

  removeItem(key) {
    this.items.delete(key)
  }
}

function run(status, result = null) {
  return { status, result }
}

test("eval lifecycle helpers distinguish active, cancellable, and published runs", () => {
  assert.equal(evalRunIsActive(run("queued")), true)
  assert.equal(evalRunIsActive(run("cancelling")), true)
  assert.equal(evalRunIsActive(run("cancelled")), false)
  assert.equal(evalRunCanCancel(run("queued")), true)
  assert.equal(evalRunCanCancel(run("running")), true)
  assert.equal(evalRunCanCancel(run("cancelling")), false)
  assert.equal(evalRunHasResult(run("completed", { revision: "sha256:result" })), true)
  assert.equal(evalRunHasResult(run("completed")), false)
  assert.equal(evalRunHasResult(run("failed", { revision: "sha256:result" })), false)
})

test("eval cancellation notices follow the returned durable status", () => {
  const returned = (status) => ({ status, spec: { run_id: "eval-1234567890abcdef" } })

  assert.equal(
    evalCancellationNotice(returned("cancelling")),
    "Cancellation requested for eval-1234567….",
  )
  assert.equal(
    evalCancellationNotice(returned("cancelled")),
    "Eval run eval-1234567… is cancelled.",
  )
  assert.equal(
    evalCancellationNotice(returned("completed")),
    "Eval run eval-1234567… completed before cancellation took effect.",
  )
  assert.equal(
    evalCancellationNotice(returned("failed")),
    "Eval run eval-1234567… failed before cancellation took effect.",
  )
})

test("eval launch notices describe the returned durable run instead of fabricating admission", () => {
  const returned = (status) => ({ status, spec: { run_id: "eval-1234567890abcdef" } })

  assert.equal(evalLaunchNotice(returned("queued")), "Opened eval run eval-1234567… (queued).")
  assert.equal(
    evalLaunchNotice(returned("completed")),
    "Opened eval run eval-1234567… (completed).",
  )
  assert.equal(evalLaunchNotice(returned("failed")), "Opened eval run eval-1234567… (failed).")
  assert.equal(
    evalLaunchNotice(returned("cancelled")),
    "Opened eval run eval-1234567… (cancelled).",
  )
})

test("eval query retries stop for definitive API failures", () => {
  assert.equal(retryEvalQuery(0, Object.assign(new Error("missing"), { status: 404 })), false)
  assert.equal(retryEvalQuery(0, Object.assign(new Error("retry"), { status: 429 })), true)
  assert.equal(retryEvalQuery(2, new Error("network")), true)
  assert.equal(retryEvalQuery(3, new Error("network")), false)
})

test("inactive complete eval results are released from the browser cache immediately", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const queryKey = ["evals", "result", "eval-complete"]
  const observer = new QueryObserver(client, {
    queryKey,
    queryFn: async () => ({ payload: "bounded-result" }),
    ...EVAL_RESULT_QUERY_RETENTION,
  })
  const unsubscribe = observer.subscribe(() => {})
  let subscribed = true

  try {
    await observer.refetch()
    assert.deepEqual(client.getQueryData(queryKey), { payload: "bounded-result" })
    unsubscribe()
    subscribed = false
    await new Promise((resolve) => setTimeout(resolve, 5))
    assert.equal(client.getQueryData(queryKey), undefined)
  } finally {
    if (subscribed) unsubscribe()
    client.clear()
  }
})

test("eval corpus import preflight bounds work without authorizing normalized content", async () => {
  const corpus = {
    schema_version: 1,
    revision: "sha256:corpus",
    target_key: "support.regressions",
    evidence_policy: { schema_version: 1 },
    suites: [],
    cases: [],
  }
  await preflightEvalCorpusFile(new Blob([JSON.stringify(corpus)]))

  await assert.rejects(preflightEvalCorpusFile(new Blob(["not json"])), /not valid JSON/)
  await assert.rejects(
    preflightEvalCorpusFile(new Blob([JSON.stringify({ ...corpus, schema_version: 2 })])),
    /corpus v1/,
  )
  await assert.rejects(
    preflightEvalCorpusFile(new Blob([new Uint8Array(MAX_EVAL_CORPUS_FILE_BYTES + 1)])),
    /larger than the supported 8 MiB/,
  )
})

test("eval identities are compact without discarding short identifiers", () => {
  assert.equal(shortEvalIdentity("sha256:1234567890abcdef"), "1234567890ab…")
  assert.equal(shortEvalIdentity("suite-1"), "suite-1")
})

test("eval launch idempotency keys use secure browser UUIDs", () => {
  assert.match(
    createEvalIdempotencyKey(),
    /^cayu-dashboard-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  )
})

test("eval launch retries reuse their key across page reloads until reconciled", () => {
  const storage = new MemoryStorage()
  const identity = evalLaunchRequestIdentity("sha256:corpus", "suite-1", 4, "sha256:profile")
  const originalKey = new EvalLaunchIdempotencyRegistry(storage, "/api").keyFor(identity)

  assert.equal(new EvalLaunchIdempotencyRegistry(storage, "/api").keyFor(identity), originalKey)
  assert.notEqual(
    identity,
    evalLaunchRequestIdentity("sha256:corpus", "suite-1", 4, "sha256:new-profile"),
  )

  new EvalLaunchIdempotencyRegistry(storage, "/api").resolve(identity)
  assert.notEqual(new EvalLaunchIdempotencyRegistry(storage, "/api").keyFor(identity), originalKey)
})

test("scenario launch retry identity binds the scenario, binding, and execution profile", () => {
  const identity = scenarioEvalLaunchRequestIdentity(
    "sha256:scenario",
    "sha256:binding",
    "sha256:profile",
  )
  assert.equal(identity, '["scenario-v2","sha256:scenario","sha256:binding","sha256:profile"]')
  assert.notEqual(
    identity,
    scenarioEvalLaunchRequestIdentity("sha256:scenario", "sha256:new-binding", "sha256:profile"),
  )
  assert.notEqual(
    identity,
    scenarioEvalLaunchRequestIdentity("sha256:scenario", "sha256:binding", "sha256:new-profile"),
  )
})

test("authored suite retry identity binds the suite, selection, and execution profiles", () => {
  const profiles = [{ case_ids: ["case-a"], execution_profile_revision: "sha256:profile-a" }]
  const identity = authoredSuiteEvalLaunchRequestIdentity(
    "sha256:suite",
    "sha256:selection",
    profiles,
  )
  assert.notEqual(
    identity,
    authoredSuiteEvalLaunchRequestIdentity("sha256:suite", "sha256:other-selection", profiles),
  )
  assert.notEqual(
    identity,
    authoredSuiteEvalLaunchRequestIdentity("sha256:suite", "sha256:selection", [
      { case_ids: ["case-a"], execution_profile_revision: "sha256:new-profile" },
    ]),
  )
})

test("captured eval retry identities include every execution contraction", () => {
  const request = {
    trial_request: { trials: 1, timeout_seconds: 45 },
    max_concurrency: 1,
    max_steps: 8,
    limits: {
      max_input_tokens: 100,
      max_output_tokens: 200,
      max_total_tokens: 300,
      max_tool_calls: 4,
      max_elapsed_seconds: 30,
      scope: "run",
    },
    cost_budget: { max_estimated_cost: "0.25", currency: "USD" },
  }
  const identity = capturedEvalLaunchRequestIdentity("session-1", "sha256:candidate", request)

  assert.notEqual(
    identity,
    capturedEvalLaunchRequestIdentity("session-1", "sha256:candidate", {
      ...request,
      limits: { ...request.limits, scope: "session" },
    }),
  )
  assert.notEqual(
    identity,
    capturedEvalLaunchRequestIdentity("session-1", "sha256:candidate", {
      ...request,
      cost_budget: { ...request.cost_budget, max_estimated_cost: "0.50" },
    }),
  )
  assert.notEqual(identity, capturedEvalLaunchRequestIdentity("session-1", "sha256:other", request))
})

test("eval launch retries are isolated between API mounts on one origin", () => {
  const storage = new MemoryStorage()
  const identity = evalLaunchRequestIdentity("sha256:corpus", "suite-1", 4, "sha256:profile")
  const first = new EvalLaunchIdempotencyRegistry(storage, "/app-a/api")
  const second = new EvalLaunchIdempotencyRegistry(storage, "/app-b/api")
  const firstKey = first.keyFor(identity)
  const secondKey = second.keyFor(identity)

  assert.notEqual(firstKey, secondKey)
  second.resolve(identity)
  assert.equal(first.keyFor(identity), firstKey)
  assert.notEqual(second.keyFor(identity), secondKey)
})

test("dynamic eval targets have bounded freshness and profile conflicts invalidate them", () => {
  assert.deepEqual(EVAL_TARGET_QUERY_KEY, ["evals", "targets"])
  assert.equal(EVAL_TARGET_STALE_TIME_MS, 15_000)
  assert.equal(evalTargetCatalogMayBeStale({ status: 409 }), true)
  assert.equal(evalTargetCatalogMayBeStale({ status: 400 }), false)
  assert.equal(evalTargetCatalogMayBeStale(new Error("offline")), false)
})

test("eval launch retry keys clear only after definitive API responses", () => {
  for (const status of [400, 401, 403, 404, 405, 409, 410, 413, 415, 422, 451, 501]) {
    assert.equal(evalLaunchFailureIsDefinitive(status), true)
  }
  for (const status of [408, 425, 429, 499, 500, 502, 503, 504]) {
    assert.equal(evalLaunchFailureIsDefinitive(status), false)
  }
})

test("permanent launch rejections release their retry-state capacity", () => {
  const storage = new MemoryStorage()
  const registry = new EvalLaunchIdempotencyRegistry(storage, "/api")

  for (let index = 0; index < 64; index += 1) {
    const identity = `permanently-rejected-request-${index}`
    registry.keyFor(identity)
    assert.equal(evalLaunchFailureIsDefinitive(index % 2 === 0 ? 405 : 501), true)
    registry.resolve(identity)
  }

  assert.equal(storage.getItem(evalLaunchRegistryKey("/api")), null)
  assert.doesNotThrow(() => registry.keyFor("subsequent-valid-request"))
})

test("eval launch retry state rejects corrupt and unavailable storage", () => {
  const storage = new MemoryStorage()
  const storageKey = evalLaunchRegistryKey("/api")
  assert.throws(
    () => new EvalLaunchIdempotencyRegistry(storage, ""),
    /API scope is outside the browser retry-state limit/,
  )
  storage.setItem(storageKey, "not-json")
  assert.throws(
    () => new EvalLaunchIdempotencyRegistry(storage, "/api").keyFor("request"),
    /retry state is invalid/,
  )

  const unavailableStorage = {
    getItem() {
      throw new Error("denied")
    },
    setItem() {},
    removeItem() {},
  }
  assert.throws(
    () => new EvalLaunchIdempotencyRegistry(unavailableStorage, "/api").keyFor("request"),
    /retry state is unavailable/,
  )

  const unwritableStorage = {
    getItem() {
      return null
    },
    setItem() {
      throw new Error("denied")
    },
    removeItem() {},
  }
  assert.throws(
    () => new EvalLaunchIdempotencyRegistry(unwritableStorage, "/api").keyFor("request"),
    /could not be persisted/,
  )

  const duplicateKey = createEvalIdempotencyKey()
  storage.setItem(
    storageKey,
    JSON.stringify({
      version: 1,
      entries: [
        ["request-1", duplicateKey],
        ["request-2", duplicateKey],
      ],
    }),
  )
  assert.throws(
    () => new EvalLaunchIdempotencyRegistry(storage, "/api").keyFor("request-3"),
    /retry state is invalid/,
  )
})

test("eval launch retry state never silently evicts an unresolved request", () => {
  const storage = new MemoryStorage()
  const registry = new EvalLaunchIdempotencyRegistry(storage, "/api")
  for (let index = 0; index < 32; index += 1) {
    registry.keyFor(`request-${index}`)
  }

  assert.throws(() => registry.keyFor("request-32"), /Too many eval launches/)
  assert.equal(JSON.parse(storage.getItem(evalLaunchRegistryKey("/api"))).entries.length, 32)
})

test("eval trial cost summary distinguishes observed, unavailable, and absent pricing", () => {
  assert.deepEqual(
    evalTrialCostSummary(
      [
        {
          detail: {
            kind: "max_estimated_cost",
            estimated_cost: "0.0125",
            currency: "USD",
            maximum: "0.02",
          },
        },
      ],
      "en-US",
    ),
    { display: "USD\u00a00.01", exact: "0.0125 USD" },
  )
  assert.deepEqual(
    evalTrialCostSummary(
      [
        {
          detail: {
            kind: "max_estimated_cost",
            estimated_cost: "1E-7",
            currency: "USD",
            maximum: "0.02",
          },
        },
      ],
      "en-US",
    ),
    { display: "<USD\u00a00.0001", exact: "1E-7 USD" },
  )
  assert.deepEqual(
    evalTrialCostSummary([
      {
        detail: {
          kind: "max_estimated_cost",
          estimated_cost: null,
          currency: "USD",
          maximum: "0.02",
          unpriced_model_steps: 2,
        },
      },
    ]),
    {
      display: "unavailable · 2 unpriced model steps",
      exact: "unavailable · 2 unpriced model steps",
    },
  )
  assert.deepEqual(evalTrialCostSummary([]), {
    display: "not evaluated",
    exact: "not evaluated",
  })
})

test("eval comparison reasons remain specific and operator-readable", () => {
  assert.equal(
    evalComparisonReasonText("pricing_profile_fingerprint_mismatch"),
    "The applicable pricing contract changed between runs.",
  )
  assert.equal(
    evalComparisonReasonText("assertion_contract_mismatch"),
    "At least one assertion contract changed between runs.",
  )
})
