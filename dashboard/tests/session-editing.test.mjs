import assert from "node:assert/strict"
import test from "node:test"

import { QueryClient, QueryObserver } from "@tanstack/react-query"

import {
  applyConfirmedSessionEdit,
  formatSessionEditDraft,
  parseSessionLabelsDraft,
  parseSessionMetadataDraft,
  sessionEditErrorMessage,
  splitSessionMetadata,
} from "../src/lib/session-editing.ts"

async function waitForRefetches(promises) {
  let timeout
  try {
    await Promise.race([
      Promise.all(promises),
      new Promise((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error("Timed out waiting for cache refetches.")),
          1000,
        )
      }),
    ])
  } finally {
    clearTimeout(timeout)
  }
}

test("session label drafts require a complete string map and permit an intentional clear", () => {
  assert.deepEqual(parseSessionLabelsDraft('{"tenant":"acme","tier":"critical"}'), {
    ok: true,
    value: { tenant: "acme", tier: "critical" },
  })
  assert.deepEqual(parseSessionLabelsDraft("{}"), { ok: true, value: {} })

  for (const draft of [
    "[]",
    '"labels"',
    '{"tenant":1}',
    '{"":"acme"}',
    '{" tenant":"acme"}',
    '{"tenant":" "}',
    '{"cayu:internal":"value"}',
  ]) {
    assert.equal(parseSessionLabelsDraft(draft).ok, false, draft)
  }

  assert.equal(parseSessionLabelsDraft(JSON.stringify({ ["😀".repeat(128)]: "value" })).ok, true)
  assert.equal(parseSessionLabelsDraft(JSON.stringify({ ["😀".repeat(129)]: "value" })).ok, false)
  assert.equal(parseSessionLabelsDraft(JSON.stringify({ key: "😀".repeat(512) })).ok, true)
  assert.equal(parseSessionLabelsDraft(JSON.stringify({ key: "😀".repeat(513) })).ok, false)

  const prototypeLabel = parseSessionLabelsDraft('{"__proto__":"allowed"}')
  assert.equal(prototypeLabel.ok, true)
  assert.equal(Object.hasOwn(prototypeLabel.value, "__proto__"), true)
  assert.equal(JSON.stringify(prototypeLabel.value), '{"__proto__":"allowed"}')
})

test("session metadata drafts require a durable top-level JSON object", () => {
  assert.deepEqual(parseSessionMetadataDraft('{"nested":{"items":[1,true,null]}}'), {
    ok: true,
    value: { nested: { items: [1, true, null] } },
  })
  assert.deepEqual(parseSessionMetadataDraft("{}"), { ok: true, value: {} })

  for (const draft of [
    "{",
    "[]",
    "null",
    '{"value":1e400}',
    '{"value":"\\u0000"}',
    '{"value":"\\ud800"}',
    '{"subagent":{}}',
    '{"cayu:taint_labels":[]}',
  ]) {
    assert.equal(parseSessionMetadataDraft(draft).ok, false, draft)
  }

  const unsafeInteger = parseSessionMetadataDraft('{"id":9007199254740993}')
  assert.equal(unsafeInteger.ok, false)
  assert.match(unsafeInteger.message, /\$\.id/)
  assert.match(unsafeInteger.message, /safe range/)
  assert.match(unsafeInteger.message, /strings/)

  assert.deepEqual(parseSessionMetadataDraft('{"id":"9007199254740993"}'), {
    ok: true,
    value: { id: "9007199254740993" },
  })
})

test("session metadata ownership separates editable and runtime-managed entries", () => {
  assert.deepEqual(
    splitSessionMetadata({
      customer: { id: "customer-1" },
      subagent: { mode: "background" },
      "cayu:taint_labels": ["untrusted"],
    }),
    {
      userMetadata: { customer: { id: "customer-1" } },
      runtimeMetadata: {
        subagent: { mode: "background" },
        "cayu:taint_labels": ["untrusted"],
      },
    },
  )
})

test("session edit drafts are formatted predictably", () => {
  assert.equal(
    formatSessionEditDraft({ tenant: "acme", nested: { enabled: true } }),
    '{\n  "tenant": "acme",\n  "nested": {\n    "enabled": true\n  }\n}',
  )
})

test("session edit errors preserve useful access and transport distinctions", () => {
  assert.equal(
    sessionEditErrorMessage(Object.assign(new Error("Invalid labels."), { status: 422 }), "labels"),
    "Invalid labels.",
  )
  assert.match(sessionEditErrorMessage({ status: 401 }, "metadata"), /Authentication/)
  assert.match(sessionEditErrorMessage({ status: 403 }, "metadata"), /permission/)
  assert.match(sessionEditErrorMessage({ status: 404 }, "metadata"), /no longer exists/)
  assert.match(sessionEditErrorMessage({ status: 503 }, "metadata"), /draft has been preserved/)
  assert.match(sessionEditErrorMessage(new TypeError("Failed to fetch"), "labels"), /reached/)
})

test("confirmed edits refresh affected bounded projections without touching history", async () => {
  const client = new QueryClient()
  const sessionKey = ["session", "session-1"]
  const otherSessionKey = ["session", "session-2"]
  const stateKey = ["session-state", "session-1"]
  const otherStateKey = ["session-state", "session-2"]
  const detailSummaryKey = ["session-summary", "session-1", "revision-1"]
  const otherDetailSummaryKey = ["session-summary", "session-2", "revision-1"]
  const indexSummaryKey = ["sessions-summary", "index", { limit: 100 }]
  const agentSummaryKey = ["sessions-summary", "agent-related", "assistant"]
  const environmentSummaryKey = ["sessions-summary", "environment-related", "local"]
  const eventKey = ["session-events", "session-1", "latest"]
  const transcriptKey = ["session-transcript", "session-1", "latest"]
  client.setQueryData(sessionKey, { id: "session-1", labels: { stage: "old" } })
  client.setQueryData(otherSessionKey, { id: "session-2", labels: { stage: "other" } })
  client.setQueryData(stateKey, { status: "completed" })
  client.setQueryData(otherStateKey, { status: "completed" })
  client.setQueryData(detailSummaryKey, { session: { id: "session-1" } })
  client.setQueryData(otherDetailSummaryKey, { session: { id: "session-2" } })
  client.setQueryData(indexSummaryKey, { sessions: [] })
  client.setQueryData(agentSummaryKey, { sessions: [] })
  client.setQueryData(environmentSummaryKey, { sessions: [] })
  client.setQueryData(eventKey, { events: [] })
  client.setQueryData(transcriptKey, { messages: [] })

  let stateRefetches = 0
  let summaryRefetches = 0
  let resolveStateRefetch
  let resolveSummaryRefetch
  const stateRefetched = new Promise((resolve) => {
    resolveStateRefetch = resolve
  })
  const summaryRefetched = new Promise((resolve) => {
    resolveSummaryRefetch = resolve
  })
  const stateObserver = new QueryObserver(client, {
    queryKey: stateKey,
    queryFn: async () => {
      stateRefetches += 1
      resolveStateRefetch()
      return { status: "completed", updated_at: "2026-07-18T10:00:00Z" }
    },
    staleTime: Number.POSITIVE_INFINITY,
  })
  const summaryObserver = new QueryObserver(client, {
    queryKey: detailSummaryKey,
    queryFn: async () => {
      summaryRefetches += 1
      resolveSummaryRefetch()
      return { session: { id: "session-1", labels: { stage: "review" } } }
    },
    staleTime: Number.POSITIVE_INFINITY,
  })
  const unsubscribeState = stateObserver.subscribe(() => {})
  const unsubscribeSummary = summaryObserver.subscribe(() => {})

  try {
    applyConfirmedSessionEdit(client, {
      id: "session-1",
      labels: { stage: "review" },
      metadata: {},
    })
    await waitForRefetches([stateRefetched, summaryRefetched])

    assert.deepEqual(client.getQueryData(sessionKey).labels, { stage: "review" })
    assert.deepEqual(client.getQueryData(otherSessionKey).labels, { stage: "other" })
    assert.equal(stateRefetches, 1)
    assert.equal(summaryRefetches, 1)
    assert.equal(client.getQueryState(otherStateKey).isInvalidated, false)
    assert.equal(client.getQueryState(otherDetailSummaryKey).isInvalidated, false)
    assert.equal(client.getQueryState(indexSummaryKey).isInvalidated, true)
    assert.equal(client.getQueryState(agentSummaryKey).isInvalidated, true)
    assert.equal(client.getQueryState(environmentSummaryKey).isInvalidated, true)
    assert.equal(client.getQueryState(eventKey).isInvalidated, false)
    assert.equal(client.getQueryState(transcriptKey).isInvalidated, false)
  } finally {
    unsubscribeState()
    unsubscribeSummary()
    client.clear()
  }
})

test("generated-contract-backed session edit requests use PATCH replacement bodies", async () => {
  const originalWindow = globalThis.window
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
  const calls = []
  const originalFetch = globalThis.fetch
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    return new Response(JSON.stringify({ id: "session/1", labels: {}, metadata: {} }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  }

  try {
    const { updateSessionLabels, updateSessionMetadata } = await import("../src/lib/api.ts")
    await updateSessionLabels("session/1", { labels: { stage: "review" } })
    await updateSessionMetadata("session/1", { metadata: { nested: [1, 2] } })
  } finally {
    globalThis.fetch = originalFetch
    if (originalWindow === undefined) delete globalThis.window
    else globalThis.window = originalWindow
  }

  assert.deepEqual(
    calls.map(({ input, init }) => ({
      input,
      method: init.method,
      body: init.body,
      contentType: new Headers(init.headers).get("content-type"),
      credentials: init.credentials,
    })),
    [
      {
        input: "/api/sessions/session%2F1/labels",
        method: "PATCH",
        body: '{"labels":{"stage":"review"}}',
        contentType: "application/json",
        credentials: "same-origin",
      },
      {
        input: "/api/sessions/session%2F1/metadata",
        method: "PATCH",
        body: '{"metadata":{"nested":[1,2]}}',
        contentType: "application/json",
        credentials: "same-origin",
      },
    ],
  )
})
