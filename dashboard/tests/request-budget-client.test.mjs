import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"

import { QueryClient, QueryObserver } from "@tanstack/react-query"
import { rolldown } from "rolldown"

import { ConditionalGetCache, REVALIDATED_HEADER } from "../src/lib/conditional-get.ts"
import {
  createDashboardQueryClient,
  QUERY_STALE_TIME_MS,
  queryErrorIsFinal,
  shouldRetryQuery,
} from "../src/lib/query-client.ts"
import {
  firstOccurrences,
  firstPageAction,
  firstPageOnly,
  OLDER_PAGES_REFRESH_MS,
  shiftedOut,
  singlePageOnly,
  withFirstPage,
} from "../src/next/app/first-page-refresh.ts"
import {
  changeForEventType,
  hasQueuedMessage,
  isSecondaryView,
  planForProgress,
  refreshSessionViews,
  SESSION_VIEWS,
  SessionRefreshThrottle,
  SessionTickRefresher,
} from "../src/next/app/session-progress-refresh.ts"
import {
  POLL_BACKOFF_CAP_MS,
  UnchangedDataBackoff,
  unchangedBackoffInterval,
} from "../src/parts/backoff.ts"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "src/next/app/specialist-activity.ts",
  platform: "node",
  logLevel: "silent",
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
const { agentActivity, changedAgents } = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)

const httpError = (status) => Object.assign(new Error(`HTTP ${status}`), { status })
const named = (name) => Object.assign(new Error(name), { name })

test("the shared query client keeps data fresh and doesn't refetch on focus", () => {
  const client = createDashboardQueryClient()
  const defaults = client.getDefaultOptions()
  assert.equal(defaults.queries.staleTime, QUERY_STALE_TIME_MS)
  assert.equal(defaults.queries.refetchOnWindowFocus, false)
  assert.equal(defaults.queries.retry, shouldRetryQuery)
  assert.equal(defaults.mutations.retry, 0)
  // Reads that decide what the viewer may do still refresh on every focus.
  for (const key of [["dashboard-identity"], ["evals", "targets"], ["session", "s1"]]) {
    const options = client.getQueryDefaults(key)
    assert.equal(options.refetchOnWindowFocus, true, key.join("/"))
    assert.equal(options.staleTime, 0, key.join("/"))
  }
  assert.equal(client.getQueryDefaults(["session-list", {}]).refetchOnWindowFocus, undefined)
  // A session's live views are read again when the session is revisited.
  for (const key of [
    ["transcript", "s1"],
    ["overview", "s1"],
    ["state", "s1"],
  ]) {
    assert.equal(client.getQueryDefaults(key).staleTime, 0, key.join("/"))
    assert.equal(client.getQueryDefaults(key).refetchOnWindowFocus, undefined, key.join("/"))
  }
})

test("reads retry once, and never for 4xx, 413, timeouts or cancellation", () => {
  for (const error of [
    httpError(400),
    httpError(404),
    httpError(413),
    httpError(422),
    httpError(408),
    httpError(504),
    named("TimeoutError"),
    named("AbortError"),
  ]) {
    assert.equal(queryErrorIsFinal(error), true, error.message)
    assert.equal(shouldRetryQuery(0, error), false, error.message)
  }
  for (const error of [httpError(500), httpError(503), new TypeError("Failed to fetch")]) {
    assert.equal(shouldRetryQuery(0, error), true, error.message)
    assert.equal(shouldRetryQuery(1, error), false, error.message)
  }
})

test("a 413 read fails after one request", async () => {
  const client = createDashboardQueryClient()
  let calls = 0
  const observer = new QueryObserver(client, {
    queryKey: ["too-large"],
    queryFn: async () => {
      calls += 1
      throw httpError(413)
    },
  })
  const unsubscribe = observer.subscribe(() => {})
  try {
    await observer.refetch()
    assert.equal(calls, 1)
    assert.equal(observer.getCurrentResult().isError, true)
  } finally {
    unsubscribe()
    client.clear()
  }
})

function server(responses) {
  const requests = []
  const fetchImpl = async (url, init) => {
    requests.push({ url, headers: new Headers(init.headers), method: init.method ?? "GET" })
    return responses.shift()(new Headers(init.headers))
  }
  return { requests, fetchImpl }
}
const json = (body, headers = {}) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json", ...headers },
  })

test("an ETag response is revalidated with If-None-Match and a 304 answers from the copy", async () => {
  const cache = new ConditionalGetCache()
  const { requests, fetchImpl } = server([
    () => json({ total: 3 }, { etag: 'W/"7"' }),
    (headers) =>
      headers.get("If-None-Match") === 'W/"7"'
        ? new Response(null, { status: 304 })
        : json({ total: -1 }),
    () => json({ total: 4 }, { etag: 'W/"8"' }),
  ])
  const first = await cache.fetch("/api/inbox/counts", {}, fetchImpl)
  assert.deepEqual(await first.json(), { total: 3 })
  assert.equal(requests[0].headers.has("If-None-Match"), false)

  const second = await cache.fetch("/api/inbox/counts", {}, fetchImpl)
  assert.equal(second.status, 200)
  assert.equal(second.headers.get(REVALIDATED_HEADER), "1")
  assert.deepEqual(await second.json(), { total: 3 })

  const third = await cache.fetch("/api/inbox/counts", {}, fetchImpl)
  assert.deepEqual(await third.json(), { total: 4 })
  assert.equal(requests[2].headers.get("If-None-Match"), 'W/"7"')
})

test("conditional GETs leave endpoints without ETags, writes and errors alone", async () => {
  const cache = new ConditionalGetCache()
  const { requests, fetchImpl } = server([
    () => json({ a: 1 }),
    () => json({ a: 2 }),
    () => json({ ok: true }, { etag: '"x"' }),
    () => json({ b: 1 }, { etag: '"b"' }),
    () => new Response("{}", { status: 403, headers: { "content-type": "application/json" } }),
    () => json({ b: 2 }),
  ])
  await cache.fetch("/api/tools/usage", {}, fetchImpl)
  await cache.fetch("/api/tools/usage", {}, fetchImpl)
  assert.equal(requests[1].headers.has("If-None-Match"), false)
  await cache.fetch("/api/inbox/x/settle", { method: "POST", body: "{}" }, fetchImpl)
  assert.equal(cache.size, 0)
  await cache.fetch("/api/b", {}, fetchImpl)
  assert.equal(cache.size, 1)
  const denied = await cache.fetch("/api/b", {}, fetchImpl)
  assert.equal(denied.status, 403)
  // Access changed: the copy is dropped, so the next read is unconditional.
  await cache.fetch("/api/b", {}, fetchImpl)
  assert.equal(requests[5].headers.has("If-None-Match"), false)
})

test("the conditional GET cache is bounded", async () => {
  const cache = new ConditionalGetCache({ maxEntries: 2, maxBodyLength: 20 })
  const fetchImpl = async (url) =>
    url === "/big" ? json({ text: "x".repeat(50) }, { etag: '"big"' }) : json({}, { etag: '"e"' })
  for (const url of ["/a", "/b", "/c", "/big"]) await cache.fetch(url, {}, fetchImpl)
  assert.equal(cache.size, 2)
})

test("an aborted request rejects instead of being retried or kept", async () => {
  const cache = new ConditionalGetCache()
  const controller = new AbortController()
  const fetchImpl = (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(named("AbortError")))
    })
  const pending = cache.fetch("/api/sessions/s/state", { signal: controller.signal }, fetchImpl)
  controller.abort()
  await assert.rejects(pending, { name: "AbortError" })
  assert.equal(cache.size, 0)
})

test("unchanged responses back off to a minute and any change resets", () => {
  assert.equal(unchangedBackoffInterval(0, 10_000), 10_000)
  assert.equal(unchangedBackoffInterval(2, 10_000), 40_000)
  assert.equal(unchangedBackoffInterval(9, 10_000), POLL_BACKOFF_CAP_MS)
  const backoff = new UnchangedDataBackoff()
  const same = { counts: 1 }
  const query = (data, dataUpdateCount, queryHash = "q") => ({
    queryHash,
    state: { data, dataUpdateCount },
  })
  assert.equal(backoff.interval(query(same, 1), 10_000), 10_000)
  // Evaluated several times per response: counted once.
  assert.equal(backoff.interval(query(same, 1), 10_000), 10_000)
  assert.equal(backoff.interval(query(same, 2), 10_000), 20_000)
  assert.equal(backoff.interval(query(same, 3), 10_000), 40_000)
  assert.equal(backoff.interval(query(same, 4), 10_000), 60_000)
  assert.equal(backoff.interval(query({ counts: 2 }, 5), 10_000), 10_000)
  assert.equal(backoff.interval(query(same, 1, "other"), 10_000), 10_000)
})

test("paged lists poll themselves only with one page and replace only page 1", () => {
  const interval = firstPageOnly((pages) => (pages.length ? 5000 : 7000))
  assert.equal(interval({ state: { data: undefined } }), 7000)
  assert.equal(interval({ state: { data: { pages: [1], pageParams: [null] } } }), 5000)
  // With older pages loaded, the whole list is re-read at most once a minute.
  assert.equal(
    interval({ state: { data: { pages: [1, 2], pageParams: [null, "c"] } } }),
    OLDER_PAGES_REFRESH_MS,
  )
  // Beside a page-1 poll, the infinite query stops polling once it holds several pages;
  // the page-1 poll re-reads everything once a minute and otherwise writes only changes.
  const beside = singlePageOnly(() => 5000)
  assert.equal(beside({ state: { data: { pages: [1], pageParams: [null] } } }), 5000)
  assert.equal(beside({ state: { data: { pages: [1, 2], pageParams: [null, "c"] } } }), false)
  assert.equal(firstPageAction(10_000, 0, true), "write")
  assert.equal(firstPageAction(10_000, 0, false), "none")
  assert.equal(firstPageAction(OLDER_PAGES_REFRESH_MS, 0, false), "refetch-all")
  assert.equal(firstPageAction(OLDER_PAGES_REFRESH_MS + 5, 0, true), "refetch-all")
  // Rows pushed past page 1 by new ones would be in no loaded page: re-read them all.
  const old = Array.from({ length: 50 }, (_, i) => `A${i + 1}`)
  const fresh = ["N1", "N2", "N3", "N4", "N5", ...old.slice(0, 45)]
  assert.equal(shiftedOut(old, fresh), true)
  assert.equal(shiftedOut(old, [...old]), false)
  assert.equal(firstPageAction(10_000, 0, true, true), "refetch-all")
  assert.equal(firstPageAction(10_000, 0, true, false), "write")
  const stopped = firstPageOnly(() => false)
  assert.equal(stopped({ state: { data: { pages: [1, 2], pageParams: [null, "c"] } } }), false)
  // A row that moved up into the refreshed page 1 keeps its fresh copy.
  const merged = firstOccurrences(
    [
      [
        { id: "t2", status: "completed" },
        { id: "t1", status: "running" },
      ],
      [
        { id: "t2", status: "running" },
        { id: "t3", status: "failed" },
      ],
    ].flat(),
    (task) => task.id,
  )
  assert.deepEqual(
    merged.map((task) => `${task.id}:${task.status}`),
    ["t2:completed", "t1:running", "t3:failed"],
  )
  const data = { pages: ["old-1", "old-2"], pageParams: [null, "c2"] }
  assert.deepEqual(withFirstPage(data, "new-1"), {
    pages: ["new-1", "old-2"],
    pageParams: [null, "c2"],
  })
  assert.equal(withFirstPage(data, "old-1"), data)
  assert.equal(withFirstPage(undefined, "x"), undefined)
})

test("a session tick refreshes only the views its events change", () => {
  const views = (type) => [
    ...(changeForEventType(type).urgent ?? []),
    ...(changeForEventType(type).progress ?? []),
  ]
  assert.deepEqual(changeForEventType("model.text.delta"), { progress: ["trace", "trace-events"] })
  assert.ok(views("model.completed").includes("transcript"))
  assert.ok(!views("tool.call.completed").includes("transcript"))
  assert.ok(changeForEventType("tool.call.approval_requested").urgent.includes("inbox"))
  assert.ok(changeForEventType("session.completed").urgent.includes("state"))
  assert.ok(changeForEventType("session.message.queued").urgent.includes("session-messages"))
  // Telemetry families (requests, context, budgets...) are progress, not lifecycle.
  for (const type of ["request.footprint.recorded", "context.compacted", "budget.updated"])
    assert.deepEqual(changeForEventType(type), { progress: ["overview", "trace", "trace-events"] })
  const tools = planForProgress(["tool.call.started", "tool.call.completed"])
  assert.equal(tools.urgent.size, 0)
  assert.deepEqual([...tools.progress].sort(), ["overview", "trace", "trace-events"])
  const empty = planForProgress([])
  assert.equal(empty.urgent.size + empty.progress.size, 0)
  // Without types, the overview (the session's state) is urgent; lifecycle views wait
  // for the drain.
  const untyped = planForProgress(null)
  // Without types, the overview (the session's state) and a still-queued message are
  // urgent; detail and runtime state refresh at the slow pace.
  assert.deepEqual([...untyped.urgent], ["overview", "queued-messages"])
  for (const view of ["session", "state", "session-state"]) {
    assert.ok(untyped.progress.has(view), view)
    assert.equal(isSecondaryView(view, false), true, view)
  }
  assert.equal(hasQueuedMessage(undefined), false)
  assert.equal(hasQueuedMessage({ pages: [{ records: [{ status: "delivered" }] }] }), false)
  assert.equal(
    hasQueuedMessage({ pages: [{ records: [] }, { records: [{ status: "queued" }] }] }),
    true,
  )
  // A view that is urgent for one event isn't also paced as progress.
  const mixed = planForProgress(["model.completed", "session.completed"])
  assert.ok(mixed.urgent.has("overview"))
  assert.equal(mixed.progress.has("overview"), false)
  assert.equal(planForProgress(null, true).urgent.size, SESSION_VIEWS.length)
  assert.equal(planForProgress(["model.text.delta"], true).urgent.size, SESSION_VIEWS.length)
})

test("session refreshes stay scoped to the session and never touch the global Inbox", async () => {
  const client = new QueryClient()
  const keys = [
    ["session", "s1"],
    ["session", "s2"],
    ["cayu", "inbox", "list", { session_id: "s1" }, null],
    ["cayu", "inbox", "list", {}, null],
    ["cayu", "inbox", "counts", {}],
    ["cayu", "session-recovery", "s1", "a1"],
    ["transcript", "s1", {}, null],
    ["trace-events", "s1", "a"],
  ]
  for (const key of keys) client.setQueryData(key, { key })
  await refreshSessionViews(client, "s1", SESSION_VIEWS, false)
  const stale = keys.filter((key) => client.getQueryState(key).isInvalidated)
  assert.deepEqual(stale, [
    ["session", "s1"],
    ["cayu", "inbox", "list", { session_id: "s1" }, null],
    ["cayu", "session-recovery", "s1", "a1"],
    ["trace-events", "s1", "a"],
  ])
  client.clear()
})

test("ticks are coalesced into at most one refresh per interval", async () => {
  let now = 0
  const timers = []
  const realSetTimeout = globalThis.setTimeout
  const realClearTimeout = globalThis.clearTimeout
  globalThis.setTimeout = (fn, ms) => {
    const timer = { fn, at: now + ms }
    timers.push(timer)
    return timer
  }
  globalThis.clearTimeout = (timer) => {
    const index = timers.indexOf(timer)
    if (index >= 0) timers.splice(index, 1)
  }
  try {
    const flushes = []
    const throttle = new SessionRefreshThrottle((views) => flushes.push([...views].sort()), {
      interval: 2000,
      now: () => now,
    })
    throttle.add(["trace"])
    assert.deepEqual(flushes, [["trace"]])
    now = 500
    throttle.add(["overview"])
    now = 900
    throttle.add(["trace", "transcript"])
    assert.equal(flushes.length, 1)
    assert.equal(timers.length, 1)
    assert.equal(timers[0].at, 2000)
    now = 2000
    timers.shift().fn()
    assert.deepEqual(flushes[1], ["overview", "trace", "transcript"])
    now = 2100
    throttle.add(["trace"])
    throttle.flushNow(["state"])
    assert.deepEqual(flushes[2], ["state", "trace"])
    assert.equal(timers.length, 0)
    throttle.add(["trace"])
    throttle.dispose()
    assert.equal(timers.length, 0)
    assert.equal(flushes.length, 3)
  } finally {
    globalThis.setTimeout = realSetTimeout
    globalThis.clearTimeout = realClearTimeout
  }
})

test("a session refresh waits for the one still running instead of overlapping it", async () => {
  let now = 0
  const flushes = []
  const releases = []
  const throttle = new SessionRefreshThrottle(
    (views) => {
      flushes.push([...views].sort())
      return new Promise((resolve) => releases.push(resolve))
    },
    { interval: 0, now: () => now },
  )
  const first = throttle.add(["trace"])
  assert.equal(flushes.length, 1)
  // Ticks while the first refresh is still reading queue up; nothing new starts.
  throttle.add(["transcript"])
  throttle.add(["trace"])
  assert.equal(flushes.length, 1)
  now = 5000
  releases.shift()()
  await first
  await new Promise((resolve) => setImmediate(resolve))
  assert.deepEqual(flushes[1], ["trace", "transcript"])
  // A terminal drain during a refresh runs after it, with everything queued.
  throttle.add(["overview"])
  const drain = throttle.flushNow(["state"])
  assert.equal(flushes.length, 2)
  releases.shift()()
  await new Promise((resolve) => setImmediate(resolve))
  assert.deepEqual(flushes[2], ["overview", "state"])
  releases.shift()()
  await drain
  throttle.dispose()
})

test("a model start refreshes the conversation for tool results published before it", () => {
  assert.ok(changeForEventType("model.started").progress.includes("transcript"))
  assert.ok(!changeForEventType("model.text.delta").progress.includes("transcript"))
})

test("the Specialists page refreshes only cards whose agent shows new activity", () => {
  const before = agentActivity({
    sessions: [
      { agent_name: "a", updated_at: "2026-11-13T10:00:00Z" },
      { agent_name: "b", updated_at: "2026-11-13T09:00:00Z" },
      { agent_name: "a", updated_at: "2026-11-13T08:00:00Z" },
    ],
  })
  assert.equal(before.get("a"), "2026-11-13T10:00:00Z")
  const after = agentActivity({
    sessions: [
      { agent_name: "b", updated_at: "2026-11-13T10:05:00Z" },
      { agent_name: "a", updated_at: "2026-11-13T10:00:00Z" },
      { agent_name: "c", updated_at: "2026-11-13T07:00:00Z" },
    ],
  })
  assert.deepEqual(changedAgents(before, after).sort(), ["b", "c"])
})

test("the view on screen refreshes every 2 s and secondary views every 10 s", async () => {
  assert.equal(isSecondaryView("trace", false), false)
  assert.equal(isSecondaryView("transcript", false), true)
  assert.equal(isSecondaryView("transcript", true), false)
  assert.equal(isSecondaryView("trace", true), true)
  assert.equal(isSecondaryView("overview", false), true)
  assert.equal(isSecondaryView("session-children", true), true)
  let now = 0
  const timers = []
  const realSetTimeout = globalThis.setTimeout
  const realClearTimeout = globalThis.clearTimeout
  globalThis.setTimeout = (fn, ms) => {
    const timer = { fn, at: now + ms }
    timers.push(timer)
    return timer
  }
  globalThis.clearTimeout = (timer) => {
    const index = timers.indexOf(timer)
    if (index >= 0) timers.splice(index, 1)
  }
  try {
    const flushed = []
    const refresher = new SessionTickRefresher((views) => flushed.push(...views), {
      now: () => now,
    })
    // One tick a second for 20 s, in the Trace view.
    for (now = 0; now < 20_000; now += 1000) {
      for (const timer of timers.filter((t) => t.at <= now)) {
        timers.splice(timers.indexOf(timer), 1)
        timer.fn()
      }
      await refresher.tick(planForProgress(null), false)
    }
    const count = (view) => flushed.filter((v) => v === view).length
    assert.equal(count("trace"), 10)
    assert.equal(count("overview"), 10)
    assert.equal(count("transcript"), 2)
    assert.equal(count("session-children"), 2)
    assert.equal(count("state"), 2)
    assert.equal(count("queued-messages"), 10)
    await refresher.drain(SESSION_VIEWS)
    // The drain refreshes every view, and flushes the pending 10 s batch with it.
    assert.equal(count("state"), 4)
    assert.equal(count("session-messages"), 1)
    refresher.dispose()
    assert.equal(timers.length, 0)
  } finally {
    globalThis.setTimeout = realSetTimeout
    globalThis.clearTimeout = realClearTimeout
  }
})

test("held tasks keep their watchers polling; finished ones stop them", async () => {
  const { taskCanChange } = await import("../src/next/app/task-polling.ts")
  for (const status of ["paused", "blocked", "needs_attention", "running", "pending"])
    assert.equal(taskCanChange(status), true, status)
  for (const status of ["completed", "failed", "cancelled", "dependency_skipped"])
    assert.equal(taskCanChange(status), false, status)
})
