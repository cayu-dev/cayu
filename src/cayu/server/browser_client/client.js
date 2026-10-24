// @ts-self-types="./client.d.ts"
/**
 * Cayu browser client.
 *
 * A dependency-free ES module served by `mount_cayu` at `{path}/client.js`.
 * It follows sessions, watches session lists and the agent's inbox with
 * bounded, visibility-aware request behavior so custom UIs do not have to
 * hand-write polling loops, and settles inbox items.
 * `cayu guide app-ui` describes the rules this module implements.
 */

/** Version of this module's public API, advertised as `client.version` by `/api/contract`. */
export const CLIENT_VERSION = "1"
/** Exact server contract version this module was released with. */
export const CONTRACT_VERSION = "60"

const TERMINAL_STATUSES = new Set(["completed", "failed", "interrupted"])
const ACTIVE_STATUSES = new Set(["pending", "running", "interrupting"])
const TRUNCATION_ISSUES = new Set(["inbox_truncated", "knowledge_truncated"])
// Answers to an inbox page cursor that cannot be followed any more.
const CONTINUATION_REFUSALS = new Set([409, 422])
// How often one inbox refresh starts over after such a refusal before it fails.
const INBOX_MAX_RESTARTS = 1
const TERMINAL_EVENT_TYPES = new Set(["session.completed", "session.failed", "session.interrupted"])
const LIFECYCLE_EVENT_TYPES = new Set([
  ...TERMINAL_EVENT_TYPES,
  "session.started",
  "session.resumed",
])

/** Minimum spacing between `watchSessions` refreshes. It cannot be lowered. */
export const MIN_WATCH_INTERVAL_MS = 15_000

const STREAM_HEALTHY_AFTER_MS = 10_000

const DEFAULTS = Object.freeze({
  followPollIntervalMs: 2_000,
  followMaxPollIntervalMs: 30_000,
  watchIntervalMs: MIN_WATCH_INTERVAL_MS,
  watchMaxIntervalMs: 120_000,
  retryInitialDelayMs: 1_000,
  retryMaxDelayMs: 60_000,
  retryAfterMaxMs: 600_000,
  jitterRatio: 0.2,
  eventPageLimit: 200,
  watchPageLimit: 100,
  inboxPageLimit: 100,
  inboxMaxPages: 10,
  maxStreamFailures: 3,
})

/**
 * Error reported by the client. `kind` is one of `auth`, `contract`,
 * `not_found`, `unavailable`, `http`, `network`, or `protocol`. Only
 * `retryable` errors are retried; the others stop the operation.
 */
export class CayuClientError extends Error {
  constructor(
    kind,
    message,
    { status = null, retryable = false, retryAfterMs = null, cause } = {},
  ) {
    super(message, cause === undefined ? undefined : { cause })
    this.name = "CayuClientError"
    this.kind = kind
    this.status = status
    this.retryable = retryable
    this.retryAfterMs = retryAfterMs
  }
}

/**
 * Read `/api/contract`, check it against this module, and return a client.
 *
 * The API base defaults to `./api` next to this module, which is where
 * `mount_cayu` places it. Requests use same-origin credentials.
 */
export async function connect(options = {}) {
  const fetchImpl = options.fetch ?? ((input, init) => globalThis.fetch(input, init))
  const apiBase = trimSlash(String(options.apiBaseUrl ?? new URL("./api", import.meta.url)))
  const context = {
    apiBase,
    fetch: fetchImpl,
    headers: { ...(options.headers ?? {}) },
    credentials: options.credentials ?? "same-origin",
    contract: null,
    streamPath: null,
  }
  const contract = await requestJson(context, "/contract", { signal: options.signal })
  if (contract === null || typeof contract !== "object") {
    throw new CayuClientError("contract", "The server returned an invalid contract.")
  }
  if (contract.contract_version !== CONTRACT_VERSION) {
    throw new CayuClientError(
      "contract",
      `This client expects Cayu server contract v${CONTRACT_VERSION}, but the server reports ` +
        `v${String(contract.contract_version)}. Reload the page to load the server's client.js.`,
    )
  }
  const advertised = contract.client
  if (advertised && typeof advertised === "object" && advertised.version !== CLIENT_VERSION) {
    throw new CayuClientError(
      "contract",
      `The server advertises client.js v${String(advertised.version)}, but v${CLIENT_VERSION} ` +
        "is loaded. Reload the page to load the server's client.js.",
    )
  }
  context.contract = contract
  context.streamPath = sessionFollowStreamPath(contract)
  return new CayuClient(context)
}

class CayuClient {
  #context
  #usageCache = new Map()

  constructor(context) {
    this.#context = context
  }

  /** The `/api/contract` response this client was checked against. */
  get contract() {
    return this.#context.contract
  }

  /** Whether `followSession` will try the session follow stream first. */
  get streamAvailable() {
    return this.#context.streamPath !== null
  }

  followSession(sessionId, options = {}) {
    return followSession(this.#context, requireSessionId(sessionId), options)
  }

  watchSessions(filter = {}, options = {}) {
    return watchSessions(this.#context, filter ?? {}, options)
  }

  /** Whether the server serves the agent's inbox. */
  get inboxAvailable() {
    return inboxAvailable(this.#context.contract)
  }

  async listInbox(options = {}) {
    requireInbox(this.#context.contract)
    return readInboxPage(
      this.#context,
      options.cursor ?? null,
      options.limit,
      options.signal,
      options,
    )
  }

  async getInboxCounts(options = {}) {
    requireInbox(this.#context.contract)
    const params = inboxFilters(options, false)
    return requestJson(this.#context, `/inbox/counts?${params}`, {
      signal: options.signal,
    })
  }

  async getInboxItem(itemId, options = {}) {
    requireInbox(this.#context.contract)
    return requestJson(this.#context, inboxItemPath(itemId), { signal: options.signal })
  }

  /**
   * Settle one inbox item as the signed-in person. A 409 is an answer, not an
   * error: the item was settled, replaced or expired, and `item` says how.
   */
  async settleInboxItem(itemId, decision, options = {}) {
    requireInbox(this.#context.contract)
    if (decision === null || typeof decision !== "object") {
      throw new TypeError("decision must be the settle request body.")
    }
    const response = await send(this.#context, `${inboxItemPath(itemId)}/settle`, {
      method: "POST",
      body: decision,
      signal: options.signal,
    })
    if (response.status === 409) {
      const detail = (await readJson(response))?.detail
      if (!detail || typeof detail.code !== "string") {
        throw new CayuClientError("contract", "The server returned an invalid inbox conflict.", {
          status: 409,
        })
      }
      return {
        settled: false,
        code: detail.code,
        message: String(detail.message ?? ""),
        item: detail.item ?? null,
      }
    }
    if (!response.ok) throw await httpError(response)
    const body = await readJson(response)
    if (!body || typeof body.item !== "object") {
      throw new CayuClientError("contract", "The server returned an invalid settlement.")
    }
    return { settled: true, replayed: body.replayed === true, item: body.item }
  }

  watchInbox(options = {}) {
    requireInbox(this.#context.contract)
    return watchInbox(this.#context, options)
  }

  async getSessionUsage(sessionId, options = {}) {
    const id = requireSessionId(sessionId)
    const usage = this.#context.contract?.capabilities?.surfaces?.usage
    if (usage?.read?.enabled === false) {
      throw new CayuClientError(
        "unavailable",
        `Session usage is unavailable: ${String(usage.read.unavailable_reason ?? "disabled")}.`,
      )
    }
    const cached = this.#usageCache.get(id)
    const headers = cached ? { "If-None-Match": cached.etag } : {}
    const response = await send(this.#context, `/sessions/${encodeURIComponent(id)}/usage`, {
      signal: options.signal,
      headers,
    })
    if (response.status === 304 && cached) return cached.body
    if (!response.ok) throw await httpError(response)
    const body = await readJson(response)
    const etag = response.headers.get("ETag")
    if (etag) {
      this.#usageCache.delete(id)
      this.#usageCache.set(id, { etag, body })
      if (this.#usageCache.size > 100) this.#usageCache.delete(this.#usageCache.keys().next().value)
    }
    return body
  }
}

function followSession(context, sessionId, options) {
  const excluded = new Set(options.excludeEventTypes ?? [])
  // Lifecycle events are always fetched so the client can see a session end;
  // excluded lifecycle types are filtered here instead of on the server.
  const serverExclude = [...excluded].find((type) => !LIFECYCLE_EVENT_TYPES.has(type)) ?? null
  const basePollMs = positive(options.pollIntervalMs, DEFAULTS.followPollIntervalMs)
  const maxPollMs = Math.max(basePollMs, DEFAULTS.followMaxPollIntervalMs)
  const runner = new Runner(options.signal)
  const state = {
    lastSequence: normalizeSequence(options.afterSequence),
    status: null,
    transport: null,
    lastError: null,
  }

  const setStatus = (status) => {
    if (typeof status !== "string") {
      throw new CayuClientError("contract", "The server returned a session without a status.")
    }
    if (status === state.status) return
    state.status = status
    invoke(options.onStatus, status)
  }

  const acceptEvent = (event) => {
    if (state.lastSequence !== null && event.sequence <= state.lastSequence) return false
    state.lastSequence = event.sequence
    if (!excluded.has(event.type)) invoke(options.onEvent, event)
    return true
  }

  const readStatus = async () => {
    const session = await runner.run((signal) =>
      requestJson(context, `/sessions/${encodeURIComponent(sessionId)}`, { signal }),
    )
    setStatus(session?.status)
  }

  const pollPage = async () => {
    const params = new URLSearchParams({ limit: String(DEFAULTS.eventPageLimit) })
    if (state.lastSequence !== null) params.set("after_sequence", String(state.lastSequence))
    if (serverExclude !== null) params.set("exclude_event_type", serverExclude)
    const page = await runner.run((signal) =>
      requestJson(context, `/sessions/${encodeURIComponent(sessionId)}/events?${params}`, {
        signal,
      }),
    )
    if (!page || !Array.isArray(page.events)) {
      throw new CayuClientError("contract", "The server returned an invalid event page.")
    }
    let received = false
    let lifecycle = null
    for (const record of page.events) {
      if (!record || !Number.isSafeInteger(record.sequence) || typeof record.type !== "string") {
        throw new CayuClientError("contract", "The server returned an invalid event record.")
      }
      if (acceptEvent(record)) {
        received = true
        if (LIFECYCLE_EVENT_TYPES.has(record.type)) lifecycle = record.type
      }
    }
    const scanned = page.scan_through_sequence
    if (
      Number.isSafeInteger(scanned) &&
      (state.lastSequence === null || scanned > state.lastSequence)
    ) {
      state.lastSequence = scanned
    }
    return { hasMore: page.has_more === true, received, lifecycle }
  }

  const readStream = async () => {
    let lifecycle = null
    let receivedEvent = false
    let openedAt = null
    const outcome = await runner.run(async (signal) => {
      // Resume with the start query rather than Last-Event-ID: the stream
      // rejects markers that do not name an existing event, and lastSequence
      // may come from afterSequence or scan_through_sequence.
      const params = new URLSearchParams()
      if (state.lastSequence !== null) params.set("after_sequence", String(state.lastSequence))
      if (serverExclude !== null) params.set("exclude_event_type", serverExclude)
      const query = params.toString() ? `?${params}` : ""
      const path = context.streamPath.replace("{session_id}", encodeURIComponent(sessionId))
      const response = await send(context, `${path}${query}`, {
        signal,
        headers: { Accept: "text/event-stream" },
      })
      if (!response.ok) throw await httpError(response)
      if (!response.body) throw new CayuClientError("protocol", "The follow stream has no body.")
      openedAt = Date.now()
      return readSse(response.body, (frame) => {
        if (frame.event === "end") {
          const data = parseJson(frame.data)
          return { type: "end", status: data?.status }
        }
        if (frame.event === "error") {
          const data = parseJson(frame.data)
          return {
            type: "error",
            error: new CayuClientError("protocol", String(data?.error ?? "The stream failed."), {
              retryable: data?.retryable === true,
            }),
          }
        }
        const sequence = sequenceFromEventId(frame.id)
        // Event frames carry a `session_id:cayu_event_<sequence>` id; other
        // named frames without one are ignored.
        if (frame.event !== "message" && sequence === null) return null
        const envelope = parseJson(frame.data)
        if (sequence === null || !envelope || typeof envelope.type !== "string") {
          return {
            type: "error",
            error: new CayuClientError("protocol", "The follow stream sent an invalid frame."),
          }
        }
        receivedEvent = true
        if (acceptEvent({ ...envelope, sequence }) && LIFECYCLE_EVENT_TYPES.has(envelope.type)) {
          lifecycle = envelope.type
        }
        return null
      })
    })
    // A stream that delivered events, or stayed open on heartbeats, was healthy
    // even if the server later closed it.
    const healthy =
      receivedEvent || (openedAt !== null && Date.now() - openedAt >= STREAM_HEALTHY_AFTER_MS)
    return { ...(outcome ?? { type: "closed" }), lifecycle, healthy }
  }

  const done = (async () => {
    let streamUsable = context.streamPath !== null
    let streamFailures = 0
    let catchUp = false
    let statusCheck = true
    let terminalSeen = false
    let attempt = 0
    let idlePolls = 0
    let lastRequestAt = Number.NEGATIVE_INFINITY
    let dueAt = 0
    let retryAt = 0
    let hideCount = runner.hideCount

    const noteLifecycle = (type) => {
      if (type === null) return
      statusCheck = true
      terminalSeen = TERMINAL_EVENT_TYPES.has(type)
    }
    const scheduleRetry = (failure) => {
      attempt += 1
      retryAt = Date.now() + Math.max(backoffDelay(attempt), failure?.retryAfterMs ?? 0)
      dueAt = retryAt
    }

    for (;;) {
      const scheduledHideCount = hideCount
      // After the page was hidden, resume promptly from the last sequence, but
      // never sooner than a pending retry or one poll interval after the last request.
      const ready = await runner.waitUntil(() =>
        runner.hideCount === scheduledHideCount
          ? dueAt
          : Math.max(retryAt, lastRequestAt + basePollMs),
      )
      if (!ready) return finish("aborted")
      hideCount = runner.hideCount
      let phase = "status"
      try {
        if (statusCheck) {
          lastRequestAt = Date.now()
          await readStatus()
          // A terminal event can become visible just before the status does;
          // keep checking at the poll cadence until they agree.
          statusCheck = terminalSeen && !TERMINAL_STATUSES.has(state.status)
        }
        if (TERMINAL_STATUSES.has(state.status)) {
          phase = "poll"
          state.transport = "poll"
          lastRequestAt = Date.now()
          const page = await pollPage()
          attempt = 0
          if (!page.hasMore) return finish("terminal")
          dueAt = 0
          continue
        }
        if (streamUsable && !catchUp) {
          phase = "stream"
          state.transport = "stream"
          lastRequestAt = Date.now()
          const outcome = await readStream()
          noteLifecycle(outcome.lifecycle)
          if (outcome.healthy) {
            attempt = 0
            streamFailures = 0
          } else if (outcome.type !== "error") {
            streamFailures += 1
            if (streamFailures >= DEFAULTS.maxStreamFailures) streamUsable = false
          }
          if (outcome.type === "error") throw outcome.error
          if (outcome.type === "end") {
            setStatus(outcome.status)
            statusCheck = false
            if (TERMINAL_STATUSES.has(state.status)) return finish("terminal")
          }
          // The stream closed before the session ended: reconnect with backoff.
          scheduleRetry(null)
          continue
        }
        phase = "poll"
        state.transport = "poll"
        lastRequestAt = Date.now()
        const page = await pollPage()
        attempt = 0
        noteLifecycle(page.lifecycle)
        if (page.hasMore || page.lifecycle !== null) {
          dueAt = 0
          continue
        }
        if (catchUp) {
          // Caught up past what the stream could not send; stream again.
          catchUp = false
          if (streamUsable) {
            dueAt = 0
            continue
          }
        }
        // Back off while nothing new arrives: base, 2x, 4x ... up to the ceiling.
        if (page.received) idlePolls = 0
        dueAt = lastRequestAt + Math.min(maxPollMs, basePollMs * 2 ** idlePolls)
        if (!page.received) idlePolls += 1
      } catch (error) {
        if (isAbortError(error)) {
          if (runner.stopped) return finish("aborted")
          continue
        }
        const failure = toClientError(error)
        state.lastError = failure
        if (failure.kind === "auth") return fail(failure)
        if (phase === "stream") {
          streamFailures += 1
          if (!failure.retryable || [404, 405, 501].includes(failure.status)) {
            // The advertised stream is not usable here. Polling stays correct,
            // so catch up by polling instead of stopping.
            catchUp = true
            if (failure.status !== null || streamFailures >= DEFAULTS.maxStreamFailures) {
              streamUsable = false
            }
            dueAt = 0
            continue
          }
          if (streamFailures >= DEFAULTS.maxStreamFailures) streamUsable = false
        } else if (!failure.retryable) {
          return fail(failure)
        }
        scheduleRetry(failure)
      }
    }
  })()

  function finish(reason) {
    runner.stop()
    return { reason, status: state.status, lastSequence: state.lastSequence }
  }

  function fail(error) {
    runner.stop()
    invoke(options.onError, error)
    throw error
  }

  return {
    done,
    stop: () => runner.stop(),
    get lastSequence() {
      return state.lastSequence
    },
    get status() {
      return state.status
    },
    get transport() {
      return state.transport
    },
    get lastError() {
      return state.lastError
    },
  }
}

function watchSessions(context, filter, options) {
  const statuses =
    filter.status === undefined || filter.status === null
      ? [null]
      : [...new Set(Array.isArray(filter.status) ? filter.status : [filter.status])]

  return startWatch(options, async (runner) => {
    const byId = new Map()
    for (const status of statuses) {
      const params = new URLSearchParams({
        limit: String(positive(filter.limit, DEFAULTS.watchPageLimit)),
      })
      if (status !== null) params.set("status", status)
      if (filter.agentName) params.set("agent_name", filter.agentName)
      if (filter.environmentName) params.set("environment_name", filter.environmentName)
      if (filter.parentSessionId) params.set("parent_session_id", filter.parentSessionId)
      for (const label of filter.labels ?? []) params.append("label", label)
      const page = await runner.run((signal) =>
        requestJson(context, `/sessions?${params}`, { signal }),
      )
      if (!page || !Array.isArray(page.sessions)) {
        throw new CayuClientError("contract", "The server returned an invalid session list.")
      }
      for (const session of page.sessions) byId.set(session.id, session)
    }
    const sessions = [...byId.values()].sort((left, right) =>
      String(right.updated_at).localeCompare(String(left.updated_at)),
    )
    return {
      value: sessions,
      key: JSON.stringify(sessions.map((s) => [s.id, s.status, s.updated_at])),
      idle: !sessions.some((session) => ACTIVE_STATUSES.has(session.status)),
    }
  })
}

function watchInbox(context, options) {
  // Something can start waiting on a person at any time, so the inbox watch never
  // stops as idle; it backs off while nothing changes and pauses while hidden.
  return startWatch(options, async (runner) => {
    let inbox = { items: [], issues: [], complete: false }
    let cursor = null
    let restarts = 0
    for (let page = 0; page < DEFAULTS.inboxMaxPages; page += 1) {
      let body
      try {
        body = await runner.run((signal) =>
          readInboxPage(context, cursor, undefined, signal, options),
        )
      } catch (error) {
        // A later page the server can no longer continue (for example it restarted
        // and its cursor key changed) starts this refresh over from the first page,
        // dropping what it read so one snapshot never holds an item twice. A second
        // refusal means the server can't follow its own cursors (such as workers
        // without a shared inbox_cursor_key), so the watch stops with that error.
        if (cursor === null || !CONTINUATION_REFUSALS.has(error?.status)) throw error
        if (restarts >= INBOX_MAX_RESTARTS) throw error
        restarts += 1
        inbox = { items: [], issues: [], complete: false }
        cursor = null
        page = -1
        continue
      }
      inbox.items.push(...body.items)
      inbox.issues.push(...body.issues)
      if (typeof body.next_cursor !== "string") {
        // A truncation issue means items were waiting that no page lists.
        inbox.complete = !inbox.issues.some((issue) => TRUNCATION_ISSUES.has(issue.code))
        break
      }
      cursor = body.next_cursor
    }
    // An item id names one exact pause or proposal revision, so ids alone say
    // whether anything changed.
    const key = JSON.stringify([inbox.items.map((item) => item.id), inbox.issues, inbox.complete])
    return { value: inbox, key, idle: false }
  })
}

/**
 * Refresh on a 15 s floor, doubling the interval while `read` reports no change,
 * and stop when it reports idle, on a non-retryable error, or when stopped.
 */
function startWatch(options, read) {
  const base = Math.max(
    MIN_WATCH_INTERVAL_MS,
    positive(options.intervalMs, DEFAULTS.watchIntervalMs),
  )
  const maxInterval = Math.max(base, positive(options.maxIntervalMs, DEFAULTS.watchMaxIntervalMs))
  const runner = new Runner(options.signal)
  let stopReason = null

  const stop = (reason = "stopped") => {
    if (runner.stopped) return
    stopReason = reason
    runner.stop()
  }

  void (async () => {
    let interval = base
    let attempt = 0
    let dueAt = 0
    let fingerprint = null
    for (;;) {
      if (!(await runner.waitUntil(() => dueAt))) break
      const startedAt = Date.now()
      try {
        const { value, key, idle } = await read(runner)
        attempt = 0
        if (key !== fingerprint) {
          fingerprint = key
          interval = base
          invoke(options.onChange, value)
        } else {
          interval = Math.min(maxInterval, interval * 2)
        }
        if (idle) {
          stop("idle")
          break
        }
        dueAt = startedAt + interval
      } catch (error) {
        if (isAbortError(error)) {
          if (runner.stopped) break
          // Hidden mid-refresh: the aborted request still counts toward spacing.
          dueAt = startedAt + base
          continue
        }
        const failure = toClientError(error)
        if (!failure.retryable) {
          invoke(options.onError, failure)
          stop("error")
          break
        }
        attempt += 1
        dueAt = Date.now() + Math.max(base, backoffDelay(attempt), failure.retryAfterMs ?? 0)
      }
    }
    invoke(options.onStop, stopReason ?? "stopped")
  })()

  return () => stop("stopped")
}

function inboxFilters(options, includeKind = true) {
  const params = new URLSearchParams()
  for (const key of includeKind
    ? ["session_id", "agent_name", "kind", "q"]
    : ["session_id", "agent_name", "q"]) {
    if (options[key] !== undefined) params.set(key, options[key])
  }
  return params
}

async function readInboxPage(context, cursor, limit, signal, options = {}) {
  const params = inboxFilters(options)
  params.set("limit", String(positive(limit, DEFAULTS.inboxPageLimit)))
  if (cursor !== null) params.set("cursor", cursor)
  const page = await requestJson(context, `/inbox?${params}`, { signal })
  if (!page || !Array.isArray(page.items) || !Array.isArray(page.issues)) {
    throw new CayuClientError("contract", "The server returned an invalid inbox page.")
  }
  return page
}

function inboxAvailable(contract) {
  const inbox = contract?.capabilities?.surfaces?.inbox
  return inbox?.read?.enabled === true
}

function requireInbox(contract) {
  if (!inboxAvailable(contract)) {
    throw new CayuClientError(
      "unavailable",
      "This server has no inbox. It needs a Cayu release that serves /api/inbox.",
    )
  }
}

function inboxItemPath(itemId) {
  if (typeof itemId !== "string" || itemId.trim() === "") {
    throw new TypeError("itemId must be a non-empty string.")
  }
  return `/inbox/${encodeURIComponent(itemId)}`
}

/**
 * Owns the visibility subscription, the in-flight request, and delays for one
 * follow or watch. Hiding the page aborts the in-flight request or delay, and
 * nothing is sent until the page is visible again.
 */
class Runner {
  constructor(signal) {
    this.stopped = false
    this.hideCount = 0
    this.active = null
    this.waiters = new Set()
    this.document = globalThis.document
    this.onVisibility = () => {
      if (this.hidden()) {
        this.hideCount += 1
        this.active?.abort()
      } else {
        this.wake()
      }
    }
    this.document?.addEventListener?.("visibilitychange", this.onVisibility)
    this.signal = signal ?? null
    this.onAbort = () => this.stop()
    if (this.signal?.aborted) this.stop()
    else this.signal?.addEventListener("abort", this.onAbort, { once: true })
  }

  hidden() {
    return this.document?.hidden === true
  }

  wake() {
    const waiters = [...this.waiters]
    this.waiters.clear()
    for (const resolve of waiters) resolve()
  }

  stop() {
    if (this.stopped) return
    this.stopped = true
    this.active?.abort()
    this.document?.removeEventListener?.("visibilitychange", this.onVisibility)
    this.signal?.removeEventListener("abort", this.onAbort)
    this.wake()
  }

  async untilVisible() {
    while (!this.stopped && this.hidden()) {
      await new Promise((resolve) => this.waiters.add(resolve))
    }
    return !this.stopped
  }

  async run(operation) {
    const controller = new AbortController()
    this.active = controller
    if (this.stopped || this.hidden()) controller.abort()
    try {
      if (controller.signal.aborted) throw abortError()
      return await operation(controller.signal)
    } finally {
      if (this.active === controller) this.active = null
    }
  }

  /** Wait until visible and `dueAt()` has passed. Returns false once stopped. */
  async waitUntil(dueAt) {
    for (;;) {
      if (!(await this.untilVisible())) return false
      const remaining = dueAt() - Date.now()
      if (remaining <= 0) return !this.stopped
      try {
        await this.run((signal) => delay(remaining, signal))
      } catch (error) {
        if (!isAbortError(error)) throw error
        if (this.stopped) return false
      }
    }
  }
}

async function readSse(body, onFrame) {
  const reader = body.getReader()
  // Streaming decode keeps a multi-byte character split across reads intact.
  const decoder = new TextDecoder()
  let buffer = ""
  // CR, LF, and CRLF each end one line. A CR that ends one read has already
  // ended its line, so an LF that starts the next read belongs to it.
  let pendingCR = false
  let frame = { event: "", data: [], id: null }
  try {
    for (;;) {
      const { value, done } = await reader.read()
      let text = done ? decoder.decode() : decoder.decode(value, { stream: true })
      if (text !== "") {
        if (pendingCR && text.startsWith("\n")) text = text.slice(1)
        pendingCR = text.endsWith("\r")
      }
      buffer += text
      const lines = buffer.split(/\r\n|\r|\n/)
      // The last piece has no line ending yet. At the end of the stream it is
      // an incomplete line, which the SSE format discards.
      buffer = lines.pop() ?? ""
      for (const line of lines) {
        if (line === "") {
          if (frame.data.length > 0 || frame.event !== "") {
            const result = onFrame({
              event: frame.event || "message",
              data: frame.data.join("\n"),
              id: frame.id,
            })
            if (result) return result
          }
          frame = { event: "", data: [], id: null }
          continue
        }
        if (line.startsWith(":")) continue
        const colon = line.indexOf(":")
        const field = colon === -1 ? line : line.slice(0, colon)
        let value = colon === -1 ? "" : line.slice(colon + 1)
        if (value.startsWith(" ")) value = value.slice(1)
        if (field === "event") frame.event = value
        else if (field === "data") frame.data.push(value)
        else if (field === "id") frame.id = value
      }
      if (done) return null
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}

function sequenceFromEventId(id) {
  if (typeof id !== "string") return null
  const marker = ":cayu_event_"
  const index = id.lastIndexOf(marker)
  if (index <= 0) return null
  const digits = id.slice(index + marker.length)
  if (!/^\d+$/.test(digits)) return null
  const sequence = Number(digits)
  return Number.isSafeInteger(sequence) ? sequence : null
}

function sessionFollowStreamPath(contract) {
  const advertised = contract?.sse?.session_follow
  if (!advertised) return null
  const capability = contract?.capabilities?.surfaces?.session_follow
  if (capability?.read?.enabled === false) return null
  // `path_template` includes the API prefix, e.g. `/api/sessions/{session_id}/events/stream`.
  const template = typeof advertised.path_template === "string" ? advertised.path_template : ""
  const prefix = typeof contract.api_prefix === "string" ? contract.api_prefix : ""
  if (prefix && template.startsWith(`${prefix}/`) && template.includes("{session_id}")) {
    return template.slice(prefix.length)
  }
  return "/sessions/{session_id}/events/stream"
}

async function send(context, path, { signal, headers = {}, method = "GET", body } = {}) {
  const init = {
    method,
    credentials: context.credentials,
    headers: { Accept: "application/json", ...context.headers, ...headers },
    signal,
  }
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json"
    init.body = JSON.stringify(body)
  }
  try {
    return await context.fetch(`${context.apiBase}${path}`, init)
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new CayuClientError("network", "The request did not reach the server.", {
      retryable: true,
      cause: error,
    })
  }
}

async function requestJson(context, path, options) {
  const response = await send(context, path, options)
  if (!response.ok) throw await httpError(response)
  return readJson(response)
}

async function readJson(response) {
  try {
    return await response.json()
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new CayuClientError("protocol", "The server returned invalid JSON.", {
      status: response.status,
      retryable: true,
      cause: error,
    })
  }
}

async function httpError(response) {
  const status = response.status
  let detail = ""
  try {
    const body = await response.json()
    if (typeof body?.detail === "string") detail = body.detail.slice(0, 500)
  } catch {
    // The status code is enough when the body is not JSON.
  }
  const suffix = detail ? `: ${detail}` : "."
  if (status === 401 || status === 403) {
    return new CayuClientError("auth", `The server rejected the request (${status})${suffix}`, {
      status,
    })
  }
  if (status === 404) {
    return new CayuClientError("not_found", `Not found (404)${suffix}`, {
      status,
    })
  }
  const retryable = status === 408 || status === 425 || status === 429 || status >= 500
  return new CayuClientError("http", `The server responded ${status}${suffix}`, {
    status,
    retryable,
    retryAfterMs: retryable ? retryAfterMs(response.headers.get("Retry-After")) : null,
  })
}

function retryAfterMs(value) {
  if (!value) return null
  const trimmed = value.trim()
  let milliseconds = null
  if (/^\d+$/.test(trimmed)) milliseconds = Number(trimmed) * 1000
  else {
    const date = Date.parse(trimmed)
    if (!Number.isNaN(date)) milliseconds = Math.max(0, date - Date.now())
  }
  return milliseconds === null ? null : Math.min(milliseconds, DEFAULTS.retryAfterMaxMs)
}

function backoffDelay(attempt) {
  const exponential = DEFAULTS.retryInitialDelayMs * 2 ** Math.max(0, attempt - 1)
  const capped = Math.min(DEFAULTS.retryMaxDelayMs, exponential)
  const jitter = 1 + (Math.random() * 2 - 1) * DEFAULTS.jitterRatio
  return Math.round(capped * jitter)
}

function toClientError(error) {
  if (error instanceof CayuClientError) return error
  return new CayuClientError("protocol", error instanceof Error ? error.message : String(error), {
    retryable: true,
    cause: error,
  })
}

function delay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort)
      resolve()
    }, milliseconds)
    const onAbort = () => {
      clearTimeout(timer)
      reject(abortError())
    }
    signal.addEventListener("abort", onAbort, { once: true })
  })
}

function invoke(callback, value) {
  if (typeof callback !== "function") return
  try {
    callback(value)
  } catch (error) {
    // A failing UI callback must not stop the follow loop; surface it the way
    // browsers report listener errors.
    if (typeof globalThis.reportError === "function") globalThis.reportError(error)
    else
      queueMicrotask(() => {
        throw error
      })
  }
}

function abortError() {
  return new DOMException("The operation was aborted.", "AbortError")
}

function isAbortError(error) {
  return error instanceof Error && error.name === "AbortError"
}

function parseJson(text) {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function normalizeSequence(value) {
  if (value === undefined || value === null) return null
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError("afterSequence must be a non-negative integer.")
  }
  return value
}

function requireSessionId(sessionId) {
  if (typeof sessionId !== "string" || sessionId.trim() === "") {
    throw new TypeError("sessionId must be a non-empty string.")
  }
  return sessionId
}

function positive(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function trimSlash(value) {
  return value.endsWith("/") ? value.slice(0, -1) : value
}
