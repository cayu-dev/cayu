// This client is deliberately separate from the shared query/mutation caches.
export type BrowserIdentity = {
  session_id: string
  session_instance_id: string
  run_epoch: number
  interaction_id: string
  execution_profile_fingerprint: string
  environment_name: string
  allocation_fingerprint: string
  browser_session_id: string
  worker_instance_id: string
  profile_checkpoint_policy: "unavailable" | "disabled" | "on_close" | "after_terminal_operation"
  operator_purpose: { code: string; expected_origins: string[] }
}

export type BrowserCheckpointConsent = "undecided" | "allow" | "deny"

export type BrowserDescriptor = {
  identity: BrowserIdentity
  revision: number
  control_epoch: number
  state: string
  sensitive_entry: boolean
  sensitive_entry_pending: boolean
  fresh_observation_required: boolean
  owned_request: {
    request_id: string
    expires_at_ms: number
    maximum_until_ms: number
    lease_until_ms: number | null
    pending_lease_until_ms: number | null
    settled_input_sequence: number
    pending_input_sequence: number | null
    checkpoint_consent: BrowserCheckpointConsent
  } | null
}

export type BrowserPage = { page_id: string; revision: string; control_epoch: number }
export type BrowserPageLocation = { page: BrowserPage; origin: string | null }
export type BrowserInputKind = "text" | "tab" | "backtab" | "enter" | "escape" | "backspace"

function unavailable(): never {
  throw new Error("Protected browser control is unavailable. Refresh discovery before retrying.")
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) unavailable()
  return value as Record<string, unknown>
}

function text(value: unknown): string {
  if (
    typeof value !== "string" ||
    !value ||
    value.length > 512 ||
    Array.from(value).some((character) => {
      const point = character.codePointAt(0) ?? 0
      return point < 32 || (point >= 0xd800 && point <= 0xdfff)
    })
  )
    unavailable()
  return value
}

function counter(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) unavailable()
  return value
}

function flag(value: unknown): boolean {
  if (typeof value !== "boolean") unavailable()
  return value
}

function descriptor(value: unknown, sessionId: string): BrowserDescriptor {
  const record = object(value)
  const identity = object(record.identity)
  const checkpointPolicy = identity.profile_checkpoint_policy
  const purpose = object(identity.operator_purpose)
  const purposeCode = text(purpose.code)
  if (
    !/^[a-z][a-z0-9_]{0,63}$/.test(purposeCode) ||
    !Array.isArray(purpose.expected_origins) ||
    purpose.expected_origins.length < 1 ||
    purpose.expected_origins.length > 32
  )
    unavailable()
  const expectedOrigins = purpose.expected_origins.map((value) => {
    const origin = text(value)
    try {
      const url = new URL(origin)
      if (url.protocol !== "https:" || url.origin !== origin || url.port) unavailable()
    } catch {
      unavailable()
    }
    return origin
  })
  if (
    new Set(expectedOrigins).size !== expectedOrigins.length ||
    [...expectedOrigins].sort().some((value, index) => value !== expectedOrigins[index])
  )
    unavailable()
  if (
    checkpointPolicy !== "unavailable" &&
    checkpointPolicy !== "disabled" &&
    checkpointPolicy !== "on_close" &&
    checkpointPolicy !== "after_terminal_operation"
  )
    unavailable()
  const owned: BrowserIdentity = {
    session_id: text(identity.session_id),
    session_instance_id: text(identity.session_instance_id),
    run_epoch: counter(identity.run_epoch),
    interaction_id: text(identity.interaction_id),
    execution_profile_fingerprint: text(identity.execution_profile_fingerprint),
    environment_name: text(identity.environment_name),
    allocation_fingerprint: text(identity.allocation_fingerprint),
    browser_session_id: text(identity.browser_session_id),
    worker_instance_id: text(identity.worker_instance_id),
    profile_checkpoint_policy: checkpointPolicy,
    operator_purpose: { code: purposeCode, expected_origins: expectedOrigins },
  }
  if (owned.session_id !== sessionId) unavailable()
  let own: BrowserDescriptor["owned_request"] = null
  if (record.owned_request !== null) {
    const request = object(record.owned_request)
    const requestId = text(request.request_id)
    const consent = request.checkpoint_consent
    if (
      !/^bt_[a-f0-9]{32}$/.test(requestId) ||
      (consent !== "undecided" && consent !== "allow" && consent !== "deny")
    )
      unavailable()
    own = {
      request_id: requestId,
      expires_at_ms: counter(request.expires_at_ms),
      maximum_until_ms: counter(request.maximum_until_ms),
      lease_until_ms: request.lease_until_ms === null ? null : counter(request.lease_until_ms),
      pending_lease_until_ms:
        request.pending_lease_until_ms === null ? null : counter(request.pending_lease_until_ms),
      settled_input_sequence: counter(request.settled_input_sequence),
      pending_input_sequence:
        request.pending_input_sequence === null ? null : counter(request.pending_input_sequence),
      checkpoint_consent: consent as "undecided" | "allow" | "deny",
    }
  }
  return {
    identity: owned,
    revision: counter(record.revision),
    control_epoch: counter(record.control_epoch),
    state: text(record.state),
    sensitive_entry: flag(record.sensitive_entry),
    sensitive_entry_pending: flag(record.sensitive_entry_pending),
    fresh_observation_required: flag(record.fresh_observation_required),
    owned_request: own,
  }
}

export function createBrowserControlClient(
  root: URL,
  sessionId: string,
  fetcher: typeof fetch = fetch,
) {
  if (root.protocol !== "https:" || root.username || root.password || root.search || root.hash)
    unavailable()
  const base = new URL(`${root.href.replace(/\/$/, "")}/`)
  const lifetime = new AbortController()
  let token = ""
  let closed = false

  async function request(path: string, body?: unknown): Promise<Record<string, unknown>> {
    if (closed) unavailable()
    try {
      const response = await fetcher(new URL(path, base), {
        method: body === undefined ? "GET" : "POST",
        credentials: "same-origin",
        cache: "no-store",
        redirect: "error",
        referrerPolicy: "no-referrer",
        signal: AbortSignal.any([lifetime.signal, AbortSignal.timeout(10_000)]),
        headers: {
          Accept: "application/json",
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          ...(token ? { "X-Cayu-Browser-Operator": token } : {}),
        },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      })
      if (!response.ok || !response.body) {
        await response.body?.cancel()
        unavailable()
      }
      const reader = response.body.getReader()
      const chunks: Uint8Array[] = []
      let size = 0
      try {
        for (;;) {
          const part = await reader.read()
          if (part.done) break
          size += part.value.byteLength
          if (size > 64 * 1024) unavailable()
          chunks.push(part.value)
        }
      } finally {
        await reader.cancel()
        reader.releaseLock()
      }
      if (closed) unavailable()
      const bytes = new Uint8Array(size)
      let offset = 0
      for (const chunk of chunks) {
        bytes.set(chunk, offset)
        offset += chunk.length
      }
      return object(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)))
    } catch {
      // Never render server details or transport exceptions containing tokens.
      unavailable()
    }
  }

  function ownerIntent(browser: BrowserDescriptor) {
    if (!browser.owned_request) unavailable()
    return {
      identity: browser.identity,
      expected_record_revision: browser.revision,
      expected_control_epoch: browser.control_epoch,
      request_id: browser.owned_request.request_id,
    }
  }

  return {
    async discover(): Promise<BrowserDescriptor[]> {
      if (!token) token = text((await request("operator-session", {})).operator_session_token)
      const result = await request(`sessions/${encodeURIComponent(sessionId)}`)
      if (!Array.isArray(result.browsers) || result.browsers.length > 32) unavailable()
      return result.browsers.map((value) => descriptor(value, sessionId))
    },
    async pages(browser: BrowserDescriptor): Promise<{
      pages: BrowserPage[]
      locations: BrowserPageLocation[]
      active_page_id: string | null
    }> {
      const result = await request("pages", {
        identity: browser.identity,
        expected_record_revision: browser.revision,
      })
      if (!Array.isArray(result.pages) || result.pages.length > 16) unavailable()
      const pages = result.pages.map((value) => {
        const page = object(value)
        return {
          page_id: text(page.page_id),
          revision: text(page.revision),
          control_epoch: counter(page.control_epoch),
        }
      })
      const active = result.active_page_id === null ? null : text(result.active_page_id)
      if (
        new Set(pages.map((page) => page.page_id)).size !== pages.length ||
        (active !== null && !pages.some((page) => page.page_id === active))
      )
        unavailable()
      if (!Array.isArray(result.locations) || result.locations.length !== pages.length)
        unavailable()
      const locations = result.locations.map((value, index) => {
        const location = object(value)
        const observed = object(location.page)
        const page = pages[index]
        if (
          !page ||
          observed.page_id !== page.page_id ||
          observed.revision !== page.revision ||
          observed.control_epoch !== page.control_epoch
        )
          unavailable()
        const origin = location.origin === null ? null : text(location.origin)
        if (origin !== null) {
          try {
            const url = new URL(origin)
            if (url.protocol !== "https:" || url.origin !== origin || url.port) unavailable()
          } catch {
            unavailable()
          }
        }
        return { page, origin }
      })
      return { pages, locations, active_page_id: active }
    },
    async takeover(
      browser: BrowserDescriptor,
      pages: BrowserPage[],
      checkpointConsent: BrowserCheckpointConsent,
    ): Promise<void> {
      if (!["undecided", "allow", "deny"].includes(checkpointConsent)) unavailable()
      if (browser.state !== "agent_controlled" || pages.length === 0) unavailable()
      const now = Date.now()
      // One explicit request, never an automatic retry with renewed authority.
      await request("takeover", {
        identity: browser.identity,
        expected_record_revision: browser.revision,
        expected_control_epoch: browser.control_epoch,
        request_id: `bt_${crypto.randomUUID().replaceAll("-", "")}`,
        pages: [...pages].sort((a, b) =>
          a.page_id < b.page_id ? -1 : a.page_id > b.page_id ? 1 : 0,
        ),
        purpose_code: browser.identity.operator_purpose.code,
        requested_at_ms: now,
        expires_at_ms: now + 30_000,
        maximum_until_ms: now + 15 * 60_000,
        checkpoint_consent: checkpointConsent,
      })
    },
    async handback(browser: BrowserDescriptor): Promise<void> {
      await request("handback", ownerIntent(browser))
    },
    async renew(browser: BrowserDescriptor): Promise<void> {
      const own = browser.owned_request
      const now = Date.now()
      if (
        !own ||
        browser.state !== "operator_controlled" ||
        own.lease_until_ms === null ||
        own.pending_lease_until_ms !== null ||
        own.lease_until_ms <= now
      )
        unavailable()
      const until = Math.min(now + 45_000, own.maximum_until_ms)
      if (until <= own.lease_until_ms) unavailable()
      await request("renew", {
        ...ownerIntent(browser),
        expected_lease_until_ms: own.lease_until_ms,
        lease_until_ms: until,
      })
    },
    async sensitiveEntry(browser: BrowserDescriptor): Promise<void> {
      await request("sensitive-entry", ownerIntent(browser))
    },
    async viewerTicket(browser: BrowserDescriptor, page: BrowserPage): Promise<string> {
      return text(
        (
          await request("view-ticket", {
            identity: browser.identity,
            expected_record_revision: browser.revision,
            page,
          })
        ).ticket,
      )
    },
    async inputTicket(
      browser: BrowserDescriptor,
      page: BrowserPage,
      kind: BrowserInputKind = "text",
    ) {
      const own = browser.owned_request
      if (
        !own ||
        browser.state !== "operator_controlled" ||
        !browser.sensitive_entry ||
        browser.sensitive_entry_pending ||
        own.pending_lease_until_ms !== null ||
        own.pending_input_sequence !== null ||
        own.lease_until_ms === null ||
        own.lease_until_ms <= Date.now()
      )
        unavailable()
      const sequence = counter(own.settled_input_sequence + 1)
      const revision = counter(browser.revision + 2)
      const epoch = browser.control_epoch
      const result = await request("input-ticket", {
        ...ownerIntent(browser),
        input_sequence: sequence,
        page,
        ...(kind === "text" ? {} : { input_kind: kind }),
      })
      return {
        ticket: text(result.ticket),
        expected: { revision, control_epoch: epoch, settled_input_sequence: sequence },
      }
    },
    inputUrl(): string {
      const url = new URL("input", base)
      url.protocol = "wss:"
      return url.href
    },
    viewerUrl(): string {
      const url = new URL("viewer", base)
      url.protocol = "wss:"
      return url.href
    },
    dispose() {
      closed = true
      token = ""
      lifetime.abort()
    },
  }
}

// Only the explicit read-only View page action uses refreshed evidence. Control
// and input intents keep their original fences and are never replayed here.
export async function requestFreshBrowserView(
  client: ReturnType<typeof createBrowserControlClient>,
  selected: BrowserDescriptor,
  pageId: string,
  current: () => boolean,
): Promise<string> {
  const browsers = await client.discover()
  if (!current()) return ""
  // Compare the complete validated identity, including allocation/run/worker
  // fences and application purpose, independently of object property order.
  const identity = descriptor(selected, selected.identity.session_id).identity
  const matches = browsers.filter(
    (browser) => JSON.stringify(browser.identity) === JSON.stringify(identity),
  )
  const browser = matches[0]
  if (matches.length !== 1 || !browser)
    throw new BrowserViewUnavailable(
      "The selected browser is no longer available. Rediscover browsers.",
    )
  if (browser.sensitive_entry || browser.sensitive_entry_pending)
    throw new BrowserViewUnavailable("Private view is paused for sensitive entry.")
  if (!["agent_controlled", "takeover_requested", "operator_controlled"].includes(browser.state))
    throw new BrowserViewUnavailable(
      "The selected browser is closed or unavailable for viewing. Refresh discovery.",
    )
  const evidence = await client.pages(browser)
  if (!current()) return ""
  const page = evidence.pages.find((candidate) => candidate.page_id === pageId)
  if (!page)
    throw new BrowserViewUnavailable("The selected page is no longer available. Rediscover pages.")
  // A race after discovery must still fail at the server; never retry it.
  return client.viewerTicket(browser, page)
}

export class BrowserViewUnavailable extends Error {}

export function browserViewFailureMessage(error: unknown): string {
  return error instanceof BrowserViewUnavailable
    ? error.message
    : "Private view unavailable. The browser or page may have changed or closed, or capture may be paused. Refresh discovery before opening a new view."
}
