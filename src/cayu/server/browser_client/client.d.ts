// Type declarations for the Cayu browser client served at `{path}/client.js`.
// See `cayu guide app-ui` for the request rules this module implements.

/** Version of this module's public API, advertised as `client.version` by `/api/contract`. */
export declare const CLIENT_VERSION: "1"
/** Exact server contract version this module was released with. */
export declare const CONTRACT_VERSION: string
/** Minimum spacing between `watchSessions` refreshes. It cannot be lowered. */
export declare const MIN_WATCH_INTERVAL_MS: 15000

export type SessionStatus =
  | "pending"
  | "running"
  | "interrupting"
  | "completed"
  | "failed"
  | "interrupted"
  | (string & {})

export type CayuClientErrorKind =
  | "auth"
  | "contract"
  | "not_found"
  | "unavailable"
  | "http"
  | "network"
  | "protocol"

export declare class CayuClientError extends Error {
  readonly name: "CayuClientError"
  readonly kind: CayuClientErrorKind
  /** HTTP status when the error came from a response. */
  readonly status: number | null
  /** Whether the client retries this error with backoff instead of stopping. */
  readonly retryable: boolean
  /** Delay requested by a `Retry-After` header, in milliseconds. */
  readonly retryAfterMs: number | null
  constructor(
    kind: CayuClientErrorKind,
    message: string,
    options?: {
      status?: number | null
      retryable?: boolean
      retryAfterMs?: number | null
      cause?: unknown
    },
  )
}

/** One durable session event, delivered once and in `sequence` order. */
export interface CayuEvent {
  sequence: number
  id: string
  type: string
  session_id: string
  interaction_id: string | null
  agent_name: string | null
  environment_name: string | null
  workflow_name: string | null
  tool_name: string | null
  payload: Record<string, unknown>
  timestamp: string
}

/** Session list item returned by `GET /api/sessions`. */
export interface CayuSession {
  id: string
  status: SessionStatus
  agent_name: string
  provider_name: string | null
  model: string | null
  parent_session_id: string | null
  causal_budget_id: string | null
  environment_name: string | null
  created_at: string
  updated_at: string
  labels: Record<string, string>
  [field: string]: unknown
}

/**
 * `GET /api/sessions/{id}/usage` response; see the OpenAPI `SessionUsageSummary`
 * schema. Token counters are decimal strings so large totals stay exact.
 */
export interface CayuSessionUsage {
  session_id: string
  model_steps: number
  tool_calls: number
  provider_names: string[]
  models: string[]
  usage: {
    input_tokens: string
    output_tokens: string
    total_tokens: string
    reasoning_output_tokens: string
    [field: string]: unknown
  }
  [field: string]: unknown
}

/** The `/api/contract` response. Only the fields this module reads are typed. */
export interface CayuContract {
  api_prefix: string
  contract_version: string
  client?: {
    module_url: string | null
    types_url: string | null
    version: string
    guide_topic: string
  }
  capabilities: Record<string, unknown>
  [field: string]: unknown
}

export interface ConnectOptions {
  /** Control-plane API base. Defaults to `./api` next to this module. */
  apiBaseUrl?: string | URL
  /** Extra request headers, for example an `Authorization` header. */
  headers?: Record<string, string>
  /** Fetch credentials mode. Defaults to `"same-origin"`. */
  credentials?: RequestCredentials
  /** Fetch implementation. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch
  signal?: AbortSignal
}

export interface FollowSessionOptions {
  /** Resume point. Only events with a greater sequence are delivered. */
  afterSequence?: number
  /** Event types not passed to `onEvent`. Lifecycle events are still read to detect the end. */
  excludeEventTypes?: readonly string[]
  /** Called once per event, deduplicated and in sequence order. */
  onEvent?: (event: CayuEvent) => void
  /** Called when the authoritative session status changes. */
  onStatus?: (status: SessionStatus) => void
  /** Called once with the non-retryable error that stopped the follow. */
  onError?: (error: CayuClientError) => void
  /** Base polling interval for the fallback transport. Defaults to 2000 ms. */
  pollIntervalMs?: number
  /** Aborting stops the follow; `done` then resolves with reason `"aborted"`. */
  signal?: AbortSignal
}

export interface FollowResult {
  /** `"terminal"` when the session ended, `"aborted"` when stopped by the caller. */
  reason: "terminal" | "aborted"
  status: SessionStatus | null
  lastSequence: number | null
}

export interface SessionFollow {
  /**
   * Resolves once the session is terminal and its events are delivered, or
   * when the follow is stopped. Rejects with a `CayuClientError` on a
   * non-retryable auth, contract, or not-found error. No requests are made
   * after it settles.
   */
  readonly done: Promise<FollowResult>
  stop(): void
  /** Highest sequence delivered or scanned so far; pass it as `afterSequence` to resume. */
  readonly lastSequence: number | null
  readonly status: SessionStatus | null
  readonly transport: "stream" | "poll" | null
  /** Most recent error, including retryable errors the client recovered from. */
  readonly lastError: CayuClientError | null
}

export interface WatchSessionsFilter {
  /** One status or several. Several statuses are read with one request each. */
  status?: SessionStatus | readonly SessionStatus[]
  agentName?: string
  environmentName?: string
  parentSessionId?: string
  /** `key=value` label filters. */
  labels?: readonly string[]
  /** Page size per request. Defaults to 100. */
  limit?: number
}

export type WatchStopReason = "idle" | "stopped" | "error"

export interface WatchSessionsOptions {
  /** Called with the merged session list whenever it changes. */
  onChange?: (sessions: CayuSession[]) => void
  /** Called once with the non-retryable error that stopped the watch. */
  onError?: (error: CayuClientError) => void
  /** Called once when the watch stops. `"idle"` means no listed session was active. */
  onStop?: (reason: WatchStopReason) => void
  /** Refresh interval. Values below 15000 ms are raised to 15000 ms. */
  intervalMs?: number
  /** Ceiling for the backoff applied while nothing changes. Defaults to 120000 ms. */
  maxIntervalMs?: number
  signal?: AbortSignal
}

export type InboxItemKind = "approval" | "question" | "recovery" | "knowledge"
export type InboxDecision = "approve" | "deny" | "answer" | "keep" | "drop"
export type InboxSettledDecision =
  | "approve"
  | "deny"
  | "expire"
  | "answer"
  | "recover"
  | "keep"
  | "drop"

/**
 * One thing waiting on a person; see the OpenAPI `InboxItem` schema. Text fields
 * are untrusted: render them as escaped text. `id` names one exact pause or
 * proposal revision.
 */
export interface InboxItem {
  id: string
  kind: InboxItemKind
  attention_id: string | null
  session_id: string | null
  agent_name: string | null
  created_at: string
  title: string
  detail: string | null
  tool_name: string | null
  arguments: null
  /**
   * Why an approval or recovery item shows no arguments: "no_review_policy" when the
   * agent configured no human review policy (such as ToolArgumentReview), "quarantined"
   * when one decides what protected human review shows.
   */
  arguments_unavailable_reason: "quarantined" | "no_review_policy" | null
  question: string | null
  options: string[]
  knowledge: { entry_id: string; revision: number; kind: string } | null
  /** What the item accepts; empty for recovery items, settled in the Developer dashboard. */
  decisions: InboxDecision[]
}

export interface InboxSettlement {
  reason: string | null
  decision: InboxSettledDecision
  /** `null` when the Runtime has no record of the person. */
  decided_by: { subject: string; tenant: string | null; source: string | null } | null
  decided_at: string | null
}

/** One item's current state, wherever it was settled. */
export interface InboxItemState {
  id: string
  kind: InboxItemKind
  status: "pending" | "settled" | "superseded" | "unavailable"
  session_id: string | null
  agent_name: string | null
  /** Present while pending. */
  item: InboxItem | null
  /** Present once settled. */
  settlement: InboxSettlement | null
}

export interface InboxIssue {
  code: string
  detail: string
  session_id: string | null
  agent_name: string | null
}

export interface InboxPage {
  items: InboxItem[]
  next_cursor: string | null
  has_more: boolean
  issues: InboxIssue[]
}

/** The body of `POST /api/inbox/{id}/settle`. */
export interface InboxSettleRequest {
  decision: InboxDecision
  /** Optional durable rationale for every decision. */
  reason?: string
  /** For `approve`: the changed arguments the tool runs with. */
  arguments?: Record<string, unknown>
  /** For `approve`: an edit the application stored itself, named by digests. */
  approval_revision?: { proposed_digest: string; approved_digest: string }
  /** For `answer`. */
  answer?: string
  /** Required when the application configures a human-review policy. */
  review_reference?: Record<string, unknown>
  /** Exact task continuation authority for a worker-owned approval or question. */
  task_worker_id?: string
  /** An elected interrupted handoff; requires task_worker_id. */
  task_handoff_id?: string
  task_continuation?: "claim"
}

export type InboxSettleResult =
  | {
      settled: true
      /** True when this exact decision was already recorded. */
      replayed: boolean
      item: InboxItemState
    }
  | {
      settled: false
      code:
        | "already_settled"
        | "superseded"
        | "expired"
        | "unavailable"
        | "not_settleable"
        | "conflict"
      message: string
      item: InboxItemState | null
    }

/** Everything waiting, read page by page up to a bound. */
export interface InboxSnapshot {
  items: InboxItem[]
  issues: InboxIssue[]
  /**
   * False when more items were waiting than the watch reads: more pages than it
   * follows, or an `inbox_truncated` or `knowledge_truncated` issue.
   */
  complete: boolean
}

export interface InboxFilters {
  /** Search only public, redacted item text. */
  q?: string
  session_id?: string
  agent_name?: string
  kind?: InboxItemKind
}

export interface InboxCounts {
  counts: Record<InboxItemKind, number> | null
  total_count: number | null
  complete: boolean
  unavailable_reason: "access_limited" | "source_incomplete" | null
}

export interface WatchInboxOptions extends InboxFilters {
  /** Called with the whole inbox whenever an item appears or leaves it. */
  onChange?: (inbox: InboxSnapshot) => void
  /**
   * Called once with the non-retryable error that stopped the watch, including a
   * later page's cursor refused (409 or 422) again after the refresh started over.
   */
  onError?: (error: CayuClientError) => void
  /** Called once when the watch stops. */
  onStop?: (reason: Exclude<WatchStopReason, "idle">) => void
  /** Refresh interval. Values below 15000 ms are raised to 15000 ms. */
  intervalMs?: number
  /** Ceiling for the backoff applied while nothing changes. Defaults to 120000 ms. */
  maxIntervalMs?: number
  signal?: AbortSignal
}

export interface CayuClient {
  readonly contract: CayuContract
  /** Whether the server advertises the session follow stream. */
  readonly streamAvailable: boolean
  followSession(sessionId: string, options?: FollowSessionOptions): SessionFollow
  /** Returns a function that stops the watch. */
  watchSessions(filter?: WatchSessionsFilter, options?: WatchSessionsOptions): () => void
  getSessionUsage(sessionId: string, options?: { signal?: AbortSignal }): Promise<CayuSessionUsage>
  /** Whether the server serves the agent's inbox. The inbox methods throw `unavailable` otherwise. */
  readonly inboxAvailable: boolean
  listInbox(
    options?: InboxFilters & {
      limit?: number
      cursor?: string
      signal?: AbortSignal
    },
  ): Promise<InboxPage>
  getInboxCounts(
    options?: Omit<InboxFilters, "kind"> & { signal?: AbortSignal },
  ): Promise<InboxCounts>
  getInboxItem(itemId: string, options?: { signal?: AbortSignal }): Promise<InboxItemState>
  /**
   * Settle one item as the signed-in person. Resolves with `settled: false` when
   * the item was already settled, replaced or expired. Repeating the same
   * decision is safe: it resolves with `replayed: true`.
   */
  settleInboxItem(
    itemId: string,
    decision: InboxSettleRequest,
    options?: { signal?: AbortSignal },
  ): Promise<InboxSettleResult>
  /** Returns a function that stops the watch. It never stops as idle. */
  watchInbox(options?: WatchInboxOptions): () => void
}

/** Read `/api/contract`, check its version, and return a client. */
export declare function connect(options?: ConnectOptions): Promise<CayuClient>
