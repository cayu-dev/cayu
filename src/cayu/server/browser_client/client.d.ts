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

export interface CayuClient {
  readonly contract: CayuContract
  /** Whether the server advertises the session follow stream. */
  readonly streamAvailable: boolean
  followSession(sessionId: string, options?: FollowSessionOptions): SessionFollow
  /** Returns a function that stops the watch. */
  watchSessions(filter?: WatchSessionsFilter, options?: WatchSessionsOptions): () => void
  getSessionUsage(sessionId: string, options?: { signal?: AbortSignal }): Promise<CayuSessionUsage>
}

/** Read `/api/contract`, check its version, and return a client. */
export declare function connect(options?: ConnectOptions): Promise<CayuClient>
