import { apiUrl } from "./config"

export const SUPPORTED_SERVER_CONTRACT_VERSION = "1"

export class ApiClientError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(message: string, status: number, detail: unknown = null) {
    super(message)
    this.name = "ApiClientError"
    this.status = status
    this.detail = detail
  }
}

export type ServerContract = {
  api_prefix: string
  contract_version: string
  versioning: {
    contract_version: string
    compatibility: string
    breaking_change_requires: string[]
  }
  sse: {
    content_type: "text/event-stream"
    event_id_format: "session_id:event_id"
    replay_header: "Last-Event-ID"
    event_data_schema: "SseEventEnvelope"
    error_event_name: "error"
    error_data_schema: "SseErrorEnvelope"
  }
  client_generation: {
    openapi_url: string | null
    supported_targets: string[]
    source_of_truth: "openapi"
  }
}

export type Session = {
  id: string
  status: string
  agent_name: string
  provider_name: string | null
  model: string | null
  parent_session_id: string | null
  causal_budget_id: string | null
  runtime_name: string
  runtime_version: string | null
  environment_name: string | null
  created_at: string
  updated_at: string
  labels: Record<string, string>
  metadata?: Record<string, unknown>
}

export type SessionEvent = {
  id: string
  type: string
  agent_name: string | null
  environment_name: string | null
  workflow_name: string | null
  tool_name: string | null
  payload: Record<string, unknown>
  timestamp: string
}

export type TranscriptMessage = {
  role: string
  content: Array<{ type: string; text?: string; tool_name?: string; [key: string]: unknown }>
}

export type SessionDetail = {
  session: Session
  events: SessionEvent[]
  transcript: TranscriptMessage[]
}

export type Task = {
  id: string
  type: string
  title: string | null
  status: string
  status_reason: string | null
  status_payload: Record<string, unknown> | null
  session_id: string | null
  worker_id: string | null
  lease_expires_at: string | null
  created_at: string
  completed_at: string | null
}

export type KnowledgeEntry = {
  entry_id: string
  namespace: string
  kind: string
  visibility: string
  status: string
  title: string | null
  labels: Record<string, string>
  aspects: string[]
  impact_targets: string[]
  source_type: string | null
  source_uri: string | null
  source_id: string | null
  created_by_type: string
  created_by: string | null
  created_at: string
  updated_at: string
  importance: number | null
  importance_source: string | null
  confidence: number | null
  chunk_count?: number
  text_preview: string | null
}

export type KnowledgeEntryDetail = KnowledgeEntry & {
  text: string
  metadata: Record<string, unknown>
  expires_at: string | null
  chunks: KnowledgeChunk[]
  chunk_limit: number
  chunk_max_bytes: number
}

export type KnowledgeChunk = {
  chunk_id: string
  entry_id: string
  chunk_index: number
  text: string
  content_hash: string | null
  source_uri: string | null
  metadata: Record<string, unknown>
}

export type KnowledgePendingPage = {
  entries: KnowledgeEntry[]
  truncated: boolean
  limit: number
  max_bytes: number
  total_entries_known: number | null
}

export type SSEEvent = {
  id: string
  type: string
  session_id: string
  agent_name: string | null
  environment_name: string | null
  workflow_name: string | null
  tool_name: string | null
  payload: Record<string, unknown>
  timestamp: string
}

type ErrorEnvelope = {
  detail?: unknown
  error?: unknown
  message?: unknown
}

type SseErrorEnvelope = {
  type: "stream.error"
  error: string
  error_type: string
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json")
  }
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  const res = await fetch(apiUrl(path), {
    ...init,
    headers,
    credentials: init.credentials ?? "same-origin",
  })
  if (!res.ok) {
    await throwResponseError(res)
  }
  return res.json() as Promise<T>
}

function postJson<T>(path: string): Promise<T> {
  return requestJson<T>(path, { method: "POST" })
}

async function throwResponseError(response: Response): Promise<never> {
  const prefix = `Request failed with HTTP ${response.status}`
  const contentType = response.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    const body = (await response.json().catch(() => null)) as ErrorEnvelope | null
    const detail = body?.detail ?? body?.error ?? body?.message
    if (typeof detail === "string" && detail.trim()) {
      throw new ApiClientError(detail, response.status, detail)
    }
    if (detail !== undefined) {
      throw new ApiClientError(`${prefix}: ${JSON.stringify(detail)}`, response.status, detail)
    }
  }
  throw new ApiClientError(prefix, response.status)
}

function parseSseEvent(raw: string): SSEEvent {
  const event = JSON.parse(raw) as SSEEvent
  if (typeof event.id !== "string" || typeof event.type !== "string") {
    throw new Error("Malformed SSE event from CAYU server.")
  }
  return event
}

function parseSseError(raw: string): SseErrorEnvelope {
  const event = JSON.parse(raw) as SseErrorEnvelope
  if (event.type !== "stream.error" || typeof event.error !== "string") {
    throw new Error("Malformed SSE error from CAYU server.")
  }
  return event
}

export function isSupportedServerContract(contract: ServerContract): boolean {
  return contract.contract_version === SUPPORTED_SERVER_CONTRACT_VERSION
}

export async function fetchServerContract(): Promise<ServerContract> {
  return requestJson<ServerContract>("/contract")
}

export async function fetchSessions(): Promise<Session[]> {
  // GET /api/sessions returns a paginated envelope; the dashboard shows the first page.
  const page = await requestJson<{ sessions: Session[] }>("/sessions")
  return page.sessions
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(`/sessions/${id}`)
}

export async function fetchTasks(): Promise<Task[]> {
  return requestJson<Task[]>("/tasks")
}

export async function fetchPendingKnowledge(): Promise<KnowledgePendingPage> {
  return requestJson<KnowledgePendingPage>("/knowledge/pending")
}

export async function fetchPendingKnowledgeEntry(entryId: string): Promise<KnowledgeEntryDetail> {
  return requestJson<KnowledgeEntryDetail>(`/knowledge/pending/${encodeURIComponent(entryId)}`)
}

export async function approveKnowledge(entryId: string): Promise<KnowledgeEntry> {
  return postJson<KnowledgeEntry>(`/knowledge/${encodeURIComponent(entryId)}/approve`)
}

export async function rejectKnowledge(entryId: string): Promise<KnowledgeEntry> {
  return postJson<KnowledgeEntry>(`/knowledge/${encodeURIComponent(entryId)}/reject`)
}

export async function streamRun(
  prompt: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  const { fetchEventSource } = await import("@microsoft/fetch-event-source")
  try {
    await fetchEventSource(apiUrl("/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ prompt }),
      async onopen(response) {
        if (!response.ok) {
          await throwResponseError(response)
        }
      },
      onmessage(msg) {
        if (!msg.data) return
        if (msg.event === "error") {
          const error = parseSseError(msg.data)
          throw new Error(error.error)
        }
        onEvent(parseSseEvent(msg.data))
      },
      onerror(error) {
        throw error
      },
    })
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  } finally {
    onDone()
  }
}

export async function streamResume(
  sessionId: string,
  prompt: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  const { fetchEventSource } = await import("@microsoft/fetch-event-source")
  try {
    await fetchEventSource(apiUrl("/resume"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ session_id: sessionId, prompt }),
      async onopen(response) {
        if (!response.ok) {
          await throwResponseError(response)
        }
      },
      onmessage(msg) {
        if (!msg.data) return
        if (msg.event === "error") {
          const error = parseSseError(msg.data)
          throw new Error(error.error)
        }
        onEvent(parseSseEvent(msg.data))
      },
      onerror(error) {
        throw error
      },
    })
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  } finally {
    onDone()
  }
}
