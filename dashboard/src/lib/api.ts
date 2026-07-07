import { apiUrl } from "./config"
import type {
  ApiKnowledgeChunk,
  ApiKnowledgeListItem,
  ApiReviewedKnowledgeEntry,
  ApiSessionBase,
  ApiSessionDetailEvent,
  ApiSessionDetailTranscriptMessage,
  ApiTaskListItem,
  ApproveKnowledgeApiKnowledgeEntryIdApprovePostResponse,
  GetContractApiContractGetResponse,
  GetSessionApiSessionsSessionIdGetResponse,
  ListSessionsApiSessionsGetResponse,
  ListTasksApiTasksGetResponse,
  PendingKnowledgeDetailResponse,
  PendingKnowledgeListResponse,
  RejectKnowledgeApiKnowledgeEntryIdRejectPostResponse,
  SseErrorEnvelope,
  SseEventEnvelope,
} from "./generated/server-api"

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

export type ServerContract = GetContractApiContractGetResponse
export type Session = ApiSessionBase
export type SessionEvent = ApiSessionDetailEvent
export type TranscriptMessage = ApiSessionDetailTranscriptMessage
export type SessionDetail = GetSessionApiSessionsSessionIdGetResponse
export type Task = ApiTaskListItem
export type KnowledgeEntry = ApiKnowledgeListItem | ApiReviewedKnowledgeEntry
export type KnowledgeEntryDetail = PendingKnowledgeDetailResponse
export type KnowledgeChunk = ApiKnowledgeChunk
export type KnowledgePendingPage = PendingKnowledgeListResponse
export type SSEEvent = SseEventEnvelope

type ErrorEnvelope = {
  detail?: unknown
  error?: unknown
  message?: unknown
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
  const page = await requestJson<ListSessionsApiSessionsGetResponse>("/sessions")
  return page.sessions
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(`/sessions/${id}`)
}

export async function fetchTasks(): Promise<Task[]> {
  return requestJson<ListTasksApiTasksGetResponse>("/tasks")
}

export async function fetchPendingKnowledge(): Promise<KnowledgePendingPage> {
  return requestJson<KnowledgePendingPage>("/knowledge/pending")
}

export async function fetchPendingKnowledgeEntry(entryId: string): Promise<KnowledgeEntryDetail> {
  return requestJson<KnowledgeEntryDetail>(`/knowledge/pending/${encodeURIComponent(entryId)}`)
}

export async function approveKnowledge(entryId: string): Promise<KnowledgeEntry> {
  return postJson<ApproveKnowledgeApiKnowledgeEntryIdApprovePostResponse>(
    `/knowledge/${encodeURIComponent(entryId)}/approve`,
  )
}

export async function rejectKnowledge(entryId: string): Promise<KnowledgeEntry> {
  return postJson<RejectKnowledgeApiKnowledgeEntryIdRejectPostResponse>(
    `/knowledge/${encodeURIComponent(entryId)}/reject`,
  )
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
