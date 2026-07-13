import { apiUrl } from "./config"
import type {
  AgentsResponse,
  ApiAgentSummary,
  ApiArtifactSummary,
  ApiEnvironmentSummary,
  ApiKnowledgeChunk,
  ApiKnowledgeListItem,
  ApiPendingAction,
  ApiPendingActionIssue,
  ApiReviewedKnowledgeEntry,
  ApiSessionBase,
  ApiSessionDetailEvent,
  ApiSessionDetailTranscriptMessage,
  ApiTaskDetail,
  ApiTaskListItem,
  ApiToolSummary,
  ApproveKnowledgeApiKnowledgeEntryIdApprovePostResponse,
  ArtifactReadResponse,
  ArtifactsResponse,
  EnvironmentsResponse,
  GetArtifactApiArtifactsArtifactIdGetData,
  GetArtifactContentApiArtifactsArtifactIdContentGetData,
  GetContractApiContractGetResponse,
  GetSessionApiSessionsSessionIdGetResponse,
  GetSessionSummaryApiSessionsSessionIdSummaryGetResponse,
  GetSessionsSummaryApiSessionsSummaryPostData,
  GetSessionsSummaryApiSessionsSummaryPostResponse,
  GetTaskApiTasksTaskIdGetResponse,
  InterruptSessionBody,
  ListArtifactsApiArtifactsGetData,
  ListPendingActionsApiPendingActionsGetData,
  ListPendingActionsApiPendingActionsGetResponse,
  ListPendingKnowledgeApiKnowledgePendingGetData,
  ListSessionsApiSessionsGetData,
  ListSessionsApiSessionsGetResponse,
  ListTasksApiTasksGetData,
  PendingKnowledgeDetailResponse,
  PendingKnowledgeListResponse,
  RejectKnowledgeApiKnowledgeEntryIdRejectPostResponse,
  ResumeTaskApiTasksTaskIdResumePostResponse,
  SessionsSummaryBody,
  SseErrorEnvelope,
  SseEventEnvelope,
  TaskHoldBody,
  ToolApprovalBody,
  ToolApprovalDecision,
  ToolApprovalRecoveryBody,
  ToolApprovalRecoveryOutcome,
  ToolRoundRecoveryBody,
  UserInputRecoveryBody,
  UserInputResolveBody,
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

export function isApiPayloadTooLarge(error: unknown): error is ApiClientError {
  return error instanceof ApiClientError && error.status === 413
}

export type ServerContract = GetContractApiContractGetResponse
export type AgentSummary = ApiAgentSummary
export type ToolSummary = ApiToolSummary
export type AgentsPage = AgentsResponse
export type EnvironmentSummary = ApiEnvironmentSummary
export type EnvironmentsPage = EnvironmentsResponse
export type ArtifactSummary = ApiArtifactSummary
export type ArtifactsPage = ArtifactsResponse
export type ArtifactRead = ArtifactReadResponse
export type ArtifactsQuery = NonNullable<ListArtifactsApiArtifactsGetData["query"]>
export type ArtifactReadQuery = NonNullable<GetArtifactApiArtifactsArtifactIdGetData["query"]>
export type ArtifactContentQuery = NonNullable<
  GetArtifactContentApiArtifactsArtifactIdContentGetData["query"]
>
export type Session = ApiSessionBase
export type SessionEvent = ApiSessionDetailEvent
export type TranscriptMessage = ApiSessionDetailTranscriptMessage
export type SessionDetail = GetSessionApiSessionsSessionIdGetResponse
export type SessionSummary = GetSessionSummaryApiSessionsSessionIdSummaryGetResponse
export type SessionsSummary = GetSessionsSummaryApiSessionsSummaryPostResponse
export type SessionListQuery = NonNullable<ListSessionsApiSessionsGetData["query"]>
export type SessionsSummaryQuery = NonNullable<
  GetSessionsSummaryApiSessionsSummaryPostData["query"]
>
export type SessionsPage = ListSessionsApiSessionsGetResponse
export type Task = ApiTaskListItem
export type TaskDetail = ApiTaskDetail
export type TaskHold = TaskHoldBody
export type TaskListQuery = NonNullable<ListTasksApiTasksGetData["query"]>
export type PendingAction = ApiPendingAction
export type PendingActionIssue = ApiPendingActionIssue
export type PendingActionsPage = ListPendingActionsApiPendingActionsGetResponse
export type PendingActionsQuery = NonNullable<ListPendingActionsApiPendingActionsGetData["query"]>
export type ApprovalDecision = ToolApprovalDecision
export type RecoveryOutcome = ToolApprovalRecoveryOutcome
export type ToolApprovalRecovery = ToolApprovalRecoveryBody
export type ToolRoundRecovery = ToolRoundRecoveryBody
export type UserInputRecovery = UserInputRecoveryBody
export type KnowledgeEntry = ApiKnowledgeListItem | ApiReviewedKnowledgeEntry
export type KnowledgeEntryDetail = PendingKnowledgeDetailResponse
export type KnowledgeChunk = ApiKnowledgeChunk
export type KnowledgePendingPage = PendingKnowledgeListResponse
export type KnowledgePendingQuery = NonNullable<
  ListPendingKnowledgeApiKnowledgePendingGetData["query"]
>
export type SSEEvent = SseEventEnvelope
export type SessionInterrupt = InterruptSessionBody

type ErrorEnvelope = {
  detail?: unknown
  error?: unknown
  message?: unknown
}

function objectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function queryString(query: Record<string, unknown> = {}): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") continue
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item !== undefined && item !== null && item !== "") {
          params.append(key, String(item))
        }
      }
      continue
    }
    params.set(key, String(value))
  }
  const encoded = params.toString()
  return encoded ? `?${encoded}` : ""
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

function postJson<T>(path: string, body?: unknown): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
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

export async function fetchAgents(): Promise<AgentsPage> {
  const page = await requestJson<unknown>("/agents")
  const pageObject = objectRecord(page)
  if (pageObject === null || !Array.isArray(pageObject.agents)) {
    throw new Error("Unexpected /agents response.")
  }
  return pageObject as AgentsPage
}

export async function fetchEnvironments(): Promise<EnvironmentsPage> {
  const page = await requestJson<unknown>("/environments")
  const pageObject = objectRecord(page)
  if (pageObject === null || !Array.isArray(pageObject.environments)) {
    throw new Error("Unexpected /environments response.")
  }
  return pageObject as EnvironmentsPage
}

export async function fetchArtifacts(query: ArtifactsQuery = {}): Promise<ArtifactsPage> {
  const page = await requestJson<unknown>(`/artifacts${queryString(query)}`)
  const pageObject = objectRecord(page)
  if (pageObject === null || !Array.isArray(pageObject.artifacts)) {
    throw new Error("Unexpected /artifacts response.")
  }
  return pageObject as ArtifactsPage
}

export async function fetchArtifact(
  artifactId: string,
  query: ArtifactReadQuery = {},
): Promise<ArtifactRead> {
  return requestJson<ArtifactRead>(
    `/artifacts/${encodeURIComponent(artifactId)}${queryString(query)}`,
  )
}

export function artifactContentUrl(artifactId: string, query: ArtifactContentQuery): string {
  return apiUrl(`/artifacts/${encodeURIComponent(artifactId)}/content${queryString(query)}`)
}

export async function fetchSessions(query: SessionListQuery = {}): Promise<Session[]> {
  // GET /api/sessions returns a paginated envelope; the dashboard shows the first page.
  const page = await fetchSessionsPage(query)
  return page.sessions
}

export async function fetchSessionsPage(query: SessionListQuery = {}): Promise<SessionsPage> {
  const page = await requestJson<unknown>(`/sessions${queryString(query)}`)
  const pageObject = objectRecord(page)
  if (pageObject === null || !Array.isArray(pageObject.sessions)) {
    throw new Error("Unexpected /sessions response.")
  }
  return pageObject as ListSessionsApiSessionsGetResponse
}

export async function fetchSessionsSummary(
  query: SessionsSummaryQuery = {},
  body: SessionsSummaryBody = {},
): Promise<SessionsSummary> {
  return requestJson<SessionsSummary>(`/sessions/summary${queryString(query)}`, {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(`/sessions/${id}`)
}

export async function fetchSessionSummary(id: string): Promise<SessionSummary> {
  return requestJson<SessionSummary>(`/sessions/${id}/summary`)
}

export async function fetchTasks(query: TaskListQuery = {}): Promise<Task[]> {
  const tasks = await requestJson<unknown>(`/tasks${queryString(query)}`)
  if (!Array.isArray(tasks)) {
    throw new Error("Unexpected /tasks response.")
  }
  return tasks as Task[]
}

export async function fetchTask(taskId: string): Promise<TaskDetail> {
  return requestJson<GetTaskApiTasksTaskIdGetResponse>(`/tasks/${encodeURIComponent(taskId)}`)
}

export async function fetchPendingActions(
  query: PendingActionsQuery = {},
): Promise<PendingActionsPage> {
  const page = await requestJson<unknown>(`/pending-actions${queryString(query)}`)
  const pageObject = objectRecord(page)
  if (pageObject === null || !Array.isArray(pageObject.actions)) {
    throw new Error("Unexpected /pending-actions response.")
  }
  return pageObject as PendingActionsPage
}

export async function pauseTask(taskId: string, body: TaskHold = {}): Promise<TaskDetail> {
  return postJson(`/tasks/${encodeURIComponent(taskId)}/pause`, body)
}

export async function blockTask(taskId: string, body: TaskHold = {}): Promise<TaskDetail> {
  return postJson(`/tasks/${encodeURIComponent(taskId)}/block`, body)
}

export async function markTaskNeedsAttention(
  taskId: string,
  body: TaskHold = {},
): Promise<TaskDetail> {
  return postJson(`/tasks/${encodeURIComponent(taskId)}/needs-attention`, body)
}

export async function resumeTask(taskId: string): Promise<TaskDetail> {
  return postJson<ResumeTaskApiTasksTaskIdResumePostResponse>(
    `/tasks/${encodeURIComponent(taskId)}/resume`,
  )
}

export async function fetchPendingKnowledge(
  query: KnowledgePendingQuery = {},
): Promise<KnowledgePendingPage> {
  const page = await requestJson<unknown>(`/knowledge/pending${queryString(query)}`)
  const pageObject = objectRecord(page)
  if (pageObject === null || !Array.isArray(pageObject.entries)) {
    throw new Error("Unexpected /knowledge/pending response.")
  }
  return pageObject as KnowledgePendingPage
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
  await streamJsonPost("/run", { prompt }, onEvent, onDone, onError)
}

export async function streamResume(
  sessionId: string,
  prompt: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  await streamJsonPost("/resume", { session_id: sessionId, prompt }, onEvent, onDone, onError)
}

export async function streamInterruptSession(
  sessionId: string,
  body: SessionInterrupt,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string, error?: unknown) => void,
) {
  await streamJsonPost(
    `/sessions/${encodeURIComponent(sessionId)}/interrupt`,
    body,
    onEvent,
    onDone,
    onError,
  )
}

export async function streamResolveToolApproval(
  body: ToolApprovalBody,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  await streamJsonPost("/tool-approvals/resolve", body, onEvent, onDone, onError)
}

export async function streamRecoverToolApproval(
  body: ToolApprovalRecoveryBody,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  await streamJsonPost("/tool-approvals/recover", body, onEvent, onDone, onError)
}

export async function streamRecoverToolRound(
  body: ToolRoundRecoveryBody,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  await streamJsonPost("/tool-rounds/recover", body, onEvent, onDone, onError)
}

export async function streamResolveUserInput(
  body: UserInputResolveBody,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  await streamJsonPost("/user-input/resolve", body, onEvent, onDone, onError)
}

export async function streamRecoverUserInput(
  body: UserInputRecoveryBody,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  await streamJsonPost("/user-input/recover", body, onEvent, onDone, onError)
}

async function streamJsonPost(
  path: string,
  body: Record<string, unknown>,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string, error?: unknown) => void,
) {
  const { fetchEventSource } = await import("@microsoft/fetch-event-source")
  try {
    await fetchEventSource(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body),
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
    onError(error instanceof Error ? error.message : String(error), error)
  } finally {
    onDone()
  }
}
