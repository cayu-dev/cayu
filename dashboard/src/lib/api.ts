import { apiUrl } from "./config.ts"
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
  ApiTaskDetail,
  ApiTaskListItem,
  ApiToolSummary,
  ApiTranscriptMessage,
  ApproveKnowledgeApiKnowledgeEntryIdApprovePostResponse,
  ArtifactReadResponse,
  ArtifactsResponse,
  EnvironmentsResponse,
  GetArtifactApiArtifactsArtifactIdGetData,
  GetArtifactContentApiArtifactsArtifactIdContentGetData,
  GetContractApiContractGetResponse,
  GetOperationalSnapshotApiOperationsSnapshotPostResponse,
  GetSessionApiSessionsSessionIdGetResponse,
  GetSessionStateApiSessionsSessionIdStateGetResponse,
  GetSessionSummaryApiSessionsSessionIdSummaryGetResponse,
  GetSessionsSummaryApiSessionsSummaryPostData,
  GetSessionsSummaryApiSessionsSummaryPostResponse,
  GetSessionTranscriptApiSessionsSessionIdTranscriptGetData,
  GetSessionTranscriptApiSessionsSessionIdTranscriptGetResponse,
  GetTaskApiTasksTaskIdGetResponse,
  GetUsageRollupApiUsageRollupPostResponse,
  InterruptSessionBody,
  ListArtifactsApiArtifactsGetData,
  ListPendingActionsApiPendingActionsGetData,
  ListPendingActionsApiPendingActionsGetResponse,
  ListPendingKnowledgeApiKnowledgePendingGetData,
  ListSessionEventsApiSessionsSessionIdEventsGetData,
  ListSessionEventsApiSessionsSessionIdEventsGetResponse,
  ListSessionsApiSessionsGetData,
  ListSessionsApiSessionsGetResponse,
  ListTasksApiTasksGetData,
  OperationalSnapshotRequest,
  PendingKnowledgeDetailResponse,
  PendingKnowledgeListResponse,
  RejectKnowledgeApiKnowledgeEntryIdRejectPostResponse,
  ResumeTaskApiTasksTaskIdResumePostResponse,
  SessionsSummaryBody,
  SseEventEnvelope,
  TaskHoldBody,
  ToolApprovalDecision,
  ToolApprovalRecoveryBody,
  ToolApprovalRecoveryOutcome,
  ToolRoundRecoveryBody,
  UpdateSessionLabelsApiSessionsSessionIdLabelsPatchResponse,
  UpdateSessionLabelsBody,
  UpdateSessionMetadataApiSessionsSessionIdMetadataPatchResponse,
  UpdateSessionMetadataBody,
  UsageRollupRequest,
  UserInputRecoveryBody,
} from "./generated/server-api"

export const SUPPORTED_SERVER_CONTRACT_VERSION = "2"

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
export type SessionEvent = SseEventEnvelope
export type TranscriptMessage = ApiTranscriptMessage
export type SessionDetail = GetSessionApiSessionsSessionIdGetResponse
export type SessionState = GetSessionStateApiSessionsSessionIdStateGetResponse
export type SessionSummary = GetSessionSummaryApiSessionsSessionIdSummaryGetResponse
export type SessionEventsPage = ListSessionEventsApiSessionsSessionIdEventsGetResponse
export type SessionEventsQuery = NonNullable<
  ListSessionEventsApiSessionsSessionIdEventsGetData["query"]
>
export type SessionTranscriptPage = GetSessionTranscriptApiSessionsSessionIdTranscriptGetResponse
export type SessionTranscriptQuery = NonNullable<
  GetSessionTranscriptApiSessionsSessionIdTranscriptGetData["query"]
>
export type SessionLabelsUpdate = UpdateSessionLabelsBody
export type SessionMetadataUpdate = UpdateSessionMetadataBody
export type SessionsSummary = GetSessionsSummaryApiSessionsSummaryPostResponse
export type SessionListQuery = NonNullable<ListSessionsApiSessionsGetData["query"]>
export type SessionsSummaryQuery = NonNullable<
  GetSessionsSummaryApiSessionsSummaryPostData["query"]
>
export type SessionsPage = ListSessionsApiSessionsGetResponse
export type OperationalSnapshot = GetOperationalSnapshotApiOperationsSnapshotPostResponse
export type OperationalSnapshotBody = OperationalSnapshotRequest
export type UsageRollup = GetUsageRollupApiUsageRollupPostResponse
export type UsageRollupBody = UsageRollupRequest
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

function patchJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, {
    method: "PATCH",
    body: JSON.stringify(body),
  })
}

export async function throwResponseError(response: Response): Promise<never> {
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
  signal?: AbortSignal,
): Promise<SessionsSummary> {
  return requestJson<SessionsSummary>(`/sessions/summary${queryString(query)}`, {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  })
}

export async function fetchOperationalSnapshot(
  body: OperationalSnapshotBody = {},
  signal?: AbortSignal,
): Promise<OperationalSnapshot> {
  return requestJson<OperationalSnapshot>("/operations/snapshot", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  })
}

export async function fetchUsageRollup(
  body: UsageRollupBody,
  signal?: AbortSignal,
): Promise<UsageRollup> {
  return requestJson<UsageRollup>("/usage/rollup", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  })
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(`/sessions/${encodeURIComponent(id)}`)
}

export async function updateSessionLabels(
  id: string,
  body: SessionLabelsUpdate,
): Promise<SessionDetail> {
  return patchJson<UpdateSessionLabelsApiSessionsSessionIdLabelsPatchResponse>(
    `/sessions/${encodeURIComponent(id)}/labels`,
    body,
  )
}

export async function updateSessionMetadata(
  id: string,
  body: SessionMetadataUpdate,
): Promise<SessionDetail> {
  return patchJson<UpdateSessionMetadataApiSessionsSessionIdMetadataPatchResponse>(
    `/sessions/${encodeURIComponent(id)}/metadata`,
    body,
  )
}

export async function fetchSessionState(id: string, signal?: AbortSignal): Promise<SessionState> {
  return requestJson<SessionState>(`/sessions/${encodeURIComponent(id)}/state`, { signal })
}

export async function fetchSessionEvents(
  id: string,
  query: SessionEventsQuery = {},
  signal?: AbortSignal,
): Promise<SessionEventsPage> {
  return requestJson<SessionEventsPage>(
    `/sessions/${encodeURIComponent(id)}/events${queryString(query)}`,
    { signal },
  )
}

export async function fetchSessionTranscript(
  id: string,
  query: SessionTranscriptQuery = {},
  signal?: AbortSignal,
): Promise<SessionTranscriptPage> {
  return requestJson<SessionTranscriptPage>(
    `/sessions/${encodeURIComponent(id)}/transcript${queryString(query)}`,
    { signal },
  )
}

export async function fetchSessionSummary(id: string): Promise<SessionSummary> {
  return requestJson<SessionSummary>(`/sessions/${encodeURIComponent(id)}/summary`)
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
