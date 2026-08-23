import { apiUrl } from "./config.ts"
import type {
  AgentsResponse,
  ApiAgentSummary,
  ApiArtifactSummary,
  ApiEnvironmentSummary,
  ApiForkGroupDetail,
  ApiKnowledgeChunk,
  ApiKnowledgeListItem,
  ApiPendingAction,
  ApiPendingActionIssue,
  ApiReviewedKnowledgeEntry,
  ApiSession,
  ApiSessionBase,
  ApiTaskDetail,
  ApiTaskListItem,
  ApiToolSummary,
  ApiTranscriptMessage,
  ApproveKnowledgeApiKnowledgeEntryIdApprovePostResponse,
  ArtifactReadResponse,
  ArtifactsResponse,
  CapturedEvaluationDraft,
  CapturedEvaluationExportRequest,
  CapturedEvaluationLaunchRequest,
  CapturedEvaluationLaunchResponse,
  CapturedEvaluationPreviewResponse,
  CapturedEvaluationSaveRequest,
  CapturedEvaluationSaveResponse,
  CompareCatalogEvalResultsApiEvalsResultComparisonsPostData,
  CreateEvalRunApiEvalsRunsPostData,
  EnvironmentsResponse,
  EvalBaselineSelectionRequest,
  EvalBaselineSelectionResponse,
  EvalCaseCatalogPage,
  EvalComparisonResponse,
  EvalCorpusCatalogEntry,
  EvalCorpusCatalogPage,
  EvalCorpusDocument,
  EvalResultComparisonResponse,
  EvalResultDetailResponse,
  EvalResultPage,
  EvalResultRecord,
  EvalResultResponse,
  EvalRunPage,
  EvalRunRecord,
  EvalRunStatus,
  EvalSuiteCatalogPage,
  EvalTargetCatalogEntry,
  EvalTargetCatalogResponse,
  EvaluationPromotionDraft,
  EvaluationPromotionExportRequest,
  EvaluationPromotionPreviewResponse,
  GetArtifactApiArtifactsArtifactIdGetData,
  GetArtifactContentApiArtifactsArtifactIdContentGetData,
  GetContractApiContractGetResponse,
  GetForkGroupApiSessionsSessionIdForkGroupsGroupIdGetResponse,
  GetOperationalSnapshotApiOperationsSnapshotPostResponse,
  GetSessionApiSessionsSessionIdGetResponse,
  GetSessionStateApiSessionsSessionIdStateGetResponse,
  GetSessionSummaryApiSessionsSessionIdSummaryGetResponse,
  GetSessionsSummaryApiSessionsSummaryPostData,
  GetSessionsSummaryApiSessionsSummaryPostResponse,
  GetSessionTopologyApiSessionsSessionIdTopologyPostResponse,
  GetSessionTranscriptApiSessionsSessionIdTranscriptGetData,
  GetSessionTranscriptApiSessionsSessionIdTranscriptGetResponse,
  GetSystemDiagnosticsApiSystemDiagnosticsGetResponse,
  GetTaskApiTasksTaskIdGetResponse,
  GetUsageRollupApiUsageRollupPostResponse,
  InterruptSessionBody,
  ListArtifactsApiArtifactsGetData,
  ListEvalCasesApiEvalsCorporaCorpusRevisionSuitesSuiteIdCasesGetData,
  ListEvalCorporaApiEvalsCorporaGetData,
  ListEvalResultsApiEvalsResultsGetData,
  ListEvalRunsApiEvalsRunsGetData,
  ListEvalSuitesApiEvalsCorporaCorpusRevisionSuitesGetData,
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
  SessionTopologyRequest,
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
import { SUPPORTED_SERVER_CONTRACT_VERSION } from "./release-metadata.ts"

export { SUPPORTED_SERVER_CONTRACT_VERSION } from "./release-metadata.ts"

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
export type SessionUpdate = ApiSession
export type SessionState = GetSessionStateApiSessionsSessionIdStateGetResponse
export type SessionSummary = GetSessionSummaryApiSessionsSessionIdSummaryGetResponse
export type SessionTopology = GetSessionTopologyApiSessionsSessionIdTopologyPostResponse
export type SessionTopologyBody = SessionTopologyRequest
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
export type SystemDiagnostics = GetSystemDiagnosticsApiSystemDiagnosticsGetResponse
export type Task = ApiTaskListItem
export type TaskDetail = ApiTaskDetail
export type ForkGroupDetail = ApiForkGroupDetail
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
export type EvaluationPromotionPreview = EvaluationPromotionPreviewResponse
export type EvaluationPromotionCandidateDraft = EvaluationPromotionDraft
export type EvaluationPromotionExport = EvaluationPromotionExportRequest
export type CapturedEvaluationPreview = CapturedEvaluationPreviewResponse
export type CapturedEvaluationCandidateDraft = CapturedEvaluationDraft
export type CapturedEvaluationExport = CapturedEvaluationExportRequest
export type CapturedEvaluationSave = CapturedEvaluationSaveRequest
export type CapturedEvaluationSaved = CapturedEvaluationSaveResponse
export type CapturedEvaluationLaunch = CapturedEvaluationLaunchRequest
export type CapturedEvaluationLaunched = CapturedEvaluationLaunchResponse
export type EvalBaselineSelection = EvalBaselineSelectionRequest
export type EvalBaselineSelected = EvalBaselineSelectionResponse
export type EvalResultDetail = EvalResultDetailResponse
export type EvalResultSummary = EvalResultRecord
export type EvalResultsPage = EvalResultPage
export type EvalCorpus = EvalCorpusDocument
export type EvalCorpusEntry = EvalCorpusCatalogEntry
export type EvalCorporaPage = EvalCorpusCatalogPage
export type EvalSuitesPage = EvalSuiteCatalogPage
export type EvalCasesPage = EvalCaseCatalogPage
export type EvalRun = EvalRunRecord
export type EvalRunsPage = EvalRunPage
export type EvalStatus = EvalRunStatus
export type EvalResult = EvalResultResponse
export type EvalComparison = EvalComparisonResponse
export type EvalResultComparison = EvalResultComparisonResponse
export type EvalTarget = EvalTargetCatalogEntry
export type EvalTargets = EvalTargetCatalogResponse
export type EvalCorporaQuery = NonNullable<ListEvalCorporaApiEvalsCorporaGetData["query"]>
export type EvalSuitesQuery = NonNullable<
  ListEvalSuitesApiEvalsCorporaCorpusRevisionSuitesGetData["query"]
>
export type EvalCasesQuery = NonNullable<
  ListEvalCasesApiEvalsCorporaCorpusRevisionSuitesSuiteIdCasesGetData["query"]
>
export type EvalRunsQuery = NonNullable<ListEvalRunsApiEvalsRunsGetData["query"]>
export type EvalResultsQuery = NonNullable<ListEvalResultsApiEvalsResultsGetData["query"]>

export type DownloadedFile = {
  blob: Blob
  filename: string
}

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

async function requestResponse(path: string, init: RequestInit = {}): Promise<Response> {
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
  return res
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await requestResponse(path, init)
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

export async function fetchSystemDiagnostics(signal?: AbortSignal): Promise<SystemDiagnostics> {
  return requestJson<SystemDiagnostics>("/system/diagnostics", { signal })
}

export async function fetchAgents(signal?: AbortSignal): Promise<AgentsPage> {
  const page = await requestJson<unknown>("/agents", { signal })
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

export async function fetchSessionTopology(
  id: string,
  body: SessionTopologyBody = {},
  signal?: AbortSignal,
): Promise<SessionTopology> {
  return requestJson<SessionTopology>(`/sessions/${encodeURIComponent(id)}/topology`, {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  })
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(`/sessions/${encodeURIComponent(id)}`)
}

export async function previewEvaluationPromotion(
  sessionId: string,
  draft?: EvaluationPromotionCandidateDraft,
  signal?: AbortSignal,
): Promise<EvaluationPromotionPreview> {
  return requestJson<EvaluationPromotionPreview>(
    `/evals/promotion/sessions/${encodeURIComponent(sessionId)}/preview`,
    {
      method: "POST",
      body: JSON.stringify({ draft: draft ?? null }),
      signal,
    },
  )
}

export async function exportEvaluationPromotion(
  sessionId: string,
  body: EvaluationPromotionExport,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  const response = await requestResponse(
    `/evals/promotion/sessions/${encodeURIComponent(sessionId)}/export`,
    {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    },
  )
  return {
    blob: await response.blob(),
    filename: evaluationPromotionFilename(response.headers.get("content-disposition")),
  }
}

export async function previewCapturedEvaluation(
  sessionId: string,
  draft?: CapturedEvaluationCandidateDraft,
  signal?: AbortSignal,
): Promise<CapturedEvaluationPreview> {
  return requestJson<CapturedEvaluationPreview>(
    `/evals/sessions/${encodeURIComponent(sessionId)}/evaluation/preview`,
    {
      method: "POST",
      body: JSON.stringify({ draft: draft ?? null }),
      signal,
    },
  )
}

export async function saveCapturedEvaluation(
  sessionId: string,
  body: CapturedEvaluationSave,
  signal?: AbortSignal,
): Promise<CapturedEvaluationSaved> {
  return requestJson<CapturedEvaluationSaved>(
    `/evals/sessions/${encodeURIComponent(sessionId)}/evaluation/save`,
    { method: "POST", body: JSON.stringify(body), signal },
  )
}

export async function launchCapturedEvaluation(
  sessionId: string,
  body: CapturedEvaluationLaunch,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CapturedEvaluationLaunched> {
  return requestJson<CapturedEvaluationLaunched>(
    `/evals/sessions/${encodeURIComponent(sessionId)}/evaluation/launch`,
    {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify(body),
      signal,
    },
  )
}

export async function exportCapturedEvaluation(
  sessionId: string,
  body: CapturedEvaluationExport,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  const response = await requestResponse(
    `/evals/sessions/${encodeURIComponent(sessionId)}/evaluation/export`,
    { method: "POST", body: JSON.stringify(body), signal },
  )
  return {
    blob: await response.blob(),
    filename: evaluationPromotionFilename(response.headers.get("content-disposition")),
  }
}

export async function fetchEvalResults(
  query: EvalResultsQuery,
  signal?: AbortSignal,
): Promise<EvalResultsPage> {
  return requestJson<EvalResultsPage>(`/evals/results${queryString(query)}`, { signal })
}

export async function fetchEvalResultDetail(
  resultRevision: string,
  signal?: AbortSignal,
): Promise<EvalResultDetail> {
  return requestJson<EvalResultDetail>(`/evals/results/${encodeURIComponent(resultRevision)}`, {
    signal,
  })
}

export async function selectEvalBaseline(
  resultRevision: string,
  body: EvalBaselineSelection,
  signal?: AbortSignal,
): Promise<EvalBaselineSelected> {
  return requestJson<EvalBaselineSelected>(
    `/evals/results/${encodeURIComponent(resultRevision)}/baseline`,
    { method: "POST", body: JSON.stringify(body), signal },
  )
}

export async function fetchEvalCorpora(
  query: EvalCorporaQuery = {},
  signal?: AbortSignal,
): Promise<EvalCorporaPage> {
  return requestJson<EvalCorporaPage>(`/evals/corpora${queryString(query)}`, { signal })
}

export async function fetchEvalTargets(signal?: AbortSignal): Promise<EvalTargets> {
  return requestJson<EvalTargets>("/evals/targets", { signal })
}

export async function importEvalCorpus(
  corpus: Blob,
  signal?: AbortSignal,
): Promise<EvalCorpusEntry> {
  return requestJson<EvalCorpusEntry>("/evals/corpora", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: corpus,
    signal,
  })
}

export async function downloadEvalCorpus(
  corpusRevision: string,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  return downloadFile(
    `/evals/corpora/${encodeURIComponent(corpusRevision)}/download`,
    "cayu-eval-corpus.json",
    signal,
  )
}

export async function fetchEvalSuites(
  corpusRevision: string,
  query: EvalSuitesQuery = {},
  signal?: AbortSignal,
): Promise<EvalSuitesPage> {
  return requestJson<EvalSuitesPage>(
    `/evals/corpora/${encodeURIComponent(corpusRevision)}/suites${queryString(query)}`,
    { signal },
  )
}

export async function fetchEvalCases(
  corpusRevision: string,
  suiteId: string,
  query: EvalCasesQuery = {},
  signal?: AbortSignal,
): Promise<EvalCasesPage> {
  return requestJson<EvalCasesPage>(
    `/evals/corpora/${encodeURIComponent(corpusRevision)}/suites/${encodeURIComponent(suiteId)}/cases${queryString(query)}`,
    { signal },
  )
}

export async function fetchEvalRuns(
  query: EvalRunsQuery = {},
  signal?: AbortSignal,
): Promise<EvalRunsPage> {
  return requestJson<EvalRunsPage>(`/evals/runs${queryString(query)}`, { signal })
}

export async function createEvalRun(
  request: CreateEvalRunApiEvalsRunsPostData["body"],
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<EvalRun> {
  return requestJson<EvalRun>("/evals/runs", {
    method: "POST",
    headers: { "Idempotency-Key": idempotencyKey },
    body: JSON.stringify(request),
    signal,
  })
}

export async function fetchEvalRun(runId: string, signal?: AbortSignal): Promise<EvalRun> {
  return requestJson<EvalRun>(`/evals/runs/${encodeURIComponent(runId)}`, { signal })
}

export async function cancelEvalRun(runId: string, signal?: AbortSignal): Promise<EvalRun> {
  return requestJson<EvalRun>(`/evals/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
    signal,
  })
}

export async function fetchEvalResult(runId: string, signal?: AbortSignal): Promise<EvalResult> {
  return requestJson<EvalResult>(`/evals/runs/${encodeURIComponent(runId)}/result`, { signal })
}

export async function compareEvalRuns(
  baselineRunId: string,
  currentRunId: string,
  signal?: AbortSignal,
): Promise<EvalComparison> {
  return requestJson<EvalComparison>("/evals/comparisons", {
    method: "POST",
    body: JSON.stringify({ baseline_run_id: baselineRunId, current_run_id: currentRunId }),
    signal,
  })
}

export async function compareEvalResults(
  baselineResultRevision: string,
  currentResultRevision: string,
  scoreTolerance = 0,
  signal?: AbortSignal,
): Promise<EvalResultComparison> {
  const body: CompareCatalogEvalResultsApiEvalsResultComparisonsPostData["body"] = {
    baseline_result_revision: baselineResultRevision,
    current_result_revision: currentResultRevision,
    score_tolerance: scoreTolerance,
  }
  return requestJson<EvalResultComparison>("/evals/result-comparisons", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  })
}

export async function downloadEvalResultJson(
  runId: string,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  return downloadFile(
    `/evals/runs/${encodeURIComponent(runId)}/report.json`,
    `${safeDownloadStem(runId)}.eval-result.json`,
    signal,
  )
}

export async function downloadEvalResultHtml(
  runId: string,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  return downloadFile(
    `/evals/runs/${encodeURIComponent(runId)}/report.html`,
    `${safeDownloadStem(runId)}.eval-report.html`,
    signal,
  )
}

export async function downloadCatalogEvalResultJson(
  resultRevision: string,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  return downloadFile(
    `/evals/results/${encodeURIComponent(resultRevision)}/report.json`,
    `${safeDownloadStem(resultRevision)}.eval-result.json`,
    signal,
  )
}

export async function downloadCatalogEvalResultHtml(
  resultRevision: string,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  return downloadFile(
    `/evals/results/${encodeURIComponent(resultRevision)}/report.html`,
    `${safeDownloadStem(resultRevision)}.eval-report.html`,
    signal,
  )
}

async function downloadFile(
  path: string,
  fallbackFilename: string,
  signal?: AbortSignal,
): Promise<DownloadedFile> {
  const response = await requestResponse(path, { signal })
  return {
    blob: await response.blob(),
    filename: responseDownloadFilename(
      response.headers.get("content-disposition"),
      fallbackFilename,
    ),
  }
}

function responseDownloadFilename(contentDisposition: string | null, fallback: string): string {
  const match = /^attachment; filename="([A-Za-z0-9][A-Za-z0-9._-]{0,191})"$/.exec(
    contentDisposition ?? "",
  )
  return match?.[1] ?? fallback
}

function safeDownloadStem(value: string): string {
  const stem = value.replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 128)
  return stem || "cayu-eval"
}

function evaluationPromotionFilename(contentDisposition: string | null): string {
  const match = /^attachment; filename="([a-z][a-z0-9._-]{0,127}\.eval\.json)"$/.exec(
    contentDisposition ?? "",
  )
  return match?.[1] ?? "cayu-eval-corpus.json"
}

export async function updateSessionLabels(
  id: string,
  body: SessionLabelsUpdate,
): Promise<SessionUpdate> {
  return patchJson<UpdateSessionLabelsApiSessionsSessionIdLabelsPatchResponse>(
    `/sessions/${encodeURIComponent(id)}/labels`,
    body,
  )
}

export async function updateSessionMetadata(
  id: string,
  body: SessionMetadataUpdate,
): Promise<SessionUpdate> {
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

export async function fetchForkGroup(
  sourceSessionId: string,
  groupId: string,
): Promise<ForkGroupDetail> {
  return requestJson<GetForkGroupApiSessionsSessionIdForkGroupsGroupIdGetResponse>(
    `/sessions/${encodeURIComponent(sourceSessionId)}/fork-groups/${encodeURIComponent(groupId)}`,
  )
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
