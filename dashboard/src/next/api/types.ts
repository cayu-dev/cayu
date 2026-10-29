/**
 * Every call the reworked dashboard makes. Existing endpoints use the server's
 * generated types, so the fake API (mock/api.ts) answers in exactly the shapes the
 * runtime returns, and switching to the real server means passing real.ts instead.
 *
 * Notes like "backend item 7" name the runtime change a call came with (PAGE-MAP.md);
 * every one of them has landed.
 */
import type {
  AgentsPage,
  SessionDetail,
  SessionEventsPage,
  SessionEventsQuery,
  SessionListQuery,
  SessionState as SessionStateRecord,
  SessionsPage,
  SessionTracePage,
  SessionTraceQuery,
  SessionTranscriptPage,
  SessionTranscriptQuery,
  Task,
  TaskDetail,
} from "@/lib/api"
import type {
  ApiTaskIndex,
  ApiTaskScheduleEvent,
  InboxCounts,
  InboxItem,
  InboxItemState,
  InboxPage,
  InboxSettleRequest,
  KeptKnowledgeListResponse,
  SessionOverviewCountsResponse,
} from "@/lib/generated/server-api"
import type { MutationExecutionOptions } from "@/lib/mutation-browser"
import type { MutationTransportSnapshot } from "@/lib/mutation-transport"
import type {
  AddToEvals,
  EvalRunDetail,
  EvalSettings,
  EvalSuiteDetail,
  EvalSuiteSummary,
  NewEvalSuite,
} from "./eval-types"
import type { TaskIndexQuery } from "./tasks"

export type {
  ApiAgentSummary as Specialist,
  ApiEventRecord as SessionEvent,
  ApiSessionBase as Session,
  ApiTranscriptMessage as TranscriptMessage,
} from "@/lib/generated/server-api"
export type {
  AgentsPage,
  InboxItem,
  InboxItemState,
  InboxPage,
  InboxSettleRequest,
  KeptKnowledgeListResponse,
  SessionDetail,
  SessionEventsPage,
  SessionListQuery,
  SessionStateRecord,
  SessionsPage,
  SessionTracePage,
  SessionTraceQuery,
  SessionTranscriptPage,
  Task,
  TaskDetail,
}
export type TaskScheduleEvent = ApiTaskScheduleEvent
export type * from "./eval-types"

/**
 * Runtime task relations are carried by the generated list/detail contracts.
 */
export type TaskItem = Task
export type TaskInfo = TaskDetail
/** The holds `POST /api/tasks/{id}/pause | block | needs-attention` put on a task. */
export type TaskHold = "pause" | "block" | "needs-attention"

/** One merged state per session; waiting states win over running (as in cloud_reporting). */
export type SessionState =
  | "running"
  | "waiting_approval"
  | "waiting_input"
  | "needs_recovery"
  | "completed"
  | "failed"
  | "interrupted"

/**
 * How a tool reads in a summary line, declared with the tool
 * (`"Read"` + `"file"` gives "read 9 files"). Built-in tools ship one; others may set it.
 */
export type ToolDisplay = import("@/lib/generated/server-api").ApiToolDisplay
/**
 * Where a tool comes from, as Cloud's inventory reports it (`CloudToolSource` in
 * cayu/cloud_inventory.py): the agent's own code, Cayu, a package (with its version), an
 * MCP server, or the model provider (hosted tools).
 */
export type ToolSource =
  | { kind: "agent" }
  | { kind: "cayu" }
  | { kind: "package"; package?: string | null; version?: string | null }
  | { kind: "mcp"; server?: string | null }
  | { kind: "provider"; provider?: string | null }
  | { kind: "unknown" }
/**
 * Whether a tool asks a person before it runs (`_Approval` in cayu/cloud_inventory.py):
 * never, always, only for some arguments, or never runs at all.
 */
export type ToolApproval = "never" | "always" | "sometimes" | "blocked" | "unknown"
export type ToolWithDisplay = import("@/lib/generated/server-api").ApiToolSummary & {
  display?: ToolDisplay | null
  /** Native source classification and approval policy. */
  source?: ToolSource | null
  approval?: ToolApproval | null
}

/** Retained public-name tool usage from the runtime. */
export type ToolUsage = {
  tool: string
  calls: number
  failed: number
  approvals: { approved: number; denied: number; waiting?: number }
  last_used_at: string | null
  recent: { session_id: string; at: string; status: "ok" | "error" | "waiting" }[]
}

/**
 * Backend item 7: a kind of task the app's own worker handles. Creating one
 * runs the app the real way, through its orchestration, not one specialist directly.
 */
export type TaskType = {
  name: string
  description: string
  input_schema: Record<string, unknown>
  /** The specialist the app's worker hands it to, when there is one. */
  agent_name: string | null
}

/** Backend item 12: knowledge the agent has kept and uses. */
export type KnowledgeEntry = {
  id: string
  text: string
  kind: string
  agent_name: string | null
  source_session_id: string | null
  kept_by: string | null
  kept_at: string
  lookups: number
  last_used_at: string | null
}

/** How an agent marks its summary: "attention" when the result needs a person to look. */
export type SummaryTone = "neutral" | "attention"

/**
 * Backend items 1 and 2: `GET /api/sessions?include=overview`.
 * `title`, `summary` and `summary_tone` are set by the agent; everything else the runtime
 * derives.
 */
export type SessionOverview = {
  session_id: string
  title: string | null
  summary: string | null
  summary_tone: SummaryTone | null
  state: SessionState
  reason: string | null
  started_at: string
  ended_at: string | null
  last_activity_at: string
  model_steps: number
  tool_calls: number
  tokens: { input: number; output: number; cached: number }
  cost: { amount: number; currency: string; complete: boolean } | null
  uses: { browser: boolean; subagents: number; artifacts: number }
  /**
   * The session ran as an eval trial (runtime-owned metadata `cayu:eval_trial`, set on
   * the trial and every session created under it). Cloud's numbers leave these out.
   */
  eval_trial?: boolean
  forked?: boolean
}

/**
 * Public server-side Inbox filters; query and cursor share the same access scope.
 */
export type InboxQuery = {
  q?: string
  limit?: number
  cursor?: string | null
  session_id?: string
  agent_name?: string
  kind?: InboxItem["kind"]
}

/**
 * Settling a manual recovery with the evidence of what happened: whether the tool call
 * that never reported a result really completed or failed, the externally verified result
 * or failure reason, and, when the call was a question, the person's answer again.
 */
export type RecoveryRequest = {
  /** The inbox item's `attention_id`, naming which pending recovery this settles. */
  attention_id?: string | null
  outcome: "completed" | "failed"
  message: string
  answer?: string | null
}

/** What the shell shows about the running app, from `GET /api/app` (backend item 8). */
export type AppInfo = { name: string; runtime_version: string }

export type Api = {
  app(): Promise<AppInfo>
  specialists(signal?: AbortSignal): Promise<AgentsPage>
  sessions(query?: SessionListQuery, signal?: AbortSignal): Promise<SessionsPage>
  session(id: string): Promise<SessionDetail>
  overviews(ids: string[]): Promise<SessionOverview[]>
  transcript(
    id: string,
    query?: SessionTranscriptQuery,
    signal?: AbortSignal,
  ): Promise<SessionTranscriptPage>
  trace?(id: string, query?: SessionTraceQuery, signal?: AbortSignal): Promise<SessionTracePage>
  events(id: string, query?: SessionEventsQuery, signal?: AbortSignal): Promise<SessionEventsPage>
  /** `GET /api/tasks`, every task including Cayu's own (`cayu.*` types). */
  tasks(): Promise<TaskItem[]>
  taskList(query: TaskIndexQuery & { offset?: number }, signal?: AbortSignal): Promise<TaskItem[]>
  taskIndex(query: TaskIndexQuery, signal?: AbortSignal): Promise<ApiTaskIndex>
  taskDependents(id: string, cursor?: string, signal?: AbortSignal): Promise<ApiTaskIndex>
  /** `GET /api/tasks/{id}`. */
  task(id: string, signal?: AbortSignal): Promise<TaskInfo>
  /** `GET /api/tasks/{id}/schedule/events`; empty for tasks without a schedule. */
  taskScheduleEvents(
    id: string,
    afterSequence?: number,
    signal?: AbortSignal,
  ): Promise<TaskScheduleEvent[]>
  holdTask(id: string, hold: TaskHold, reason?: string): Promise<TaskInfo>
  /** `POST /api/tasks/{id}/resume`: back to pending. */
  resumeTask(id: string): Promise<TaskInfo>
  inbox(query?: InboxQuery, signal?: AbortSignal): Promise<InboxPage>
  inboxCounts(query?: Pick<InboxQuery, "session_id" | "agent_name" | "q">): Promise<InboxCounts>
  overviewCounts(query: { since: string; until: string }): Promise<SessionOverviewCountsResponse>
  /** `POST /api/inbox/{id}/settle`. Recovery items can't be settled here (409 `not_settleable`). */
  settle(itemId: string, body: InboxSettleRequest): Promise<InboxItemState>
  /**
   * Settle the session's manual recovery: `POST /api/user-input/recover`,
   * `/api/tool-approvals/recover` or `/api/tool-rounds/recover`, whichever the pending
   * recovery names from the exact public pending-action projection.
   */
  recover(
    sessionId: string,
    body: RecoveryRequest,
    options?: MutationExecutionOptions,
  ): Promise<MutationTransportSnapshot | undefined>
  /** `POST /api/run`: start a session with one specialist. Returns the new session's ID. */
  start(
    body: { agent: string; prompt: string },
    options?: MutationExecutionOptions,
  ): Promise<{ session_id: string; snapshot?: MutationTransportSnapshot }>
  /** `POST /api/resume`; the caller retains uncertain observation through options. */
  reply(
    id: string,
    prompt: string,
    options?: MutationExecutionOptions,
  ): Promise<MutationTransportSnapshot | undefined>
  /** `POST /api/sessions/{id}/interrupt`, with an optional reason. */
  interrupt(
    id: string,
    reason?: string,
    options?: MutationExecutionOptions,
  ): Promise<MutationTransportSnapshot | undefined>
  /** `GET /api/sessions/{id}/state`: execution state, snapshots, interruption cascade. */
  state(id: string): Promise<SessionStateRecord>
  /** Declared task types from the current worker-owned runtime catalog. */
  taskTypes(): Promise<TaskType[]>
  /** Exact operation identity is retained by the creating form across retries. */
  createTask(
    body: { type: string; input: Record<string, unknown> },
    operationId?: string,
  ): Promise<{
    task_id: string
    session_id: string | null
  }>
  /**
   * Start the same specialist with its first user text and labels (`POST /api/run`).
   * Task-origin sessions require the separate server-owned task replay operation.
   */
  runAgain(
    sessionId: string,
    options?: MutationExecutionOptions,
  ): Promise<{ session_id: string; snapshot?: MutationTransportSnapshot }>
  /** Native retained per-tool usage, shared by public tool name. */
  toolUsage(signal?: AbortSignal): Promise<ToolUsage[]>
  toolCalls?(
    name: string,
    cursor?: string,
    signal?: AbortSignal,
  ): Promise<import("@/lib/generated/server-api").ToolCallsResponse>
  /** Public kept entries, bounded cursor page with scoped usage. */
  knowledge(
    query?: { cursor?: string | null; limit?: number },
    signal?: AbortSignal,
  ): Promise<KeptKnowledgeListResponse>
  /** Retire the exact observed active revision. */
  removeKnowledge(id: string, expectedRevision: number): Promise<void>
  /** Backend item 14: every suite with its last run and baseline (prototype pages only). */
  evalSuites(): Promise<EvalSuiteSummary[]>
  evalSuite(id: string): Promise<EvalSuiteDetail>
  /** A run, compared with the suite's baseline unless `compareWith` names another run. */
  evalRun(id: string, compareWith?: string): Promise<EvalRunDetail>
  /** `POST /api/evals/runs` for the suite; returns the queued run. */
  runEvalSuite(suiteId: string): Promise<{ run_id: string }>
  /** `POST /api/evals/results/{revision}/baseline` for the run's result. */
  approveBaseline(runId: string): Promise<void>
  createEvalSuite(body: NewEvalSuite): Promise<{ suite_id: string }>
  /** The captured-evaluation and promotion flows (`/api/evals/sessions/{id}/evaluation/*`). */
  addToEvals(body: AddToEvals): Promise<{ suite_id: string; run_id: string | null }>
  evalSettings(): Promise<EvalSettings>
}
