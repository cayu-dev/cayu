import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useParams, useSearch } from "@tanstack/react-router"
import {
  ChevronRight,
  Copy,
  FlaskConical,
  GitBranch,
  PanelRight,
  RotateCw,
  TriangleAlert,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { EvaluationPromotionAction } from "@/components/dashboard/evaluation-promotion"
import {
  MutationTransportStatus,
  mutationTransportErrorMessage,
} from "@/components/dashboard/mutation-transport-status"
import { ProviderOperationResolutionBanner } from "@/components/dashboard/provider-operation-resolution"
import { TranscriptPart } from "@/components/dashboard/transcript-part"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { executeResolveProviderOperationMutation } from "@/lib/mutation-browser"
import {
  historySearchPatch,
  type SessionDetailSearch,
  workflowSearchFromSession,
  workflowSearchPatch,
} from "@/lib/session-detail-search"
import {
  initialFilteredPageNavigation,
  navigateToNewerPage,
  navigateToOlderPage,
  olderTranscriptPage,
  pageNavigationForFilter,
  type TranscriptPageParam,
  transcriptHistoryFilterKey,
  transcriptHistoryFilters,
  transcriptHistoryQuery,
} from "@/lib/session-history"
import { validateSessionHistorySearch } from "@/lib/session-history-search"
import { prepareSessionTaskReplay, type TaskReplayAcceptance } from "@/lib/session-task-replay"
import { cn } from "@/lib/utils"
import type { ToolWithDisplay } from "../../api/types"
import {
  useInbox,
  useOverview,
  useSessionDetail,
  useSessionState,
  useSpecialists,
  useTranscript,
} from "../../app/queries"
import { useApi, useNow, useServices } from "../../app/services"
import { useSessionMessageQueue } from "../../app/session-message-queue"
import {
  buildBlocks,
  formatDuration,
  formatTokens,
  isActive,
  isWaiting,
  sessionDisplayTitle,
} from "../../app/session-model"
import { useSessionMutation } from "../../app/session-mutation"
import { useSessionProgress } from "../../app/session-progress"
import { useSessionRelations } from "../../app/session-relations"
import { toolSpanId } from "../../app/trace-model"
import {
  ActionsMenu,
  ErrorState,
  EvalTrialTag,
  LabelChips,
  NavLink,
  SessionSummary,
  StateBadge,
  TopBar,
  useGo,
} from "../common"
import { selectClassName } from "../session-filters"
import { AddToEvalsDialog } from "./add-to-evals"
import { Composer } from "./composer"
import { Conversation, SubagentRows } from "./conversation"
import { DetailsPanel, PANEL_TABS, type PanelTab } from "./details-panel"
import { InboxCard } from "./inbox-card"
import { TraceView } from "./trace-view"

export function SessionPage({
  structuredOutputEnabled = true,
  artifactsEnabled = true,
  interruptionEnabled = false,
  providerResolutionEnabled = false,
  recoveryEnabled = false,
  taskReplayEnabled = false,
  workflowEnabled = true,
}: {
  structuredOutputEnabled?: boolean
  artifactsEnabled?: boolean
  interruptionEnabled?: boolean
  providerResolutionEnabled?: boolean
  recoveryEnabled?: boolean
  taskReplayEnabled?: boolean
  workflowEnabled?: boolean
} = {}) {
  const { sessionId } = useParams({ strict: false }) as { sessionId: string }
  const search = useSearch({ strict: false }) as SessionDetailSearch
  const go = useGo()
  const api = useApi()
  const viewer = useViewer()
  const services = useServices()
  const canExecute =
    !!services.prototype || (viewer?.full_access === true && services.executionEnabled === true)
  const canInterrupt = !!services.prototype || (viewer?.full_access === true && interruptionEnabled)
  const canResolveProvider =
    !services.prototype && viewer?.full_access === true && providerResolutionEnabled
  const canRecover = !!services.prototype || (viewer?.full_access === true && recoveryEnabled)
  const mutation = useSessionMutation()
  const interruptMutation = useSessionMutation()
  const queue = useSessionMessageQueue(sessionId, canExecute && !services.prototype)
  const [draft, setDraft] = useState("")
  const submittedDraft = useRef("")
  const now = useNow()
  const queryClient = useQueryClient()
  const detail = useSessionDetail(sessionId)
  const overview = useOverview(sessionId, !!services.prototype)
  const firstUser = useQuery({
    queryKey: ["session-list-first-user", sessionId],
    queryFn: ({ signal }) =>
      api.transcript(sessionId, { role: "user", offset: 0, limit: 1 }, signal),
    enabled:
      !!detail.data &&
      overview.isSuccess &&
      !overview.data?.title?.trim() &&
      Object.keys(detail.data.labels).length === 0,
    staleTime: 30_000,
    retry: false,
  })
  const state = overview.data?.state
  const runtimeState = useSessionState(sessionId, true)
  const filters = transcriptHistoryFilters(validateSessionHistorySearch(search))
  const filterKey = `${sessionId}:${transcriptHistoryFilterKey(filters)}`
  const [history, setHistory] = useState(() =>
    initialFilteredPageNavigation<TranscriptPageParam>(filterKey, null),
  )
  const navigation = pageNavigationForFilter(history, filterKey, null)
  const [pinned, setPinned] = useState(true)
  const [newActivity, setNewActivity] = useState(false)
  const [transcriptPosition, setTranscriptPosition] = useState<{
    messageIndex: number
    partIndex: number
  }>()
  const transcript = useTranscript(
    sessionId,
    !!services.prototype && isActive(state) && pinned,
    navigation.current,
    { ...transcriptHistoryQuery(filters), include_thinking: true },
  )
  const olderPage = transcript.data ? olderTranscriptPage(transcript.data.offset, 100) : undefined
  const inbox = useInbox({ session_id: sessionId }, !!services.prototype)
  const relations = useSessionRelations(sessionId, detail.data?.parent_session_id)
  const specialists = useSpecialists()
  const [menuOpen, setMenuOpen] = useState(false)
  const detailsOpener = useRef<HTMLButtonElement>(null)
  const [prototypeEvals, setPrototypeEvals] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  const panel = PANEL_TABS.find((t) => t === search.panel) as PanelTab | undefined
  const trace = search.view === "trace"
  const setSearch = (patch: Record<string, unknown>) => {
    const next = { ...search, ...patch }
    for (const k of Object.keys(next))
      if (next[k as keyof typeof next] === undefined) delete next[k as keyof typeof next]
    go(`/sessions/${encodeURIComponent(sessionId)}`, next)
  }
  const setPanel = (tab: PanelTab | undefined) => setSearch({ panel: tab })
  const openTrace = (span?: string) => setSearch({ view: "trace", span, log: undefined })
  const displays = useMemo(() => {
    const agent = specialists.data?.agents.find((a) => a.name === detail.data?.agent_name)
    return Object.fromEntries(
      (agent?.tools ?? []).map((t) => [t.name, (t as ToolWithDisplay).display ?? undefined]),
    )
  }, [specialists.data, detail.data])
  const blocks = useMemo(
    () =>
      buildBlocks(
        transcript.data?.messages ?? [],
        structuredOutputEnabled,
        filters.includeThinking,
      ),
    [transcript.data, structuredOutputEnabled, filters.includeThinking],
  )
  const refresh = () => {
    for (const key of [
      "overview",
      "transcript",
      "sessions",
      "inbox",
      "state",
      "session-state",
      "session-list",
      "inbox-counts",
      "session-children",
      "session-parent",
      "session-messages",
      "session-recovery",
    ])
      void queryClient.invalidateQueries({ queryKey: [key] })
  }
  const progress = useSessionProgress({
    sessionId,
    enabled: !services.prototype,
    active: state === "running",
    mutationActive: mutation.busy || interruptMutation.busy,
    onProgress: async () => {
      const following = pinned && navigation.current === null
      if (!following) setNewActivity(true)
      await Promise.all([
        ...[
          "session",
          "overview",
          "state",
          "session-state",
          "session-children",
          "session-parent",
          "session-messages",
          "session-recovery",
        ].map((key) => queryClient.invalidateQueries({ queryKey: [key, sessionId] })),
        queryClient.invalidateQueries({ queryKey: ["inbox"] }),
        queryClient.invalidateQueries({ queryKey: ["inbox-counts"] }),
        queryClient.invalidateQueries({ queryKey: ["trace", sessionId] }),
        queryClient.invalidateQueries({
          queryKey: ["trace-events", sessionId],
          predicate: (query) => query.meta?.sessionProgress !== false,
        }),
        ...(following
          ? [queryClient.invalidateQueries({ queryKey: ["transcript", sessionId] })]
          : []),
      ])
    },
  })
  const reply = useMutation({
    mutationFn: (text: string) => {
      if (!canExecute) throw new Error("Session execution is unavailable.")
      if (state === "running" && !services.prototype)
        throw new Error("Queued messaging has not been authorized for this session.")
      submittedDraft.current = text
      return mutation.execute("reply", (options) => api.reply(sessionId, text, options))
    },
    onSuccess: (result, text) => {
      if (result === null) return
      if (!result || result.phase === "terminal")
        setDraft((current) => (current.trim() === text ? "" : current))
      refresh()
    },
  })
  const replayOperation = useRef<Awaited<ReturnType<typeof prepareSessionTaskReplay>> | null>(null)
  const replayAbort = useRef<AbortController | null>(null)
  const [replayAcceptance, setReplayAcceptance] = useState<TaskReplayAcceptance | null>(null)
  useEffect(() => () => replayAbort.current?.abort(), [])
  const runAgain = useMutation({
    mutationFn: async () => {
      if (!canExecute) throw new Error("Session execution is unavailable.")
      if (!services.prototype && detail.data?.invocation.source === "task") {
        if (!taskReplayEnabled) throw new Error("Task replay is unavailable in this deployment.")
        if (replayAbort.current || mutation.fenced || interruptMutation.fenced)
          throw new Error("Resolve the pending action before replaying this task.")
        const abort = new AbortController()
        replayAbort.current = abort
        try {
          replayOperation.current ??= await prepareSessionTaskReplay(sessionId, abort.signal)
          const receipt = await replayOperation.current.submit(abort.signal)
          if (abort.signal.aborted) return null
          setReplayAcceptance(receipt)
          if (receipt.phase === "rejected") replayOperation.current = null
          return { task_id: receipt.taskId, session_id: undefined, snapshot: undefined }
        } finally {
          if (replayAbort.current === abort) replayAbort.current = null
        }
      }
      let destination: string | undefined
      const snapshot = await mutation.execute("rerun", async (options) => {
        const result = await api.runAgain(sessionId, options)
        destination = result.session_id
        return result.snapshot
      })
      if (snapshot === null) return null
      return { session_id: destination, snapshot }
    },
    onSuccess: (result) => {
      if (result && "task_id" in result && result.task_id) {
        setMenuOpen(false)
        go("/tasks", { task_id: result.task_id })
        return
      }
      if (
        !result?.session_id ||
        (result.snapshot && !["terminal", "runtime_failed"].includes(result.snapshot.phase))
      )
        return
      setMenuOpen(false)
      refresh()
      go(`/sessions/${encodeURIComponent(result.session_id)}`)
    },
  })
  const [confirmingStop, setConfirmingStop] = useState(false)
  const resolveProvider = useMutation({
    mutationFn: ({ action, reason }: { action: "fallback_retry" | "fail"; reason: string }) => {
      const operation = runtimeState.data?.provider_operation
      if (
        !canResolveProvider ||
        runtimeState.error ||
        !operation?.stage_id ||
        operation.run_epoch == null ||
        !operation.allowed_resolutions?.includes(action) ||
        !reason.trim()
      )
        throw new Error(
          "The exact provider-operation resolution is unavailable. Refresh the session.",
        )
      return mutation.execute("provider", (options) =>
        executeResolveProviderOperationMutation(
          {
            session_id: sessionId,
            stage_id: operation.stage_id!,
            expected_run_epoch: operation.run_epoch!,
            action,
            reason,
          },
          options,
        ),
      )
    },
    onSuccess: (result) => {
      if (result !== null) refresh()
    },
  })
  const stop = useMutation({
    mutationFn: (reason: string | undefined) => {
      if (!canInterrupt) throw new Error("Session interruption is unavailable.")
      return interruptMutation.execute("interrupt", (options) =>
        api.interrupt(sessionId, reason, options),
      )
    },
    onSuccess: (result) => {
      if (result === null) return
      if (!result || result.phase === "terminal") setConfirmingStop(false)
      refresh()
    },
  })

  useEffect(() => {
    if (!trace && transcriptPosition && transcript.data)
      document
        .getElementById(
          `transcript-tool-${transcriptPosition.messageIndex}-${transcriptPosition.partIndex}`,
        )
        ?.scrollIntoView({ block: "center" })
  }, [trace, transcriptPosition, transcript.data])

  // biome-ignore lint/correctness/useExhaustiveDependencies: follow new transcript and decision content only while pinned
  useEffect(() => {
    if (pinned && navigation.current === null) endRef.current?.scrollIntoView({ block: "end" })
  }, [pinned, navigation.current, transcript.data, inbox.data?.items.length])
  // biome-ignore lint/correctness/useExhaustiveDependencies: close the menus when the session changes
  useEffect(() => {
    setMenuOpen(false)
    setConfirmingStop(false)
  }, [sessionId])

  if (detail.error) {
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <TopBar>Session</TopBar>
        <div className="p-6">
          <ErrorState error={detail.error} />
        </div>
      </div>
    )
  }

  const o = overview.data
  const title = detail.data
    ? sessionDisplayTitle(detail.data, o?.title, firstUser.data?.messages[0])
    : sessionId
  const running = state === "running"
  const finished = state === "completed" || state === "failed"
  const seconds = o
    ? ((o.ended_at ? Date.parse(o.ended_at) : now()) - Date.parse(o.started_at)) / 1000
    : null
  const items = inbox.data?.items ?? []

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1">
      {services.prototype && prototypeEvals && detail.data && (
        <AddToEvalsDialog
          sessionId={sessionId}
          specialist={detail.data.agent_name}
          onClose={() => setPrototypeEvals(false)}
        />
      )}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <TopBar
          actions={
            <>
              <div
                className="mr-1 flex rounded-lg border border-border p-0.5"
                role="tablist"
                aria-label="Session view"
                onKeyDown={(event) => {
                  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return
                  const tabs = Array.from(
                    event.currentTarget.querySelectorAll<HTMLButtonElement>(
                      'button[role="tab"]:not(:disabled)',
                    ),
                  )
                  const current = tabs.indexOf(event.target as HTMLButtonElement)
                  if (current < 0) return
                  event.preventDefault()
                  const next =
                    event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? tabs.length - 1
                        : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
                          tabs.length
                  tabs[next]?.click()
                  tabs[next]?.focus()
                }}
              >
                {(
                  [
                    ["conversation", "Conversation"],
                    ["trace", "Trace"],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="tab"
                    aria-selected={(value === "trace") === trace}
                    tabIndex={(value === "trace") === trace ? 0 : -1}
                    onClick={() =>
                      setSearch({
                        view: value,
                      })
                    }
                    className={cn(
                      "rounded-md px-2.5 py-0.5 text-[13px]",
                      (value === "trace") === trace
                        ? "bg-muted font-medium"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <Button
                variant="ghost"
                size="icon-sm"
                ref={detailsOpener}
                aria-label="Session details"
                aria-pressed={!!panel}
                className={cn(panel && "bg-muted")}
                onClick={() => setPanel(panel ? undefined : "overview")}
              >
                <PanelRight />
              </Button>
              <ActionsMenu open={menuOpen} onOpenChange={setMenuOpen}>
                {canExecute && (
                  <MenuItem
                    icon={RotateCw}
                    label={
                      runAgain.isPending
                        ? "Starting…"
                        : replayAcceptance?.phase === "unknown"
                          ? "Retry task replay"
                          : "Run again"
                    }
                    disabled={runAgain.isPending}
                    onClick={() => runAgain.mutate()}
                  />
                )}
                <MenuItem
                  icon={Copy}
                  label="Copy session ID"
                  onClick={() => {
                    void navigator.clipboard?.writeText(sessionId).catch(() => undefined)
                    setMenuOpen(false)
                  }}
                />
                <MenuItem
                  icon={GitBranch}
                  label="Open trace"
                  onClick={() => {
                    setMenuOpen(false)
                    openTrace()
                  }}
                />
                {!!services.prototype && finished && (
                  <MenuItem
                    icon={FlaskConical}
                    label="Add to evals"
                    onClick={() => {
                      setMenuOpen(false)
                      setPrototypeEvals(true)
                    }}
                  />
                )}
                {!services.prototype && viewer?.full_access === true && detail.data && (
                  <EvaluationPromotionAction
                    key={`${sessionId}:${detail.data.status}`}
                    sessionId={sessionId}
                    status={detail.data.status}
                    label="Add to evals"
                  />
                )}
              </ActionsMenu>
            </>
          }
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <NavLink
              to="/sessions"
              className="shrink-0 font-normal text-muted-foreground hover:text-foreground"
            >
              Sessions
            </NavLink>
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="truncate" title={title}>
              {title}
            </span>
          </span>
        </TopBar>
        {!!progress.error && (
          <div role="status" className="px-4 py-2 text-sm">
            Live updates are unavailable.{" "}
            <Button variant="outline" size="sm" onClick={progress.retry}>
              Retry live updates
            </Button>
          </div>
        )}

        {trace && search.workflow === "1" && !workflowEnabled && (
          <div role="status" className="px-4 py-2 text-sm">
            Workflow is unavailable for this deployment or role. Session trace remains available.
          </div>
        )}
        {trace && detail.data ? (
          <TraceView
            session={detail.data}
            state={state}
            displays={displays}
            transcriptPosition={transcriptPosition}
            onConversation={(position) => {
              setTranscriptPosition(position)
              const nextFilterKey = `${sessionId}:${transcriptHistoryFilterKey({ role: "all", includeThinking: true })}`
              setHistory({
                filterKey: nextFilterKey,
                navigation: {
                  current: { offset: Math.floor(position.messageIndex / 100) * 100, limit: 100 },
                  newer: [null],
                },
              })
              setPinned(false)
              setSearch({
                view: "conversation",
                transcript_role: undefined,
                include_thinking: undefined,
              })
            }}
            historySearch={validateSessionHistorySearch(search)}
            onHistorySearch={(next) => setSearch(historySearchPatch(next))}
            workflowOpen={search.workflow === "1" || !!search.expanded_session_id?.length}
            workflowSearch={workflowSearchFromSession(search)}
            onWorkflowSearch={(next) => setSearch(workflowSearchPatch(next))}
            mutationSnapshots={[
              {
                label: "Session action",
                snapshot: mutation.snapshot,
                retryObservation: () => {
                  void mutation.reobserve().then(refresh)
                },
                retrying: mutation.busy,
              },
              {
                label: "Interruption",
                snapshot: interruptMutation.snapshot,
                retryObservation: () => {
                  void interruptMutation.reobserve().then(refresh)
                },
                retrying: interruptMutation.busy,
              },
            ]}
            onRetryInterruption={
              canInterrupt && !interruptMutation.fenced
                ? () => stop.mutate("Retry incomplete background interruption.")
                : undefined
            }
            retryInterruptionBusy={stop.isPending || interruptMutation.busy}
            subagents={relations.rows}
            spanId={search.span}
            log={search.log === "1"}
            onSelect={(span) => {
              setTranscriptPosition(undefined)
              setSearch({ span })
            }}
            eventId={search.event}
            onSelectEvent={(event) => setSearch({ event })}
            onShowSpan={(span) => {
              setTranscriptPosition(undefined)
              setSearch({ log: undefined, span })
            }}
            onLog={(log) => setSearch({ log: log ? "1" : undefined })}
            toolCallOrder={blocks.flatMap((b) =>
              b.kind === "work" ? b.tools.map((t) => t.id) : [],
            )}
          />
        ) : (
          <>
            <div
              className="relative min-h-0 flex-1 overflow-y-auto"
              onScroll={(event) => {
                const node = event.currentTarget
                const atTail = node.scrollHeight - node.scrollTop - node.clientHeight <= 32
                setPinned(atTail)
                if (atTail && newActivity && navigation.current === null) {
                  void transcript.refetch().then((result) => {
                    if (result.isSuccess) setNewActivity(false)
                  })
                }
              }}
            >
              <div className="mx-auto w-full max-w-3xl px-4 pt-5 pb-6 sm:px-6">
                <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted-foreground">
                  <NavLink
                    to="/agents"
                    search={{ q: detail.data?.agent_name }}
                    className="font-medium text-foreground hover:underline"
                  >
                    {detail.data?.agent_name}
                  </NavLink>
                  {detail.data?.environment_name && (
                    <NavLink
                      to="/environments"
                      search={{ q: detail.data.environment_name }}
                      className="hover:underline"
                    >
                      {detail.data.environment_name}
                    </NavLink>
                  )}
                  {artifactsEnabled && (
                    <NavLink
                      to="/artifacts"
                      search={{ session_id: sessionId }}
                      className="hover:underline"
                    >
                      Artifacts
                    </NavLink>
                  )}
                  <StateBadge overview={o} />
                  {o?.eval_trial && <EvalTrialTag />}
                  {seconds != null && (
                    <span className="tabular-nums">
                      {running ? "running " : ""}
                      {formatDuration(seconds)}
                    </span>
                  )}
                  {o && (
                    <span className="tabular-nums">
                      {formatTokens(o.tokens.input + o.tokens.output)} tokens
                    </span>
                  )}
                  {o?.cost && (
                    <span className="tabular-nums">
                      {o.cost.currency === "USD" ? "$" : `${o.cost.currency} `}
                      {o.cost.amount.toFixed(2)}
                      {o.cost.complete ? "" : " (partial)"}
                    </span>
                  )}
                  {detail.data && <LabelChips labels={detail.data.labels} />}
                </div>

                {o?.summary && (
                  <div className="mb-4">
                    <SessionSummary text={o.summary} tone={o.summary_tone} />
                  </div>
                )}
                {firstUser.error && (
                  <p className="mb-4 text-xs text-muted-foreground">Message title unavailable</p>
                )}

                <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
                  <label>
                    <span className="sr-only">Message role</span>
                    <select
                      aria-label="Message role"
                      value={filters.role}
                      onChange={(event) =>
                        setSearch({
                          transcript_role:
                            event.target.value === "all" ? undefined : event.target.value,
                        })
                      }
                      className={selectClassName}
                    >
                      <option value="all">All roles</option>
                      <option value="user">User</option>
                      <option value="assistant">Assistant</option>
                      <option value="system">System</option>
                      <option value="tool">Tool</option>
                    </select>
                  </label>
                  <label className="flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={filters.includeThinking}
                      onChange={(event) =>
                        setSearch({ include_thinking: event.target.checked ? undefined : "false" })
                      }
                    />
                    Include thinking
                  </label>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={olderPage === undefined || transcript.isFetching}
                    onClick={() => {
                      if (olderPage)
                        setHistory({
                          filterKey,
                          navigation: navigateToOlderPage(navigation, olderPage),
                        })
                    }}
                  >
                    Older messages
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!navigation.newer.length || transcript.isFetching}
                    onClick={() =>
                      setHistory({ filterKey, navigation: navigateToNewerPage(navigation) })
                    }
                  >
                    Newer messages
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={navigation.current === null && pinned && !newActivity}
                    onClick={() => {
                      setHistory(initialFilteredPageNavigation(filterKey, null))
                      setPinned(true)
                      setNewActivity(false)
                      if (navigation.current === null) void transcript.refetch()
                    }}
                  >
                    Latest messages
                  </Button>
                  {newActivity && <span role="status">New activity is available.</span>}
                  <span className="text-xs text-muted-foreground">
                    {transcript.data
                      ? `${transcript.data.messages.length} of ${transcript.data.total_messages} shown`
                      : "Loading messages…"}
                  </span>
                </div>
                {transcript.error && <ErrorState error={transcript.error} />}
                {runtimeState.error && <ErrorState error={runtimeState.error} />}
                {runtimeState.data?.provider_operation.allowed_resolutions?.length ? (
                  <ProviderOperationResolutionBanner
                    key={runtimeState.data.provider_operation.stage_id}
                    operation={runtimeState.data.provider_operation}
                    resolving={mutation.fenced || interruptMutation.fenced}
                    unavailableReason={
                      canResolveProvider && !runtimeState.error
                        ? null
                        : "Full access and a current state read are required."
                    }
                    error={
                      resolveProvider.error instanceof Error ? resolveProvider.error.message : null
                    }
                    onResolve={(action, reason) => resolveProvider.mutate({ action, reason })}
                  />
                ) : null}
                {transcript.isPending && (
                  <p className="text-sm text-muted-foreground">Loading conversation…</p>
                )}
                {(filters.role === "system" || filters.role === "tool") &&
                  transcript.data?.messages.map((message) => (
                    <article
                      key={message.index}
                      className="mb-3 space-y-2 rounded-lg border border-border p-3"
                    >
                      <span className="text-xs text-muted-foreground">{message.role}</span>
                      {message.content.map((part, index) =>
                        part.type === "structured_output" && !structuredOutputEnabled ? null : (
                          // biome-ignore lint/suspicious/noArrayIndexKey: parts have stable positions within immutable transcript records
                          <TranscriptPart key={index} part={part} />
                        ),
                      )}
                    </article>
                  ))}
                <Conversation
                  position={transcriptPosition}
                  blocks={filters.role === "system" || filters.role === "tool" ? [] : blocks}
                  displays={displays}
                  onTrace={(tool) => {
                    if (services.prototype) openTrace(toolSpanId(tool.id))
                    else {
                      setTranscriptPosition({
                        messageIndex: tool.messageIndex,
                        partIndex: tool.partIndex,
                      })
                      openTrace()
                    }
                  }}
                  pending="unknown"
                >
                  <SubagentRows rows={relations.rows} />
                  {relations.parent.error && <ErrorState error={relations.parent.error} />}
                  {relations.children.error && <ErrorState error={relations.children.error} />}
                  {relations.children.hasNextPage && (
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={relations.children.isFetchingNextPage}
                      onClick={() => void relations.children.fetchNextPage()}
                    >
                      Load more subagents
                    </Button>
                  )}
                  {state === "failed" && (
                    <p className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                      <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
                      {o?.reason ?? "This session failed."}
                    </p>
                  )}
                  {state === "interrupted" && items.length === 0 && (
                    <p className="text-sm text-muted-foreground">
                      This session was stopped. Send a message to continue it.
                    </p>
                  )}
                  {items.map((item) => (
                    <InboxCard
                      key={item.id}
                      item={item}
                      recoveryEnabled={canRecover}
                      mutationBlocked={
                        mutation.fenced ||
                        mutation.busy ||
                        interruptMutation.fenced ||
                        interruptMutation.busy
                      }
                      onRecover={async (body) => {
                        if (!canRecover)
                          throw new Error(
                            "Recovery requires full access and pending-action capability.",
                          )
                        const result = await mutation.execute("recovery", (options) =>
                          api.recover(sessionId, body, options),
                        )
                        if (result === null)
                          throw new DOMException("Observation stopped.", "AbortError")
                        if (result?.phase === "request_failed")
                          throw new Error(
                            mutationTransportErrorMessage(result) ?? "Recovery was rejected.",
                          )
                        refresh()
                        return result?.phase === "transport_failed" ? "unknown" : "observed"
                      }}
                    />
                  ))}
                  {running && (
                    <p className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span
                        className="size-2 animate-pulse rounded-full bg-info motion-reduce:animate-none"
                        aria-hidden="true"
                      />
                      Working…
                    </p>
                  )}
                </Conversation>
                <div ref={endRef} />
              </div>
            </div>

            <div className="relative mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
              {confirmingStop && running && (
                <StopConfirm
                  childCount={o?.uses.subagents ?? relations.rows.length}
                  busy={stop.isPending}
                  error={stop.error}
                  onCancel={() => {
                    setConfirmingStop(false)
                    stop.reset()
                  }}
                  onStop={(reason) => stop.mutate(reason)}
                />
              )}
              <MutationTransportStatus
                snapshot={mutation.snapshot}
                retryingObservation={mutation.busy}
                onRetryObservation={() => {
                  void mutation.reobserve().then((result) => {
                    if (!result) return
                    if (
                      mutation.kind === "rerun" &&
                      ["terminal", "runtime_failed"].includes(result.phase)
                    )
                      go(`/sessions/${encodeURIComponent(result.sessionId)}`)
                    if (result.phase === "terminal" && mutation.kind === "reply")
                      setDraft((current) =>
                        current.trim() === submittedDraft.current ? "" : current,
                      )
                    refresh()
                  })
                }}
              />
              <MutationTransportStatus
                snapshot={interruptMutation.snapshot}
                retryingObservation={interruptMutation.busy}
                onRetryObservation={() => {
                  void interruptMutation.reobserve().then((result) => {
                    if (!result) return
                    if (result.phase === "terminal") setConfirmingStop(false)
                    refresh()
                  })
                }}
              />
              {interruptMutation.error != null && <ErrorState error={interruptMutation.error} />}
              {canExecute && !services.prototype && (
                <div className="mb-2 space-y-2 text-sm">
                  {running && queue.unavailable && (
                    <p className="text-muted-foreground">
                      Queued messages require authenticated access and an application-configured
                      session-message policy. They are unavailable for this session.
                    </p>
                  )}
                  {queue.inspection.error && !queue.unavailable && (
                    <ErrorState error={queue.inspection.error} />
                  )}
                  {queue.unreadable && (
                    <p role="alert">
                      Some queued-message records cannot be read. Their delivery state is unknown.
                    </p>
                  )}
                  {queue.acceptance?.phase === "accepted" &&
                    queue.rows.length === 0 &&
                    queue.inspection.hasNextPage && (
                      <p role="status">
                        Message accepted. Load more queue records to inspect its current delivery
                        state.
                      </p>
                    )}
                  {queue.rows.length > 0 && (
                    <p>
                      {queue.rows.length}
                      {queue.inspection.hasNextPage ? " loaded" : ""}{" "}
                      {queue.rows.length === 1 ? "message" : "messages"} queued
                    </p>
                  )}
                  {queue.rows.map((record) => (
                    <div
                      key={record.queue_id}
                      className="flex items-start gap-2 rounded border border-border p-2"
                    >
                      <span className="min-w-0 flex-1 break-words">{record.message?.content}</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={queue.fenced}
                        aria-label="Withdraw queued message"
                        onClick={() => {
                          if (queue.instance) void queue.withdraw(record, queue.instance)
                        }}
                      >
                        ×
                      </Button>
                    </div>
                  ))}
                  {queue.inspection.hasNextPage && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={queue.inspection.isFetchingNextPage}
                      onClick={() => void queue.inspection.fetchNextPage()}
                    >
                      Load more queued messages
                    </Button>
                  )}
                  {queue.acceptance?.message && <p role="status">{queue.acceptance.message}</p>}
                  {queue.acceptance?.phase === "unknown" && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={queue.busy}
                      onClick={() => {
                        void queue.submit().then((content) => {
                          if (content !== null)
                            setDraft((current) => (current.trim() === content ? "" : current))
                        })
                      }}
                    >
                      Retry message acceptance
                    </Button>
                  )}
                  {queue.actionError && <p role="alert">{queue.actionError}</p>}
                </div>
              )}
              {canExecute && (
                <Composer
                  id="session-composer"
                  value={draft}
                  onChange={setDraft}
                  placeholder={
                    running
                      ? `${detail.data?.agent_name ?? "The specialist"} is working…`
                      : isWaiting(state)
                        ? "Answer the card above first, or send a message"
                        : "Message this session…"
                  }
                  running={running}
                  onStop={canInterrupt ? () => setConfirmingStop(true) : undefined}
                  disabled={
                    mutation.fenced ||
                    mutation.busy ||
                    interruptMutation.fenced ||
                    interruptMutation.busy ||
                    queue.fenced ||
                    (running &&
                      !services.prototype &&
                      (!queue.inspection.data || !!queue.inspection.error))
                  }
                  onSend={(text) => {
                    if (running && !services.prototype) {
                      void queue.submit(text).then((content) => {
                        if (content !== null)
                          setDraft((current) => (current.trim() === content ? "" : current))
                      })
                    } else reply.mutate(text)
                  }}
                />
              )}
              {mutation.error != null && <ErrorState error={mutation.error} />}
              {replayAcceptance?.message && <p role="status">{replayAcceptance.message}</p>}
              {(reply.error || runAgain.error || (stop.error && !confirmingStop)) && (
                <div className="mt-2">
                  <ErrorState error={reply.error ?? runAgain.error ?? stop.error} />
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {panel && detail.data && (
        <>
          <button
            type="button"
            aria-label="Close details"
            className="fixed inset-0 z-30 bg-black/20 xl:hidden"
            onClick={() => setPanel(undefined)}
          />
          <DetailsPanel
            returnFocus={detailsOpener}
            tab={panel === "artifacts" && !artifactsEnabled ? "overview" : panel}
            artifactsEnabled={artifactsEnabled}
            setTab={(t) => setPanel(t)}
            onClose={() => setPanel(undefined)}
            session={detail.data}
            overview={o ?? null}
            parent={relations.parent.data ?? null}
            onOpenTrace={() => setSearch({ view: "trace", panel: undefined })}
          />
        </>
      )}
    </div>
  )
}

/**
 * Confirm before stopping, as today's session page does: stopping doesn't undo finished
 * tool calls or cancel linked tasks, and active child sessions may be interrupted too.
 * The optional reason goes with the interrupt.
 */
function StopConfirm({
  childCount,
  busy,
  error,
  onCancel,
  onStop,
}: {
  childCount: number
  busy: boolean
  error: unknown
  onCancel: () => void
  onStop: (reason: string | undefined) => void
}) {
  const [reason, setReason] = useState("")
  return (
    <div
      role="dialog"
      aria-labelledby="stop-confirm-title"
      className="absolute right-4 bottom-full z-20 mb-2 w-80 max-w-[calc(100%-2rem)] rounded-xl border border-border bg-popover p-3 text-sm shadow-lg sm:right-6"
      onKeyDown={(e) => {
        if (e.key === "Escape" && !busy) onCancel()
      }}
    >
      <p id="stop-confirm-title" className="font-medium">
        Stop this session?
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        It ends as stopped. Tool calls that already finished, and their effects, aren't undone, and
        a task linked to it isn't cancelled.
      </p>
      {childCount > 0 && (
        <p className="mt-2 flex items-start gap-1.5 text-xs text-warning">
          <TriangleAlert className="mt-px size-3.5 shrink-0" aria-hidden="true" />
          {childCount === 1
            ? "It has a child session, which may be interrupted too if still active."
            : `It has ${childCount} child sessions, which may be interrupted too if still active.`}
        </p>
      )}
      <label htmlFor="stop-reason" className="mt-3 block text-xs font-medium">
        Reason <span className="font-normal text-muted-foreground">(optional)</span>
      </label>
      <textarea
        id="stop-reason"
        // biome-ignore lint/a11y/noAutofocus: the popover opens on Stop, so typing the reason comes next
        autoFocus
        rows={2}
        value={reason}
        disabled={busy}
        onChange={(e) => setReason(e.target.value)}
        placeholder="Why stop it?"
        className="mt-1 block w-full resize-none rounded-lg border border-input bg-background px-2.5 py-1.5 text-sm outline-none focus:border-ring"
      />
      {error != null && (
        <div className="mt-2">
          <ErrorState error={error} />
        </div>
      )}
      <div className="mt-3 flex justify-end gap-2">
        <Button size="sm" variant="outline" disabled={busy} onClick={onCancel}>
          Keep running
        </Button>
        <Button
          size="sm"
          variant="destructive"
          disabled={busy}
          onClick={() => onStop(reason.trim() || undefined)}
        >
          {busy ? "Stopping…" : "Stop session"}
        </Button>
      </div>
    </div>
  )
}

function MenuItem({
  icon: Icon,
  label,
  onClick,
  disabled,
}: {
  icon: typeof Copy
  label: string
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted"
    >
      <Icon className="size-3.5" aria-hidden="true" />
      {label}
    </button>
  )
}
