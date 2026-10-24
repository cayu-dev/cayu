/**
 * Everything that happened in a session, exactly: a waterfall of spans built from the
 * session's events (turns, model steps, tool rounds, tool calls, commands, waits), the
 * selected span's detail, and the flat event log with the server's filters.
 */
import { useQuery } from "@tanstack/react-query"
import { ChevronRight, Copy, Radio, X } from "lucide-react"
import { type ReactNode, useEffect, useMemo, useState } from "react"
import { MutationTransportStatus } from "@/components/dashboard/mutation-transport-status"
import { useDashboardCapability } from "@/components/dashboard/server-contract"
import { ApiClientError } from "@/lib/api"
import type { MutationTransportSnapshot } from "@/lib/mutation-transport"
import type { SessionHistorySearch } from "@/lib/session-history-search"
import { cn } from "@/lib/utils"
import type { WorkflowSearch } from "@/lib/workflow-search"
import { EVENT_CAP } from "../../api/paging"
import type { SessionDetail, SessionEvent, SessionState } from "../../api/types"
import type { SessionRow } from "../../app/queries"
import { useEvents, useSessionState } from "../../app/queries"
import { useApi, useNow, useServices } from "../../app/services"
import { argPreview, isActive, type ToolDisplays } from "../../app/session-model"
import { useTraceHistory } from "../../app/trace-history"
import {
  buildTrace,
  findSpan,
  flatten,
  formatSpanDuration,
  isHiddenValue,
  NOISY_EVENT_TYPES,
  type Span,
  type SpanStatus,
} from "../../app/trace-model"
import {
  buildRuntimeTrace,
  displayTraceValue,
  mergeTracePages,
  type TranscriptPosition,
  traceSpanForPosition,
} from "../../app/trace-runtime"
import { ErrorState, NavLink, needsAttention, StateDot, sessionPath } from "../common"
import { TraceEventLog } from "./trace-event-log"
import { TraceWorkflow } from "./trace-workflow"

export type TraceMutationObservation = {
  label: string
  snapshot: MutationTransportSnapshot | null
  retryObservation?: () => void
  retrying?: boolean
}
type TraceControls = {
  mutationSnapshots?: TraceMutationObservation[]
  onRetryInterruption?: () => void
  retryInterruptionBusy?: boolean
}
type Payload = Record<string, unknown>
const payload = (e: SessionEvent) => (e.payload ?? {}) as Payload
const ts = (e: SessionEvent) => Date.parse((e as { timestamp: string }).timestamp)
const toolOf = (e: SessionEvent) => (e as { tool_name?: string | null }).tool_name ?? null
const clock = (ms: number, seconds = true) =>
  new Date(ms).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    ...(seconds ? { second: "2-digit" } : {}),
  })

const TERMINAL: Partial<Record<SessionState, SpanStatus>> = {
  completed: "ok",
  failed: "error",
  interrupted: "error",
}

const BAR: Record<Span["kind"], string> = {
  session: "bg-foreground/25",
  interaction: "bg-foreground/20",
  model: "bg-info/70",
  round: "bg-foreground/15",
  tool: "bg-foreground/45",
  exec: "bg-foreground/30",
  wait: "bg-warning/40",
  knowledge: "bg-success/40",
}
const KIND_LABEL: Record<Span["kind"], string> = {
  session: "Session",
  interaction: "Turn",
  model: "Model step",
  round: "Tool round",
  tool: "Tool call",
  exec: "Command",
  wait: "Waiting",
  knowledge: "Knowledge lookup",
}
const STATUS_TEXT: Record<SpanStatus, string> = {
  ok: "text-success",
  error: "text-destructive",
  running: "text-info",
  waiting: "text-warning",
  interrupted: "text-warning",
  blocked: "text-warning",
  unknown: "text-muted-foreground",
}
const STATUS_LABEL: Record<SpanStatus, string> = {
  ok: "OK",
  error: "Failed",
  running: "Running",
  waiting: "Waiting",
  interrupted: "Interrupted",
  blocked: "Blocked",
  unknown: "Unknown",
}

/** A span's label in the waterfall: tool calls read like the conversation does. */
function spanLabel(s: Span, displays: ToolDisplays): { name: string; detail: string } {
  if (s.kind === "tool" && (s.status === "running" || s.status === "waiting")) {
    return { name: s.name, detail: "" }
  }
  if (s.kind === "tool") {
    const done = s.events.find(
      (e) => e.type === "tool.call.completed" || e.type === "tool.call.failed",
    )
    const args = payload(done ?? s.events[0] ?? ({} as SessionEvent)).arguments
    const display = displays[s.name]
    return { name: display ? display.verb : s.name, detail: args ? argPreview(args) : "" }
  }
  return { name: s.name, detail: s.detail }
}

type TraceViewProps = Omit<
  Parameters<typeof TraceViewContent>[0],
  "traceEnabled" | "workflowEnabled" | "retryEnabled"
>

export function TraceView(props: TraceViewProps) {
  const { prototype } = useServices()
  return prototype ? (
    <TraceViewContent
      {...props}
      traceEnabled={false}
      workflowEnabled={false}
      retryEnabled={false}
    />
  ) : (
    <RuntimeTraceView {...props} />
  )
}

function RuntimeTraceView(props: TraceViewProps) {
  const trace = useDashboardCapability({ kind: "surface", surface: "session_trace" })
  const workflow = useDashboardCapability({ kind: "surface", surface: "workflow" })
  const retry = useDashboardCapability({ kind: "mutation", mutation: "session_interruption" })
  return (
    <TraceViewContent
      {...props}
      traceEnabled={trace.enabled}
      workflowEnabled={workflow.enabled}
      retryEnabled={retry.enabled}
    />
  )
}

function TraceViewContent({
  session,
  state,
  displays,
  subagents,
  spanId,
  eventId,
  log,
  onSelect,
  onSelectEvent,
  onShowSpan,
  onLog,
  toolCallOrder,
  transcriptPosition,
  onConversation,
  historySearch = {},
  onHistorySearch,
  workflowSearch = {},
  onWorkflowSearch,
  mutationSnapshots,
  onRetryInterruption,
  retryInterruptionBusy,
  traceEnabled,
  workflowEnabled,
  retryEnabled,
  workflowOpen,
}: TraceControls & {
  traceEnabled: boolean
  workflowEnabled: boolean
  retryEnabled: boolean
  workflowOpen?: boolean
  session: SessionDetail
  state: SessionState | undefined
  displays: ToolDisplays
  subagents: SessionRow[]
  spanId: string | undefined
  /** The event selected in the event log; each tab keeps its own selection. */
  eventId: string | undefined
  log: boolean
  onSelect: (spanId: string) => void
  onSelectEvent: (eventId: string | undefined) => void
  /** Switch to the waterfall with this span selected. */
  onShowSpan: (spanId: string) => void
  onLog: (log: boolean) => void
  /** The conversation's tool call IDs in order, to find a call when the server hides IDs. */
  toolCallOrder: string[]
  historySearch?: SessionHistorySearch
  onHistorySearch?: (search: SessionHistorySearch) => void
  workflowSearch?: WorkflowSearch
  onWorkflowSearch?: (search: WorkflowSearch) => void
  transcriptPosition?: TranscriptPosition
  onConversation?: (position: TranscriptPosition) => void
}) {
  const now = useNow()
  const api = useApi()
  const { prototype } = useServices()
  const runtime = useTraceHistory(session.id, traceEnabled)
  const focusSequence = spanId?.match(
    /^(?:interaction|model|round|tool|exec|wait|knowledge):(\d+)$/,
  )?.[1]
  const focused = useQuery({
    queryKey: ["trace-focus", session.id, spanId, runtime.data?.page.session_instance_id],
    enabled: traceEnabled && !!focusSequence && !!runtime.data?.page,
    queryFn: ({ signal }) =>
      api.trace!(
        session.id,
        {
          after_sequence: Math.max(0, Number(focusSequence) - 1),
          limit: 1,
          session_instance_id: runtime.data?.page.session_instance_id,
        },
        signal,
      ),
  })
  const runtimePage = useMemo(
    () =>
      runtime.data?.page && focused.data
        ? mergeTracePages(runtime.data.page, focused.data)
        : runtime.data?.page,
    [runtime.data?.page, focused.data],
  )
  useEffect(() => {
    if (
      runtime.data?.hasMore &&
      !runtime.loadingMore &&
      !runtime.pageError &&
      ((transcriptPosition && !traceSpanForPosition(runtime.data.page, transcriptPosition)) ||
        (spanId &&
          spanId !== "session" &&
          !focused.isFetching &&
          !runtimePage?.spans.some((span) => span.id === spanId)))
    )
      void runtime.loadMore()
  }, [
    transcriptPosition,
    spanId,
    runtime.data,
    runtime.loadingMore,
    runtime.pageError,
    runtime.loadMore,
    runtimePage,
    focused.isFetching,
  ])

  const active = isActive(state)
  const prototypeEvents = useEvents(session.id, !!prototype, active)
  const events = prototype
    ? prototypeEvents
    : {
        ...runtime,
        data: runtime.data ? { events: runtimePage?.events ?? [], truncated: false } : undefined,
      }
  const all = events.data?.events ?? []
  const trace = useMemo(
    () =>
      runtimePage && !prototype
        ? buildRuntimeTrace(runtimePage)
        : buildTrace(session.id, all, now(), state ? (TERMINAL[state] ?? null) : null),
    [session.id, all, now, state, runtimePage, prototype],
  )
  const selected = useMemo(() => {
    if (transcriptPosition && runtime.data && !prototype) {
      const exact = traceSpanForPosition(runtime.data.page, transcriptPosition)
      if (exact) return findSpan(trace, exact) ?? trace
    }
    if (!spanId) return trace
    const found = findSpan(trace, spanId)
    if (found) return found
    // Hidden IDs: the n-th tool call in the conversation is the n-th tool span.
    const index =
      prototype && spanId.startsWith("tool:") ? toolCallOrder.indexOf(spanId.slice(5)) : -1
    if (index >= 0) {
      const tools: Span[] = []
      const walk = (s: Span) => {
        if (s.kind === "tool") tools.push(s)
        for (const c of s.children) walk(c)
      }
      walk(trace)
      tools.sort((a, b) => a.start - b.start)
      if (tools[index]) return tools[index]
    }
    return trace
  }, [trace, spanId, toolCallOrder, transcriptPosition, runtime.data, prototype])
  const spanOf = useMemo(() => {
    const map = new Map<string, Span>()
    const walk = (s: Span) => {
      for (const e of s.events) map.set(e.id, s)
      for (const c of s.children) walk(c)
    }
    walk(trace)
    return map
  }, [trace])
  const selectedEvent = eventId ? all.find((e) => e.id === eventId) : undefined

  if (!prototype && !traceEnabled)
    return (
      <p className="p-6 text-sm text-muted-foreground">
        Exact session traces are unavailable in this deployment.
      </p>
    )

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2 text-sm sm:px-6">
        <div
          className="flex rounded-lg border border-border p-0.5"
          role="tablist"
          aria-label="Trace view"
        >
          {[
            [false, "Waterfall"],
            [true, "Event log"],
          ].map(([value, label]) => (
            <button
              key={String(label)}
              type="button"
              role="tab"
              aria-selected={log === value}
              onClick={() => onLog(value as boolean)}
              className={cn(
                "rounded-md px-2.5 py-1 text-[13px]",
                log === value
                  ? "bg-muted font-medium"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">
          {all.length.toLocaleString()} loaded events
          {trace.end ? ` · ${formatSpanDuration(trace.end - trace.start)}` : ""}
        </span>
        {events.data?.truncated && (
          <span
            className="text-xs text-warning"
            title={`The trace keeps the newest ${EVENT_CAP.toLocaleString()} events. Older ones are still on the server.`}
          >
            Older events not loaded
          </span>
        )}
        {trace.linkedByOrder && (
          <span
            className="text-xs text-muted-foreground"
            title="The server's public events hide model-step and tool-call IDs, so spans are linked by event order. A server-side trace endpoint would link them exactly."
          >
            Linked by event order
          </span>
        )}
        {active && (
          <span className="flex items-center gap-1 text-xs text-info">
            <Radio
              className="size-3.5 animate-pulse motion-reduce:animate-none"
              aria-hidden="true"
            />
            Live
          </span>
        )}
      </div>
      {!prototype && runtime.data?.hasMore && (
        <button
          type="button"
          className="m-2 rounded border px-3 py-2 text-sm"
          disabled={runtime.loadingMore}
          onClick={() => void runtime.loadMore()}
        >
          {runtime.loadingMore ? "Loading history…" : "Load more trace history"}
        </button>
      )}
      {!prototype && runtime.pageError != null && <ErrorState error={runtime.pageError} />}
      {focused.error && <ErrorState error={focused.error} />}
      {transcriptPosition &&
        runtime.data &&
        !traceSpanForPosition(runtime.data.page, transcriptPosition) && (
          <p className="p-3 text-sm">
            {runtime.data.hasMore
              ? "Loading the linked transcript step…"
              : "No retained exact span matches this transcript part."}
          </p>
        )}
      {events.error && (
        <div className="p-4">
          <ErrorState error={events.error} />
          <button
            type="button"
            className="mt-2 rounded border px-3 py-2 text-sm"
            onClick={() =>
              events.error instanceof ApiClientError && events.error.status === 409
                ? window.location.reload()
                : void events.refetch()
            }
          >
            {events.error instanceof ApiClientError && events.error.status === 409
              ? "Reload session trace"
              : "Retry trace update"}
          </button>
        </div>
      )}
      {events.isPending && <p className="p-6 text-sm text-muted-foreground">Loading events…</p>}
      {!prototype && log && (
        <TraceEventLog
          key={session.id}
          sessionId={session.id}
          search={historySearch}
          onSearch={onHistorySearch ?? (() => {})}
          selected={eventId ?? historySearch.event_id}
          onSelect={onSelectEvent}
          renderDetail={(event, observedAt) => (
            <RuntimeEventDetail
              sessionId={session.id}
              event={event}
              sessionInstanceId={runtimePage?.session_instance_id}
              observedAt={observedAt}
              traceUnavailable={!!(runtime.error || runtime.pageError || focused.error)}
              displays={displays}
              onClose={() => onSelectEvent(undefined)}
              onShowSpan={onShowSpan}
            />
          )}
        />
      )}
      {events.data && (prototype || !log) && (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="min-h-0 min-w-0 flex-1 overflow-auto">
            {log ? (
              <EventLog events={all} selected={eventId} onSelect={onSelectEvent} />
            ) : (
              <Waterfall
                trace={trace}
                selected={selected.id}
                displays={displays}
                onSelect={onSelect}
              />
            )}
          </div>
          {(!log || selectedEvent) && (
            <aside
              aria-label={log ? "Event detail" : "Span detail"}
              className="max-h-[45vh] min-h-0 shrink-0 overflow-y-auto border-t border-border lg:max-h-none lg:w-[420px] lg:border-t-0 lg:border-l"
            >
              {log && selectedEvent ? (
                <EventDetail
                  event={selectedEvent}
                  span={spanOf.get(selectedEvent.id) ?? null}
                  displays={displays}
                  onClose={() => onSelectEvent(undefined)}
                  onShowSpan={onShowSpan}
                />
              ) : (
                <SpanDetail
                  span={selected}
                  session={session}
                  displays={displays}
                  subagents={subagents}
                  onConversation={onConversation}
                  controls={{
                    mutationSnapshots,
                    onRetryInterruption: retryEnabled ? onRetryInterruption : undefined,
                    retryInterruptionBusy,
                  }}
                  workflowOpen={workflowOpen || !!workflowSearch.expanded_session_id?.length}
                  workflow={
                    !prototype && workflowEnabled && onWorkflowSearch ? (
                      <TraceWorkflow
                        key={session.id}
                        sessionId={session.id}
                        search={workflowSearch}
                        onSearch={onWorkflowSearch}
                      />
                    ) : undefined
                  }
                />
              )}
            </aside>
          )}
        </div>
      )}
    </div>
  )
}

function Waterfall({
  trace,
  selected,
  displays,
  onSelect,
}: {
  trace: Span
  selected: string
  displays: ToolDisplays
  onSelect: (id: string) => void
}) {
  const now = useNow()
  const [, tick] = useState(0)
  useEffect(() => {
    if (trace.status !== "running" && trace.status !== "waiting") return
    const timer = setInterval(() => tick((value) => value + 1), 1000)
    return () => clearInterval(timer)
  }, [trace.status])
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const rows = flatten(trace, collapsed)
  const displayEnd = (span: Span) =>
    span.end ?? (span.status === "running" || span.status === "waiting" ? now() : span.start)
  const total = Math.max(1, displayEnd(trace) - trace.start)
  const toggle = (id: string) =>
    setCollapsed((c) => {
      const next = new Set(c)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <div className="min-w-[640px] py-1 text-[13px]" role="tree" aria-label="Session trace">
      <div className="sticky top-0 z-10 grid grid-cols-[minmax(16rem,2fr)_3fr_4.5rem] gap-3 border-b border-border bg-background px-4 py-1.5 text-xs text-muted-foreground sm:px-6">
        <span>Span</span>
        <span className="flex justify-between tabular-nums">
          <span>{clock(trace.start)}</span>
          <span>{trace.end ? clock(trace.end) : ""}</span>
        </span>
        <span className="text-right">Duration</span>
      </div>
      {rows.map(({ span, depth }) => {
        const { name, detail } = spanLabel(span, displays)
        const left = ((span.start - trace.start) / total) * 100
        const width = Math.max(0.4, ((displayEnd(span) - span.start) / total) * 100)
        return (
          <div
            key={span.id}
            role="treeitem"
            aria-selected={selected === span.id}
            aria-expanded={span.children.length ? !collapsed.has(span.id) : undefined}
            tabIndex={0}
            onClick={() => onSelect(span.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSelect(span.id)
              if (e.key === "ArrowLeft" && span.children.length && !collapsed.has(span.id))
                toggle(span.id)
              if (e.key === "ArrowRight" && collapsed.has(span.id)) toggle(span.id)
            }}
            className={cn(
              "grid cursor-pointer grid-cols-[minmax(16rem,2fr)_3fr_4.5rem] items-center gap-3 px-4 py-1 hover:bg-muted/60 focus-visible:bg-muted/60 focus-visible:outline-none sm:px-6",
              selected === span.id && "bg-muted",
            )}
          >
            <span className="flex min-w-0 items-center gap-1" style={{ paddingLeft: depth * 14 }}>
              {span.children.length ? (
                <button
                  type="button"
                  aria-label={collapsed.has(span.id) ? "Expand" : "Collapse"}
                  onClick={(e) => {
                    e.stopPropagation()
                    toggle(span.id)
                  }}
                  className="rounded p-0.5 text-muted-foreground hover:text-foreground"
                >
                  <ChevronRight
                    className={cn(
                      "size-3.5 transition-transform",
                      !collapsed.has(span.id) && "rotate-90",
                    )}
                  />
                </button>
              ) : (
                <span className="w-[18px] shrink-0" />
              )}
              <span className={cn("shrink-0", span.status === "error" && "text-destructive")}>
                {name}
              </span>
              {detail && (
                <span className="min-w-0 truncate font-mono text-xs text-muted-foreground">
                  {detail}
                </span>
              )}
            </span>
            <span className="relative h-3.5">
              {span.startKnown !== false && (
                <span
                  className={cn(
                    "absolute top-0 h-full rounded-sm",
                    span.status === "error"
                      ? "bg-destructive/70"
                      : span.status === "interrupted" || span.status === "blocked"
                        ? "bg-warning/70"
                        : BAR[span.kind],
                    span.status === "waiting" &&
                      "bg-[repeating-linear-gradient(45deg,var(--warning)_0_4px,transparent_4px_8px)] opacity-60",
                  )}
                  style={{
                    left: `${Math.max(0, left)}%`,
                    width: `${Math.max(0, Math.min(width, 100 - left))}%`,
                  }}
                />
              )}
            </span>
            <span
              className={cn(
                "text-right text-xs tabular-nums",
                span.status === "error"
                  ? "text-destructive"
                  : span.status === "waiting"
                    ? "text-warning"
                    : "text-muted-foreground",
              )}
            >
              {span.startKnown === false
                ? "Unknown"
                : formatSpanDuration(displayEnd(span) - span.start)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function Fact({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[8.5rem_1fr] gap-3 py-1 text-sm">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

function Json({ value }: { value: unknown }) {
  const shown = displayTraceValue(value)
  return (
    <pre className="max-h-72 overflow-auto rounded-md bg-muted p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-words">
      {typeof shown === "string" ? shown : JSON.stringify(shown, null, 2)}
    </pre>
  )
}

const num = (v: unknown) =>
  isHiddenValue(v)
    ? null
    : typeof v === "number"
      ? v.toLocaleString()
      : typeof v === "string"
        ? v
        : null

function SpanDetail({
  span,
  session,
  displays,
  subagents,
  onConversation,
  workflow,
  controls,
  workflowOpen,
}: {
  workflowOpen?: boolean
  span: Span
  controls?: TraceControls
  workflow?: ReactNode
  session: SessionDetail
  displays: ToolDisplays
  subagents: SessionRow[]
  onConversation?: (position: TranscriptPosition) => void
}) {
  const of = (type: string) => span.events.find((e) => e.type === type)
  const facts: ReactNode[] = []
  const sections: ReactNode[] = []
  const duration =
    span.startKnown === false
      ? "Start not retained"
      : formatSpanDuration((span.end ?? span.start) - span.start)

  if (span.kind === "session") {
    const started = payload(of("session.started") ?? ({} as SessionEvent))
    const env = span.children
      .flatMap((c) => c.events)
      .concat(span.events)
      .find((e) => e.type === "environment.factory.completed")
    facts.push(
      <Fact
        key="rt"
        k="Runtime"
      >{`${session.runtime_name} ${session.runtime_version ?? ""}`}</Fact>,
      <Fact key="env" k="Environment">
        {session.environment_name ?? "–"}
        {env && num(payload(env).factory_type) ? ` (${payload(env).factory_type})` : ""}
      </Fact>,
      <Fact key="budget" k="Causal budget">
        <span className="font-mono text-xs">{session.causal_budget_id ?? "–"}</span>
      </Fact>,
      <Fact key="started" k="Started by">
        {session.invocation.source.replaceAll("_", " ")}
      </Fact>,
      <Fact key="epoch" k="Run epoch">
        {num(started.run_epoch) ?? "–"}
      </Fact>,
    )
    if (started.execution_deadline) {
      sections.push(
        <Section key="deadline" title="Execution deadline">
          <Json value={started.execution_deadline} />
        </Section>,
      )
    }
    if (workflow)
      sections.push(
        <WorkflowSection key="workflow" initiallyOpen={workflowOpen}>
          {workflow}
        </WorkflowSection>,
      )
    sections.push(
      <Section key="subagents" title="Parent and subagent sessions">
        {session.parent_session_id && (
          <NavLink
            to={sessionPath(session.parent_session_id)}
            className="block text-sm hover:underline"
          >
            Parent: {session.parent_session_id}
          </NavLink>
        )}
        {subagents.length === 0 && !session.parent_session_id && (
          <p className="text-sm text-muted-foreground">None.</p>
        )}
        {subagents.map((r) => (
          <NavLink
            key={r.session.id}
            to={sessionPath(r.session.id)}
            className="flex items-center gap-2 rounded-md py-1 text-sm hover:underline"
          >
            <StateDot state={r.overview?.state} attention={needsAttention(r.overview)} />
            <span className="truncate">{r.overview?.title ?? r.session.id}</span>
            <span className="ml-auto shrink-0 text-xs text-muted-foreground">
              {r.session.agent_name}
            </span>
          </NavLink>
        ))}
      </Section>,
      <RuntimeInternals key="internals" sessionId={session.id} {...controls} />,
      <Section key="side" title="Side effects">
        <p className="text-sm text-muted-foreground">
          Outbox deliveries for these events come from{" "}
          <code className="text-xs">/api/event-side-effects</code>. None recorded for this session.
        </p>
      </Section>,
    )
  } else if (span.kind === "interaction") {
    const done = payload(of("interaction.completed") ?? ({} as SessionEvent))
    facts.push(
      <Fact key="status" k="Status">
        {String(done.status ?? "running")}
      </Fact>,
      <Fact key="steps" k="Model steps">
        {num(done.model_step_count) ?? "–"}
      </Fact>,
      <Fact key="active" k="Active time">
        {typeof done.active_duration_ms === "number"
          ? formatSpanDuration(done.active_duration_ms)
          : "–"}
      </Fact>,
    )
    if (done.pending_action_kind)
      facts.push(
        <Fact key="pending" k="Waiting on">
          {String(done.pending_action_kind)}
        </Fact>,
      )
  } else if (span.kind === "model") {
    const started = span.events.filter((e) => e.type === "model.started")
    const done = payload(of("model.completed") ?? ({} as SessionEvent))
    const usage = (done.usage ?? {}) as Payload
    const completion = (done.completion ?? {}) as Payload
    const footprint = payload(of("request.footprint.recorded") ?? ({} as SessionEvent))
    const exposure = payload(of("tool.exposure.recorded") ?? ({} as SessionEvent))
    facts.push(
      <Fact key="model" k="Model">
        {String(done.model ?? payload(started[0] ?? ({} as SessionEvent)).model ?? "–")}
        {done.provider_name ? ` via ${done.provider_name}` : ""}
      </Fact>,
      <Fact key="attempts" k="Attempts">
        {started.length || "–"}
      </Fact>,
      <Fact key="finish" k="Finish reason">
        {String(completion.finish_reason ?? done.status ?? "–")}
      </Fact>,
      <Fact key="tokens" k="Tokens">
        {num(usage.input_tokens) ?? "–"} in ·{" "}
        {num(usage.output_tokens ?? completion.output_tokens) ?? "–"} out
      </Fact>,
    )
    const components = (footprint.component_tokens ?? null) as Payload | null
    if (components && Object.keys(components).length) {
      sections.push(
        <Section key="request" title="What went into the request">
          <dl>
            {Object.entries(components).map(([k, v]) => (
              <Fact key={k} k={k.replaceAll("_", " ")}>
                <span className="tabular-nums">{num(v) ?? JSON.stringify(v)}</span>
              </Fact>
            ))}
            {footprint.cache_breakpoints != null && (
              <Fact k="cache breakpoints">{num(footprint.cache_breakpoints)}</Fact>
            )}
          </dl>
          {footprint.context_pressure != null && <Json value={footprint.context_pressure} />}
        </Section>,
      )
    }
    if (Object.keys(exposure).length) {
      sections.push(
        <Section key="exposure" title="Tools offered">
          <p className="text-sm">
            {num(exposure.exposed_count) ?? "?"} of {num(exposure.registered_count) ?? "?"}{" "}
            registered tools
            {exposure.ceiling_count != null ? ` (limit ${num(exposure.ceiling_count)})` : ""}
          </p>
        </Section>,
      )
    }
  } else if (span.kind === "tool" || span.kind === "wait") {
    const done = span.events.find(
      (e) => e.type === "tool.call.completed" || e.type === "tool.call.failed",
    )
    const first = span.events[0]
    const name = first ? toolOf(first) : null
    const p = payload(done ?? first ?? ({} as SessionEvent))
    facts.push(
      <Fact key="tool" k="Tool">
        <span className="font-mono text-xs">{name ?? span.detail ?? "–"}</span>
        {name && displays[name] ? ` (${displays[name]?.verb})` : ""}
      </Fact>,
    )
    if (p.effect)
      facts.push(
        <Fact key="effect" k="Effect">
          {String(p.effect)}
        </Fact>,
      )
    if (p.idempotency_key && !isHiddenValue(p.idempotency_key))
      facts.push(
        <Fact key="idem" k="Idempotency key">
          <span className="font-mono text-xs">{String(p.idempotency_key)}</span>
        </Fact>,
      )
    if (p.reason)
      facts.push(
        <Fact key="reason" k="Reason">
          {String(p.reason)}
        </Fact>,
      )
    if (p.arguments !== undefined)
      sections.push(
        <Section key="args" title="Arguments">
          <Json value={p.arguments} />
        </Section>,
      )
    if (p.result !== undefined)
      sections.push(
        <Section key="result" title="Result">
          <Json value={p.result} />
        </Section>,
      )
  } else if (span.kind === "knowledge") {
    sections.push(
      <Section key="knowledge" title="Knowledge lookup">
        <Json value={payload(span.events[0] ?? ({} as SessionEvent))} />
      </Section>,
    )
  } else if (span.kind === "exec") {
    const done = payload(of("runner.exec.completed") ?? ({} as SessionEvent))
    sections.push(
      <Section key="command" title="Command evidence">
        <Json
          value={done.command ?? payload(of("runner.exec.started") ?? ({} as SessionEvent)).command}
        />
      </Section>,
    )
    facts.push(
      <Fact key="cmd" k="Command">
        <span className="font-mono text-xs">{span.detail || "–"}</span>
      </Fact>,
      <Fact key="adapter" k="Adapter">
        {String(done.adapter ?? "–")}
      </Fact>,
      <Fact key="exit" k="Exit code">
        {num(done.exit_code) ?? "–"}
      </Fact>,
      <Fact key="timeout" k="Timed out">
        {done.timed_out ? "Yes" : "No"}
      </Fact>,
    )
  } else if (span.kind === "round") {
    facts.push(
      <Fact key="calls" k="Tool calls">
        {span.children.length}
      </Fact>,
    )
  }

  const copy = () => {
    void navigator.clipboard
      ?.writeText(JSON.stringify(displayTraceValue(span.events), null, 2))
      .catch(() => undefined)
  }

  return (
    <div className="space-y-4 p-4">
      <div className="space-y-1">
        <p className="text-xs font-medium text-muted-foreground">{KIND_LABEL[span.kind]}</p>
        <p className="font-medium break-words">{spanLabel(span, displays).name}</p>
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span className={cn("font-medium", STATUS_TEXT[span.status])}>
            {STATUS_LABEL[span.status]}
          </span>
          <span className="tabular-nums">{duration}</span>
          <span className="tabular-nums">started {clock(span.start)}</span>
          {span.kind === "tool" && onConversation && span.transcriptLinks?.[0] ? (
            <button
              type="button"
              className="underline"
              onClick={() =>
                onConversation({
                  messageIndex: span.transcriptLinks![0]!.message_index,
                  partIndex: span.transcriptLinks![0]!.part_index,
                })
              }
            >
              Show in conversation
            </button>
          ) : (
            span.kind === "tool" && (
              <NavLink
                to={sessionPath(session.id)}
                search={{ view: "conversation" }}
                className="text-foreground hover:underline"
              >
                Show in conversation
              </NavLink>
            )
          )}
        </p>
      </div>
      {facts.length > 0 && <dl className="divide-y divide-border">{facts}</dl>}
      {sections}
      <Section
        title={`Events (${span.events.length})`}
        action={
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            <Copy className="size-3" aria-hidden="true" />
            Copy as JSON
          </button>
        }
      >
        <ol className="space-y-0.5">
          {span.events.map((e) => (
            <EventRow key={e.id} event={e} />
          ))}
        </ol>
      </Section>
    </div>
  )
}

/**
 * What runtime developers need when a session misbehaves, collapsed until asked for:
 * execution state and owner, snapshots, the interruption cascade, the provider operation,
 * and the mutation transport when the state reports it. Read from `/sessions/{id}/state`.
 */
function WorkflowSection({
  initiallyOpen,
  children,
}: {
  initiallyOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(!!initiallyOpen)
  useEffect(() => {
    if (initiallyOpen) setOpen(true)
  }, [initiallyOpen])
  return (
    <details
      className="rounded border"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer p-3 text-sm">Session tree and causal usage</summary>
      {open && children}
    </details>
  )
}

function RuntimeInternals({
  sessionId,
  mutationSnapshots = [],
  onRetryInterruption,
  retryInterruptionBusy,
}: { sessionId: string } & TraceControls) {
  const [open, setOpen] = useState(false)
  const state = useSessionState(sessionId, open)
  const data = state.data
  const cascade = data?.interruption_cascade
  return (
    <details
      className="rounded-lg border border-border"
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground">
        Runtime internals
      </summary>
      <div className="space-y-3 border-t border-border px-3 py-3">
        {state.isPending && <p className="text-xs text-muted-foreground">Loading…</p>}
        {state.error && <ErrorState error={state.error} />}
        {data && (
          <>
            <dl className="divide-y divide-border">
              <Fact k="Status">{data.status}</Fact>
              <Fact k="Interruption cascade">
                <span
                  className={cn(
                    cascade === "failed" && "text-destructive",
                    cascade === "pending" && "text-warning",
                  )}
                >
                  {cascade}
                </span>
              </Fact>
              <Fact k="Execution owner">
                {data.execution?.owner_kind
                  ? `${data.execution.owner_kind}${data.execution.owner_label ? ` (${data.execution.owner_label})` : ""}`
                  : "none"}
              </Fact>
              <Fact k="Mutation transport">
                {mutationSnapshots.some((item) => item.snapshot)
                  ? "Observed in this browser"
                  : "No action observed in this browser"}
              </Fact>
            </dl>
            {mutationSnapshots.map((item) => (
              <section key={item.label} aria-label={item.label}>
                <h4 className="text-xs font-medium">{item.label}</h4>
                <MutationTransportStatus
                  snapshot={item.snapshot}
                  onRetryObservation={item.retryObservation}
                  retryingObservation={item.retrying}
                />
              </section>
            ))}
            {(cascade === "pending" || cascade === "failed") && onRetryInterruption && (
              <button
                type="button"
                className="rounded border px-3 py-2 text-sm"
                disabled={retryInterruptionBusy}
                onClick={onRetryInterruption}
              >
                {retryInterruptionBusy ? "Retrying interruption…" : "Retry background interruption"}
              </button>
            )}
            <Section title="Execution">
              <Json value={data.execution} />
            </Section>
            <Section title={`Execution snapshots (${data.execution_snapshots?.length ?? 0})`}>
              {data.execution_snapshots?.length ? (
                <Json value={data.execution_snapshots} />
              ) : (
                <p className="text-xs text-muted-foreground">None recorded.</p>
              )}
            </Section>
            <Section title="Complete runtime state">
              <Json value={data} />
            </Section>
            <Section title="Provider operation">
              <Json value={data.provider_operation} />
            </Section>
          </>
        )}
      </div>
    </details>
  )
}

function Section({
  title,
  action,
  children,
}: {
  title: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="space-y-1.5">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-muted-foreground">{title}</h3>
        {action}
      </div>
      {children}
    </section>
  )
}

/** The first few readable scalar fields of an event, without values the server hid. */
function eventSummary(event: SessionEvent, count = 3) {
  return Object.entries(payload(event))
    .filter(
      ([, v]) =>
        (typeof v === "string" || typeof v === "number" || typeof v === "boolean") &&
        !isHiddenValue(v),
    )
    .slice(0, count)
    .map(([k, v]) => `${k}=${String(v)}`)
    .join("  ")
}

function EventTypeName({ event }: { event: SessionEvent }) {
  return (
    <>
      <span
        className={cn(
          event.type.includes("failed")
            ? "text-destructive"
            : event.type.startsWith("tool.")
              ? "text-info"
              : "",
        )}
      >
        {event.type}
      </span>
      {toolOf(event) && <span className="text-muted-foreground"> {toolOf(event)}</span>}
    </>
  )
}

/** An event in a span's list: expands in place to its payload. */
function EventRow({ event }: { event: SessionEvent }) {
  const [open, setOpen] = useState(false)
  const p = payload(event)
  const summary = eventSummary(event)
  return (
    <li>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="grid w-full grid-cols-[3.5rem_5.5rem_minmax(0,1fr)] items-baseline gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-muted"
      >
        <span className="font-mono tabular-nums text-muted-foreground">
          {String(event.sequence).slice(-5)}
        </span>
        <span className="tabular-nums text-muted-foreground">{clock(ts(event))}</span>
        <span className="min-w-0 truncate font-mono">
          <EventTypeName event={event} />
          {summary && <span className="text-muted-foreground"> · {summary}</span>}
        </span>
      </button>
      {open && (
        <div className="mt-1 mb-2 ml-1">
          <Json value={{ id: event.id, interaction_id: event.interaction_id, ...p }} />
        </div>
      )}
    </li>
  )
}

/** The event selected in the log: its fields, the span it belongs to, and its raw payload. */
function RuntimeEventDetail({
  sessionId,
  sessionInstanceId,
  observedAt,
  traceUnavailable,
  ...props
}: Omit<Parameters<typeof EventDetail>[0], "span"> & {
  sessionId: string
  sessionInstanceId?: string
  observedAt: number
  traceUnavailable: boolean
}) {
  const api = useApi()
  // /events has no incarnation field. Every fresh event observation must be
  // verified by the bound trace read, including events already in the cached tree.
  const query = useQuery({
    queryKey: ["trace-event-span", sessionId, sessionInstanceId, props.event.id, observedAt],
    enabled: !!sessionInstanceId && !traceUnavailable,
    retry: false,
    queryFn: ({ signal }) =>
      api.trace!(
        sessionId,
        {
          after_sequence: Math.max(0, props.event.sequence - 1),
          session_instance_id: sessionInstanceId,
          limit: 1,
        },
        signal,
      ),
  })
  const verified =
    !traceUnavailable &&
    !query.isFetching &&
    !query.error &&
    query.data?.session_instance_id === sessionInstanceId
      ? query.data
      : undefined
  const event = verified?.events.find((event) => event.id === props.event.id)
  const exact = event && verified?.spans.find((span) => span.event_ids.includes(event.id))
  const span = verified && exact ? findSpan(buildRuntimeTrace(verified), exact.id) : null
  return (
    <>
      {query.error && <ErrorState error={query.error} />}
      <EventDetail {...props} event={event ?? props.event} span={span} />
    </>
  )
}

function EventDetail({
  event,
  span,
  displays,
  onClose,
  onShowSpan,
}: {
  event: SessionEvent
  span: Span | null
  displays: ToolDisplays
  onClose: () => void
  onShowSpan: (spanId: string) => void
}) {
  const p = payload(event)
  const entries = Object.entries(p)
  const hidden = entries.filter(([, v]) => isHiddenValue(v)).map(([k]) => k)
  const fields = entries.filter(
    ([, v]) =>
      !isHiddenValue(v) &&
      (typeof v === "string" || typeof v === "number" || typeof v === "boolean"),
  )
  const copy = () => {
    void navigator.clipboard
      ?.writeText(JSON.stringify(displayTraceValue(event), null, 2))
      .catch(() => undefined)
  }
  return (
    <div className="space-y-4 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <p className="text-xs font-medium text-muted-foreground">Event</p>
          <p className="font-mono text-sm font-medium break-words">
            <EventTypeName event={event} />
          </p>
          <p className="text-xs text-muted-foreground tabular-nums">
            #{event.sequence} · {clock(ts(event))}
          </p>
        </div>
        <button
          type="button"
          aria-label="Close event"
          onClick={onClose}
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <X className="size-4" />
        </button>
      </div>
      {span && (
        <p className="text-sm">
          Part of{" "}
          <button
            type="button"
            onClick={() => onShowSpan(span.id)}
            className="font-medium underline-offset-2 hover:underline"
          >
            {KIND_LABEL[span.kind].toLowerCase()} {spanLabel(span, displays).name}
          </button>{" "}
          <span className="text-muted-foreground">(show in waterfall)</span>
        </p>
      )}
      {fields.length > 0 && (
        <dl className="divide-y divide-border">
          {fields.map(([k, v]) => (
            <Fact key={k} k={k.replaceAll("_", " ")}>
              <span className="font-mono text-xs">{String(v)}</span>
            </Fact>
          ))}
        </dl>
      )}
      {hidden.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Hidden by the server: {hidden.join(", ")}. Public events don't carry these IDs.
        </p>
      )}
      <Section
        title="Raw event"
        action={
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            <Copy className="size-3" aria-hidden="true" />
            Copy as JSON
          </button>
        }
      >
        <Json value={{ id: event.id, interaction_id: event.interaction_id, ...p }} />
      </Section>
    </div>
  )
}

function EventLog({
  events,
  selected,
  onSelect,
}: {
  events: SessionEvent[]
  selected: string | undefined
  onSelect: (eventId: string | undefined) => void
}) {
  const [hideNoise, setHideNoise] = useState(true)
  const [type, setType] = useState("")
  const [tool, setTool] = useState("")
  const [turn, setTurn] = useState("")
  const types = [...new Set(events.map((e) => e.type))].sort()
  const tools = [...new Set(events.map(toolOf).filter((t): t is string => !!t))].sort()
  const turns = [...new Set(events.map((e) => e.interaction_id).filter((t): t is string => !!t))]
  const rows = events.filter(
    (e) =>
      (!hideNoise || type === e.type || !NOISY_EVENT_TYPES.has(e.type)) &&
      (!type || e.type === type) &&
      (!tool || toolOf(e) === tool) &&
      (!turn || e.interaction_id === turn),
  )
  const field = "h-7 rounded-md border border-input bg-background px-2 text-xs"

  return (
    <div className="px-4 py-3 sm:px-6">
      <div className="flex flex-wrap items-center gap-2 pb-2">
        <select
          id="log-type"
          aria-label="Event type"
          className={field}
          value={type}
          onChange={(e) => setType(e.target.value)}
        >
          <option value="">All event types</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        {tools.length > 0 && (
          <select
            id="log-tool"
            aria-label="Tool"
            className={field}
            value={tool}
            onChange={(e) => setTool(e.target.value)}
          >
            <option value="">All tools</option>
            {tools.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        )}
        {turns.length > 1 && (
          <select
            id="log-turn"
            aria-label="Turn"
            className={field}
            value={turn}
            onChange={(e) => setTurn(e.target.value)}
          >
            <option value="">All turns</option>
            {turns.map((t, i) => (
              <option key={t} value={t}>
                Turn {i + 1}
              </option>
            ))}
          </select>
        )}
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <input
            id="log-hide-noise"
            type="checkbox"
            checked={hideNoise}
            onChange={(e) => setHideNoise(e.target.checked)}
          />
          Hide thinking, footprints and exposure records
        </label>
        <span className="ml-auto text-xs text-muted-foreground tabular-nums">
          {rows.length} of {events.length}
        </span>
      </div>
      <ol className="space-y-0.5">
        {rows.map((e) => {
          const summary = eventSummary(e)
          return (
            <li key={e.id}>
              <button
                type="button"
                aria-pressed={selected === e.id}
                onClick={() => onSelect(selected === e.id ? undefined : e.id)}
                className={cn(
                  "grid w-full grid-cols-[3.5rem_5.5rem_minmax(0,1fr)] items-baseline gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-muted",
                  selected === e.id && "bg-muted",
                )}
              >
                <span className="font-mono tabular-nums text-muted-foreground">
                  {String(e.sequence).slice(-5)}
                </span>
                <span className="tabular-nums text-muted-foreground">{clock(ts(e))}</span>
                <span className="min-w-0 truncate font-mono">
                  <EventTypeName event={e} />
                  {summary && <span className="text-muted-foreground"> · {summary}</span>}
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
