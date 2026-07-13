import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Link, useParams } from "@tanstack/react-router"
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  Bot,
  CheckCircle,
  ChevronRight,
  CircleStop,
  Clock,
  Cpu,
  Database,
  Hash,
  LoaderCircle,
  type LucideIcon,
  MessageSquare,
  RefreshCw,
  Send,
  ShieldCheck,
  UserRound,
  Wrench,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Page, PayloadViewer, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button, buttonVariants } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Input } from "../components/ui/input"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "../components/ui/sheet"
import { Textarea } from "../components/ui/textarea"
import {
  ApiClientError,
  type PendingAction as ApiPendingAction,
  type ArtifactSummary,
  fetchArtifacts,
  fetchPendingActions,
  fetchSession,
  fetchSessionEvents,
  fetchSessionState,
  fetchSessionSummary,
  fetchSessionTranscript,
  isApiPayloadTooLarge,
  type RecoveryOutcome,
  type SessionEvent,
  type SessionEventsPage,
  type SessionSummary,
  type SessionTranscriptPage,
  type SSEEvent,
  streamInterruptSession,
  streamRecoverToolApproval,
  streamRecoverToolRound,
  streamRecoverUserInput,
  streamResolveToolApproval,
  streamResolveUserInput,
  streamResume,
} from "../lib/api"
import {
  formatBytes,
  formatCount,
  formatDateTime,
  formatTime,
  modelUsagePayload,
  numericValue,
} from "../lib/format"
import { dashboardPath } from "../lib/links"
import {
  isFailureEventType,
  latestFailureEvent,
  objectPayload,
  optionalString,
  summarizeFailureEvent,
} from "../lib/session-debug"
import {
  CoalescedAsyncRunner,
  durationDetail,
  durationEndTimestamp,
  eventScanCursor,
  historyModeLabel,
  initialPageNavigation,
  jumpToLatestPage,
  loadStableTranscriptTailPage,
  mergeLatestEventWindow,
  mergeLatestTranscriptWindow,
  mergeStoredAndLiveRecords,
  navigateToNewerPage,
  navigateToOlderPage,
  olderTranscriptPage,
  pendingTailCanReconcile,
  queryReadErrorIsFatal,
  sessionMetadataNeedsRefresh,
  sessionSummaryRevision,
  statePollInterval,
  summaryStateIsStable,
  type TranscriptPageParam,
  tailActivityAction,
} from "../lib/session-history"
import { cn } from "../lib/utils"

const INTERRUPTIBLE_SESSION_STATUSES = new Set(["pending", "running"])
const ACTIVE_SESSION_STATUSES = new Set(["pending", "running", "interrupting"])
const PENDING_ACTION_SESSION_STATUSES = new Set(["interrupted", "failed", "completed"])
const SESSION_EVENT_PAGE_SIZE = 100
const SESSION_TRANSCRIPT_PAGE_SIZE = 100
const SESSION_TRANSCRIPT_TAIL_READ_LIMIT = 3

async function fetchTranscriptTailPage(
  sessionId: string,
  totalMessages: number,
  signal?: AbortSignal,
) {
  return loadStableTranscriptTailPage(
    totalMessages,
    SESSION_TRANSCRIPT_PAGE_SIZE,
    SESSION_TRANSCRIPT_TAIL_READ_LIMIT,
    (offset) =>
      fetchSessionTranscript(sessionId, { offset, limit: SESSION_TRANSCRIPT_PAGE_SIZE }, signal),
  )
}

async function fetchTranscriptPage(
  sessionId: string,
  pageParam: TranscriptPageParam,
  signal?: AbortSignal,
) {
  if (pageParam !== null) {
    return fetchSessionTranscript(sessionId, pageParam, signal)
  }

  const probe = await fetchSessionTranscript(sessionId, { offset: 0, limit: 1 }, signal)
  if (probe.total_messages <= 1) return probe
  return fetchTranscriptTailPage(sessionId, probe.total_messages, signal)
}

function isInterruptionConflict(error: unknown) {
  return error instanceof ApiClientError && error.status === 409
}

function isNonRetryableStateError(error: unknown) {
  return (
    error instanceof ApiClientError &&
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 429
  )
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError"
}

function EventIcon({ type }: { type: string }) {
  if (type.includes("tool.call.started")) return <Wrench className="h-3.5 w-3.5 text-primary" />
  if (type.includes("tool.call.completed"))
    return <CheckCircle className="h-3.5 w-3.5 text-chart-2" />
  if (type.includes("tool.call.failed"))
    return <AlertCircle className="h-3.5 w-3.5 text-destructive" />
  if (type.includes("model.")) return <Bot className="h-3.5 w-3.5 text-chart-1" />
  if (type.includes("task.")) return <CheckCircle className="h-3.5 w-3.5 text-chart-4" />
  return <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
}

function timestampMs(value: string | null | undefined) {
  if (!value) return null
  const time = new Date(value).getTime()
  return Number.isFinite(time) ? time : null
}

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default"
  if (status === "running") return "secondary"
  if (status === "failed") return "destructive"
  return "outline"
}

function totalTokens(summary: SessionSummary | undefined) {
  const usage = summary?.usage.usage
  return usage?.total_tokens ?? (usage?.input_tokens ?? 0) + (usage?.output_tokens ?? 0)
}

function summaryModels(summary: SessionSummary | undefined) {
  return summary?.usage.models ?? []
}

function sortedEventCounts(summary: SessionSummary | undefined) {
  return Object.entries(summary?.events.counts_by_type ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
}

function eventTone(type: string) {
  if (type === "model.completed") return "bg-chart-2"
  if (type.startsWith("model.")) return "bg-chart-1"
  if (isFailureEventType(type)) return "bg-destructive"
  if (type.startsWith("tool.call")) return "bg-chart-3"
  if (type.startsWith("session.")) return "bg-primary"
  return "bg-muted-foreground"
}

type PendingAction =
  | {
      kind: "approval"
      approvalId: string
      toolName: string
      reason: string | null
      arguments: unknown
    }
  | {
      kind: "user_input"
      inputId: string
      toolCallId: string | null
      question: string
      options: string[]
    }
  | {
      kind: "manual_recovery"
      approvalId: string | null
      inputId: string | null
      roundId: string | null
      toolCallId: string
      toolName: string | null
      question: string | null
      detail: string
      arguments: unknown
    }

function pendingActionFromApi(action: ApiPendingAction | undefined): PendingAction | null {
  if (!action) return null
  if (action.kind === "tool_approval" && action.approval_id) {
    return {
      kind: "approval",
      approvalId: action.approval_id,
      toolName: action.tool_name || "tool",
      reason: action.detail ?? null,
      arguments: action.arguments || {},
    }
  }
  if (action.kind === "user_input" && action.input_id) {
    return {
      kind: "user_input",
      inputId: action.input_id,
      toolCallId: action.tool_call_id ?? null,
      question: action.question || action.detail || "Input required",
      options: action.options ?? [],
    }
  }
  if (action.kind === "manual_recovery" && action.tool_call_id) {
    return {
      kind: "manual_recovery",
      approvalId: action.approval_id ?? null,
      inputId: action.input_id ?? null,
      roundId: action.round_id ?? null,
      toolCallId: action.tool_call_id,
      toolName: action.tool_name ?? null,
      question: action.question ?? null,
      detail:
        action.detail ||
        "A previously started tool result must be reconciled before the session can continue.",
      arguments: action.arguments || action.event.payload,
    }
  }
  return null
}

function eventSummary(event: {
  type: string
  tool_name: string | null
  payload: Record<string, unknown>
}) {
  if (event.type === "model.completed") {
    const usage = modelUsagePayload(event.payload)
    const input = numericValue(usage.input_tokens)
    const output = numericValue(usage.output_tokens)
    const total = numericValue(usage.total_tokens) || input + output
    const model = typeof usage.model === "string" ? usage.model : null
    return `model.completed - ${formatCount(total)} total tokens${model ? ` - ${model}` : ""}`
  }
  if (event.type === "model.started") {
    const model = typeof event.payload.model === "string" ? event.payload.model : null
    return model ? `model.started - ${model}` : event.type
  }
  if (event.type === "tool.call.started" || event.type === "tool.call.completed") {
    return event.tool_name ? `${event.type} - ${event.tool_name}` : event.type
  }
  if (event.type === "tool.call.failed") {
    const error = typeof event.payload.error === "string" ? event.payload.error : null
    return event.tool_name
      ? `${event.type} - ${event.tool_name}${error ? ` - ${error}` : ""}`
      : `${event.type}${error ? ` - ${error}` : ""}`
  }
  if (event.type === "session.failed") {
    const error = typeof event.payload.error === "string" ? event.payload.error : null
    return error ? `session.failed - ${error}` : event.type
  }
  if (event.type === "session.interrupted") {
    const reason = typeof event.payload.reason === "string" ? event.payload.reason : null
    const interruptionType =
      typeof event.payload.interruption_type === "string" ? event.payload.interruption_type : null
    if (interruptionType === "tool_approval_required")
      return "session.interrupted - approval required"
    if (interruptionType === "user_input_required")
      return "session.interrupted - user input required"
    return reason ? `session.interrupted - ${reason}` : event.type
  }
  if (event.type === "tool.call.approval_requested") {
    const approval = objectPayload(event.payload.approval)
    const toolName = optionalString(approval?.tool_name) || event.tool_name
    return toolName ? `approval requested - ${toolName}` : "approval requested"
  }
  if (event.type === "session.awaiting_user_input") {
    const question = optionalString(event.payload.question)
    return question ? `awaiting user input - ${question}` : "awaiting user input"
  }
  return event.tool_name ? `${event.type} - ${event.tool_name}` : event.type
}

function SummaryStat({
  icon: Icon,
  label,
  value,
  detail,
  className,
  tone = "neutral",
}: {
  icon: LucideIcon
  label: string
  value: string
  detail?: string
  className?: string
  tone?: "neutral" | "green" | "blue" | "amber" | "red" | "violet"
}) {
  const toneClass = {
    neutral: "text-muted-foreground",
    green: "text-chart-2",
    blue: "text-chart-3",
    amber: "text-chart-1",
    red: "text-destructive",
    violet: "text-chart-5",
  }[tone]

  return (
    <div className={cn("flex min-w-0 gap-3 px-5 py-4", className)}>
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", toneClass)} />
      <div className="min-w-0">
        <div className="text-sm font-medium text-muted-foreground">{label}</div>
        <div className="mt-1 truncate text-xl font-semibold leading-tight">{value}</div>
        {detail && <div className="mt-1 truncate text-xs text-muted-foreground">{detail}</div>}
      </div>
    </div>
  )
}

function EventRow({
  event,
  selected,
  onSelect,
}: {
  event: SessionEvent
  selected: boolean
  onSelect: (eventId: string) => void
}) {
  const time = formatTime(event.timestamp)
  const summary = eventSummary(event)
  const failure = isFailureEventType(event.type)

  return (
    <div
      className={cn("relative border-b border-border/70 last:border-0", selected && "bg-primary/5")}
    >
      <button
        type="button"
        onClick={() => onSelect(event.id)}
        className={cn(
          "grid w-full grid-cols-[0.75rem_5.25rem_minmax(0,1fr)_1rem] items-center gap-2 px-3 py-3 text-left transition-colors hover:bg-muted/50 sm:grid-cols-[1rem_7.5rem_minmax(0,1fr)_1.5rem] sm:gap-3 sm:px-4",
          selected && "bg-primary/5",
        )}
      >
        <span className={cn("h-2.5 w-2.5 rounded-full", eventTone(event.type))} />
        <span className="truncate font-mono text-xs text-muted-foreground">{time}</span>
        <span className="flex min-w-0 items-center gap-2">
          <EventIcon type={event.type} />
          <span className={cn("truncate text-sm", failure && "font-medium text-destructive")}>
            {summary}
          </span>
        </span>
        <ChevronRight
          className={cn(
            "h-4 w-4 text-muted-foreground transition-transform",
            selected && "rotate-90",
          )}
        />
      </button>
    </div>
  )
}

function EventDetailPanel({ event }: { event: SessionEvent | null }) {
  if (event === null) {
    return (
      <div className="flex min-h-72 items-center justify-center p-6 text-center text-sm text-muted-foreground">
        Select an event to inspect its structured payload.
      </div>
    )
  }

  const failure = isFailureEventType(event.type)
  const failureSummary = failure ? summarizeFailureEvent(event) : null

  return (
    <div className="min-h-0 space-y-4 p-4">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={failure ? "destructive" : "outline"}>{event.type}</Badge>
          {event.tool_name && <Badge variant="secondary">{event.tool_name}</Badge>}
        </div>
        <div className="mt-2 break-all font-mono text-xs text-muted-foreground">{event.id}</div>
      </div>

      {failure && (
        <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm">
          <div className="mb-1 flex items-center gap-2 font-medium text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {failureSummary?.label || "Failure Event"}
          </div>
          <div className="break-words text-muted-foreground">
            {failureSummary?.detail || "Inspect the payload for failure details."}
          </div>
        </div>
      )}

      <div className="grid gap-3 rounded-md border border-border bg-muted/30 p-3 text-sm">
        <DetailPair label="Timestamp" value={formatDateTime(event.timestamp)} />
        <DetailPair label="Agent" value={event.agent_name || "-"} />
        <DetailPair label="Environment" value={event.environment_name || "-"} />
        <DetailPair label="Workflow" value={event.workflow_name || "-"} />
      </div>

      <section>
        <h3 className="mb-2 text-sm font-medium">Payload</h3>
        <PayloadViewer value={event.payload} maxHeight="max-h-[34rem]" />
      </section>
    </div>
  )
}

function DetailPair({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-3">
      <div className="text-muted-foreground">{label}</div>
      <div className="min-w-0 break-words">{value}</div>
    </div>
  )
}

function TranscriptPart({ part }: { part: Record<string, unknown> }) {
  const [expanded, setExpanded] = useState(false)
  const type = typeof part.type === "string" ? part.type : "part"
  const text = typeof part.text === "string" ? part.text : null
  const toolName = typeof part.tool_name === "string" ? part.tool_name : ""
  const content = typeof part.content === "string" ? part.content : null
  const shouldTruncate =
    (text !== null && text.length > 500) ||
    (content !== null && content.length > 300) ||
    type === "tool_call" ||
    type === "tool_result"

  if (type === "text" && text !== null) {
    return (
      <div className="min-w-0">
        <span className="whitespace-pre-wrap break-words leading-relaxed">
          {expanded ? text : text.slice(0, 500)}
          {!expanded && text.length > 500 ? "..." : ""}
        </span>
        {text.length > 500 && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 h-7 px-2"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Collapse" : "Show full text"}
          </Button>
        )}
      </div>
    )
  }

  if (type === "tool_call") {
    return (
      <div className="min-w-0">
        <span className="block overflow-x-auto rounded bg-background/80 px-2 py-1 font-mono text-xs">
          call {toolName}({JSON.stringify(part.arguments || {}).slice(0, 140)})
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="mt-2 h-7 px-2"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Hide arguments" : "Show arguments"}
        </Button>
        {expanded && <PayloadViewer value={part.arguments || {}} className="mt-2" />}
      </div>
    )
  }

  if (type === "tool_result") {
    return (
      <div className="min-w-0">
        <span className="block overflow-x-auto rounded bg-background/80 px-2 py-1 font-mono text-xs">
          result {String(content || "").slice(0, 300)}
          {content && content.length > 300 ? "..." : ""}
        </span>
        {shouldTruncate && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 h-7 px-2"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Hide result" : "Show result payload"}
          </Button>
        )}
        {expanded && <PayloadViewer value={part} className="mt-2" />}
      </div>
    )
  }

  return <PayloadViewer value={part} maxHeight="max-h-48" />
}

function PendingActionBanner({
  action,
  resolving,
  error,
  onApprove,
  onDeny,
  onAnswer,
  onRecover,
}: {
  action: PendingAction
  resolving: boolean
  error: string | null
  onApprove: (reason: string | null) => void
  onDeny: (reason: string | null) => void
  onAnswer: (answer: string) => void
  onRecover: (outcome: RecoveryOutcome, message: string, answer: string | null) => void
}) {
  const [answer, setAnswer] = useState("")
  const [reason, setReason] = useState("")
  const [recoveryOutcome, setRecoveryOutcome] = useState<RecoveryOutcome>("completed")
  const [recoveryMessage, setRecoveryMessage] = useState("")
  const [recoveryAnswer, setRecoveryAnswer] = useState("")

  if (action.kind === "approval") {
    return (
      <Card className="border-chart-1/30 bg-chart-1/5">
        <CardContent className="space-y-4 p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" className="border-chart-1/40 bg-background">
                  <ShieldCheck className="mr-1 h-3.5 w-3.5 text-chart-1" />
                  Awaiting approval
                </Badge>
                <Badge variant="secondary">{action.toolName}</Badge>
              </div>
              <p className="mt-2 text-sm font-medium">
                This session is paused until the tool call is approved or denied.
              </p>
              <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
                approval_id: {action.approvalId}
              </p>
              {action.reason && (
                <p className="mt-2 text-sm text-muted-foreground">{action.reason}</p>
              )}
            </div>
            <div className="flex shrink-0 gap-2">
              <Button
                variant="outline"
                disabled={resolving}
                onClick={() => onDeny(reason.trim() || null)}
              >
                Deny
              </Button>
              <Button disabled={resolving} onClick={() => onApprove(reason.trim() || null)}>
                Approve
              </Button>
            </div>
          </div>
          <Input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="Optional decision reason"
            disabled={resolving}
          />
          <PayloadViewer value={action.arguments} maxHeight="max-h-48" />
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>
    )
  }

  if (action.kind === "manual_recovery") {
    const requiresAnswer = action.inputId !== null
    const recoveryDisabled =
      resolving || !recoveryMessage.trim() || (requiresAnswer && !recoveryAnswer.trim())

    return (
      <Card className="border-destructive/30 bg-destructive/5">
        <CardContent className="space-y-4 p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="destructive">
                  <AlertTriangle className="mr-1 h-3.5 w-3.5" />
                  Manual recovery required
                </Badge>
                {action.toolName && <Badge variant="secondary">{action.toolName}</Badge>}
              </div>
              <p className="mt-2 text-sm font-medium">
                A tool call started before its terminal result was recorded.
              </p>
              <p className="mt-1 break-words text-sm text-muted-foreground">{action.detail}</p>
              <div className="mt-2 space-y-1 break-all font-mono text-xs text-muted-foreground">
                <div>tool_call_id: {action.toolCallId}</div>
                {action.approvalId && <div>approval_id: {action.approvalId}</div>}
                {action.inputId && <div>input_id: {action.inputId}</div>}
                {action.roundId && <div>round_id: {action.roundId}</div>}
              </div>
              {action.question && (
                <p className="mt-2 break-words text-sm text-muted-foreground">
                  Question: {action.question}
                </p>
              )}
            </div>
            <div className="grid shrink-0 gap-2 sm:grid-cols-[8.5rem_auto] lg:grid-cols-1">
              <select
                value={recoveryOutcome}
                onChange={(event) => setRecoveryOutcome(event.target.value as RecoveryOutcome)}
                disabled={resolving}
                className="h-9 rounded-md border border-input bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Recovery outcome"
              >
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
              <Button
                disabled={recoveryDisabled}
                onClick={() =>
                  onRecover(
                    recoveryOutcome,
                    recoveryMessage.trim(),
                    requiresAnswer ? recoveryAnswer.trim() : null,
                  )
                }
              >
                Recover
              </Button>
            </div>
          </div>
          {requiresAnswer && (
            <Input
              value={recoveryAnswer}
              onChange={(event) => setRecoveryAnswer(event.target.value)}
              placeholder="Re-supply the user answer for recovery"
              disabled={resolving}
            />
          )}
          <Input
            value={recoveryMessage}
            onChange={(event) => setRecoveryMessage(event.target.value)}
            placeholder="Externally verified result or failure reason"
            disabled={resolving}
          />
          <div className="space-y-2">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Tool arguments
            </div>
            <PayloadViewer value={action.arguments} maxHeight="max-h-48" />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-chart-3/30 bg-chart-3/5">
      <CardContent className="space-y-4 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="border-chart-3/40 bg-background">
                <UserRound className="mr-1 h-3.5 w-3.5 text-chart-3" />
                Awaiting user input
              </Badge>
              {action.toolCallId && <Badge variant="secondary">{action.toolCallId}</Badge>}
            </div>
            <p className="mt-2 text-sm font-medium">{action.question}</p>
            <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
              input_id: {action.inputId}
            </p>
          </div>
          <Button disabled={resolving || !answer.trim()} onClick={() => onAnswer(answer.trim())}>
            Submit Answer
          </Button>
        </div>
        {action.options.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {action.options.map((option) => (
              <Button
                key={option}
                variant={answer === option ? "default" : "outline"}
                size="sm"
                disabled={resolving}
                onClick={() => setAnswer(option)}
              >
                {option}
              </Button>
            ))}
          </div>
        )}
        <Input
          value={answer}
          onChange={(event) => setAnswer(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && answer.trim() && onAnswer(answer.trim())}
          placeholder="Answer the paused session..."
          disabled={resolving}
        />
        {error && <p className="text-sm text-destructive">{error}</p>}
      </CardContent>
    </Card>
  )
}

function FailureDebugBanner({
  event,
  selected,
  onInspect,
}: {
  event: SessionEvent
  selected: boolean
  onInspect: () => void
}) {
  const summary = summarizeFailureEvent(event)
  const badgeLabel = summary.kind === "tool_issue" ? "Tool issue" : "Debug failure"

  return (
    <Card className="border-destructive/30 bg-destructive/5">
      <CardContent className="flex flex-col gap-3 p-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="destructive">
              <AlertTriangle className="mr-1 h-3.5 w-3.5" />
              {badgeLabel}
            </Badge>
            <Badge variant="outline">{summary.eventType}</Badge>
            {summary.toolName && <Badge variant="secondary">{summary.toolName}</Badge>}
          </div>
          <p className="mt-2 text-sm font-medium">{summary.label}</p>
          <p className="mt-1 break-words text-sm text-muted-foreground">{summary.detail}</p>
        </div>
        <Button variant={selected ? "secondary" : "outline"} onClick={onInspect}>
          {selected ? "Inspecting event" : "Inspect event"}
        </Button>
      </CardContent>
    </Card>
  )
}

function transcriptMessageKey(message: {
  index: number
  role: string
  content: Array<Record<string, unknown>>
}) {
  return `${message.index}:${message.role}`
}

function transcriptPartKey(index: number) {
  return String(index)
}

function SessionArtifacts({
  artifacts,
  sessionId,
}: {
  artifacts: ArtifactSummary[]
  sessionId: string
}) {
  if (artifacts.length === 0) return null
  return (
    <Card className="flex-shrink-0">
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle>Session Artifacts</CardTitle>
          <p className="mt-1 text-sm text-muted-foreground">
            Files and generated artifacts linked to this session.
          </p>
        </div>
        <a
          href={dashboardPath("/artifacts", { session_id: sessionId })}
          className={buttonVariants({ variant: "outline", size: "sm" })}
        >
          Open artifacts
        </a>
      </CardHeader>
      <CardContent className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {artifacts.slice(0, 6).map((artifact) => (
          <a
            key={`${artifact.artifact_store_id}:${artifact.id}`}
            href={dashboardPath("/artifacts", {
              artifact_store_id: artifact.artifact_store_id,
              session_id: sessionId,
              q: artifact.id,
            })}
            className="block min-w-0 rounded-md border border-border p-3 transition-colors hover:bg-muted/40"
          >
            <div className="truncate text-sm font-medium">{artifact.filename}</div>
            <div className="mt-1 flex min-w-0 items-center justify-between gap-3 text-xs text-muted-foreground">
              <span className="truncate">{artifact.content_type}</span>
              <span className="shrink-0">{formatBytes(artifact.size_bytes)}</span>
            </div>
          </a>
        ))}
      </CardContent>
    </Card>
  )
}

export function SessionDetailPage() {
  const { sessionId } = useParams({ from: "/sessions/$sessionId" })
  return <SessionDetail key={sessionId} sessionId={sessionId} />
}

function SessionDetail({ sessionId }: { sessionId: string }) {
  const queryClient = useQueryClient()
  const [interruptSheetOpen, setInterruptSheetOpen] = useState(false)
  const [interruptReason, setInterruptReason] = useState("")
  const [interruptRequested, setInterruptRequested] = useState(false)
  const [interruptError, setInterruptError] = useState<string | null>(null)
  const [retryingCascade, setRetryingCascade] = useState(false)
  const [cascadeRetryError, setCascadeRetryError] = useState<string | null>(null)
  const [resumePrompt, setResumePrompt] = useState("")
  const [resuming, setResuming] = useState(false)
  const [liveEvents, setLiveEvents] = useState<SSEEvent[]>([])
  const [resumeError, setResumeError] = useState<string | null>(null)
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null)
  const [resolvingAction, setResolvingAction] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [eventPinnedToTail, setEventPinnedToTail] = useState(true)
  const [transcriptPinnedToTail, setTranscriptPinnedToTail] = useState(true)
  const [transcriptVisible, setTranscriptVisible] = useState(false)
  const [eventReconcileError, setEventReconcileError] = useState<string | null>(null)
  const [transcriptReconcileError, setTranscriptReconcileError] = useState<string | null>(null)
  const [eventNavigation, setEventNavigation] = useState(() =>
    initialPageNavigation<number | null>(null),
  )
  const [transcriptNavigation, setTranscriptNavigation] = useState(() =>
    initialPageNavigation<TranscriptPageParam>(null),
  )
  const mutationActive = resuming || resolvingAction || retryingCascade
  const detailQuery = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => fetchSession(sessionId),
    staleTime: Number.POSITIVE_INFINITY,
  })
  const stateQuery = useQuery({
    queryKey: ["session-state", sessionId],
    queryFn: () => fetchSessionState(sessionId),
    enabled: detailQuery.data !== undefined,
    retry: (failureCount, error) => !isNonRetryableStateError(error) && failureCount < 2,
    refetchInterval: (query) =>
      statePollInterval({
        status: query.state.data?.status,
        interruptionCascade: query.state.data?.interruption_cascade,
        interruptRequested,
        mutationActive,
        hasError: query.state.error !== null,
        failureCount: query.state.fetchFailureCount,
        nonRetryableError: isNonRetryableStateError(query.state.error),
      }),
    refetchIntervalInBackground: false,
  })
  const state = stateQuery.data
  const interruptionCascade = state?.interruption_cascade ?? "none"
  const summaryRevision =
    state === undefined
      ? null
      : sessionSummaryRevision({
          status: state.status,
          lastActivityAt: state.last_activity_at,
          interruptionCascade,
        })
  const summaryQuery = useQuery({
    queryKey: ["session-summary", sessionId, summaryRevision],
    queryFn: () => fetchSessionSummary(sessionId),
    enabled: summaryRevision !== null && detailQuery.data !== undefined,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
  })
  const eventsQuery = useQuery({
    queryKey: ["session-events", sessionId, eventNavigation.current ?? "latest"],
    queryFn: ({ signal }) =>
      fetchSessionEvents(
        sessionId,
        {
          order_by: "sequence_desc",
          limit: SESSION_EVENT_PAGE_SIZE,
          exclude_event_type: "model.text.delta",
          ...(eventNavigation.current === null ? {} : { before_sequence: eventNavigation.current }),
        },
        signal,
      ),
    // Establish the activity baseline before reading history. A later append is
    // then either present in this page or observed by the next state heartbeat.
    enabled: state !== undefined,
    gcTime: 0,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
  })
  const transcriptQuery = useQuery({
    queryKey: ["session-transcript", sessionId, transcriptNavigation.current ?? "latest"],
    queryFn: ({ signal }) => fetchTranscriptPage(sessionId, transcriptNavigation.current, signal),
    enabled: state !== undefined,
    gcTime: 0,
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnWindowFocus: false,
  })

  const session = detailQuery.data
    ? {
        ...detailQuery.data,
        ...(state === undefined ? {} : { status: state.status, updated_at: state.updated_at }),
      }
    : undefined
  const error = detailQuery.error ?? stateQuery.error
  const permanentStateError = isNonRetryableStateError(stateQuery.error)
  const isError =
    queryReadErrorIsFatal(detailQuery.isError, detailQuery.data !== undefined) ||
    queryReadErrorIsFatal(stateQuery.isError, state !== undefined) ||
    (stateQuery.isRefetchError && permanentStateError)
  const isLoading = detailQuery.isLoading || stateQuery.isLoading
  const stateRefetchError = stateQuery.isRefetchError && !permanentStateError
  const detailRefetchError = detailQuery.isRefetchError

  const pendingActionQuery = useQuery({
    queryKey: [
      "pending-actions",
      "session",
      sessionId,
      session?.status ?? null,
      session?.updated_at ?? null,
    ],
    queryFn: () => fetchPendingActions({ session_id: sessionId, limit: 1 }),
    retry: (failureCount, error) => !isApiPayloadTooLarge(error) && failureCount < 3,
    refetchInterval: (query) => {
      const result = query.state.data
      if (result !== undefined && result.actions.length === 0 && result.issues.length === 0) {
        return false
      }
      return 5000
    },
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: (query) => !isApiPayloadTooLarge(query.state.error),
    enabled:
      !isError && session !== undefined && PENDING_ACTION_SESSION_STATUSES.has(session.status),
  })
  const sessionArtifacts = useQuery({
    queryKey: ["session-artifacts", sessionId],
    queryFn: () => fetchArtifacts({ session_id: sessionId, limit: 6 }),
    enabled: !isError,
    staleTime: 10_000,
  })

  const eventsViewportRef = useRef<HTMLDivElement>(null)
  const transcriptViewportRef = useRef<HTMLDivElement>(null)
  const observedActivityRef = useRef<string | null>(null)
  const suppressActivityReconcileRef = useRef(false)
  const eventAtLatestRef = useRef(true)
  const transcriptAtLatestRef = useRef(true)
  const eventPinnedToTailRef = useRef(true)
  const transcriptPinnedToTailRef = useRef(true)
  const transcriptVisibleRef = useRef(false)
  const attemptedMetadataRevisionRef = useRef<string | null>(null)
  const eventsPageRef = useRef<SessionEventsPage | undefined>(eventsQuery.data)
  const transcriptPageRef = useRef<SessionTranscriptPage | undefined>(transcriptQuery.data)
  const eventReconcileAbortRef = useRef<AbortController | null>(null)
  const transcriptReconcileAbortRef = useRef<AbortController | null>(null)
  const eventReconcilePendingRef = useRef(false)
  const transcriptReconcilePendingRef = useRef(false)
  const eventScanCursorRef = useRef(0)
  const mutationActiveRef = useRef(mutationActive)
  const eventReconcileRunnerRef = useRef<CoalescedAsyncRunner | null>(null)
  const transcriptReconcileRunnerRef = useRef<CoalescedAsyncRunner | null>(null)
  if (eventReconcileRunnerRef.current === null) {
    eventReconcileRunnerRef.current = new CoalescedAsyncRunner()
  }
  if (transcriptReconcileRunnerRef.current === null) {
    transcriptReconcileRunnerRef.current = new CoalescedAsyncRunner()
  }
  const eventReconcileRunner = eventReconcileRunnerRef.current
  const transcriptReconcileRunner = transcriptReconcileRunnerRef.current
  eventAtLatestRef.current = eventNavigation.current === null
  transcriptAtLatestRef.current = transcriptNavigation.current === null
  eventPinnedToTailRef.current = eventPinnedToTail
  transcriptPinnedToTailRef.current = transcriptPinnedToTail
  transcriptVisibleRef.current = transcriptVisible
  mutationActiveRef.current = mutationActive
  eventsPageRef.current = eventsQuery.data
  transcriptPageRef.current = transcriptQuery.data
  const events = useMemo(() => {
    return [...(eventsQuery.data?.events ?? [])].sort(
      (left, right) => left.sequence - right.sequence,
    )
  }, [eventsQuery.data])
  const transcript = useMemo(() => {
    return [...(transcriptQuery.data?.messages ?? [])].sort(
      (left, right) => left.index - right.index,
    )
  }, [transcriptQuery.data])
  const liveTimelineEventCount = useMemo(
    () => liveEvents.filter((event) => event.type !== "model.text.delta").length,
    [liveEvents],
  )
  const eventTailRevision = events.at(-1)?.sequence ?? null
  const transcriptTailRevision = transcript.at(-1)?.index ?? null

  const performLatestEventReconciliation = useCallback(async (): Promise<boolean> => {
    if (!eventAtLatestRef.current || mutationActiveRef.current) return false
    eventReconcilePendingRef.current = false
    setEventReconcileError(null)
    const current = eventsPageRef.current
    if (current === undefined) {
      const result = await eventsQuery.refetch()
      eventReconcilePendingRef.current = result.isError
      return !result.isError
    }

    const controller = new AbortController()
    eventReconcileAbortRef.current = controller
    try {
      const scanCursor = eventScanCursor(current, eventScanCursorRef.current)
      const incremental = await fetchSessionEvents(
        sessionId,
        {
          after_sequence: scanCursor,
          order_by: "sequence_asc",
          limit: SESSION_EVENT_PAGE_SIZE,
          exclude_event_type: "model.text.delta",
        },
        controller.signal,
      )
      if (controller.signal.aborted || !eventAtLatestRef.current) return false
      if (incremental.has_more) {
        const result = await eventsQuery.refetch()
        eventReconcilePendingRef.current = result.isError
        return !result.isError
      }
      const nextScanCursor = eventScanCursor(incremental, scanCursor)
      eventScanCursorRef.current = nextScanCursor
      if (incremental.events.length === 0) {
        queryClient.setQueryData<SessionEventsPage>(
          ["session-events", sessionId, "latest"],
          (existing) =>
            existing === undefined
              ? existing
              : { ...existing, scan_through_sequence: nextScanCursor },
        )
        return true
      }

      queryClient.setQueryData<SessionEventsPage>(
        ["session-events", sessionId, "latest"],
        (existing) => {
          if (existing === undefined) return existing
          const merged = mergeLatestEventWindow(
            existing.events,
            incremental.events,
            SESSION_EVENT_PAGE_SIZE,
          )
          const oldest = merged.records.at(-1)
          return {
            ...existing,
            events: merged.records,
            order_by: "sequence_desc",
            next_sequence: oldest?.sequence ?? null,
            scan_through_sequence: nextScanCursor,
            has_more: existing.has_more || merged.dropped,
          }
        },
      )
      return true
    } catch (error) {
      if (isAbortError(error)) return false
      throw error
    } finally {
      if (eventReconcileAbortRef.current === controller) {
        eventReconcileAbortRef.current = null
      }
    }
  }, [eventsQuery.refetch, queryClient, sessionId])

  const reconcileLatestEvents = useCallback(
    () =>
      eventReconcileRunner.request(performLatestEventReconciliation).catch((error) => {
        if (isAbortError(error)) return
        eventReconcilePendingRef.current = true
        setEventReconcileError(
          error instanceof Error ? error.message : "Failed to refresh the event tail.",
        )
      }),
    [eventReconcileRunner, performLatestEventReconciliation],
  )

  useEffect(() => {
    if (eventNavigation.current !== null || eventsQuery.data === undefined) return
    eventScanCursorRef.current = eventScanCursor(eventsQuery.data, eventScanCursorRef.current)
  }, [eventNavigation.current, eventsQuery.data])

  const performLatestTranscriptReconciliation = useCallback(async (): Promise<boolean> => {
    if (!transcriptAtLatestRef.current || mutationActiveRef.current) return false
    transcriptReconcilePendingRef.current = false
    setTranscriptReconcileError(null)
    const current = transcriptPageRef.current
    if (current === undefined) {
      const result = await transcriptQuery.refetch()
      transcriptReconcilePendingRef.current = result.isError
      return !result.isError
    }

    const controller = new AbortController()
    transcriptReconcileAbortRef.current = controller
    try {
      const incremental = await fetchSessionTranscript(
        sessionId,
        { offset: current.total_messages, limit: SESSION_TRANSCRIPT_PAGE_SIZE },
        controller.signal,
      )
      if (controller.signal.aborted || !transcriptAtLatestRef.current) return false

      if (incremental.has_more) {
        const tail = await fetchTranscriptTailPage(
          sessionId,
          incremental.total_messages,
          controller.signal,
        )
        if (controller.signal.aborted || !transcriptAtLatestRef.current) return false
        queryClient.setQueryData<SessionTranscriptPage>(
          ["session-transcript", sessionId, "latest"],
          tail,
        )
        transcriptReconcilePendingRef.current = false
        return true
      }
      if (
        incremental.messages.length === 0 &&
        incremental.total_messages === current.total_messages
      ) {
        queryClient.setQueryData<SessionTranscriptPage>(
          ["session-transcript", sessionId, "latest"],
          (existing) => existing,
        )
        transcriptReconcilePendingRef.current = false
        return true
      }

      queryClient.setQueryData<SessionTranscriptPage>(
        ["session-transcript", sessionId, "latest"],
        (existing) => {
          if (existing === undefined) return existing
          const merged = mergeLatestTranscriptWindow(
            existing.messages,
            incremental.messages,
            SESSION_TRANSCRIPT_PAGE_SIZE,
          )
          const totalMessages = Math.max(existing.total_messages, incremental.total_messages)
          return {
            ...existing,
            messages: merged.records,
            offset: merged.records[0]?.index ?? totalMessages,
            next_offset: totalMessages,
            has_more: false,
            total_messages: totalMessages,
          }
        },
      )
      transcriptReconcilePendingRef.current = false
      return true
    } catch (error) {
      if (isAbortError(error)) return false
      throw error
    } finally {
      if (transcriptReconcileAbortRef.current === controller) {
        transcriptReconcileAbortRef.current = null
      }
    }
  }, [queryClient, sessionId, transcriptQuery.refetch])

  const reconcileLatestTranscript = useCallback(
    () =>
      transcriptReconcileRunner.request(performLatestTranscriptReconciliation).catch((error) => {
        if (isAbortError(error)) return
        transcriptReconcilePendingRef.current = true
        setTranscriptReconcileError(
          error instanceof Error ? error.message : "Failed to refresh the transcript tail.",
        )
      }),
    [performLatestTranscriptReconciliation, transcriptReconcileRunner],
  )

  useEffect(() => {
    const detailUpdatedAt = detailQuery.data?.updated_at
    const stateUpdatedAt = state?.updated_at
    if (!sessionMetadataNeedsRefresh(detailUpdatedAt, stateUpdatedAt)) {
      if (detailUpdatedAt !== undefined && detailUpdatedAt === stateUpdatedAt) {
        attemptedMetadataRevisionRef.current = null
      }
      return
    }
    if (
      detailQuery.isFetching ||
      stateUpdatedAt === undefined ||
      attemptedMetadataRevisionRef.current === stateUpdatedAt
    ) {
      return
    }
    attemptedMetadataRevisionRef.current = stateUpdatedAt
    void detailQuery.refetch()
  }, [detailQuery.data?.updated_at, detailQuery.isFetching, detailQuery.refetch, state?.updated_at])

  useEffect(() => {
    const lastActivityAt = state?.last_activity_at
    if (lastActivityAt === undefined) return
    const previous = observedActivityRef.current
    observedActivityRef.current = lastActivityAt
    if (suppressActivityReconcileRef.current) {
      suppressActivityReconcileRef.current = false
      return
    }
    if (previous === null || previous === lastActivityAt) return
    if (mutationActive) return

    const eventAction = tailActivityAction({
      atLatest: eventNavigation.current === null,
      pinned: eventPinnedToTail,
      visible: true,
    })
    if (eventAction === "reconcile") {
      if (eventsQuery.isFetching) eventReconcilePendingRef.current = true
      else void reconcileLatestEvents()
    } else if (eventAction === "defer") {
      eventReconcilePendingRef.current = true
    }

    const transcriptAction = tailActivityAction({
      atLatest: transcriptNavigation.current === null,
      pinned: transcriptPinnedToTail,
      visible: transcriptVisible,
    })
    if (transcriptAction === "reconcile") {
      if (transcriptQuery.isFetching) transcriptReconcilePendingRef.current = true
      else void reconcileLatestTranscript()
    } else if (transcriptAction === "defer") {
      transcriptReconcilePendingRef.current = true
    }
  }, [
    eventNavigation.current,
    eventPinnedToTail,
    eventsQuery.isFetching,
    mutationActive,
    reconcileLatestEvents,
    reconcileLatestTranscript,
    state?.last_activity_at,
    transcriptNavigation.current,
    transcriptPinnedToTail,
    transcriptQuery.isFetching,
    transcriptVisible,
  ])

  useEffect(() => {
    if (
      eventNavigation.current === null &&
      eventPinnedToTail &&
      (eventTailRevision !== null || liveTimelineEventCount > 0)
    ) {
      const viewport = eventsViewportRef.current
      if (viewport !== null) viewport.scrollTop = viewport.scrollHeight
    }
  }, [eventNavigation.current, eventPinnedToTail, eventTailRevision, liveTimelineEventCount])

  useEffect(() => {
    if (
      transcriptNavigation.current === null &&
      transcriptPinnedToTail &&
      transcriptTailRevision !== null
    ) {
      const viewport = transcriptViewportRef.current
      if (viewport !== null) viewport.scrollTop = viewport.scrollHeight
    }
  }, [transcriptNavigation.current, transcriptPinnedToTail, transcriptTailRevision])

  useEffect(() => {
    const node = transcriptViewportRef.current
    if (isLoading || node === null) return
    if (typeof IntersectionObserver === "undefined") {
      setTranscriptVisible(true)
      return
    }
    const observer = new IntersectionObserver(
      ([entry]) => setTranscriptVisible(entry?.isIntersecting ?? false),
      { threshold: 0.01 },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [isLoading])

  useEffect(() => {
    if (
      pendingTailCanReconcile({
        pending: eventReconcilePendingRef.current,
        atLatest: eventNavigation.current === null,
        pinned: eventPinnedToTail,
        visible: true,
        mutationActive,
        queryFetching: eventsQuery.isFetching,
        hasError: eventReconcileError !== null || eventsQuery.isError,
      })
    ) {
      eventReconcilePendingRef.current = false
      void reconcileLatestEvents()
    }
  }, [
    eventNavigation.current,
    eventPinnedToTail,
    eventReconcileError,
    eventsQuery.isError,
    eventsQuery.isFetching,
    mutationActive,
    reconcileLatestEvents,
  ])

  useEffect(() => {
    if (
      pendingTailCanReconcile({
        pending: transcriptReconcilePendingRef.current,
        atLatest: transcriptNavigation.current === null,
        pinned: transcriptPinnedToTail,
        visible: transcriptVisible,
        mutationActive,
        queryFetching: transcriptQuery.isFetching,
        hasError: transcriptReconcileError !== null || transcriptQuery.isError,
      })
    ) {
      transcriptReconcilePendingRef.current = false
      void reconcileLatestTranscript()
    }
  }, [
    mutationActive,
    reconcileLatestTranscript,
    transcriptNavigation.current,
    transcriptPinnedToTail,
    transcriptQuery.isError,
    transcriptQuery.isFetching,
    transcriptReconcileError,
    transcriptVisible,
  ])

  useEffect(
    () => () => {
      eventReconcileRunner.clearPending()
      transcriptReconcileRunner.clearPending()
      eventReconcileAbortRef.current?.abort()
      transcriptReconcileAbortRef.current?.abort()
    },
    [eventReconcileRunner, transcriptReconcileRunner],
  )

  useEffect(() => {
    if (!interruptRequested) return
    const status = session?.status
    if (status && !["pending", "running", "interrupting"].includes(status)) {
      setInterruptRequested(false)
      setInterruptReason("")
    }
  }, [session?.status, interruptRequested])

  const refreshSessionReadModels = () => {
    eventReconcileRunner.clearPending()
    transcriptReconcileRunner.clearPending()
    eventReconcileAbortRef.current?.abort()
    transcriptReconcileAbortRef.current?.abort()
    const historyRefreshes: Promise<unknown>[] = []
    if (eventAtLatestRef.current) {
      if (eventPinnedToTailRef.current) {
        eventReconcilePendingRef.current = false
        historyRefreshes.push(
          queryClient.refetchQueries({
            queryKey: ["session-events", sessionId, "latest"],
            exact: true,
          }),
        )
      } else {
        eventReconcilePendingRef.current = true
      }
    } else {
      eventReconcilePendingRef.current = false
    }
    if (transcriptAtLatestRef.current) {
      if (transcriptPinnedToTailRef.current && transcriptVisibleRef.current) {
        transcriptReconcilePendingRef.current = false
        historyRefreshes.push(
          queryClient.refetchQueries({
            queryKey: ["session-transcript", sessionId, "latest"],
            exact: true,
          }),
        )
      } else {
        transcriptReconcilePendingRef.current = true
      }
    } else {
      transcriptReconcilePendingRef.current = false
    }
    setEventReconcileError(null)
    setTranscriptReconcileError(null)
    suppressActivityReconcileRef.current = true
    const stateRefresh = stateQuery.refetch().then((result) => {
      if (result.data !== undefined) {
        observedActivityRef.current = result.data.last_activity_at
      }
      suppressActivityReconcileRef.current = false
      return result
    })
    return Promise.all([stateRefresh, ...historyRefreshes])
  }

  const refreshAfterInterrupt = () =>
    Promise.all([
      refreshSessionReadModels(),
      pendingActionQuery.refetch(),
      queryClient.invalidateQueries({ queryKey: ["sessions-summary"] }),
      queryClient.invalidateQueries({ queryKey: ["pending-actions"] }),
      queryClient.invalidateQueries({ queryKey: ["tasks"] }),
    ])

  const handleInterrupt = async () => {
    let streamFailed = false
    setInterruptRequested(true)
    setInterruptError(null)
    setLiveEvents([])
    const reason = interruptReason.trim()

    await streamInterruptSession(
      sessionId,
      reason ? { reason } : {},
      (event) => setLiveEvents((previous) => [...previous, event]),
      () => {
        if (!streamFailed) {
          setInterruptSheetOpen(false)
          setInterruptReason("")
        }
        void refreshAfterInterrupt().finally(() => {
          setLiveEvents([])
          if (streamFailed) setInterruptRequested(false)
        })
      },
      (message, streamError) => {
        if (isInterruptionConflict(streamError)) {
          setInterruptSheetOpen(false)
          setInterruptReason("")
          setInterruptRequested(true)
          return
        }
        streamFailed = true
        setInterruptRequested(false)
        setInterruptError(message)
      },
    )
  }

  const handleResume = async () => {
    if (!resumePrompt.trim()) return
    let streamFailed = false
    setResuming(true)
    setLiveEvents([])
    setResumeError(null)
    await streamResume(
      sessionId,
      resumePrompt,
      (event) => setLiveEvents((prev) => [...prev, event]),
      () => {
        setResuming(false)
        if (!streamFailed) {
          setResumePrompt("")
        }
        void Promise.all([refreshSessionReadModels(), pendingActionQuery.refetch()]).finally(() => {
          setLiveEvents([])
        })
      },
      (message) => {
        streamFailed = true
        setResumeError(message)
      },
    )
  }

  const handleRetryCascade = async () => {
    let streamFailed = false
    setRetryingCascade(true)
    setCascadeRetryError(null)
    setLiveEvents([])
    await streamInterruptSession(
      sessionId,
      { reason: "Retry incomplete background interruption." },
      (event) => setLiveEvents((previous) => [...previous, event]),
      () => {
        void refreshAfterInterrupt().finally(() => {
          setRetryingCascade(false)
          setLiveEvents([])
        })
      },
      (message) => {
        streamFailed = true
        setRetryingCascade(false)
        setCascadeRetryError(message)
      },
    )
    if (streamFailed) setLiveEvents([])
  }

  const finishContinuation = (streamFailed: boolean) => {
    void Promise.all([refreshSessionReadModels(), pendingActionQuery.refetch()]).finally(() => {
      if (!streamFailed) {
        setLiveEvents([])
      }
    })
  }

  const handleApprovalDecision = async (
    action: Extract<PendingAction, { kind: "approval" }>,
    decision: "approve" | "deny",
    reason: string | null,
  ) => {
    let streamFailed = false
    setResolvingAction(true)
    setLiveEvents([])
    setActionError(null)
    await streamResolveToolApproval(
      {
        session_id: sessionId,
        approval_id: action.approvalId,
        decision,
        reason,
      },
      (event) => setLiveEvents((prev) => [...prev, event]),
      () => {
        setResolvingAction(false)
        finishContinuation(streamFailed)
      },
      (message) => {
        streamFailed = true
        setActionError(message)
      },
    )
  }

  const handleUserInputAnswer = async (
    action: Extract<PendingAction, { kind: "user_input" }>,
    answer: string,
  ) => {
    let streamFailed = false
    setResolvingAction(true)
    setLiveEvents([])
    setActionError(null)
    await streamResolveUserInput(
      {
        session_id: sessionId,
        input_id: action.inputId,
        answer,
      },
      (event) => setLiveEvents((prev) => [...prev, event]),
      () => {
        setResolvingAction(false)
        finishContinuation(streamFailed)
      },
      (message) => {
        streamFailed = true
        setActionError(message)
      },
    )
  }

  const handleManualRecovery = async (
    action: Extract<PendingAction, { kind: "manual_recovery" }>,
    outcome: RecoveryOutcome,
    message: string,
    answer: string | null,
  ) => {
    let streamFailed = false
    setResolvingAction(true)
    setLiveEvents([])
    setActionError(null)
    const onEvent = (event: SSEEvent) => setLiveEvents((prev) => [...prev, event])
    const onDone = () => {
      setResolvingAction(false)
      finishContinuation(streamFailed)
    }
    const onError = (message: string) => {
      streamFailed = true
      setActionError(message)
    }

    if (action.inputId) {
      if (!answer) {
        setActionError("Manual user-input recovery requires an answer.")
        setResolvingAction(false)
        return
      }
      await streamRecoverUserInput(
        {
          session_id: sessionId,
          input_id: action.inputId,
          tool_call_id: action.toolCallId,
          answer,
          outcome,
          message,
        },
        onEvent,
        onDone,
        onError,
      )
      return
    }

    if (action.approvalId) {
      await streamRecoverToolApproval(
        {
          session_id: sessionId,
          approval_id: action.approvalId,
          tool_call_id: action.toolCallId,
          outcome,
          message,
        },
        onEvent,
        onDone,
        onError,
      )
      return
    }

    if (action.roundId) {
      await streamRecoverToolRound(
        {
          session_id: sessionId,
          round_id: action.roundId,
          tool_call_id: action.toolCallId,
          outcome,
          message,
        },
        onEvent,
        onDone,
        onError,
      )
      return
    }

    setActionError("Manual recovery requires an approval_id, input_id, or round_id.")
    setResolvingAction(false)
  }

  if (isLoading) return <div className="text-muted-foreground">Loading...</div>
  if (isError) {
    return (
      <div className="text-destructive">
        {error instanceof Error ? error.message : "Failed to load session."}
      </div>
    )
  }
  if (!session) return <div className="text-destructive">Session not found</div>

  const stateIsFresh = state !== undefined && !stateRefetchError
  const canInterrupt =
    stateIsFresh && INTERRUPTIBLE_SESSION_STATUSES.has(session.status) && !interruptRequested
  const interruptionFinalizing = session.status === "interrupting" || interruptRequested
  const summary = summaryQuery.data
  const allEvents =
    eventNavigation.current === null && eventPinnedToTail
      ? mergeStoredAndLiveRecords<SessionEvent>(events, liveEvents)
      : events
  const filteredEvents = allEvents.filter((e) => e.type !== "model.text.delta")
  const selectedEvent = filteredEvents.find((event) => event.id === selectedEventId) ?? null
  const pendingActionStatus = PENDING_ACTION_SESSION_STATUSES.has(session.status)
  const pendingAction = pendingActionStatus
    ? pendingActionFromApi(pendingActionQuery.data?.actions[0])
    : null
  const pendingActionIssue = pendingActionStatus ? pendingActionQuery.data?.issues[0] : undefined
  const pendingActionStateUnknown =
    pendingActionStatus &&
    (pendingActionQuery.data === undefined ||
      pendingActionQuery.isFetching ||
      pendingActionQuery.isError)
  const hasPendingInterruptionCascade = interruptionCascade !== "none"
  const failureEvent = latestFailureEvent(
    interruptionCascade === "failed"
      ? filteredEvents
      : filteredEvents.filter((event) => event.type !== "session.interruption_cascade_failed"),
  )
  const canRetryCascade =
    stateIsFresh && session.status === "interrupted" && interruptionCascade === "failed"
  const cascadePending = session.status === "interrupted" && interruptionCascade === "pending"
  const canResume =
    pendingActionStatus &&
    !pendingAction &&
    !pendingActionIssue &&
    !pendingActionStateUnknown &&
    !hasPendingInterruptionCascade &&
    stateIsFresh
  const summaryIsCurrent =
    summary !== undefined &&
    state !== undefined &&
    summaryStateIsStable({
      status: state.status,
      lastActivityAt: state.last_activity_at,
      interruptionCascade,
    })
  const models = summaryModels(summary)
  const modelSteps = summary?.usage.model_steps
  const toolCalls = summary?.usage.tool_calls
  const tokenInput = summary?.usage.usage?.input_tokens
  const tokenOutput = summary?.usage.usage?.output_tokens
  const tokenTotal = summary === undefined ? null : totalTokens(summary)
  const outcomeReason = summaryIsCurrent
    ? summary.outcome.reason || summary.session.status
    : session.status
  const latestEvent = summary?.events.latest_event
  const eventCounts = summaryIsCurrent ? sortedEventCounts(summary) : []
  const totalEvents = summaryIsCurrent ? summary.events.total_events : undefined
  const totalTranscriptMessages =
    transcriptQuery.data?.total_messages ?? summary?.transcript.total_messages ?? transcript.length
  const statusDetail =
    summaryIsCurrent && summary.outcome.retry ? "retry recorded" : "current lifecycle state"
  const summaryUnavailableDetail = summaryQuery.isError ? "summary unavailable" : "loading summary"
  const summarySnapshotDetail = summaryIsCurrent ? "" : " · snapshot"
  const createdAtMs = timestampMs(session.created_at)
  const terminalEventAt = summaryIsCurrent ? summary.outcome.terminal_event?.timestamp : null
  const updatedAtMs = timestampMs(
    durationEndTimestamp({
      terminalEventAt,
      lastActivityAt: state?.last_activity_at,
      updatedAt: session.updated_at,
    }),
  )
  const duration =
    createdAtMs !== null && updatedAtMs !== null ? Math.max(0, updatedAtMs - createdAtMs) : null
  const durationText =
    duration === null
      ? null
      : duration < 60_000
        ? `${Math.round(duration / 1000)}s duration`
        : `${Math.round(duration / 60_000)}m duration`
  const summaryDurationValue =
    duration === null
      ? "-"
      : duration < 60_000
        ? `${Math.round(duration / 1000)}s`
        : `${Math.floor(duration / 60_000)}m ${Math.round((duration % 60_000) / 1000)}s`
  const olderEventCursor =
    eventsQuery.data?.has_more && typeof eventsQuery.data.next_sequence === "number"
      ? eventsQuery.data.next_sequence
      : null
  const olderTranscript =
    transcriptQuery.data === undefined
      ? undefined
      : olderTranscriptPage(transcriptQuery.data.offset, SESSION_TRANSCRIPT_PAGE_SIZE)
  const lifecycleActive = ACTIVE_SESSION_STATUSES.has(session.status) || mutationActive
  const eventHistoryMode = historyModeLabel({
    atLatest: eventNavigation.current === null,
    active: lifecycleActive,
    pinned: eventPinnedToTail,
  })
  const transcriptHistoryMode = historyModeLabel({
    atLatest: transcriptNavigation.current === null,
    active: lifecycleActive,
    pinned: transcriptPinnedToTail,
  })
  const durationDetailText = durationDetail(session.status, terminalEventAt != null)
  const showNewerEvents = () => {
    eventReconcileRunner.clearPending()
    eventReconcileAbortRef.current?.abort()
    eventReconcilePendingRef.current = false
    const next = navigateToNewerPage(eventNavigation)
    setEventReconcileError(null)
    setEventNavigation(next)
    if (next.current === null) setEventPinnedToTail(true)
  }
  const showOlderEvents = () => {
    if (olderEventCursor === null) return
    eventReconcileRunner.clearPending()
    eventReconcileAbortRef.current?.abort()
    eventReconcilePendingRef.current = false
    setEventReconcileError(null)
    setEventPinnedToTail(false)
    setEventNavigation(navigateToOlderPage(eventNavigation, olderEventCursor))
  }
  const jumpToEventTail = () => {
    eventReconcileRunner.clearPending()
    eventReconcileAbortRef.current?.abort()
    if (eventNavigation.current !== null) eventReconcilePendingRef.current = false
    setEventReconcileError(null)
    setEventPinnedToTail(true)
    setEventNavigation(jumpToLatestPage(null))
    const viewport = eventsViewportRef.current
    if (viewport !== null) viewport.scrollTop = viewport.scrollHeight
  }
  const showNewerTranscript = () => {
    transcriptReconcileRunner.clearPending()
    transcriptReconcileAbortRef.current?.abort()
    transcriptReconcilePendingRef.current = false
    const next = navigateToNewerPage(transcriptNavigation)
    setTranscriptReconcileError(null)
    setTranscriptNavigation(next)
    if (next.current === null) setTranscriptPinnedToTail(true)
  }
  const showOlderTranscript = () => {
    if (olderTranscript === undefined) return
    transcriptReconcileRunner.clearPending()
    transcriptReconcileAbortRef.current?.abort()
    transcriptReconcilePendingRef.current = false
    setTranscriptReconcileError(null)
    setTranscriptPinnedToTail(false)
    setTranscriptNavigation(navigateToOlderPage(transcriptNavigation, olderTranscript))
  }
  const jumpToTranscriptTail = () => {
    transcriptReconcileRunner.clearPending()
    transcriptReconcileAbortRef.current?.abort()
    if (transcriptNavigation.current !== null) transcriptReconcilePendingRef.current = false
    setTranscriptReconcileError(null)
    setTranscriptPinnedToTail(true)
    setTranscriptNavigation(jumpToLatestPage(null))
    const viewport = transcriptViewportRef.current
    if (viewport !== null) viewport.scrollTop = viewport.scrollHeight
  }

  return (
    <Page className="space-y-4">
      <div className="flex flex-shrink-0 items-start gap-3">
        <Link
          to="/sessions"
          className="mt-1 rounded-md border border-border p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={statusVariant(session.status)}>{session.status}</Badge>
            <span className="text-sm font-medium text-muted-foreground">{session.agent_name}</span>
            {durationText && <span className="text-sm text-muted-foreground">{durationText}</span>}
          </div>
          <h1 className="mt-2 truncate font-mono text-2xl font-semibold leading-tight">
            {session.id}
          </h1>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <span>Created {formatDateTime(session.created_at)}</span>
            {session.provider_name && <Badge variant="outline">{session.provider_name}</Badge>}
            {session.model && <Badge variant="outline">{session.model}</Badge>}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <a
              href={dashboardPath("/agents", { q: session.agent_name })}
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Agent
            </a>
            {session.environment_name && (
              <a
                href={dashboardPath("/environments", { q: session.environment_name })}
                className={buttonVariants({ variant: "outline", size: "sm" })}
              >
                Environment
              </a>
            )}
            <a
              href={dashboardPath("/artifacts", { session_id: session.id })}
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Artifacts
            </a>
            {canInterrupt && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => {
                  setInterruptError(null)
                  setInterruptSheetOpen(true)
                }}
              >
                <CircleStop className="mr-1.5 h-4 w-4" />
                Interrupt session
              </Button>
            )}
            {interruptionFinalizing && (
              <Button variant="outline" size="sm" disabled>
                <LoaderCircle className="mr-1.5 h-4 w-4 animate-spin" />
                Interruption finalizing...
              </Button>
            )}
            {cascadePending && (
              <Button variant="outline" size="sm" disabled>
                <LoaderCircle className="mr-1.5 h-4 w-4 animate-spin" />
                Stopping background sessions...
              </Button>
            )}
            {canRetryCascade && (
              <Button
                variant="outline"
                size="sm"
                disabled={retryingCascade}
                onClick={handleRetryCascade}
              >
                {retryingCascade ? (
                  <LoaderCircle className="mr-1.5 h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="mr-1.5 h-4 w-4" />
                )}
                {retryingCascade ? "Retrying interruption..." : "Retry background interruption"}
              </Button>
            )}
          </div>
          {cascadeRetryError && (
            <p className="mt-2 text-sm text-destructive">{cascadeRetryError}</p>
          )}
        </div>
      </div>

      {interruptionFinalizing && (
        <div className="flex items-start gap-3 rounded-md border border-chart-1/30 bg-chart-1/5 px-4 py-3 text-sm">
          <LoaderCircle className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-chart-1" />
          <div>
            <div className="font-medium">Cayu is stopping active runtime work.</div>
            <div className="mt-1 text-muted-foreground">
              This session cannot be resumed until its durable status becomes interrupted. Any
              linked task remains application-owned and is not cancelled automatically.
            </div>
          </div>
        </div>
      )}

      {cascadePending && (
        <div className="flex items-start gap-3 rounded-md border border-chart-1/30 bg-chart-1/5 px-4 py-3 text-sm">
          <LoaderCircle className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-chart-1" />
          <div>
            <div className="font-medium">The parent session is interrupted.</div>
            <div className="mt-1 text-muted-foreground">
              Cayu is still stopping background subagent sessions. Resume becomes available after
              that durable cascade completes; linked application tasks remain application-owned.
            </div>
          </div>
        </div>
      )}

      {stateRefetchError && (
        <StateMessage tone="danger" className="rounded-md border border-destructive/30 p-4">
          <span className="font-medium">Live session state is temporarily unavailable.</span> The
          page is showing the last confirmed state, and mutation controls are disabled until polling
          recovers.
        </StateMessage>
      )}

      {detailRefetchError && (
        <StateMessage tone="danger" className="rounded-md border border-destructive/30 p-4">
          <div className="flex items-center justify-between gap-3">
            <span>
              <span className="font-medium">Session metadata could not be refreshed.</span> The page
              is preserving the last confirmed metadata while lifecycle state continues
              independently.
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={detailQuery.isFetching}
              onClick={() => {
                void detailQuery.refetch()
              }}
            >
              {detailQuery.isFetching ? (
                <LoaderCircle className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : null}
              Retry
            </Button>
          </div>
        </StateMessage>
      )}

      {summaryQuery.isError && (
        <StateMessage tone="danger" className="rounded-md border border-destructive/30 p-4">
          <div className="flex items-center justify-between gap-3">
            <span>
              <span className="font-medium">Session summary is temporarily unavailable.</span>{" "}
              Metadata and bounded history remain usable while you retry the aggregate read.
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={summaryQuery.isFetching}
              onClick={() => {
                void summaryQuery.refetch()
              }}
            >
              {summaryQuery.isFetching ? (
                <LoaderCircle className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : null}
              Retry
            </Button>
          </div>
        </StateMessage>
      )}

      <Card className="flex-shrink-0 gap-0 py-0">
        <CardContent className="grid grid-cols-1 p-0 md:grid-cols-2 xl:grid-cols-6">
          <SummaryStat
            icon={Activity}
            label="Outcome"
            value={outcomeReason}
            detail={statusDetail}
            className="border-b border-border md:border-r xl:border-b-0"
            tone={
              session.status === "failed"
                ? "red"
                : session.status === "running"
                  ? "amber"
                  : session.status === "completed"
                    ? "green"
                    : "neutral"
            }
          />
          <SummaryStat
            icon={Hash}
            label="Events"
            value={summary === undefined ? "-" : formatCount(summary.events.total_events)}
            detail={
              latestEvent
                ? `latest ${latestEvent.type}${summarySnapshotDetail}`
                : summaryUnavailableDetail
            }
            className="border-b border-border xl:border-b-0 xl:border-r"
            tone="blue"
          />
          <SummaryStat
            icon={Cpu}
            label="Model Steps"
            value={summary === undefined ? "-" : formatCount(modelSteps)}
            detail={
              models.length
                ? `${models.slice(0, 2).join(", ")}${summarySnapshotDetail}`
                : summary === undefined
                  ? summaryUnavailableDetail
                  : `no model usage${summarySnapshotDetail}`
            }
            className="border-b border-border md:border-r xl:border-b-0"
            tone="violet"
          />
          <SummaryStat
            icon={Wrench}
            label="Tool Calls"
            value={summary === undefined ? "-" : formatCount(toolCalls)}
            detail={
              summary === undefined
                ? summaryUnavailableDetail
                : `completed and failed calls${summarySnapshotDetail}`
            }
            className="border-b border-border xl:border-b-0 xl:border-r"
            tone="amber"
          />
          <SummaryStat
            icon={Database}
            label="Tokens"
            value={summary === undefined ? "-" : formatCount(tokenTotal)}
            detail={
              summary === undefined
                ? summaryUnavailableDetail
                : `${formatCount(tokenInput)} in / ${formatCount(tokenOutput)} out${summarySnapshotDetail}`
            }
            className="border-b border-border md:border-b-0 md:border-r"
            tone="blue"
          />
          <SummaryStat
            icon={Clock}
            label="Duration"
            value={summaryDurationValue}
            detail={durationDetailText}
            tone="neutral"
          />
        </CardContent>
      </Card>

      <SessionArtifacts artifacts={sessionArtifacts.data?.artifacts ?? []} sessionId={session.id} />

      {pendingAction && (
        <PendingActionBanner
          key={
            pendingAction.kind === "approval"
              ? pendingAction.approvalId
              : pendingAction.kind === "user_input"
                ? pendingAction.inputId
                : pendingAction.toolCallId
          }
          action={pendingAction}
          resolving={resolvingAction}
          error={actionError}
          onApprove={(reason) =>
            pendingAction.kind === "approval" &&
            void handleApprovalDecision(pendingAction, "approve", reason)
          }
          onDeny={(reason) =>
            pendingAction.kind === "approval" &&
            void handleApprovalDecision(pendingAction, "deny", reason)
          }
          onAnswer={(answer) =>
            pendingAction.kind === "user_input" && void handleUserInputAnswer(pendingAction, answer)
          }
          onRecover={(outcome, message, answer) =>
            pendingAction.kind === "manual_recovery" &&
            void handleManualRecovery(pendingAction, outcome, message, answer)
          }
        />
      )}

      {pendingActionIssue && (
        <StateMessage tone="danger" className="rounded-md border border-destructive/30 p-4">
          <span className="font-medium">Pending state requires direct inspection.</span>{" "}
          {pendingActionIssue.detail}
        </StateMessage>
      )}

      {pendingActionQuery.isError && pendingActionStatus && (
        <StateMessage tone="danger" className="rounded-md border border-destructive/30 p-4">
          <span className="font-medium">Pending state could not be verified.</span>{" "}
          {pendingActionQuery.error instanceof Error
            ? pendingActionQuery.error.message
            : "Reload the page before resuming this session."}
        </StateMessage>
      )}

      {failureEvent && (
        <FailureDebugBanner
          event={failureEvent}
          selected={selectedEventId === failureEvent.id}
          onInspect={() => setSelectedEventId(failureEvent.id)}
        />
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="flex min-h-[30rem] flex-col">
          <Card className="flex min-h-0 flex-1 gap-0 py-0">
            <CardHeader className="border-b border-border py-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-base">Event Timeline</CardTitle>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Stored runtime events, excluding streaming text deltas.
                  </p>
                </div>
                <div className="flex max-w-[62%] flex-wrap justify-end gap-1.5">
                  <Badge variant="secondary" className="font-mono text-[10px]">
                    {formatCount(filteredEvents.length)} visible
                    {totalEvents === undefined ? "" : ` · ${formatCount(totalEvents)} stored`}
                  </Badge>
                  <Badge variant={eventHistoryMode === "Live tail" ? "secondary" : "outline"}>
                    {eventHistoryMode}
                  </Badge>
                  {eventCounts.slice(0, 4).map(([type, count]) => (
                    <Badge key={type} variant="outline" className="font-mono text-[10px]">
                      {type} {count}
                    </Badge>
                  ))}
                  {eventNavigation.newer.length > 0 && (
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={eventsQuery.isFetching}
                      onClick={showNewerEvents}
                    >
                      Newer events
                    </Button>
                  )}
                  {eventNavigation.newer.length > 1 && (
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={eventsQuery.isFetching}
                      onClick={jumpToEventTail}
                    >
                      Jump to latest
                    </Button>
                  )}
                  {olderEventCursor !== null && (
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={eventsQuery.isFetching}
                      onClick={showOlderEvents}
                    >
                      {eventsQuery.isFetching ? (
                        <LoaderCircle className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                      ) : null}
                      {eventsQuery.isFetching ? "Loading events..." : "Older events"}
                    </Button>
                  )}
                  {eventNavigation.current === null && eventHistoryMode === "Paused" && (
                    <Button variant="outline" size="sm" onClick={jumpToEventTail}>
                      {lifecycleActive ? "Jump to live" : "Jump to latest"}
                    </Button>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent className="min-h-0 flex-1 p-0">
              <div
                ref={eventsViewportRef}
                className="max-h-[42rem] overflow-auto"
                onScroll={(event) => {
                  const viewport = event.currentTarget
                  const pinned =
                    viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 32
                  if (!pinned && eventPinnedToTail) {
                    eventReconcilePendingRef.current = true
                    eventReconcileRunner.clearPending()
                    eventReconcileAbortRef.current?.abort()
                    void queryClient.cancelQueries({
                      queryKey: ["session-events", sessionId, "latest"],
                      exact: true,
                    })
                  }
                  setEventPinnedToTail(pinned)
                }}
              >
                <div className="sticky top-0 z-10 grid grid-cols-[0.75rem_5.25rem_minmax(0,1fr)_1rem] gap-2 border-b border-border bg-background/95 px-3 py-2 text-[11px] font-medium uppercase text-muted-foreground backdrop-blur sm:grid-cols-[1rem_7.5rem_minmax(0,1fr)_1.5rem] sm:gap-3 sm:px-4">
                  <span />
                  <span>Time</span>
                  <span>Event</span>
                  <span />
                </div>
                {(eventsQuery.isError || eventReconcileError) && (
                  <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3 text-sm text-destructive">
                    <span>
                      {eventReconcileError ??
                        (eventsQuery.error instanceof Error
                          ? eventsQuery.error.message
                          : "Failed to load events.")}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        if (eventNavigation.current === null) void reconcileLatestEvents()
                        else void eventsQuery.refetch()
                      }}
                    >
                      Retry
                    </Button>
                  </div>
                )}
                {filteredEvents.map((e) => (
                  <EventRow
                    key={e.id}
                    event={e}
                    selected={e.id === selectedEventId}
                    onSelect={setSelectedEventId}
                  />
                ))}
                {filteredEvents.length === 0 && !eventsQuery.isError && !eventReconcileError && (
                  <p className="px-4 py-6 text-sm text-muted-foreground">
                    {eventsQuery.isLoading
                      ? "Loading events..."
                      : allEvents.length > 0
                        ? olderEventCursor !== null
                          ? "This page contains only hidden streaming text deltas. Load older events to continue."
                          : "Only streaming text deltas were recorded; they are hidden from this timeline."
                        : "No events yet"}
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="flex min-h-[30rem] gap-0 py-0">
          <CardHeader className="border-b border-border py-4">
            <CardTitle className="text-base">Event Detail</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              Structured payload and runtime metadata.
            </p>
          </CardHeader>
          <CardContent className="min-h-0 flex-1 overflow-auto p-0">
            <EventDetailPanel event={selectedEvent} />
          </CardContent>
        </Card>
      </div>

      <Card className="flex min-h-[30rem] gap-0 py-0">
        <CardHeader className="border-b border-border py-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base">Transcript</CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">Model-facing conversation state.</p>
            </div>
            <div className="flex flex-wrap justify-end gap-2">
              <Badge variant="outline" className="font-mono text-[10px]">
                {formatCount(transcript.length)} of {formatCount(totalTranscriptMessages)} shown
              </Badge>
              <Badge variant={transcriptHistoryMode === "Live tail" ? "secondary" : "outline"}>
                {transcriptHistoryMode}
              </Badge>
              {transcriptNavigation.newer.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={transcriptQuery.isFetching}
                  onClick={showNewerTranscript}
                >
                  Newer messages
                </Button>
              )}
              {transcriptNavigation.newer.length > 1 && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={transcriptQuery.isFetching}
                  onClick={jumpToTranscriptTail}
                >
                  Jump to latest
                </Button>
              )}
              {olderTranscript !== undefined && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={transcriptQuery.isFetching}
                  onClick={showOlderTranscript}
                >
                  {transcriptQuery.isFetching ? (
                    <LoaderCircle className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : null}
                  {transcriptQuery.isFetching ? "Loading messages..." : "Older messages"}
                </Button>
              )}
              {transcriptNavigation.current === null && transcriptHistoryMode === "Paused" && (
                <Button variant="outline" size="sm" onClick={jumpToTranscriptTail}>
                  {lifecycleActive ? "Jump to live" : "Jump to latest"}
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 p-0">
          <div
            ref={transcriptViewportRef}
            className="max-h-[42rem] space-y-3 overflow-auto p-4"
            onScroll={(event) => {
              const viewport = event.currentTarget
              const pinned =
                viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 32
              if (!pinned && transcriptPinnedToTail) {
                transcriptReconcilePendingRef.current = true
                transcriptReconcileRunner.clearPending()
                transcriptReconcileAbortRef.current?.abort()
                void queryClient.cancelQueries({
                  queryKey: ["session-transcript", sessionId, "latest"],
                  exact: true,
                })
              }
              setTranscriptPinnedToTail(pinned)
            }}
          >
            {(transcriptQuery.isError || transcriptReconcileError) && (
              <div className="flex items-center justify-between gap-3 text-sm text-destructive">
                <span>
                  {transcriptReconcileError ??
                    (transcriptQuery.error instanceof Error
                      ? transcriptQuery.error.message
                      : "Failed to load the transcript.")}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (transcriptNavigation.current === null) void reconcileLatestTranscript()
                    else void transcriptQuery.refetch()
                  }}
                >
                  Retry
                </Button>
              </div>
            )}
            {transcript.map((msg) => (
              <div
                key={transcriptMessageKey(msg)}
                className={cn(
                  "rounded-md border border-border p-3 text-sm",
                  msg.role === "user"
                    ? "border-chart-3/20 bg-chart-3/5"
                    : msg.role === "assistant"
                      ? "border-border bg-muted/60"
                      : msg.role === "system"
                        ? "border-border bg-background"
                        : "border-chart-4/25 bg-chart-4/10",
                )}
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <Badge variant="outline" className="font-mono text-[10px]">
                    {msg.role}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {msg.content.length} part{msg.content.length === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="space-y-2">
                  {msg.content.map((part, partIndex) => (
                    <TranscriptPart key={transcriptPartKey(partIndex)} part={part} />
                  ))}
                </div>
              </div>
            ))}
            {transcript.length === 0 && !transcriptQuery.isError && !transcriptReconcileError && (
              <p className="text-sm text-muted-foreground">
                {transcriptQuery.isLoading ? "Loading transcript..." : "No transcript yet"}
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {canResume && (
        <div className="flex-shrink-0">
          <Card className="gap-0 py-0">
            <CardContent className="py-3">
              {resumeError && <p className="mb-3 text-sm text-destructive">{resumeError}</p>}
              <div className="flex items-center gap-2">
                <Input
                  value={resumePrompt}
                  onChange={(e) => setResumePrompt(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleResume()}
                  placeholder="Continue with a new prompt..."
                  disabled={resuming}
                />
                <Button onClick={handleResume} disabled={resuming || !resumePrompt.trim()}>
                  <Send className="h-4 w-4 mr-2" />
                  {resuming ? "Running..." : "Resume"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Sheet
        open={interruptSheetOpen}
        onOpenChange={(open) => {
          if (interruptRequested) return
          setInterruptSheetOpen(open)
          if (!open) {
            setInterruptError(null)
            setInterruptReason("")
          }
        }}
      >
        <SheetContent side="right" showCloseButton={!interruptRequested}>
          <SheetHeader className="border-b border-border">
            <SheetTitle>Interrupt session?</SheetTitle>
            <SheetDescription>
              Stop Cayu-owned execution for this session and finalize it as interrupted.
            </SheetDescription>
          </SheetHeader>
          <div className="space-y-4 overflow-auto px-4">
            <div className="rounded-md border border-destructive/25 bg-destructive/5 p-3 text-sm">
              <div className="font-medium text-destructive">This is a runtime interruption.</div>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                <li>It does not cancel any linked task or business operation.</li>
                <li>
                  It does not reverse tool calls or external side effects that already completed.
                </li>
                <li>Active background child sessions owned by this run may also be interrupted.</li>
              </ul>
            </div>
            <div className="space-y-2">
              <label htmlFor="interrupt-reason" className="text-sm font-medium">
                Reason <span className="font-normal text-muted-foreground">(optional)</span>
              </label>
              <Textarea
                id="interrupt-reason"
                value={interruptReason}
                onChange={(event) => setInterruptReason(event.target.value)}
                placeholder="Why is this session being interrupted?"
                rows={4}
                disabled={interruptRequested}
              />
            </div>
            {interruptError && (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {interruptError}
              </div>
            )}
          </div>
          <SheetFooter className="border-t border-border">
            <Button
              variant="outline"
              disabled={interruptRequested}
              onClick={() => setInterruptSheetOpen(false)}
            >
              Keep running
            </Button>
            <Button variant="destructive" disabled={interruptRequested} onClick={handleInterrupt}>
              {interruptRequested ? (
                <LoaderCircle className="mr-1.5 h-4 w-4 animate-spin" />
              ) : (
                <CircleStop className="mr-1.5 h-4 w-4" />
              )}
              {interruptRequested ? "Interrupting..." : "Interrupt session"}
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </Page>
  )
}
