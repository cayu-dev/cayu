import { useQuery } from "@tanstack/react-query"
import { Link, useParams } from "@tanstack/react-router"
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  Bot,
  CheckCircle,
  ChevronRight,
  Clock,
  Cpu,
  Database,
  Hash,
  type LucideIcon,
  MessageSquare,
  Send,
  ShieldCheck,
  UserRound,
  Wrench,
} from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { Page, PayloadViewer } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Input } from "../components/ui/input"
import {
  fetchSession,
  fetchSessionSummary,
  type SessionEvent,
  type SessionSummary,
  type SSEEvent,
  streamResolveToolApproval,
  streamResolveUserInput,
  streamResume,
} from "../lib/api"
import {
  formatCount,
  formatDateTime,
  formatTime,
  modelUsagePayload,
  numericValue,
} from "../lib/format"
import {
  isFailureEventType,
  latestFailureEvent,
  objectPayload,
  optionalString,
  summarizeFailureEvent,
} from "../lib/session-debug"
import { cn } from "../lib/utils"

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

function fallbackModelSteps(events: Array<{ type: string }>) {
  return events.filter((event) => event.type === "model.completed").length
}

function fallbackToolCalls(events: Array<{ type: string }>) {
  const terminalCount = events.filter(
    (event) =>
      event.type === "tool.call.completed" ||
      event.type === "tool.call.failed" ||
      event.type === "tool.call.blocked",
  ).length
  if (terminalCount > 0) return terminalCount
  return events.filter((event) => event.type === "tool.call.started").length
}

function fallbackUsage(
  events: Array<{
    type: string
    payload: Record<string, unknown>
  }>,
) {
  return events.reduce(
    (usage, event) => {
      if (event.type !== "model.completed") return usage
      const eventUsage = modelUsagePayload(event.payload)
      const inputTokens = numericValue(eventUsage.input_tokens)
      const outputTokens = numericValue(eventUsage.output_tokens)
      const totalTokens = numericValue(eventUsage.total_tokens) || inputTokens + outputTokens
      const model = typeof eventUsage.model === "string" ? eventUsage.model : null
      return {
        inputTokens: usage.inputTokens + inputTokens,
        outputTokens: usage.outputTokens + outputTokens,
        totalTokens: usage.totalTokens + totalTokens,
        models:
          model === null || usage.models.includes(model) ? usage.models : [...usage.models, model],
      }
    },
    { inputTokens: 0, outputTokens: 0, totalTokens: 0, models: [] as string[] },
  )
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

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : []
}

function pendingActionFromEvents(
  sessionStatus: string,
  events: SessionEvent[],
): PendingAction | null {
  if (sessionStatus !== "interrupted") return null

  for (const event of [...events].reverse()) {
    if (
      event.type === "session.resumed" ||
      event.type === "session.completed" ||
      event.type === "session.failed"
    ) {
      return null
    }

    if (event.type === "tool.call.approval_requested") {
      const approval = objectPayload(event.payload.approval)
      const approvalId = optionalString(approval?.approval_id)
      if (approval && approvalId) {
        return {
          kind: "approval",
          approvalId,
          toolName: optionalString(approval.tool_name) || event.tool_name || "tool",
          reason: optionalString(approval.reason),
          arguments: approval.arguments || {},
        }
      }
    }

    if (event.type === "session.awaiting_user_input") {
      const inputId = optionalString(event.payload.input_id)
      if (inputId) {
        return {
          kind: "user_input",
          inputId,
          toolCallId: optionalString(event.payload.tool_call_id),
          question: optionalString(event.payload.question) || "Input required",
          options: stringList(event.payload.options),
        }
      }
    }

    if (event.type === "session.interrupted") {
      if (event.payload.interruption_type === "tool_approval_required") {
        const approval = objectPayload(event.payload.approval)
        const approvalId = optionalString(approval?.approval_id)
        if (approval && approvalId) {
          return {
            kind: "approval",
            approvalId,
            toolName: optionalString(approval.tool_name) || event.tool_name || "tool",
            reason: optionalString(approval.reason),
            arguments: approval.arguments || {},
          }
        }
      }
      if (event.payload.interruption_type === "user_input_required") {
        const userInput = objectPayload(event.payload.user_input)
        const inputId = optionalString(userInput?.input_id)
        if (userInput && inputId) {
          return {
            kind: "user_input",
            inputId,
            toolCallId: optionalString(userInput.tool_call_id),
            question: optionalString(userInput.question) || "Input required",
            options: stringList(userInput.options),
          }
        }
      }
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
}: {
  action: PendingAction
  resolving: boolean
  error: string | null
  onApprove: (reason: string | null) => void
  onDeny: (reason: string | null) => void
  onAnswer: (answer: string) => void
}) {
  const [answer, setAnswer] = useState("")
  const [reason, setReason] = useState("")

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

function transcriptMessageKey(
  message: { role: string; content: Array<Record<string, unknown>> },
  index: number,
) {
  return `${index}:${message.role}`
}

function transcriptPartKey(index: number) {
  return String(index)
}

export function SessionDetailPage({ live }: { live?: boolean }) {
  const { sessionId } = useParams({ from: "/sessions/$sessionId" })
  const { data, error, isError, isLoading, refetch } = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => fetchSession(sessionId),
    refetchInterval: live ? 2000 : 5000,
  })
  const summaryQuery = useQuery({
    queryKey: ["session-summary", sessionId],
    queryFn: () => fetchSessionSummary(sessionId),
    refetchInterval: live ? 2000 : 5000,
    enabled: !isError,
  })

  const [resumePrompt, setResumePrompt] = useState("")
  const [resuming, setResuming] = useState(false)
  const [liveEvents, setLiveEvents] = useState<SSEEvent[]>([])
  const [resumeError, setResumeError] = useState<string | null>(null)
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null)
  const [resolvingAction, setResolvingAction] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const eventsEndRef = useRef<HTMLDivElement>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)
  const transcriptCount = data?.transcript.length ?? 0
  const liveEventCount = liveEvents.length

  useEffect(() => {
    if (liveEventCount > 0) {
      eventsEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [liveEventCount])

  useEffect(() => {
    if (resuming && transcriptCount > 0) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [resuming, transcriptCount])

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
        void Promise.all([refetch(), summaryQuery.refetch()]).finally(() => {
          setLiveEvents([])
        })
      },
      (message) => {
        streamFailed = true
        setResumeError(message)
      },
    )
  }

  const finishContinuation = (streamFailed: boolean) => {
    void Promise.all([refetch(), summaryQuery.refetch()]).finally(() => {
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

  if (isLoading) return <div className="text-muted-foreground">Loading...</div>
  if (isError) {
    return (
      <div className="text-destructive">
        {error instanceof Error ? error.message : "Failed to load session."}
      </div>
    )
  }
  if (!data) return <div className="text-destructive">Session not found</div>

  const { session, events, transcript } = data
  const summary = summaryQuery.data
  const allEvents = [...events, ...liveEvents.map((e, i) => ({ ...e, id: `live-${i}` }))]
  const filteredEvents = allEvents.filter((e) => e.type !== "model.text.delta")
  const selectedEvent = filteredEvents.find((event) => event.id === selectedEventId) ?? null
  const pendingAction = pendingActionFromEvents(session.status, filteredEvents)
  const failureEvent = latestFailureEvent(filteredEvents)
  const canResume =
    ["completed", "failed", "interrupted"].includes(session.status) && !pendingAction
  const eventUsage = fallbackUsage(filteredEvents)
  const models = summaryModels(summary)
  const modelSteps = summary?.usage.model_steps ?? fallbackModelSteps(filteredEvents)
  const toolCalls = summary?.usage.tool_calls ?? fallbackToolCalls(filteredEvents)
  const tokenInput = summary?.usage.usage?.input_tokens ?? eventUsage.inputTokens
  const tokenOutput = summary?.usage.usage?.output_tokens ?? eventUsage.outputTokens
  const tokenTotal = summary === undefined ? eventUsage.totalTokens : totalTokens(summary)
  const outcomeReason = summary?.outcome.reason || summary?.session.status || session.status
  const latestEvent = summary?.events.latest_event
  const eventCounts = sortedEventCounts(summary)
  const statusDetail = summary?.outcome.retry ? "retry recorded" : "latest session state"
  const durationSession = summary?.session ?? session
  const createdAtMs = timestampMs(durationSession.created_at)
  const updatedAtMs = timestampMs(durationSession.updated_at)
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
        </div>
      </div>

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
            value={formatCount(summary?.events.total_events ?? filteredEvents.length)}
            detail={latestEvent ? `latest ${latestEvent.type}` : "stored timeline"}
            className="border-b border-border xl:border-b-0 xl:border-r"
            tone="blue"
          />
          <SummaryStat
            icon={Cpu}
            label="Model Steps"
            value={formatCount(modelSteps)}
            detail={
              models.length
                ? models.slice(0, 2).join(", ")
                : eventUsage.models.length
                  ? eventUsage.models.slice(0, 2).join(", ")
                  : "no model usage"
            }
            className="border-b border-border md:border-r xl:border-b-0"
            tone="violet"
          />
          <SummaryStat
            icon={Wrench}
            label="Tool Calls"
            value={formatCount(toolCalls)}
            detail="completed and failed tool calls"
            className="border-b border-border xl:border-b-0 xl:border-r"
            tone="amber"
          />
          <SummaryStat
            icon={Database}
            label="Tokens"
            value={formatCount(tokenTotal)}
            detail={`${formatCount(tokenInput)} in / ${formatCount(tokenOutput)} out`}
            className="border-b border-border md:border-b-0 md:border-r"
            tone="blue"
          />
          <SummaryStat
            icon={Clock}
            label="Duration"
            value={summaryDurationValue}
            detail="started to completed"
            tone="neutral"
          />
        </CardContent>
      </Card>

      {pendingAction && (
        <PendingActionBanner
          key={pendingAction.kind === "approval" ? pendingAction.approvalId : pendingAction.inputId}
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
        />
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
                  {eventCounts.slice(0, 4).map(([type, count]) => (
                    <Badge key={type} variant="outline" className="font-mono text-[10px]">
                      {type} {count}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent className="min-h-0 flex-1 p-0">
              <div className="h-full overflow-auto">
                <div className="sticky top-0 z-10 grid grid-cols-[0.75rem_5.25rem_minmax(0,1fr)_1rem] gap-2 border-b border-border bg-background/95 px-3 py-2 text-[11px] font-medium uppercase text-muted-foreground backdrop-blur sm:grid-cols-[1rem_7.5rem_minmax(0,1fr)_1.5rem] sm:gap-3 sm:px-4">
                  <span />
                  <span>Time</span>
                  <span>Event</span>
                  <span />
                </div>
                {filteredEvents.map((e) => (
                  <EventRow
                    key={e.id}
                    event={e}
                    selected={e.id === selectedEventId}
                    onSelect={setSelectedEventId}
                  />
                ))}
                {filteredEvents.length === 0 && (
                  <p className="px-4 py-6 text-sm text-muted-foreground">No events yet</p>
                )}
                <div ref={eventsEndRef} />
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
            {summary?.transcript && (
              <Badge variant="outline" className="font-mono text-[10px]">
                {formatCount(summary.transcript.total_messages)} messages
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 p-0">
          <div className="max-h-[42rem] space-y-3 overflow-auto p-4">
            {transcript.map((msg, messageIndex) => (
              <div
                key={transcriptMessageKey(msg, messageIndex)}
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
            {transcript.length === 0 && (
              <p className="text-sm text-muted-foreground">No transcript yet</p>
            )}
            <div ref={transcriptEndRef} />
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
    </Page>
  )
}
