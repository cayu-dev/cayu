import { useQuery } from "@tanstack/react-query"
import { Link, useParams } from "@tanstack/react-router"
import {
  Activity,
  AlertCircle,
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
  type SessionSummary,
  type SSEEvent,
  streamResume,
} from "../lib/api"
import {
  formatCount,
  formatDateTime,
  formatTime,
  modelUsagePayload,
  numericValue,
} from "../lib/format"
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
  if (type === "tool.call.failed" || type === "session.failed") return "bg-destructive"
  if (type.startsWith("tool.call")) return "bg-chart-3"
  if (type.startsWith("session.")) return "bg-primary"
  return "bg-muted-foreground"
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
    return reason ? `session.interrupted - ${reason}` : event.type
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
}: {
  event: {
    type: string
    tool_name: string | null
    payload: Record<string, unknown>
    timestamp: string
  }
}) {
  const [expanded, setExpanded] = useState(false)
  const time = formatTime(event.timestamp)
  const summary = eventSummary(event)

  return (
    <div className="relative border-b border-border/70 last:border-0">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="grid w-full grid-cols-[0.75rem_5.25rem_minmax(0,1fr)_1rem] items-center gap-2 px-3 py-3 text-left transition-colors hover:bg-muted/50 sm:grid-cols-[1rem_7.5rem_minmax(0,1fr)_1.5rem] sm:gap-3 sm:px-4"
      >
        <span className={cn("h-2.5 w-2.5 rounded-full", eventTone(event.type))} />
        <span className="truncate font-mono text-xs text-muted-foreground">{time}</span>
        <span className="flex min-w-0 items-center gap-2">
          <EventIcon type={event.type} />
          <span className="truncate text-sm">{summary}</span>
        </span>
        <ChevronRight
          className={cn(
            "h-4 w-4 text-muted-foreground transition-transform",
            expanded && "rotate-90",
          )}
        />
      </button>
      {expanded && <PayloadViewer value={event.payload} className="mx-4 mb-3 bg-muted/70" />}
    </div>
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
  const eventsEndRef = useRef<HTMLDivElement>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)
  const storedEventCount = data?.events.length ?? 0
  const transcriptCount = data?.transcript.length ?? 0
  const liveEventCount = liveEvents.length

  useEffect(() => {
    if (storedEventCount + liveEventCount > 0) {
      eventsEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [storedEventCount, liveEventCount])

  useEffect(() => {
    if (transcriptCount > 0) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [transcriptCount])

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
  const canResume = ["completed", "failed", "interrupted"].includes(session.status)
  const allEvents = [...events, ...liveEvents.map((e, i) => ({ ...e, id: `live-${i}` }))]
  const filteredEvents = allEvents.filter((e) => e.type !== "model.text.delta")
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

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <div className="flex min-h-[28rem] flex-col xl:col-span-3">
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
                  <EventRow key={e.id} event={e} />
                ))}
                {filteredEvents.length === 0 && (
                  <p className="px-4 py-6 text-sm text-muted-foreground">No events yet</p>
                )}
                <div ref={eventsEndRef} />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex min-h-[28rem] flex-col xl:col-span-2">
          <Card className="flex min-h-0 flex-1 gap-0 py-0">
            <CardHeader className="border-b border-border py-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-base">Transcript</CardTitle>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Model-facing conversation state.
                  </p>
                </div>
                {summary?.transcript && (
                  <Badge variant="outline" className="font-mono text-[10px]">
                    {formatCount(summary.transcript.total_messages)} messages
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="min-h-0 flex-1 p-0">
              <div className="h-full space-y-3 overflow-auto p-4">
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
                    {msg.content.map((part, partIndex) => (
                      <div key={transcriptPartKey(partIndex)} className="min-w-0">
                        {part.type === "text" && (
                          <span className="whitespace-pre-wrap break-words leading-relaxed">
                            {String(part.text || "").slice(0, 500)}
                            {String(part.text || "").length > 500 ? "..." : ""}
                          </span>
                        )}
                        {part.type === "tool_call" && (
                          <span className="block overflow-x-auto rounded bg-background/80 px-2 py-1 font-mono text-xs">
                            call {String(part.tool_name || "")}(
                            {JSON.stringify(part.arguments || {}).slice(0, 100)})
                          </span>
                        )}
                        {part.type === "tool_result" && (
                          <span className="block overflow-x-auto rounded bg-background/80 px-2 py-1 font-mono text-xs">
                            result {String(part.content || "").slice(0, 300)}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                ))}
                {transcript.length === 0 && (
                  <p className="text-sm text-muted-foreground">No transcript yet</p>
                )}
                <div ref={transcriptEndRef} />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

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
