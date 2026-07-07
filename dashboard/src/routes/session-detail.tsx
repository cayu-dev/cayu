import { useQuery } from "@tanstack/react-query"
import { Link, useParams } from "@tanstack/react-router"
import { AlertCircle, ArrowLeft, Bot, CheckCircle, MessageSquare, Send, Wrench } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Input } from "../components/ui/input"
import { fetchSession, type SSEEvent, streamResume } from "../lib/api"

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
  const time = new Date(event.timestamp).toLocaleTimeString()

  let summary = event.type
  if (event.tool_name) summary = `${event.type} — ${event.tool_name}`
  if (event.type === "model.completed") {
    const usage = event.payload.usage as Record<string, number> | undefined
    if (usage) {
      summary = `model.completed — ${usage.input_tokens || 0} in / ${usage.output_tokens || 0} out`
      if (usage.cached_tokens) summary += ` (${usage.cached_tokens} cached)`
    }
  }

  return (
    <div className="border-b border-border last:border-0">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-muted/50 transition-colors"
      >
        <EventIcon type={event.type} />
        <span className="text-xs text-muted-foreground font-mono w-20 flex-shrink-0">{time}</span>
        <span className="text-sm truncate flex-1">{summary}</span>
      </button>
      {expanded && (
        <pre className="mx-4 mb-3 p-3 bg-muted rounded-md text-xs overflow-auto max-h-48">
          {JSON.stringify(event.payload, null, 2)}
        </pre>
      )}
    </div>
  )
}

function transcriptMessageKey(message: { role: string; content: Array<Record<string, unknown>> }) {
  return `${message.role}:${JSON.stringify(message.content)}`
}

function transcriptPartKey(part: Record<string, unknown>) {
  return JSON.stringify(part)
}

export function SessionDetailPage({ live }: { live?: boolean }) {
  const { sessionId } = useParams({ from: "/sessions/$sessionId" })
  const { data, error, isError, isLoading, refetch } = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => fetchSession(sessionId),
    refetchInterval: live ? 2000 : 5000,
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
    setResuming(true)
    setLiveEvents([])
    setResumeError(null)
    await streamResume(
      sessionId,
      resumePrompt,
      (event) => setLiveEvents((prev) => [...prev, event]),
      () => {
        setResuming(false)
        setResumePrompt("")
        refetch()
      },
      setResumeError,
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
  const canResume = ["completed", "failed", "interrupted"].includes(session.status)
  const allEvents = [...events, ...liveEvents.map((e, i) => ({ ...e, id: `live-${i}` }))]
  const filteredEvents = allEvents.filter((e) => e.type !== "model.text.delta")

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] gap-4">
      <div className="flex items-center gap-3 flex-shrink-0">
        <Link to="/sessions" className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold tracking-tight">{session.id}</h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>{session.agent_name}</span>
            <span>·</span>
            <Badge
              variant={
                session.status === "completed"
                  ? "default"
                  : session.status === "running"
                    ? "secondary"
                    : "destructive"
              }
            >
              {session.status}
            </Badge>
            <span>·</span>
            <span>{new Date(session.created_at).toLocaleString()}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-6 flex-1 min-h-0">
        <div className="col-span-3 flex flex-col min-h-0">
          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Event Timeline</CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1 min-h-0">
              <div className="h-full overflow-auto">
                {filteredEvents.map((e) => (
                  <EventRow key={e.id} event={e} />
                ))}
                <div ref={eventsEndRef} />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="col-span-2 flex flex-col min-h-0">
          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Transcript</CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1 min-h-0">
              <div className="h-full overflow-auto p-4 space-y-3">
                {transcript.map((msg) => (
                  <div
                    key={transcriptMessageKey(msg)}
                    className={`text-sm rounded-md p-3 ${
                      msg.role === "user"
                        ? "bg-primary/5 border border-primary/10"
                        : msg.role === "assistant"
                          ? "bg-muted"
                          : msg.role === "system"
                            ? "bg-muted/50 border border-border"
                            : "bg-chart-4/10 border border-chart-4/20"
                    }`}
                  >
                    <div className="text-xs font-semibold text-muted-foreground mb-1">
                      {msg.role}
                    </div>
                    {msg.content.map((part) => (
                      <div key={transcriptPartKey(part)}>
                        {part.type === "text" && (
                          <span className="whitespace-pre-wrap">
                            {String(part.text || "").slice(0, 500)}
                            {String(part.text || "").length > 500 ? "..." : ""}
                          </span>
                        )}
                        {part.type === "tool_call" && (
                          <span className="font-mono text-xs">
                            → {String(part.tool_name || "")}(
                            {JSON.stringify(part.arguments || {}).slice(0, 100)})
                          </span>
                        )}
                        {part.type === "tool_result" && (
                          <span className="font-mono text-xs">
                            ← {String(part.content || "").slice(0, 300)}
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
          <Card>
            <CardContent className="py-3">
              {resumeError && <p className="mb-3 text-sm text-destructive">{resumeError}</p>}
              <div className="flex gap-2 items-center">
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
    </div>
  )
}
