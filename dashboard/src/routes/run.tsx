import { useNavigate } from "@tanstack/react-router"
import { AlertCircle, Bot, CheckCircle, Play, Wrench } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { DataCard, Page, PageHeader } from "../components/dashboard/layout"
import { Button } from "../components/ui/button"
import { Textarea } from "../components/ui/textarea"
import { type SSEEvent, streamRun } from "../lib/api"
import { formatCount, modelUsagePayload, numericValue } from "../lib/format"

function LiveEvent({ event }: { event: SSEEvent }) {
  const icons: Record<string, React.ReactNode> = {
    "tool.call.started": <Wrench className="h-3.5 w-3.5 text-primary" />,
    "tool.call.completed": <CheckCircle className="h-3.5 w-3.5 text-chart-2" />,
    "tool.call.failed": <AlertCircle className="h-3.5 w-3.5 text-destructive" />,
    "model.started": <Bot className="h-3.5 w-3.5 text-chart-1" />,
    "model.completed": <Bot className="h-3.5 w-3.5 text-chart-1" />,
  }

  let text = event.type
  if (event.tool_name) text += ` — ${event.tool_name}`
  if (event.type === "model.completed") {
    const usage = modelUsagePayload(event.payload)
    const inputTokens = numericValue(usage.input_tokens)
    const outputTokens = numericValue(usage.output_tokens)
    if (inputTokens > 0 || outputTokens > 0) {
      text = `model.completed (${formatCount(inputTokens)} in / ${formatCount(outputTokens)} out)`
    }
  }
  if (event.type === "session.completed") text = "Session completed"
  if (event.type === "session.failed") text = `Session failed: ${event.payload.error || "unknown"}`
  if (event.type === "task.started") text = "Task started"
  if (event.type === "task.completed") text = "Task completed"

  return (
    <div className="flex items-center gap-2 py-1.5">
      {icons[event.type] || <span className="w-3.5" />}
      <span className="text-sm text-muted-foreground">{text}</span>
    </div>
  )
}

export function RunPage() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState("")
  const [running, setRunning] = useState(false)
  const [events, setEvents] = useState<SSEEvent[]>([])
  const [textOutput, setTextOutput] = useState("")
  const [error, setError] = useState<string | null>(null)
  const eventsEndRef = useRef<HTMLDivElement>(null)
  const outputEndRef = useRef<HTMLDivElement>(null)
  const visibleEventCount = events.filter((event) => event.type !== "model.text.delta").length

  useEffect(() => {
    if (visibleEventCount > 0) {
      eventsEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [visibleEventCount])

  useEffect(() => {
    if (textOutput.length > 0) {
      outputEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [textOutput.length])

  const handleRun = async () => {
    if (!prompt.trim()) return
    setRunning(true)
    setEvents([])
    setTextOutput("")
    setError(null)

    await streamRun(
      prompt,
      (event) => {
        setEvents((prev) => [...prev, event])
        if (event.type === "model.text.delta") {
          setTextOutput((prev) => prev + String(event.payload.delta || ""))
        }
      },
      () => setRunning(false),
      setError,
    )
  }

  const lastSessionId = events.at(-1)?.session_id ?? null

  return (
    <Page>
      <PageHeader title="New Run" />

      <DataCard title="Prompt" headerClassName="pb-3" contentClassName="space-y-3 p-4">
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Analyze the customer dataset in data/. Write a Python script that extracts top 10 countries by customer count..."
          disabled={running}
          rows={4}
          className="resize-none"
        />
        {error && <p className="text-sm text-destructive">{error}</p>}
        <div className="flex items-center justify-end">
          <Button onClick={handleRun} disabled={running || !prompt.trim()}>
            <Play className="h-4 w-4 mr-2" />
            {running ? "Running..." : "Run"}
          </Button>
        </div>
      </DataCard>

      {events.length > 0 && (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <DataCard
            title="Events"
            className="min-h-[22rem]"
            contentClassName="min-h-0 flex-1"
            actions={
              lastSessionId && !running ? (
                <Button
                  variant="link"
                  size="sm"
                  className="h-auto p-0 text-xs"
                  onClick={() =>
                    navigate({ to: "/sessions/$sessionId", params: { sessionId: lastSessionId } })
                  }
                >
                  View Session →
                </Button>
              ) : null
            }
          >
            <div className="h-full overflow-auto px-4 py-2">
              {events
                .filter((e) => e.type !== "model.text.delta")
                .map((e) => (
                  <LiveEvent key={e.id} event={e} />
                ))}
              <div ref={eventsEndRef} />
            </div>
          </DataCard>

          <DataCard
            title="Agent Output"
            className="min-h-[22rem]"
            contentClassName="min-h-0 flex-1"
          >
            <div className="h-full overflow-auto p-4">
              <pre className="whitespace-pre-wrap break-words font-sans text-sm">
                {textOutput || "Waiting for output..."}
              </pre>
              <div ref={outputEndRef} />
            </div>
          </DataCard>
        </div>
      )}
    </Page>
  )
}
