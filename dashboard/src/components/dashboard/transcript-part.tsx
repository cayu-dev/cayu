import { useState } from "react"
import { hostedSearchSourceRows } from "@/lib/hosted-search-sources"
import { Badge } from "../ui/badge"
import { Button } from "../ui/button"
import { PayloadViewer } from "./layout"

export function TranscriptPart({ part }: { part: Record<string, unknown> }) {
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
    if (part.arguments_state === "unavailable") {
      return (
        <div className="min-w-0">
          <span className="block rounded bg-background/80 px-2 py-1 font-mono text-xs">
            call {toolName} — arguments unavailable
          </span>
          <p className="mt-2 text-xs text-muted-foreground">
            The submitted arguments were suppressed. See the tool result for recovery guidance.
          </p>
        </div>
      )
    }
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

  if (type === "citation") {
    const url = typeof part.url === "string" ? part.url : ""
    const title = typeof part.title === "string" ? part.title : url
    return (
      <div className="min-w-0 rounded border border-chart-3/25 bg-chart-3/5 px-3 py-2">
        <div className="mb-1 flex items-center gap-2">
          <Badge variant="outline">Web citation</Badge>
          <span className="text-xs text-muted-foreground">Untrusted external evidence</span>
        </div>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer nofollow"
          className="break-all text-sm text-primary underline underline-offset-2"
        >
          {title || url}
        </a>
      </div>
    )
  }

  if (type === "hosted_tool_call") {
    const status = typeof part.status === "string" ? part.status : "unknown"
    const action =
      part.action && typeof part.action === "object"
        ? (part.action as Record<string, unknown>)
        : null
    const sources = Array.isArray(action?.sources)
      ? action.sources.filter(
          (source): source is Record<string, unknown> =>
            source !== null && typeof source === "object",
        )
      : []
    return (
      <div className="min-w-0 rounded border border-border bg-background/60 px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">OpenAI hosted web search</Badge>
          <Badge variant="outline">{status.replaceAll("_", " ")}</Badge>
          <span className="text-xs text-muted-foreground">{sources.length} sources</span>
        </div>
        {sources.length > 0 && (
          <ul className="mt-2 space-y-1 text-sm">
            {hostedSearchSourceRows(sources).map((source) => (
              <li key={source.key}>
                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer nofollow"
                    className="break-all text-primary underline underline-offset-2"
                  >
                    {source.label}
                  </a>
                ) : (
                  <span className="break-all">{source.label}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  return <PayloadViewer value={part} maxHeight="max-h-48" />
}
