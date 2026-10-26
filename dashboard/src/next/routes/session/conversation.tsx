import { ChevronRight, CornerDownRight, LoaderCircle } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { TranscriptPart } from "@/components/dashboard/transcript-part"
import { cn } from "@/lib/utils"
import type { SessionRow } from "../../app/queries"
import {
  type Block,
  stepLabel,
  type ToolDisplays,
  type ToolUse,
  workSummary,
} from "../../app/session-model"
import { NavLink, StateBadge, sessionPath } from "../common"
import { ValueView } from "./value-view"

const LONG_MESSAGE = 1200

export type Pending = "running" | "approval" | "recovery" | "stopped" | "unknown"
const PENDING: Record<Pending, { row: string; tone: string; block: (tool: string) => string }> = {
  unknown: {
    row: "result not loaded",
    tone: "text-muted-foreground",
    block: (tool) => `${tool} — result not loaded`,
  },
  running: { row: "running", tone: "text-info", block: (t) => `Running ${t}…` },
  approval: {
    row: "waiting for approval",
    tone: "text-warning",
    block: (t) => `Waiting for your approval to run ${t}`,
  },
  recovery: {
    row: "no result",
    tone: "text-warning",
    block: (t) => `${t} never reported a result`,
  },
  stopped: {
    row: "stopped",
    tone: "text-muted-foreground",
    block: (t) => `Stopped while running ${t}`,
  },
}

function UserMessage({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const long = text.length > LONG_MESSAGE
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] min-w-0 rounded-2xl bg-muted px-4 py-2.5 text-[15px] leading-relaxed">
        <p className="whitespace-pre-wrap break-words">
          {long && !open ? `${text.slice(0, 400).trimEnd()}…` : text}
        </p>
        {long && (
          <button
            type="button"
            className="mt-1.5 text-xs text-muted-foreground hover:text-foreground"
            onClick={() => setOpen(!open)}
          >
            {open ? "Show less" : `Show all (${Math.round(text.length / 1024)} KB)`}
          </button>
        )}
      </div>
    </div>
  )
}

/** One step inside an expanded line: "Read src/app.py", with its arguments and result on click. */
function ToolRow({
  tool,
  pending,
  displays,
  onTrace,
  position,
}: {
  tool: ToolUse
  pending: Pending
  displays: ToolDisplays
  onTrace?: (tool: ToolUse) => void
  position?: { messageIndex: number; partIndex: number }
}) {
  const [open, setOpen] = useState(false)
  const node = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (position?.messageIndex === tool.messageIndex && position.partIndex === tool.partIndex) {
      setOpen(true)
      node.current?.scrollIntoView({ block: "center" })
    }
  }, [position, tool.messageIndex, tool.partIndex])
  const { verb, target } = stepLabel(tool, displays)
  const named = verb !== tool.name
  return (
    <div ref={node} id={`transcript-tool-${tool.messageIndex}-${tool.partIndex}`}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="group flex w-full items-center gap-1.5 rounded-md py-0.5 text-left text-sm text-muted-foreground hover:text-foreground"
      >
        <span
          className={cn(
            "min-w-0 break-words [overflow-wrap:anywhere]",
            target && "max-w-[60%] shrink-0",
            !named && "font-mono text-[13px]",
          )}
        >
          {verb}
        </span>
        {target && <span className="min-w-0 truncate font-mono text-[13px]">{target}</span>}
        {tool.running && (
          <span className={cn("shrink-0 text-xs", PENDING[pending].tone)}>
            {PENDING[pending].row}
          </span>
        )}
        {tool.error && (
          <span className="shrink-0 rounded bg-destructive/10 px-1 text-xs text-destructive">
            failed
          </span>
        )}
        <ChevronRight
          className={cn(
            "size-3.5 shrink-0 opacity-0 group-hover:opacity-100",
            open && "rotate-90 opacity-100",
          )}
          aria-hidden="true"
        />
      </button>
      {open && (
        <div className="mt-1 mb-2 space-y-1.5">
          {onTrace && (
            <button
              type="button"
              onClick={() => onTrace(tool)}
              className="text-xs text-muted-foreground hover:text-foreground hover:underline"
            >
              View in trace
            </button>
          )}
          {tool.argumentsUnavailable ? (
            <p className="text-xs text-muted-foreground">
              Original tool arguments are unavailable. Read the result for recovery guidance.
            </p>
          ) : (
            <pre className="overflow-x-auto rounded-md bg-muted p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-words">
              {JSON.stringify(tool.args, null, 2)}
            </pre>
          )}
          {tool.structured != null && <ValueView value={tool.structured} label="Tool output" />}
          {tool.result != null && (
            <pre
              className={cn(
                "max-h-64 overflow-auto rounded-md p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-words",
                tool.error
                  ? "bg-destructive/10 text-destructive"
                  : "bg-muted text-muted-foreground",
              )}
            >
              {tool.result}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * A run of tool calls as one muted line, the way Claude shows "Ran 2 commands, read 8
 * files". The phrasing comes from each tool's display label, so it fits any app.
 */
function WorkBlock({
  block,
  pending,
  displays,
  onTrace,
  position,
}: {
  block: Extract<Block, { kind: "work" }>
  pending: Pending
  displays: ToolDisplays
  onTrace?: (tool: ToolUse) => void
  position?: { messageIndex: number; partIndex: number }
}) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (
      position &&
      block.tools.some(
        (tool) =>
          tool.messageIndex === position.messageIndex && tool.partIndex === position.partIndex,
      )
    )
      setOpen(true)
  }, [position, block.tools])
  const failed = block.tools.filter((t) => t.error).length
  const running = block.tools.find((t) => t.running)
  const label = running
    ? PENDING[pending].block(running.name)
    : block.tools.length
      ? workSummary(block.tools, displays)
      : "Thought it through"
  return (
    <div>
      <div className="flex max-w-full items-center gap-2">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className="group flex min-w-0 items-center gap-2 text-left text-[15px] text-muted-foreground hover:text-foreground"
        >
          {running && pending === "running" && (
            <LoaderCircle
              className="size-3.5 shrink-0 animate-spin motion-reduce:animate-none"
              aria-hidden="true"
            />
          )}
          <span className="truncate">{label}</span>
          <ChevronRight
            className={cn("size-4 shrink-0 transition-transform", open && "rotate-90")}
            aria-hidden="true"
          />
        </button>
        {failed > 0 && (
          <button
            type="button"
            title="Show the failed call in the trace"
            onClick={() => {
              const first = block.tools.find((t) => t.error)
              if (first && onTrace) onTrace(first)
              else setOpen(true)
            }}
            className="shrink-0 rounded bg-destructive/10 px-1.5 text-xs font-medium tabular-nums text-destructive hover:bg-destructive/20"
          >
            {failed} failed
          </button>
        )}
      </div>
      {open && (
        <div className="mt-2 space-y-0.5 border-l border-border pl-4">
          {block.thoughts.length > 0 && (
            <p className="pb-1 text-sm text-muted-foreground italic">
              {block.thoughts.join(" · ")}
            </p>
          )}
          {block.tools.map((t) => (
            <ToolRow
              key={t.key}
              tool={t}
              pending={pending}
              displays={displays}
              onTrace={onTrace}
              position={position}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function Conversation({
  blocks,
  pending = "unknown",
  displays = {},
  onTrace,
  position,
  children,
}: {
  blocks: Block[]
  /** Display labels for the specialist's tools. */
  displays?: ToolDisplays
  /** Opens the trace at a tool call's span. */
  onTrace?: (tool: ToolUse) => void
  position?: { messageIndex: number; partIndex: number }
  /** Use a stronger label only when the caller has evidence for this call. */
  pending?: Pending
  children?: React.ReactNode
}) {
  return (
    <div className="space-y-4">
      {blocks.map((b) => {
        if (b.kind === "user") return <UserMessage key={b.key} text={b.text} />
        if (b.kind === "text")
          return (
            <p key={b.key} className="text-[15px] leading-relaxed whitespace-pre-wrap break-words">
              {b.text}
            </p>
          )
        if (b.kind === "work")
          return (
            <WorkBlock
              key={b.key}
              block={b}
              pending={pending}
              displays={displays}
              onTrace={onTrace}
              position={position}
            />
          )
        if (b.kind === "part")
          return (
            <div
              key={b.key}
              data-transcript-part={b.key}
              data-transcript-role={b.user ? "user" : undefined}
              className={b.user ? "flex justify-end" : undefined}
            >
              <div
                className={
                  b.user ? "max-w-[85%] min-w-0 rounded-2xl bg-muted px-4 py-2.5" : undefined
                }
              >
                <TranscriptPart part={b.part} />
              </div>
            </div>
          )
        return <ValueView key={b.key} value={b.value} />
      })}
      {children}
    </div>
  )
}

/** Sessions this one started as subagents, shown where the conversation ends. */
export function SubagentRows({ rows }: { rows: SessionRow[] }) {
  if (!rows.length) return null
  return (
    <div className="space-y-1">
      {rows.map(({ session, overview, title }) => (
        <NavLink
          key={session.id}
          to={sessionPath(session.id)}
          className="flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm hover:bg-muted"
        >
          <CornerDownRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="shrink-0 text-muted-foreground">Subagent {session.agent_name}</span>
          <span className="min-w-0 flex-1 truncate">{title ?? overview?.title ?? session.id}</span>
          <StateBadge overview={overview} />
        </NavLink>
      ))}
    </div>
  )
}
