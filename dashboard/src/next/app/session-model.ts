/**
 * Pure helpers that turn runtime records into what the pages show. Nothing here knows
 * about a particular app: titles, summaries, labels and structured output are generic.
 */
import type { SessionState, ToolDisplay, TranscriptMessage } from "../api/types.ts"

export const STATE_LABEL: Record<SessionState, string> = {
  running: "Running",
  waiting_approval: "Needs your approval",
  waiting_input: "Needs your answer",
  needs_recovery: "Needs recovery",
  completed: "Completed",
  failed: "Failed",
  interrupted: "Stopped",
}

export const isWaiting = (state: SessionState | undefined) =>
  state === "waiting_approval" || state === "waiting_input" || state === "needs_recovery"

export const isActive = (state: SessionState | undefined) => state === "running" || isWaiting(state)

/** Runtime-owned session metadata marking an eval trial (cayu/sessions/_run_identity.py). */
export const EVAL_TRIAL_METADATA_KEY = "cayu:eval_trial"

/** Whether session metadata marks an eval trial; only `true` counts, as in `is_eval_trial`. */
export const isEvalTrialMetadata = (metadata: Record<string, unknown> | null | undefined) =>
  metadata?.[EVAL_TRIAL_METADATA_KEY] === true

/**
 * Whether a list row's click should navigate. Clicks with a modifier key or a button
 * other than the main one belong to the browser (new tab, new window), and clicks on a
 * link or control inside the row are handled by that element.
 */
export function shouldFollowRowClick(click: {
  button: number
  metaKey: boolean
  ctrlKey: boolean
  shiftKey: boolean
  altKey: boolean
  defaultPrevented: boolean
  /** The click landed on a link, button or form control inside the row. */
  onControl: boolean
}): boolean {
  return (
    !click.defaultPrevented &&
    click.button === 0 &&
    !click.metaKey &&
    !click.ctrlKey &&
    !click.shiftKey &&
    !click.altKey &&
    !click.onControl
  )
}

/** The structured-output tool the runtime adds; its result is the session's answer. */
export const STRUCTURED_OUTPUT_TOOL = "__cayu_submit_structured_output"

export type ToolUse = {
  id: string
  name: string
  args: unknown
  result: string | null
  structured: unknown
  error: boolean
  running: boolean
}

export type Block =
  | { kind: "user"; key: string; text: string }
  | { kind: "text"; key: string; text: string }
  | { kind: "work"; key: string; tools: ToolUse[]; thoughts: string[] }
  | { kind: "output"; key: string; value: unknown }

type Part = { [key: string]: unknown }
const str = (v: unknown) => (typeof v === "string" ? v : "")

function resultText(part: Part): string {
  const content = part.content
  if (typeof content === "string") return content
  if (Array.isArray(content)) {
    return content.map((c) => (typeof c === "object" && c ? str((c as Part).text) : "")).join("")
  }
  return content == null ? "" : JSON.stringify(content)
}

/**
 * Group a transcript into conversation blocks: user messages, assistant text, and one
 * "work" block for each run of tool calls and thinking between them.
 */
export function buildBlocks(messages: TranscriptMessage[]): Block[] {
  const results = new Map<string, Part>()
  for (const m of messages) {
    if (m.role !== "tool") continue
    for (const p of m.content) {
      if (p.type === "tool_result" && typeof p.tool_call_id === "string")
        results.set(p.tool_call_id, p)
    }
  }
  const blocks: Block[] = []
  let work: Extract<Block, { kind: "work" }> | null = null
  const ensureWork = (key: string) => {
    if (!work) {
      work = { kind: "work", key, tools: [], thoughts: [] }
      blocks.push(work)
    }
    return work
  }
  for (const m of messages) {
    if (m.role === "system" || m.role === "tool") continue
    const key = `m${m.index}`
    if (m.role === "user") {
      const text = m.content.map((p) => str(p.text)).join("")
      if (!text) continue
      work = null
      blocks.push({ kind: "user", key, text })
      continue
    }
    m.content.forEach((p, i) => {
      const pkey = `${key}.${i}`
      if (p.type === "text" && str(p.text).trim()) {
        work = null
        blocks.push({ kind: "text", key: pkey, text: str(p.text) })
      } else if (p.type === "thinking" && str(p.text).trim()) {
        ensureWork(pkey).thoughts.push(
          str(p.text).replaceAll("****", " · ").replaceAll("**", "").trim(),
        )
      } else if (p.type === "tool_call") {
        const id = str(p.tool_call_id)
        const r = results.get(id)
        if (p.tool_name === STRUCTURED_OUTPUT_TOOL) {
          const output =
            r && typeof r.structured === "object" && r.structured
              ? (r.structured as Part).output
              : undefined
          if (output !== undefined) {
            work = null
            blocks.push({ kind: "output", key: pkey, value: output })
          }
          return
        }
        ensureWork(pkey).tools.push({
          id,
          name: str(p.tool_name),
          args: p.arguments ?? {},
          result: r ? resultText(r) : null,
          structured: r?.structured ?? null,
          error: r?.is_error === true,
          running: !r,
        })
      }
    })
  }
  return blocks
}

/**
 * How a tool reads in a summary line, set by whoever defines the tool: "Read" + "file"
 * gives "read 9 files", "Searched" alone gives "searched 10 times". Tools without one
 * show their name.
 */
export type ToolDisplays = Record<string, ToolDisplay | undefined>

const lowerFirst = (s: string) => s.charAt(0).toLowerCase() + s.slice(1)
const upperFirst = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

function phrase(name: string, n: number, display: ToolDisplay | undefined): string {
  if (!display) return n > 1 ? `used ${name} ${n} times` : `used ${name}`
  const verb = lowerFirst(display.verb)
  if (display.noun)
    return `${verb} ${n === 1 ? `${/^[aeiou]/i.test(display.noun) ? "an" : "a"} ${display.noun}` : `${n} ${display.plural ?? `${display.noun}s`}`}`
  return n > 1 ? `${verb} ${n} times` : verb
}

/** "Read 9 files, searched 10 times, listed files": one phrase per tool, in first-use order. */
export function workSummary(tools: ToolUse[], displays: ToolDisplays = {}): string {
  const counts = new Map<string, number>()
  for (const t of tools) counts.set(t.name, (counts.get(t.name) ?? 0) + 1)
  return upperFirst([...counts].map(([name, n]) => phrase(name, n, displays[name])).join(", "))
}

/** One step in an expanded line: "Read src/app.py", or the tool name when it has no label. */
export function stepLabel(
  tool: ToolUse,
  displays: ToolDisplays = {},
): { verb: string; target: string } {
  const display = displays[tool.name]
  return { verb: display ? upperFirst(display.verb) : tool.name, target: argPreview(tool.args) }
}

/** A short one-line view of tool arguments: the first string value, else compact JSON. */
export function argPreview(args: unknown): string {
  if (args && typeof args === "object" && !Array.isArray(args)) {
    const first = Object.values(args).find((v) => typeof v === "string" && v.length > 0)
    if (typeof first === "string") return first
  }
  const json = JSON.stringify(args)
  return json === "{}" ? "" : (json ?? "")
}

const TZ = "America/Los_Angeles"
const dayKey = (ms: number, timeZone = TZ) => new Date(ms).toLocaleDateString("en-CA", { timeZone })

export function relativeTime(ms: number, now: number, timeZone = TZ): string {
  const s = Math.max(0, (now - ms) / 1000)
  if (s < 60) return "now"
  if (s < 3600) return `${Math.round(s / 60)}m`
  if (dayKey(ms, timeZone) === dayKey(now, timeZone)) return `${Math.round(s / 3600)}h`
  if (dayKey(ms, timeZone) === dayKey(now - 864e5, timeZone)) return "Yesterday"
  return new Date(ms).toLocaleDateString("en-US", { timeZone, month: "short", day: "numeric" })
}

export function formatDuration(seconds: number | null): string {
  if (seconds == null) return "–"
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m < 60 ? `${m}m ${s}s` : `${Math.floor(m / 60)}h ${m % 60}m`
}

export function formatTokens(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`
  return String(n)
}

/** Columns whose value is the same on every row are hidden and named once instead. */
export function constantColumns<T>(rows: T[], columns: Record<string, (row: T) => string | null>) {
  const constant: Record<string, string> = {}
  const varying: string[] = []
  for (const [name, get] of Object.entries(columns)) {
    const values = new Set(rows.map(get))
    if (rows.length > 1 && values.size === 1) {
      const [only] = values
      if (only) constant[name] = only
    } else {
      varying.push(name)
    }
  }
  return { constant, varying }
}
