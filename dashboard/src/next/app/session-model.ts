/**
 * Pure helpers that turn runtime records into what the pages show. Nothing here knows
 * about a particular app: titles, summaries, labels and structured output are generic.
 */
import type { AgentsPage, SessionState, ToolDisplay, TranscriptMessage } from "../api/types.ts"

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

export type ToolUse = {
  key: string
  messageIndex: number
  partIndex: number
  id: string
  name: string
  args: unknown
  argumentsUnavailable: boolean
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
  | { kind: "part"; key: string; part: Record<string, unknown>; user?: boolean }

type Part = { [key: string]: unknown }
const str = (v: unknown) => (typeof v === "string" ? v : "")
const redacted = (value: string) =>
  value.includes("[REDACTED_SECRET]") || value.includes("[PRIVATE_EVENT_AUTHORITY]")

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
export function buildBlocks(
  messages: TranscriptMessage[],
  structuredOutputEnabled = true,
  includeThinking = true,
): Block[] {
  // Correlation fields are public display evidence, not mutation authority. A
  // redacted or ambiguous tuple cannot attach a result to a particular call.
  const identity = (part: Part): string | null => {
    const fields = [
      part.tool_call_id,
      part.model_step_id,
      part.model_attempt_id,
      part.tool_round_id,
    ]
    if (typeof fields[0] !== "string" || !fields[0].trim()) return null
    if (fields.some((field) => typeof field !== "string" || !field.trim() || redacted(field)))
      return null
    return JSON.stringify(fields)
  }
  const positions = new Map<Part, number>()
  const calls = new Map<string, Part[]>()
  const results = new Map<string, Part[]>()
  for (const message of messages) {
    for (const part of message.content) {
      positions.set(part, positions.size)
      const collection =
        message.role === "assistant" && part.type === "tool_call"
          ? calls
          : message.role === "tool" && part.type === "tool_result"
            ? results
            : null
      const key = collection && identity(part)
      if (collection && key) collection.set(key, [...(collection.get(key) ?? []), part])
    }
  }
  const matched = new Map<Part, Part>()
  const consumed = new Set<Part>()
  for (const [key, candidates] of calls) {
    const outcomes = results.get(key)
    if (
      candidates.length === 1 &&
      outcomes?.length === 1 &&
      positions.get(candidates[0]!)! < positions.get(outcomes[0]!)!
    ) {
      matched.set(candidates[0]!, outcomes[0]!)
      consumed.add(outcomes[0]!)
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
    if (m.role === "system") continue
    const key = `m${m.index}`
    if (m.role === "user") {
      work = null
      m.content.forEach((part, index) => {
        if (part.type === "thinking" && !includeThinking) return
        const partKey = `${key}.${index}`
        if (part.type === "text") {
          blocks.push({ kind: "user", key: partKey, text: str(part.text) })
        } else {
          blocks.push({ kind: "part", key: partKey, part, user: true })
        }
      })
      continue
    }
    m.content.forEach((p, i) => {
      if (p.type === "thinking" && !includeThinking) return
      const pkey = `${key}.${i}`
      if (p.type === "structured_output" && structuredOutputEnabled) {
        work = null
        blocks.push({ kind: "output", key: pkey, value: p.output })
      } else if (m.role === "tool") {
        if (p.type === "tool_result" && !consumed.has(p)) {
          work = null
          blocks.push({ kind: "part", key: pkey, part: p })
        }
        return
      } else if (p.type === "text" && str(p.text).trim()) {
        work = null
        blocks.push({ kind: "text", key: pkey, text: str(p.text) })
      } else if (p.type === "thinking" && includeThinking && str(p.text).trim()) {
        ensureWork(pkey).thoughts.push(
          str(p.text).replaceAll("****", " · ").replaceAll("**", "").trim(),
        )
      } else if (p.type === "tool_call") {
        const id = str(p.tool_call_id)
        const r = matched.get(p)
        ensureWork(pkey).tools.push({
          key: pkey,
          messageIndex: m.index,
          partIndex: i,
          id,
          name: str(p.tool_name),
          args: p.arguments_state === "unavailable" ? null : (p.arguments ?? {}),
          argumentsUnavailable: p.arguments_state === "unavailable",
          result: r ? resultText(r) : null,
          structured: r?.structured ?? null,
          error: r?.is_error === true,
          running: !r,
        })
      } else if (p.type !== "structured_output" && p.type !== "text" && p.type !== "thinking") {
        work = null
        blocks.push({ kind: "part", key: pkey, part: p })
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

/** Current declarations for the session's specialist, never a global tool-name lookup. */
export function sessionToolDisplays(
  inventory: AgentsPage | undefined,
  agentName: string | undefined,
): ToolDisplays {
  if (!agentName || redacted(agentName)) return {}
  const agents = inventory?.agents.filter((agent) => agent.name === agentName) ?? []
  if (agents.length !== 1) return {}
  const declarations = new Map<string, ToolDisplay | undefined>()
  for (const tool of agents[0]!.tools) {
    // Public redaction can collapse distinct registered names. Neither the first
    // nor the last declaration establishes which one the transcript called.
    declarations.set(
      tool.name,
      declarations.has(tool.name) || !tool.name || redacted(tool.name)
        ? undefined
        : (tool.display ?? undefined),
    )
  }
  return Object.fromEntries(declarations)
}

/** Only declared own keys count: names such as `constructor` are valid tool names. */
export function toolDisplay(name: string, displays: ToolDisplays): ToolDisplay | undefined {
  return Object.hasOwn(displays, name) ? displays[name] : undefined
}

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
  return upperFirst(
    [...counts].map(([name, n]) => phrase(name, n, toolDisplay(name, displays))).join(", "),
  )
}

/** One step in an expanded line: "Read src/app.py", or the tool name when it has no label. */
export function stepLabel(
  tool: ToolUse,
  displays: ToolDisplays = {},
): { verb: string; target: string } {
  const display = toolDisplay(tool.name, displays)
  return {
    verb: display ? upperFirst(display.verb) : tool.name,
    target: tool.argumentsUnavailable ? "" : argPreview(tool.args),
  }
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

const viewerTimeZone = () => Intl.DateTimeFormat().resolvedOptions().timeZone
const dayKey = (ms: number, timeZone = viewerTimeZone()) =>
  new Date(ms).toLocaleDateString("en-CA", { timeZone })

export function relativeTime(ms: number, now: number, timeZone = viewerTimeZone()): string {
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

/** Authored title first; callers fetch only the first user record when labels are absent. */
export function sessionDisplayTitle(
  session: { id: string; labels: Record<string, string> },
  title: string | null | undefined,
  firstUser: { role: string; content: Record<string, unknown>[] } | null | undefined,
): string {
  if (title?.trim()) return title
  const labels = Object.entries(session.labels)
    .map(([key, value]) => `${key}=${value}`)
    .join(" · ")
  if (labels) return labels
  const text =
    firstUser?.role === "user"
      ? firstUser.content
          .filter((part) => part.type === "text")
          .map((part) => (typeof part.text === "string" ? part.text : ""))
          .join("")
          .trim()
      : ""
  return text || session.id
}
