/**
 * Builds a trace (a tree of timed spans) from a session's events. Every runtime event
 * carries the IDs that place it: interaction_id for a turn, model_step_id and
 * model_attempt_id for a model call, tool_round_id and tool_call_id for tool work, and
 * execution_id for a command run in the environment. Nothing here is app-specific.
 */
import type { SessionEvent } from "../api/types.ts"

export type SpanKind =
  | "session"
  | "interaction"
  | "model"
  | "round"
  | "tool"
  | "exec"
  | "wait"
  | "knowledge"
export type SpanStatus =
  | "ok"
  | "error"
  | "running"
  | "waiting"
  | "interrupted"
  | "blocked"
  | "unknown"

export type Span = {
  id: string
  kind: SpanKind
  name: string
  detail: string
  start: number
  startKnown?: boolean
  transcriptLinks?: { message_index: number; part_index: number; tool_call_id: string }[]
  end: number | null
  status: SpanStatus
  events: SessionEvent[]
  children: Span[]
  /** Set on the root when spans had to be linked by event order (see resolveLinks). */
  linkedByOrder?: boolean
}

/** Event types that are mostly noise when reading a trace or an event log. */
export const NOISY_EVENT_TYPES = new Set([
  "model.thinking.delta",
  "tool.exposure.recorded",
  "request.footprint.recorded",
  "environment.lifecycle.transition",
])

const APPROVAL_REQUESTS = new Set(["tool.approval.requested", "tool.call.approval_requested"])
const APPROVAL_RESOLUTIONS = new Set([
  "tool.approval.resolved",
  "tool.approval.recovered",
  "tool.call.approved",
  "tool.call.approval_denied",
  "tool.call.approval_expired",
])
const approvalOutcome = (e: SessionEvent): SpanStatus | null => {
  const payload = p(e)
  if (
    e.type === "tool.call.approval_denied" ||
    e.type === "tool.call.approval_expired" ||
    payload.decision === "deny" ||
    payload.decision === "denied"
  )
    return "error"
  if (e.type === "tool.approval.recovered") {
    if (payload.outcome === "completed") return "ok"
    if (payload.outcome === "failed") return "error"
  }
  return null
}

const WAIT_STARTS: Record<string, string> = {
  "tool.call.approval_requested": "Waiting for approval",
  "tool.approval.requested": "Waiting for approval",
  "user_input.requested": "Waiting for an answer",
}
const WAIT_ENDS = new Set([
  ...APPROVAL_RESOLUTIONS,
  "user_input.resolved",
  "user_input.recovered",
  "session.resumed",
])

type Payload = Record<string, unknown>

/**
 * The public event projection hides the IDs that tie events together: model_step_id and
 * execution_id come back as this marker, and tool_call_id / tool_round_id as per-event
 * aliases ("cayu_event_42:tool_call_id") that never match across events. Only
 * interaction_id survives as a stable alias.
 */
const PRIVATE = "[PRIVATE_EVENT_AUTHORITY]"
/** True for a value the server withheld: the private marker or a per-event alias. */
export const isHiddenValue = (v: unknown): boolean =>
  v === PRIVATE || (typeof v === "string" && /^cayu_event_\d+:[a-z_]+$/.test(v))

const linkable = (v: unknown): v is string =>
  typeof v === "string" && v !== "" && v !== PRIVATE && !/^cayu_event_\d+:/.test(v)

/**
 * Fill in hidden linkage from event order: a model step runs from model.started to
 * model.completed; tool calls requested after it belong to it; a tool call's completion
 * matches a unique open call of the same tool in the same round/model step and turn; approvals remain
 * pending across turns until a matching start or terminal. Ambiguous matches are
 * left separate, never treated as exact linkage. A command belongs to
 * the open call of its tool. Real IDs are kept wherever the server provides them.
 */
export function resolveLinks(events: SessionEvent[]): {
  events: SessionEvent[]
  inferred: boolean
} {
  let inferred = false
  type Turn = {
    step: string | null
    steps: number
    calls: Map<string, string[]>
    execs: Map<string, string[]>
  }
  const turns = new Map<string, Turn>()
  const callLinks = new Map<string, { step: string | null; round: string }>()
  const execCalls = new Map<string, string>()
  const pendingApprovals = new Map<
    string,
    { tool: string; approval: string | null; settled: boolean }
  >()
  const callApprovals = new Map<string, string | null>()
  const canonicalCalls = new Map<string, string>()
  const out = [...events]
    .sort((a, b) => a.sequence - b.sequence)
    .map((e) => {
      const key = e.interaction_id ?? "session"
      let turn = turns.get(key)
      if (!turn) {
        turn = { step: null, steps: 0, calls: new Map(), execs: new Map() }
        turns.set(key, turn)
      }
      const before = p(e)
      const payload: Payload = { ...before }
      const tool = toolName(e) ?? "tool"
      const queue = (m: Map<string, string[]>) => {
        // Never let an unresolved parallel round become a candidate for later work.
        // Public round/step IDs scope directly; otherwise the latest model.started
        // supplies an order-inferred scope. Keep old spans as unmatched evidence.
        const scope = linkable(before.tool_round_id)
          ? before.tool_round_id
          : linkable(before.model_step_id)
            ? before.model_step_id
            : (turn.step ?? key)
        const bucket = JSON.stringify([
          linkable(before.tool_round_id) ? "round" : "step",
          scope,
          tool,
        ])
        let q = m.get(bucket)
        if (!q) {
          q = []
          m.set(bucket, q)
        }
        return q
      }
      // Missing and withheld model IDs need the same per-turn linkage on the
      // whole model lifecycle, not only on its start. Explicit IDs always win.
      const hiddenStep =
        (e.type.startsWith("model.") || "model_step_id" in before) &&
        !linkable(before.model_step_id)
      if (e.type === "model.started") {
        turn.steps++
        turn.step = linkable(before.model_step_id)
          ? before.model_step_id
          : `inferred-step:${key}:${turn.steps}`
        payload.model_step_id = turn.step
      }
      if (hiddenStep && turn.step) payload.model_step_id = turn.step
      const round = `inferred-round:${turn.step ?? key}`

      const approval = linkable(before.approval_id) ? before.approval_id : null
      const pending = (includeSettled = false) => {
        const candidates = [...pendingApprovals].filter(([id, candidate]) => {
          if (candidate.settled && !includeSettled) return false
          if (toolName(e) && candidate.tool !== tool) return false
          if (
            linkable(before.tool_call_id) &&
            !id.startsWith("inferred-call:") &&
            id !== before.tool_call_id
          )
            return false
          return approval ? candidate.approval === approval : true
        })
        return candidates.length === 1 ? candidates[0]?.[0] : undefined
      }
      const place = (id: string) => {
        const link = callLinks.get(id)
        payload.tool_call_id = id
        if (!linkable(before.tool_round_id)) payload.tool_round_id = link?.round ?? round
        if (!linkable(before.model_step_id) && link?.step) payload.model_step_id = link.step
      }
      if (APPROVAL_REQUESTS.has(e.type)) {
        const id = linkable(before.tool_call_id)
          ? before.tool_call_id
          : `inferred-call:${e.sequence}`
        callLinks.set(id, {
          step: turn.step,
          round: linkable(before.tool_round_id) ? before.tool_round_id : round,
        })
        pendingApprovals.set(id, { tool, approval, settled: false })
        callApprovals.set(id, approval)
        place(id)
      } else if (APPROVAL_RESOLUTIONS.has(e.type)) {
        const id = pending()
        if (id) {
          place(id)
          const record = pendingApprovals.get(id)
          if (approvalOutcome(e) && record) record.settled = true
        }
      } else if (e.type === "tool.call.started") {
        const approved = pending()
        const id = linkable(before.tool_call_id)
          ? before.tool_call_id
          : (approved ?? `inferred-call:${e.sequence}`)
        if (approved) {
          pendingApprovals.delete(approved)
          if (approved !== id) {
            canonicalCalls.set(approved, id)
            const link = callLinks.get(approved)
            if (link) callLinks.set(id, link)
          }
        }
        if (!callLinks.has(id))
          callLinks.set(id, {
            step: turn.step,
            round: linkable(before.tool_round_id) ? before.tool_round_id : round,
          })
        callApprovals.set(id, approval)
        if (!queue(turn.calls).includes(id)) queue(turn.calls).push(id)
        place(id)
      } else if (e.type === "tool.call.completed" || e.type === "tool.call.failed") {
        const calls = queue(turn.calls)
        const candidates = calls.filter((id) => !approval || callApprovals.get(id) === approval)
        const id = linkable(before.tool_call_id)
          ? before.tool_call_id
          : ((candidates.length === 1
              ? candidates[0]
              : calls.length === 0 || approval
                ? pending(true)
                : undefined) ?? `inferred-call:${e.sequence}`)
        const index = calls.indexOf(id)
        if (index >= 0) calls.splice(index, 1)
        pendingApprovals.delete(id)
        place(id)
      } else if (e.type === "runner.exec.started" || e.type === "runner.exec.completed") {
        if (!linkable(before.execution_id)) {
          const id =
            e.type === "runner.exec.started"
              ? `inferred-exec:${e.sequence}`
              : (queue(turn.execs).shift() ?? `inferred-exec:${e.sequence}`)
          if (e.type === "runner.exec.started") queue(turn.execs).push(id)
          payload.execution_id = id
          if (!linkable(before.tool_call_id)) {
            const calls = queue(turn.calls)
            const call = execCalls.get(id) ?? (calls.length === 1 ? calls[0] : undefined)
            if (call) {
              execCalls.set(id, call)
              const link = callLinks.get(call)
              payload.tool_call_id = call
              payload.tool_round_id = link?.round ?? round
              if (link?.step) payload.model_step_id = link.step
            }
          }
        }
      } else if (
        (e.type.startsWith("structured_output.") || e.type.startsWith("tool.")) &&
        "tool_round_id" in before &&
        !linkable(before.tool_round_id)
      ) {
        payload.tool_round_id = round
      }
      const changed = Object.keys(payload).some((k) => payload[k] !== before[k])
      if (!changed) return e
      inferred = true
      return { ...e, payload } as SessionEvent
    })
  return {
    events: out.map((e) => {
      const id = str(p(e).tool_call_id)
      const canonical = id ? canonicalCalls.get(id) : undefined
      return canonical
        ? ({ ...e, payload: { ...p(e), tool_call_id: canonical } } as SessionEvent)
        : e
    }),
    inferred,
  }
}
const p = (e: SessionEvent) => (e.payload ?? {}) as Payload
const str = (v: unknown) => (typeof v === "string" && v ? v : null)
const time = (e: SessionEvent) => Date.parse((e as { timestamp: string }).timestamp)
const toolName = (e: SessionEvent) => (e as { tool_name?: string | null }).tool_name ?? null

function span(id: string, kind: SpanKind, name: string, start: number): Span {
  return {
    id,
    kind,
    name,
    detail: "",
    start,
    end: null,
    status: "running",
    events: [],
    children: [],
  }
}

function touch(s: Span, e: SessionEvent) {
  s.events.push(e)
  const t = time(e)
  if (t < s.start) s.start = t
}

function finish(s: Span, e: SessionEvent, status: SpanStatus) {
  touch(s, e)
  s.end = Math.max(time(e), s.start)
  s.status = status
}

/**
 * `now` closes spans that are still open (running or waiting). Pass the session's
 * terminal state so open spans in a finished session read as stopped, not running.
 */
export function buildTrace(
  sessionId: string,
  events: SessionEvent[],
  now: number,
  terminal: SpanStatus | null = null,
): Span {
  const { events: sorted, inferred } = resolveLinks(events)
  const first = sorted[0] ? time(sorted[0]) : now
  const root = span(`session:${sessionId}`, "session", "Session", first)
  const interactions = new Map<string, Span>()
  const models = new Map<string, Span>()
  const rounds = new Map<string, Span>()
  const tools = new Map<string, Span>()
  const execs = new Map<string, Span>()
  let openWait: Span | null = null
  const approvalWaits = new Map<string, Span>()

  const interactionFor = (e: SessionEvent): Span => {
    const id = e.interaction_id
    if (!id) return root
    let s = interactions.get(id)
    if (!s) {
      s = span(`interaction:${id}`, "interaction", `Turn ${interactions.size + 1}`, time(e))
      interactions.set(id, s)
      root.children.push(s)
    }
    return s
  }
  const modelFor = (e: SessionEvent, parent: Span): Span | null => {
    const id = str(p(e).model_step_id)
    if (!id || !linkable(id)) return null
    let s = models.get(id)
    if (!s) {
      const step = p(e).step
      s = span(
        `model:${id}`,
        "model",
        `Model step${typeof step === "number" ? ` ${step}` : ""}`,
        time(e),
      )
      models.set(id, s)
      parent.children.push(s)
    }
    return s
  }
  const roundFor = (e: SessionEvent, parent: Span): Span | null => {
    const id = str(p(e).tool_round_id)
    if (!id || !linkable(id)) return null
    let s = rounds.get(id)
    if (!s) {
      s = span(`round:${id}`, "round", "Tool round", time(e))
      rounds.set(id, s)
      parent.children.push(s)
    }
    return s
  }
  const toolFor = (e: SessionEvent, parent: Span): Span | null => {
    const id = str(p(e).tool_call_id)
    if (!id || !linkable(id)) return null
    let s = tools.get(id)
    if (!s) {
      s = span(`tool:${id}`, "tool", toolName(e) ?? "tool", time(e))
      tools.set(id, s)
      parent.children.push(s)
    }
    return s
  }

  for (const e of sorted) {
    const payload = p(e)
    if (e.type.startsWith("session.")) {
      touch(root, e)
      if (e.type === "session.completed") root.status = "ok"
      if (e.type === "session.failed") root.status = "error"
      if (
        e.type === "session.completed" ||
        e.type === "session.failed" ||
        e.type === "session.interrupted"
      ) {
        root.end = time(e)
        if (e.type === "session.interrupted") root.status = "error"
      }
      continue
    }
    const interaction = interactionFor(e)
    if (e.type === "interaction.started") {
      touch(interaction, e)
      continue
    }
    if (e.type === "interaction.completed") {
      finish(interaction, e, payload.status === "failed" ? "error" : "ok")
      continue
    }
    if (WAIT_STARTS[e.type]) {
      openWait = span(`wait:${e.id}`, "wait", WAIT_STARTS[e.type] ?? "Waiting", time(e))
      openWait.status = "waiting"
      openWait.detail = toolName(e) ?? ""
      openWait.events.push(e)
      interaction.children.push(openWait)
      if (APPROVAL_REQUESTS.has(e.type)) {
        const id = str(payload.tool_call_id)
        if (id) approvalWaits.set(id, openWait)
        openWait = null
      }
    } else if (
      APPROVAL_RESOLUTIONS.has(e.type) ||
      e.type === "tool.call.started" ||
      e.type === "tool.call.failed" ||
      e.type === "tool.call.completed"
    ) {
      const id = str(payload.tool_call_id)
      const waiting = id ? approvalWaits.get(id) : undefined
      if (waiting && id) {
        finish(waiting, e, approvalOutcome(e) ?? "ok")
        approvalWaits.delete(id)
      }
    } else if (WAIT_ENDS.has(e.type) && openWait) {
      finish(openWait, e, "ok")
      openWait = null
    }

    const model = modelFor(e, interaction)
    const round =
      e.type.startsWith("tool.") ||
      e.type.startsWith("runner.") ||
      e.type.startsWith("structured_output.")
        ? roundFor(e, model ?? interaction)
        : null
    const tool =
      e.type.startsWith("tool.call.") ||
      e.type.startsWith("runner.") ||
      APPROVAL_REQUESTS.has(e.type) ||
      (APPROVAL_RESOLUTIONS.has(e.type) && approvalOutcome(e) !== null)
        ? toolFor(e, round ?? model ?? interaction)
        : null

    if (e.type.startsWith("runner.exec.") && tool) {
      const id = str(payload.execution_id) ?? e.id
      let s = execs.get(id)
      if (!s) {
        s = span(`exec:${id}`, "exec", "Command", time(e))
        s.detail = str(payload.command) ?? str(payload.adapter) ?? ""
        execs.set(id, s)
        tool.children.push(s)
      }
      if (e.type === "runner.exec.completed") {
        const failed =
          payload.timed_out === true ||
          payload.cancelled === true ||
          (typeof payload.exit_code === "number" && payload.exit_code !== 0)
        finish(s, e, failed ? "error" : "ok")
      } else touch(s, e)
      touch(tool, e)
      continue
    }
    if (tool) {
      if (APPROVAL_RESOLUTIONS.has(e.type) && approvalOutcome(e))
        finish(tool, e, approvalOutcome(e) ?? "error")
      else if (APPROVAL_REQUESTS.has(e.type)) {
        touch(tool, e)
        tool.status = "waiting"
      } else if (e.type === "tool.call.started") {
        touch(tool, e)
        tool.status = "running"
      } else if (e.type === "tool.call.completed") finish(tool, e, "ok")
      else if (e.type === "tool.call.failed") finish(tool, e, "error")
      else touch(tool, e)
      if (round) {
        touch(round, e)
        if (round.end === null || time(e) > round.end) round.end = tool.end ?? round.end
      }
      if (model) touch(model, e)
      continue
    }
    if (round) touch(round, e)
    if (model) {
      if (e.type === "model.completed") {
        finish(model, e, payload.status === "failed" || payload.status === "error" ? "error" : "ok")
        model.detail = str(payload.model) ?? model.detail
      } else if (e.type === "model.failed") finish(model, e, "error")
      else {
        touch(model, e)
        if (!model.detail) model.detail = str(payload.model) ?? ""
      }
      continue
    }
    touch(interaction, e)
  }

  // Close what is still open, and give parents the extent of their children.
  const close = (s: Span): number => {
    let end = s.end ?? s.start
    for (const c of s.children) end = Math.max(end, close(c))
    if (s.end === null) {
      if (s.status === "waiting" || (s.status === "running" && !terminal)) end = Math.max(end, now)
      else if (s.status === "running" && terminal) s.status = terminal === "ok" ? "ok" : "error"
      s.end = end
    }
    if (s.kind === "round") {
      s.status = s.children.some((c) => c.status === "error")
        ? "error"
        : s.children.some((c) => c.status === "waiting" || c.status === "running")
          ? "running"
          : "ok"
    }
    if (s.kind === "round" || s.kind === "interaction" || s.kind === "session")
      s.end = Math.max(s.end, end)
    return s.end
  }
  if (root.status === "running" && terminal) root.status = terminal
  close(root)
  // A round with a single call adds nothing: show the call directly.
  const unwrap = (s: Span) => {
    s.children = s.children.flatMap((c) =>
      c.kind === "round" && c.children.length === 1 && c.children[0] ? [c.children[0]] : [c],
    )
    for (const c of s.children) unwrap(c)
  }
  unwrap(root)
  root.linkedByOrder = inferred
  for (const r of rounds.values()) {
    const names = r.children.map((c) => c.name)
    r.name = names.length === 1 ? (names[0] ?? "Tool round") : `Tool round · ${names.length} calls`
  }
  return root
}

/** Depth-first rows for the waterfall, skipping children of collapsed spans. */
export function flatten(
  root: Span,
  collapsed: Set<string>,
  depth = 0,
): { span: Span; depth: number }[] {
  const rows = [{ span: root, depth }]
  if (!collapsed.has(root.id)) {
    for (const c of root.children) rows.push(...flatten(c, collapsed, depth + 1))
  }
  return rows
}

export function findSpan(root: Span, id: string): Span | null {
  if (root.id === id) return root
  for (const c of root.children) {
    const found = findSpan(c, id)
    if (found) return found
  }
  return null
}

/** The span for a tool call, so the conversation can link straight to it. */
export const toolSpanId = (toolCallId: string) => `tool:${toolCallId}`

export function formatSpanDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`
  const m = Math.floor(ms / 60_000)
  if (m < 60) return `${m}m ${Math.round((ms % 60_000) / 1000)}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}
