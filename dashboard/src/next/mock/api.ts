/**
 * The prototype's API: the same calls real.ts makes, answered from an in-memory copy of
 * one app's seed. Writes change that copy and play out a short simulated run, so a
 * decision made on one page shows everywhere (answer a question, and the session runs on).
 */
import { ApiClientError } from "@/lib/api"
import { summarizeSessionDebugState } from "@/lib/session-debug"
import { buildSessionAggregateFilter } from "@/lib/usage-rollup"
import { validateUsageRollupSearch } from "@/lib/usage-rollup-search"
import { filterInboxItems } from "../api/paging"
import type {
  Api,
  InboxItem,
  InboxItemState,
  SessionEvent,
  SessionState,
  SessionStateRecord,
  ToolApproval,
  ToolSource,
  ToolUsage,
  ToolWithDisplay,
  TranscriptMessage,
} from "../api/types"
import { isEvalTrialMetadata } from "../app/session-model"
import { BUILTIN_DISPLAY, defaultToolSource } from "./builder"
import { createEvalStore } from "./evals"
import {
  holdTask,
  resumeTask,
  scheduleEvents,
  sessionMatches,
  taskInfo,
  taskList,
} from "./legacy/core"
import type { AppSeed, SeedSession } from "./model"
import { NotFound } from "./not-found"
import { SEEDS } from "./seeds"

const LATENCY_MS = 120
const wait = <T>(value: T, ms = LATENCY_MS) =>
  new Promise<T>((resolve) => setTimeout(() => resolve(structuredClone(value)), ms))

/**
 * A seed's tool in today's inventory shape. Snapshots taken before the Cloud inventory
 * shapes may carry `source: "built-in" | "app" | …` and `requires_approval: boolean`.
 */
function inventoryTool(
  t: ToolWithDisplay & { requires_approval?: boolean | null; source?: unknown },
): ToolWithDisplay {
  const { requires_approval, source, ...rest } = t
  const legacy: Record<string, ToolSource> = {
    "built-in": { kind: "cayu" },
    app: { kind: "agent" },
    package: { kind: "package" },
    mcp: { kind: "mcp" },
  }
  const normalized: ToolSource =
    typeof source === "string"
      ? (legacy[source] ?? { kind: "unknown" })
      : source && typeof source === "object" && "kind" in source
        ? (source as ToolSource)
        : defaultToolSource(t.name)
  const approval: ToolApproval = t.approval ?? (requires_approval ? "always" : "never")
  return {
    ...rest,
    display: t.display ?? BUILTIN_DISPLAY[t.name] ?? null,
    source: normalized,
    approval,
  }
}

export type MockApi = Api & {
  seeds: { id: string; label: string; description: string }[]
  seedId(): string
  setSeed(id: string): void
  reset(): void
  /** Whether sessions carry the app-set title and summary (backend item 1), or only today's fields. */
  setAppTitles(on: boolean): void
  /** The app's clock: the seed's "now", moving forward in real time. */
  now(): number
  /** The live records, for the fake server behind today's unchanged pages. Read only. */
  records(): AppSeed
}

export function createMockApi(initial = SEEDS[0]?.id ?? "month-end"): MockApi {
  let seedId = initial
  let data: AppSeed
  let loadedAt = 0
  let base = 0

  const load = () => {
    const seed = SEEDS.find((s) => s.id === seedId) ?? SEEDS[0]
    if (!seed) throw new Error("No prototype data.")
    data = structuredClone(seed)
    loadedAt = Date.now()
    base = Date.parse(seed.now)
  }
  load()

  const now = () => base + (Date.now() - loadedAt)
  const tasksContext = { records: () => data, now }
  /** Suites and runs per seed, so switching seeds or resetting starts them fresh. */
  const evals = createEvalStore({ records: () => data, now })
  const iso = () => new Date(now()).toISOString()
  const find = (id: string) => {
    const s = data.sessions.find((x) => x.session.id === id)
    if (!s) throw new NotFound(`Session ${id}`)
    return s
  }
  const append = (s: SeedSession, role: string, content: TranscriptMessage["content"]) => {
    s.transcript.push({ index: s.transcript.length, interaction_id: null, role, content })
  }
  /** Append a runtime event, so simulated work shows in the trace and event log too. */
  const record = (
    s: SeedSession,
    type: string,
    payload: Record<string, unknown> = {},
    tool: string | null = null,
    turn = true,
  ) => {
    const sequence = (s.events.at(-1)?.sequence ?? 0) + 1
    s.events.push({
      sequence,
      id: `cayu_event_sim_${sequence}`,
      type,
      session_id: s.session.id,
      interaction_id: turn ? `int_sim_${s.session.id}_${turns.get(s.session.id) ?? 0}` : null,
      agent_name: s.session.agent_name,
      environment_name: s.session.environment_name,
      tool_name: tool,
      payload,
      timestamp: iso(),
    } as SeedSession["events"][number])
  }
  const turns = new Map<string, number>()
  const newTurn = (s: SeedSession) => {
    turns.set(s.session.id, (turns.get(s.session.id) ?? 0) + 1)
    record(s, "interaction.started", { status: "active" })
  }
  let stepCounter = 0
  const modelStep = (s: SeedSession) => {
    const ids = {
      model_step_id: `mstep_sim_${++stepCounter}`,
      model_attempt_id: `matt_sim_${stepCounter}`,
    }
    record(s, "model.started", { ...ids, model: s.session.model })
    record(s, "model.completed", { ...ids, model: s.session.model, status: "completed" })
    s.overview.model_steps++
    return ids
  }
  const setState = (s: SeedSession, state: SessionState, summary?: string) => {
    const terminal = {
      completed: "session.completed",
      failed: "session.failed",
      interrupted: "session.interrupted",
    } as const
    if (state in terminal && s.overview.state !== state) {
      record(s, "interaction.completed", { status: state })
      record(s, terminal[state as keyof typeof terminal], {}, null, false)
    }
    s.overview.state = state
    s.overview.last_activity_at = iso()
    s.session.updated_at = iso()
    s.session.status =
      state === "running" ? "active" : state.startsWith("waiting") ? "interrupted" : state
    if (state === "completed" || state === "failed" || state === "interrupted")
      s.overview.ended_at = iso()
    else s.overview.ended_at = null
    if (summary !== undefined) {
      s.overview.summary = summary
      s.overview.summary_tone = null
    }
    // Only tasks created by this simulator follow its transitions. Seed tasks and
    // their operator holds remain owned by the task mock, not inferred on reads.
    for (const task of data.tasks) {
      if (task.session_id !== s.session.id || !taskInputs.has(task.id)) continue
      task.status =
        state === "completed"
          ? "completed"
          : state === "failed"
            ? "failed"
            : state === "interrupted"
              ? "paused"
              : "running"
      task.status_reason = state === "interrupted" ? "The session was interrupted." : null
      task.updated_at = s.session.updated_at
      task.completed_at = state === "completed" || state === "failed" ? s.overview.ended_at : null
      if (state === "completed" || state === "failed" || state === "interrupted") {
        task.worker_id = null
        task.lease_expires_at = null
      }
    }
  }
  let callCounter = 0
  const callId = () => `call_sim_${++callCounter}`
  const toolCall = (
    s: SeedSession,
    name: string,
    args: Record<string, unknown>,
    result: string,
  ) => {
    const id = callId()
    const ids = { ...modelStep(s), tool_round_id: `tround_${id}`, tool_call_id: id }
    record(s, "tool.call.started", ids, name)
    record(s, "tool.call.completed", { ...ids, arguments: args, result }, name)
    append(s, "assistant", [
      {
        type: "tool_call",
        tool_call_id: id,
        tool_name: name,
        arguments: args,
        arguments_state: "finalized",
      },
    ])
    append(s, "tool", [
      { type: "tool_result", tool_call_id: id, tool_name: name, content: result, is_error: false },
    ])
    s.overview.tool_calls++
  }
  /** Finish open tool calls (the one an approval or recovery was about). */
  const closeOpenCalls = (s: SeedSession, result: string, isError = false) => {
    const answered = new Set(
      s.transcript.flatMap((m) =>
        m.content.filter((p) => p.type === "tool_result").map((p) => p.tool_call_id),
      ),
    )
    for (const m of [...s.transcript]) {
      for (const p of m.content) {
        if (p.type === "tool_call" && !answered.has(p.tool_call_id)) {
          record(
            s,
            isError ? "tool.call.failed" : "tool.call.completed",
            {
              tool_call_id: p.tool_call_id,
              tool_round_id: p.tool_round_id,
              model_step_id: p.model_step_id,
              result,
            },
            typeof p.tool_name === "string" ? p.tool_name : null,
          )
          append(s, "tool", [
            {
              type: "tool_result",
              tool_call_id: p.tool_call_id,
              tool_name: p.tool_name,
              content: result,
              is_error: isError,
            },
          ])
        }
      }
    }
  }
  /** A short simulated continuation using the specialist's own tools. */
  const runOn = (s: SeedSession, finalText: string, withTool = true) => {
    const spec = data.specialists.agents.find((a) => a.name === s.session.agent_name)
    const toolName = spec?.tools[0]?.name
    const generation = data
    setTimeout(() => {
      if (data !== generation || s.overview.state !== "running") return
      if (toolName && withTool) toolCall(s, toolName, {}, "(simulated result)")
      s.overview.last_activity_at = iso()
    }, 900)
    setTimeout(() => {
      if (data !== generation || s.overview.state !== "running") return
      modelStep(s)
      append(s, "assistant", [{ type: "text", text: finalText }])
      setState(s, "completed")
    }, 2200)
  }

  let appTitles = true
  /**
   * What a session can be called without an app-set title: its labels, then its first
   * message, then nothing (the UI shows the ID).
   */
  const fallbackTitle = (s: SeedSession): string | null => {
    const labels = Object.entries(s.session.labels)
    if (labels.length) return labels.map(([k, v]) => `${k}=${v}`).join(" · ")
    const first = s.transcript.find((m) => m.role === "user")
    const text =
      first?.content
        .map((p) => (typeof p.text === "string" ? p.text : ""))
        .join("")
        .trim() ?? ""
    return text ? (text.length > 140 ? `${text.slice(0, 139)}…` : text) : null
  }
  /** Each prototype task's input, so Run again can replay it. */
  const taskInputs = new Map<string, { type: string; input: Record<string, unknown> }>()
  /** How each session started, for the session detail's invocation. */
  const startedBy = new Map<string, "http_run" | "task">()

  /** Start a simulated session with one specialist. */
  const launch = (
    agent: string,
    prompt: string,
    options: { title: string | null; labels: Record<string, string>; source: "http_run" | "task" },
  ) => {
    const id = `${agent}-${Math.random().toString(16).slice(2, 14)}`
    const t = iso()
    const s: SeedSession = {
      session: {
        id,
        status: "active",
        agent_name: agent,
        provider_name: data.sessions[0]?.session.provider_name ?? null,
        model: data.specialists.agents.find((a) => a.name === agent)?.model ?? null,
        parent_session_id: null,
        causal_budget_id: id,
        runtime_name: "cayu",
        runtime_version: data.app.runtime_version,
        runtime_build_provenance: {
          availability: "available",
          origin: "wheel_record",
          strength: "structural",
        },
        environment_name: data.sessions[0]?.session.environment_name ?? "local",
        created_at: t,
        updated_at: t,
        labels: options.labels,
      },
      metadata: {},
      usage: { session_id: id, model_steps: 0, tool_calls: 0 },
      transcript: [],
      events: [],
      overview: {
        session_id: id,
        title: options.title ?? (prompt.length > 140 ? `${prompt.slice(0, 139)}…` : prompt),
        summary: null,
        summary_tone: null,
        state: "running",
        reason: null,
        started_at: t,
        ended_at: null,
        last_activity_at: t,
        model_steps: 0,
        tool_calls: 0,
        tokens: { input: 0, output: 0, cached: 0 },
        cost: null,
        uses: { browser: false, subagents: 0, artifacts: 0 },
      },
    }
    startedBy.set(id, options.source)
    record(s, "session.started", { agent_name: agent }, null, false)
    newTurn(s)
    append(s, "user", [{ type: "text", text: prompt }])
    data.sessions.unshift(s)
    runOn(s, "This is where the specialist's reply streams in. The prototype only simulates it.")
    return id
  }

  const api: MockApi = {
    seeds: SEEDS.map(({ id, label, description }) => ({ id, label, description })),
    seedId: () => seedId,
    setAppTitles(on) {
      appTitles = on
    },
    setSeed(id) {
      seedId = id
      load()
    },
    reset: load,
    now,
    records: () => data,

    app: () => wait(data.app),
    specialists: () =>
      wait({
        ...data.specialists,
        agents: data.specialists.agents.map((a) => ({
          ...a,
          tools: a.tools.map((t) => inventoryTool(t)),
        })),
      }),
    async sessions(query = {}, signal) {
      signal?.throwIfAborted()
      let rows = data.sessions
      if (query.include_eval === false)
        rows = rows.filter((row) => !isEvalTrialMetadata(row.metadata))
      if (query.include_forks === false) rows = rows.filter((row) => !row.overview.forked)
      if (query.overview_state?.length)
        rows = rows.filter((row) => query.overview_state?.includes(row.overview.state))
      const filter = buildSessionAggregateFilter(validateUsageRollupSearch(query))
      rows = rows.filter((row) => sessionMatches(row, filter))
      if (query.debug_state)
        rows = rows.filter((row) => {
          const counts: Record<string, number> = {}
          for (const event of row.events) counts[event.type] = (counts[event.type] ?? 0) + 1
          const debug = summarizeSessionDebugState({
            status: row.session.status,
            latestEvent: row.events.at(-1),
            terminalEvent: row.events.findLast((event) =>
              ["session.completed", "session.failed", "session.interrupted"].includes(event.type),
            ),
            countsByType: counts,
          })
          return query.debug_state === "needs_attention"
            ? debug !== null
            : debug?.kind ===
                (query.debug_state === "session_failure" ? "failure" : query.debug_state)
        })
      if (query.status)
        rows = rows.filter(
          (row) =>
            (row.session.status === "active" ? "running" : row.session.status) === query.status,
        )
      if (query.agent_name) rows = rows.filter((s) => s.session.agent_name === query.agent_name)
      if (query.parent_session_id)
        rows = rows.filter((s) => s.session.parent_session_id === query.parent_session_id)
      if (query.q) {
        const q = query.q.toLowerCase()
        rows = rows.filter((s) =>
          [
            s.session.id,
            appTitles ? (s.overview.title ?? "") : "",
            ...Object.entries(s.session.labels).map(([k, v]) => `${k}=${v}`),
          ]
            .join(" ")
            .toLowerCase()
            .includes(q),
        )
      }
      const order = query.order_by ?? "updated_at_desc"
      const field = order.startsWith("created_at") ? "created_at" : "updated_at"
      const direction = order.endsWith("_asc") ? 1 : -1
      rows = [...rows].sort((a, b) => direction * a.session[field].localeCompare(b.session[field]))
      const limit = query.limit ?? 100
      const offset = query.cursor?.startsWith("mock-session:") ? Number(query.cursor.slice(13)) : 0
      const page = rows.slice(offset, offset + limit)
      const result = await wait({
        sessions: page.map((row) => row.session),
        usage: page.map((row) => row.usage),
        overviews: page.map((row) => ({
          ...row.overview,
          run_id: row.session.id,
          reason: row.overview.reason ?? "",
          title: appTitles ? row.overview.title : null,
          summary: appTitles ? row.overview.summary : null,
          summary_tone: appTitles ? row.overview.summary_tone : null,
          model_steps: String(row.overview.model_steps),
          tool_calls: String(row.overview.tool_calls),
          tokens: {
            input: String(row.overview.tokens.input),
            output: String(row.overview.tokens.output),
            cached: String(row.overview.tokens.cached),
          },
          cost: row.overview.cost
            ? { ...row.overview.cost, amount: String(row.overview.cost.amount) }
            : null,
          uses: {
            ...row.overview.uses,
            subagents: String(row.overview.uses.subagents),
            artifacts: String(row.overview.uses.artifacts),
          },
          eval: isEvalTrialMetadata(row.metadata),
          forked: row.overview.forked === true,
        })),
        next_cursor: offset + limit < rows.length ? `mock-session:${offset + limit}` : null,
        total_count: null,
      })
      signal?.throwIfAborted()
      return result
    },
    async session(id) {
      const s = find(id)
      return wait({
        ...s.session,
        metadata: s.metadata,
        invocation: {
          schema_version: 1,
          origin: { trust: "unattributed", subject: null, tenant: null },
          root_invocation_id: "00000000-0000-4000-8000-000000000000",
          root_session_id: s.session.parent_session_id ?? s.session.id,
          source: s.session.parent_session_id
            ? "subagent"
            : (startedBy.get(s.session.id) ?? "http_run"),
        },
      })
    },
    async overviews(ids) {
      const wanted = new Set(ids)
      return wait(
        data.sessions
          .filter((s) => wanted.has(s.session.id))
          .map((s) => ({
            ...(appTitles
              ? s.overview
              : { ...s.overview, title: fallbackTitle(s), summary: null, summary_tone: null }),
            eval_trial: isEvalTrialMetadata(s.metadata),
          })),
      )
    },
    async transcript(id, query = {}) {
      const s = find(id)
      const offset = query.offset ?? 0
      const limit = query.limit ?? 500
      const selected = query.role
        ? s.transcript.filter((message) => message.role === query.role)
        : s.transcript
      const messages = selected.slice(offset, offset + limit)
      return wait({
        session_id: id,
        messages,
        offset,
        next_offset: offset + messages.length,
        has_more: offset + messages.length < selected.length,
        total_messages: selected.length,
      })
    },
    /** Pages like the server: at most 1,000 per page, either direction, and `wait_seconds`. */
    async events(id, query = {}) {
      const s = find(id)
      const limit = query.limit ?? 100
      if (limit < 1 || limit > 1000)
        throw new ApiClientError("limit must be between 1 and 1000.", 422)
      const descending = query.order_by === "sequence_desc"
      const generation = data
      const select = () => {
        let rows: SessionEvent[] = s.events
        if (query.event_type) rows = rows.filter((e) => e.type === query.event_type)
        if (query.exclude_event_type) rows = rows.filter((e) => e.type !== query.exclude_event_type)
        if (query.tool_name) rows = rows.filter((e) => e.tool_name === query.tool_name)
        if (query.interaction_id)
          rows = rows.filter((e) => e.interaction_id === query.interaction_id)
        if (query.after_sequence != null)
          rows = rows.filter((e) => e.sequence > (query.after_sequence ?? 0))
        if (query.before_sequence != null)
          rows = rows.filter((e) => e.sequence < (query.before_sequence ?? 0))
        return descending ? [...rows].reverse() : rows
      }
      let rows = select()
      // Long-poll: wait for a new event while the session runs, as the server does. A
      // session waiting on a person isn't running, so its read returns right away.
      if (query.wait_seconds && !descending) {
        const deadline = Date.now() + Math.min(query.wait_seconds, 25) * 1000
        while (
          !rows.length &&
          Date.now() < deadline &&
          data === generation &&
          s.overview.state === "running"
        ) {
          await new Promise((resolve) => setTimeout(resolve, 200))
          rows = select()
        }
      }
      const page = rows.slice(0, limit)
      const cursor = descending ? query.before_sequence : query.after_sequence
      return wait({
        session_id: id,
        events: page,
        has_more: rows.length > limit,
        order_by: descending ? ("sequence_desc" as const) : ("sequence_asc" as const),
        scan_through_sequence: descending
          ? null
          : (page.at(-1)?.sequence ?? query.after_sequence ?? null),
        next_sequence: page.at(-1)?.sequence ?? cursor ?? null,
      })
    },
    // Tasks come from the same records as today's Tasks page (legacy/core.ts), so both agree.
    tasks: () => wait(taskList(tasksContext)),
    task: async (id) => wait(taskInfo(tasksContext, id)),
    taskScheduleEvents: async (id) => wait(scheduleEvents(tasksContext, id)),
    holdTask: async (id, hold, reason) =>
      wait(
        holdTask(
          tasksContext,
          id,
          hold === "pause" ? "paused" : hold === "block" ? "blocked" : "needs_attention",
          reason ? { reason } : {},
        ),
      ),
    resumeTask: async (id) => wait(resumeTask(tasksContext, id)),
    async inboxCounts(query = {}) {
      const items = filterInboxItems(data.inbox, query)
      const counts = { approval: 0, question: 0, recovery: 0, knowledge: 0 }
      for (const item of items) counts[item.kind]++
      return wait({ counts, total_count: items.length, complete: true, unavailable_reason: null })
    },
    async overviewCounts(query) {
      const rows = data.sessions.filter(
        (row) => !isEvalTrialMetadata(row.metadata) && !row.overview.forked,
      )
      const day = rows.filter(
        (row) => row.overview.started_at >= query.since && row.overview.started_at < query.until,
      )
      const counts: Record<string, string> = {}
      for (const row of day)
        counts[row.overview.state] = String(Number(counts[row.overview.state] ?? 0) + 1)
      return wait({
        ...query,
        tokens: {
          input: String(day.reduce((total, row) => total + row.overview.tokens.input, 0)),
          output: String(day.reduce((total, row) => total + row.overview.tokens.output, 0)),
          cached: String(day.reduce((total, row) => total + row.overview.tokens.cached, 0)),
        },
        tokens_complete: true,
        cost: null,
        counts_by_state: counts,
        total_count: String(day.length),
        failed: counts.failed ?? "0",
        running_now: String(rows.filter((row) => row.overview.state === "running").length),
      })
    },
    async inbox(query = {}) {
      const items = filterInboxItems(data.inbox, query)
      // The prototype returns all its synthetic items in one page.
      return wait({
        items: items.sort((a, b) => b.created_at.localeCompare(a.created_at)),
        next_cursor: null,
        has_more: false,
        issues: [],
      })
    },
    async settle(itemId, body) {
      const item = data.inbox.find((i) => i.id === itemId)
      if (!item) throw new NotFound(`Inbox item ${itemId}`)
      // As the server: recovery is settled on the session page, with evidence (409).
      if (item.kind === "recovery")
        throw new ApiClientError(
          "Recovery items are settled in the Developer dashboard, with the evidence of what happened.",
          409,
        )
      data.inbox = data.inbox.filter((i) => i.id !== itemId)
      const state: InboxItemState = {
        id: item.id,
        kind: item.kind,
        status: "settled",
        session_id: item.session_id,
        agent_name: item.agent_name,
        item,
        settlement: {
          decision: body.decision,
          decided_at: iso(),
          decided_by: { subject: "you" },
        } as InboxItemState["settlement"],
      }
      const s = item.session_id
        ? data.sessions.find((x) => x.session.id === item.session_id)
        : undefined
      if (s && item.kind !== "knowledge") settleSession(s, item, body.decision, body.answer ?? null)
      if (item.kind === "knowledge" && body.decision === "keep") {
        data.knowledge = [
          {
            id: item.knowledge?.entry_id ?? item.id,
            text: item.detail ?? item.title,
            kind: item.knowledge?.kind ?? "fact",
            agent_name: item.agent_name ?? null,
            source_session_id: item.session_id ?? null,
            kept_by: "you",
            kept_at: iso(),
            lookups: 0,
            last_used_at: null,
          },
          ...(data.knowledge ?? []),
        ]
      }
      return wait(state)
    },
    async start({ agent, prompt }) {
      return wait({
        session_id: launch(agent, prompt, { title: null, labels: {}, source: "http_run" }),
      })
    },
    taskTypes: () => wait(data.taskTypes ?? []),
    async createTask({ type, input }) {
      const taskType = (data.taskTypes ?? []).find((t) => t.name === type)
      if (!taskType) throw new NotFound(`Task type ${type}`)
      const taskId = `task-${Math.random().toString(16).slice(2, 12)}`
      const agent = taskType.agent_name ?? data.specialists.agents[0]?.name ?? "agent"
      const title = `${type}: ${Object.values(input)
        .flatMap((v) => (Array.isArray(v) ? v : [v]))
        .filter((v) => typeof v === "string" || typeof v === "number")
        .join(" ")}`
      const sessionId = launch(agent, `${type}\n${JSON.stringify(input, null, 2)}`, {
        title,
        labels: Object.fromEntries(
          Object.entries(input)
            .filter(([, v]) => typeof v === "string" || typeof v === "number")
            .map(([k, v]) => [k, String(v)]),
        ),
        source: "task",
      })
      taskInputs.set(taskId, { type, input })
      data.taskInputs = { ...data.taskInputs, [taskId]: input }
      data.tasks.unshift({
        id: taskId,
        type,
        title,
        description: null,
        status: "running",
        status_reason: null,
        status_payload: null,
        session_id: sessionId,
        graph_id: null,
        prerequisite_task_ids: [],
        parent_task_id: null,
        assigned_agent_name: agent,
        available_at: null,
        worker_id: "worker-1",
        lease_expires_at: null,
        created_at: iso(),
        updated_at: iso(),
        completed_at: null,
        retry_series: null,
        schedule: null,
      })
      return wait({ task_id: taskId, session_id: sessionId })
    },
    async runAgain(sessionId) {
      const task = data.tasks.find((t) => t.session_id === sessionId && taskInputs.has(t.id))
      if (task) {
        const original = taskInputs.get(task.id)
        if (original) {
          const created = await api.createTask(original)
          if (created.session_id) return { session_id: created.session_id }
        }
      }
      const s = find(sessionId)
      const first = s.transcript.find((m) => m.role === "user")
      const prompt =
        first?.content.map((p) => (typeof p.text === "string" ? p.text : "")).join("") ?? ""
      return wait({
        session_id: launch(s.session.agent_name, prompt, {
          title: s.overview.title,
          labels: { ...s.session.labels },
          source: "http_run",
        }),
      })
    },
    async state(id) {
      const s = find(id)
      return wait({
        session_id: id,
        status:
          s.overview.state === "running"
            ? "running"
            : s.overview.state === "failed"
              ? "failed"
              : s.overview.state === "completed"
                ? "completed"
                : "interrupted",
        last_activity_at: s.overview.last_activity_at,
        updated_at: s.session.updated_at,
        interruption_cascade: "none",
        execution:
          s.overview.state === "running"
            ? {
                state: "running",
                run_epoch: 1,
                owner_kind:
                  startedBy.get(s.session.id) === "task" ? "task_worker" : "server_stream",
                local_owner: true,
              }
            : { state: "idle", run_epoch: 1 },
        execution_snapshots: [],
        provider_operation: { status: "none" },
      } as unknown as SessionStateRecord)
    },
    async reply(id, prompt) {
      const s = find(id)
      record(s, "session.resumed", {}, null, false)
      newTurn(s)
      append(s, "user", [{ type: "text", text: prompt }])
      setState(s, "running")
      runOn(s, "Reply simulated by the prototype. In the dashboard this streams from the session.")
      return wait(undefined)
    },
    async toolUsage() {
      const usage = new Map<string, ToolUsage>()
      const get = (tool: string) => {
        let u = usage.get(tool)
        if (!u) {
          u = {
            tool,
            calls: 0,
            failed: 0,
            approvals: { approved: 0, denied: 0, waiting: 0 },
            last_used_at: null,
            recent: [],
          }
          usage.set(tool, u)
        }
        return u
      }
      for (const s of data.sessions) {
        for (const e of s.events) {
          const tool = e.tool_name
          if (!tool) continue
          const u = get(tool)
          const at = (e as { timestamp: string }).timestamp
          if (e.type === "tool.call.started") u.calls++
          if (e.type === "tool.approval.requested") {
            u.calls++
            u.approvals.waiting++
            u.recent.push({ session_id: s.session.id, at, status: "waiting" })
          }
          if (e.type === "tool.approval.resolved") {
            u.approvals.waiting = Math.max(0, u.approvals.waiting - 1)
            if ((e.payload as { decision?: string }).decision === "approve") u.approvals.approved++
            else u.approvals.denied++
          }
          if (e.type === "tool.call.completed" || e.type === "tool.call.failed") {
            if (e.type === "tool.call.failed") u.failed++
            u.recent.push({
              session_id: s.session.id,
              at,
              status: e.type === "tool.call.failed" ? "error" : "ok",
            })
            if (!u.last_used_at || at > u.last_used_at) u.last_used_at = at
          }
        }
      }
      for (const u of usage.values())
        u.recent = u.recent.sort((a, b) => b.at.localeCompare(a.at)).slice(0, 8)
      return wait([...usage.values()])
    },
    knowledge: () => wait(data.knowledge ?? []),
    async removeKnowledge(id) {
      data.knowledge = (data.knowledge ?? []).filter((k) => k.id !== id)
      return wait(undefined)
    },
    async recover(sessionId, body) {
      const s = find(sessionId)
      const item = data.inbox.find(
        (i) =>
          i.kind === "recovery" &&
          i.session_id === sessionId &&
          (!body.attention_id || i.attention_id === body.attention_id),
      )
      if (!item || s.overview.state !== "needs_recovery")
        throw new ApiClientError("This recovery is no longer pending. Refresh the session.", 409)
      if (item.question && !body.answer)
        throw new ApiClientError("Recovering a question needs the person's answer.", 422)
      if (!body.message.trim()) throw new ApiClientError("message must not be blank.", 422)
      data.inbox = data.inbox.filter((i) => i.id !== item.id)
      const failed = body.outcome === "failed"
      closeOpenCalls(s, item.question ? (body.answer ?? body.message) : body.message, failed)
      setState(s, "running")
      runOn(
        s,
        failed
          ? `Noted that ${item.tool_name ?? "the call"} failed (${body.message}). Continuing without it.`
          : `Recovered ${item.tool_name ?? "the call"} with the verified result and continued.`,
        false,
      )
      return wait(undefined)
    },
    async interrupt(id, reason) {
      const s = find(id)
      closeOpenCalls(s, reason ? `Interrupted: ${reason}` : "Interrupted.", true)
      setState(s, "interrupted")
      // Stopping a run interrupts its active background children too.
      for (const child of data.sessions)
        if (child.session.parent_session_id === id && child.overview.state === "running") {
          closeOpenCalls(child, "Interrupted with its parent.", true)
          setState(child, "interrupted")
        }
      const last = s.events.findLast((e) => e.type === "session.interrupted")
      if (last && reason) last.payload = { ...last.payload, reason }
      return wait(undefined)
    },
    evalSuites: async () => wait(evals.evalSuites()),
    evalSuite: async (id) => wait(evals.evalSuite(id)),
    evalRun: async (id, compareWith) => wait(evals.evalRun(id, compareWith)),
    runEvalSuite: async (suiteId) => wait(evals.runEvalSuite(suiteId)),
    approveBaseline: async (runId) => wait(evals.approveBaseline(runId)),
    createEvalSuite: async (body) => wait(evals.createEvalSuite(body)),
    addToEvals: async (body) => wait(evals.addToEvals(body)),
    evalSettings: async () => wait(evals.evalSettings()),
  }

  function settleSession(s: SeedSession, item: InboxItem, decision: string, answer: string | null) {
    const resolved = {
      approval: "tool.approval.resolved",
      question: "user_input.resolved",
    } as const
    if (item.kind in resolved)
      record(s, resolved[item.kind as keyof typeof resolved], { decision }, item.tool_name ?? null)
    if (item.kind === "approval") {
      closeOpenCalls(
        s,
        decision === "approve" ? "Done." : "Denied by a person.",
        decision !== "approve",
      )
      setState(s, "running")
      runOn(
        s,
        decision === "approve"
          ? `Ran ${item.tool_name} after your approval.`
          : `Okay, I didn't run ${item.tool_name}.`,
        false,
      )
    } else if (item.kind === "question") {
      append(s, "user", [{ type: "text", text: answer ?? decision }])
      setState(s, "running")
      runOn(s, "Thanks, continuing with that.")
    }
  }

  return api
}
