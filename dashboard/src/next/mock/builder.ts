/**
 * Builds runtime-shaped records for a made-up app from a short script per session:
 * the transcript, events, usage, overview and any inbox item all agree with each other.
 */
import type { ApiAgentSummary } from "@/lib/generated/server-api"
import type {
  InboxItem,
  SessionState,
  SummaryTone,
  TaskItem,
  ToolApproval,
  ToolDisplay,
  ToolSource,
  ToolWithDisplay,
  TranscriptMessage,
} from "../api/types"
import type { SeedSession } from "./model"

export type Step =
  | { user: string }
  | { think: string }
  | { tool: string; args?: Record<string, unknown>; result: string; error?: boolean }
  | { say: string }
  | { output: unknown }
  /** A tool call with no result yet: running, or waiting for approval. */
  | { pending: string; args?: Record<string, unknown> }

export type Waiting =
  | { kind: "approval"; tool: string; reason: string }
  | { kind: "question"; question: string; options?: string[] }
  | { kind: "recovery"; tool: string; detail: string }

export type SessionSpec = {
  id: string
  agent: string
  status: "running" | "completed" | "failed" | "interrupted"
  /** Minutes before the seed's now that the session started. */
  startedAgo: number
  /** Seconds it ran; omit while running or waiting. */
  seconds?: number
  title?: string
  summary?: string
  summaryTone?: SummaryTone
  reason?: string
  labels?: Record<string, string>
  metadata?: Record<string, unknown>
  parent?: string
  environment?: string
  waiting?: Waiting
  steps: Step[]
}

export type AppContext = {
  now: number
  runtime: string
  provider: string
  model: string
}

const iso = (ms: number) => new Date(ms).toISOString()
let counter = 0
const nextId = (prefix: string) => `${prefix}_${(++counter).toString(16).padStart(8, "0")}`

export function tool(
  name: string,
  description: string,
  display?: ToolDisplay,
  effect = "none",
  options: {
    approval?: ToolApproval
    source?: ToolSource
    schema?: Record<string, unknown>
  } = {},
): ToolWithDisplay {
  return {
    name,
    description,
    effect,
    input_schema: options.schema ?? { type: "object" },
    parallel_safe: effect === "none",
    workspace_mutation: false,
    display: display ?? BUILTIN_DISPLAY[name] ?? null,
    source: options.source ?? defaultToolSource(name),
    approval: options.approval ?? "never",
  }
}

/** Cayu's built-in tools come from Cayu; anything else from the agent's own code. */
export const defaultToolSource = (name: string): ToolSource =>
  BUILTIN_DISPLAY[name] ? { kind: "cayu" } : { kind: "agent" }

/** Labels the runtime's built-in tools would ship with (backend item 9). */
export const BUILTIN_DISPLAY: Record<string, ToolDisplay> = {
  read_file: { verb: "Read", noun: "file" },
  write_file: { verb: "Wrote", noun: "file" },
  edit_file: { verb: "Edited", noun: "file" },
  delete_file: { verb: "Deleted", noun: "file" },
  apply_patch: { verb: "Applied", noun: "patch", plural: "patches" },
  list_files: { verb: "Listed files" },
  search_text: { verb: "Searched" },
  workspace_text_search: { verb: "Searched" },
  run_command: { verb: "Ran", noun: "command" },
  exec_command: { verb: "Ran", noun: "command" },
  run_check: { verb: "Ran", noun: "check" },
  git: { verb: "Ran git" },
  git_changes: { verb: "Checked git changes" },
  browser_session: { verb: "Used the browser" },
  screenshot_page: { verb: "Took", noun: "screenshot" },
  web_search: { verb: "Searched the web" },
  web_fetch: { verb: "Fetched", noun: "page" },
  subagent: { verb: "Started", noun: "subagent" },
  subagent_result: { verb: "Read subagent results" },
  ask_user: { verb: "Asked", noun: "question" },
  search_tools: { verb: "Searched tools" },
  call_tool: { verb: "Called", noun: "tool" },
  list_artifacts: { verb: "Listed artifacts" },
  publish_workspace_artifact: { verb: "Published", noun: "artifact" },
  list_knowledge: { verb: "Listed knowledge" },
  read_knowledge: { verb: "Read knowledge" },
  search_knowledge: { verb: "Searched knowledge" },
  remember_knowledge: { verb: "Proposed", noun: "memory", plural: "memories" },
}

export function specialist(
  ctx: AppContext,
  name: string,
  description: string,
  tools: ToolWithDisplay[],
): ApiAgentSummary {
  return {
    name,
    model: ctx.model,
    provider_name: ctx.provider,
    has_system_prompt: true,
    metadata: { "cayu:description": description },
    provider_options: {},
    thinking: null,
    tool_count: tools.length,
    tools,
  }
}

export function buildSession(
  ctx: AppContext,
  spec: SessionSpec,
): { seed: SeedSession; inbox: InboxItem | null } {
  const started = ctx.now - spec.startedAgo * 60_000
  const ended = spec.seconds != null ? started + spec.seconds * 1000 : null
  const waitingState: Record<Waiting["kind"], SessionState> = {
    approval: "waiting_approval",
    question: "waiting_input",
    recovery: "needs_recovery",
  }
  const state: SessionState = spec.waiting ? waitingState[spec.waiting.kind] : spec.status
  const interaction = nextId("int")
  const transcript: TranscriptMessage[] = []
  // The last moment anything happened; a waiting session has waited since then.
  const last =
    ended ??
    (spec.waiting
      ? Math.min(ctx.now - 60_000, started + spec.steps.length * 40_000)
      : ctx.now - 20_000)

  // Events are drafted on a relative clock, then stretched to fit [started, last].
  type Draft = {
    at: number
    type: string
    payload: Record<string, unknown>
    tool: string | null
    turn: boolean
  }
  const drafts: Draft[] = []
  let at = 0
  const random = seeded(spec.id)
  const pause = (min: number, max: number) => {
    at += Math.round(min + random() * (max - min))
  }
  const emit = (
    type: string,
    payload: Record<string, unknown> = {},
    tool: string | null = null,
    turn = true,
  ) => drafts.push({ at, type, payload, tool, turn })

  let modelSteps = 0
  let toolCalls = 0
  let usesBrowser = false
  let firstUser: string | null = null
  let step: { model_step_id: string; model_attempt_id: string; step: number } | null = null
  let round: string | null = null
  const push = (role: string, content: TranscriptMessage["content"]) =>
    transcript.push({ index: transcript.length, interaction_id: interaction, role, content })
  const modelStep = () => {
    modelSteps++
    step = { model_step_id: nextId("mstep"), model_attempt_id: nextId("matt"), step: modelSteps }
    round = null
    const base = { ...step, model: ctx.model, provider: ctx.provider, attempt: 1, max_attempts: 3 }
    emit("model.started", base)
    pause(20, 60)
    const conversation = 1800 + transcript.length * 900
    emit("request.footprint.recorded", {
      ...step,
      model: ctx.model,
      component_tokens: { system: 1240, tools: 860, conversation, tool_results: toolCalls * 1500 },
      cache_breakpoints: 2,
      context_pressure: {
        used_tokens: 2100 + conversation + toolCalls * 1500,
        window_tokens: 400_000,
      },
    })
    emit("tool.exposure.recorded", {
      ...step,
      registered_count: 4,
      exposed_count: 4,
      ceiling_count: 64,
    })
    pause(1200, 5200)
    emit("model.completed", {
      ...base,
      status: "completed",
      completion: { finish_reason: "tool_calls" },
      usage: {
        input_tokens: 2100 + conversation + toolCalls * 1500,
        output_tokens: Math.round(120 + random() * 500),
      },
    })
    return step
  }
  const toolIds = () => {
    const s = step ?? modelStep()
    round ??= nextId("tround")
    return {
      model_step_id: s.model_step_id,
      model_attempt_id: s.model_attempt_id,
      tool_round_id: round,
    }
  }

  emit("session.started", { agent_name: spec.agent, run_epoch: 1 }, null, false)
  emit("interaction.started", { status: "active" })
  for (const item of spec.steps) {
    if ("user" in item) {
      firstUser ??= item.user
      push("user", [{ type: "text", text: item.user }])
      step = null
      round = null
      pause(30, 80)
    } else if ("think" in item) {
      modelStep()
      push("assistant", [{ type: "thinking", text: item.think }])
    } else if ("say" in item) {
      modelStep()
      push("assistant", [{ type: "text", text: item.say }])
      step = null
    } else if ("tool" in item) {
      const id = nextId("call")
      const ids = { ...toolIds(), tool_call_id: id }
      toolCalls++
      usesBrowser ||= item.tool === "browser_session"
      push("assistant", [
        {
          type: "tool_call",
          ...ids,
          tool_name: item.tool,
          arguments: item.args ?? {},
          arguments_state: "finalized",
        },
      ])
      emit("tool.call.started", { ...ids, arguments_state: "finalized", effect: "none" }, item.tool)
      const execution = nextId("exec")
      pause(10, 30)
      emit(
        "runner.exec.started",
        { ...ids, execution_id: execution, adapter: "local", command: item.tool },
        item.tool,
      )
      const ms = Math.round(150 + random() * 2400)
      pause(ms, ms)
      emit(
        "runner.exec.completed",
        {
          ...ids,
          execution_id: execution,
          adapter: "local",
          duration_ms: ms,
          exit_code: item.error ? 1 : 0,
          timed_out: false,
        },
        item.tool,
      )
      pause(10, 30)
      push("tool", [
        {
          type: "tool_result",
          ...ids,
          tool_name: item.tool,
          content: item.result,
          is_error: item.error === true,
        },
      ])
      emit(
        item.error ? "tool.call.failed" : "tool.call.completed",
        { ...ids, arguments: item.args ?? {}, arguments_exact: true, result: item.result },
        item.tool,
      )
    } else if ("output" in item) {
      const ids = toolIds()
      push("assistant", [{ type: "structured_output", name: "output", output: item.output }])
      emit("structured_output.validating", { ...ids, name: "output", attempt: 1 })
      pause(20, 60)
      emit("structured_output.validated", {
        ...ids,
        name: "output",
        valid: true,
        errors: [],
        output: item.output,
      })
    } else if ("pending" in item) {
      const id = nextId("call")
      const ids = { ...toolIds(), tool_call_id: id }
      toolCalls++
      push("assistant", [
        {
          type: "tool_call",
          ...ids,
          tool_name: item.pending,
          arguments: item.args ?? {},
          arguments_state: "finalized",
        },
      ])
      if (spec.waiting?.kind === "approval") {
        emit(
          "tool.approval.requested",
          { ...ids, arguments: item.args ?? {}, reason: spec.waiting.reason },
          item.pending,
        )
      } else {
        emit(
          "tool.call.started",
          { ...ids, arguments_state: "finalized", effect: "write" },
          item.pending,
        )
        pause(10, 30)
        emit(
          "runner.exec.started",
          { ...ids, execution_id: nextId("exec"), adapter: "local", command: item.pending },
          item.pending,
        )
      }
    }
  }
  if (spec.waiting?.kind === "question") {
    emit("user_input.requested", {
      question: spec.waiting.question,
      options: spec.waiting.options ?? [],
    })
  }
  if (spec.waiting) {
    emit("interaction.completed", {
      status: "waiting",
      pending_action_kind: spec.waiting.kind,
      model_step_count: modelSteps,
    })
  } else if (state !== "running") {
    pause(20, 60)
    emit("interaction.completed", {
      status: state === "completed" ? "completed" : state,
      model_step_count: modelSteps,
    })
  }
  if (state === "completed") emit("session.completed", {}, null, false)
  if (state === "failed") emit("session.failed", { reason: spec.reason ?? null }, null, false)
  if (state === "interrupted") emit("session.interrupted", {}, null, false)

  const stretch = at > 0 ? (last - started) / at : 1
  let seq = Math.floor(started / 1000)
  const events: SeedSession["events"] = drafts.map(
    (d) =>
      ({
        sequence: ++seq,
        id: `cayu_event_${seq}`,
        type: d.type,
        session_id: spec.id,
        interaction_id: d.turn ? interaction : null,
        agent_name: spec.agent,
        environment_name: spec.environment ?? "local",
        tool_name: d.tool,
        payload: d.payload,
        timestamp: iso(started + Math.round(d.at * stretch)),
      }) as SeedSession["events"][number],
  )

  const inputTokens = 4000 + toolCalls * 5200 + modelSteps * 1800
  const metadata: Record<string, unknown> = { ...(spec.metadata ?? {}) }
  if (spec.title) metadata["cayu:title"] = spec.title
  if (spec.summary) metadata["cayu:summary"] = spec.summary
  if (spec.summaryTone) metadata["cayu:summary_tone"] = spec.summaryTone
  const seed: SeedSession = {
    session: {
      id: spec.id,
      status:
        spec.waiting && spec.waiting.kind !== "recovery"
          ? "interrupted"
          : spec.status === "running"
            ? "active"
            : spec.status,
      agent_name: spec.agent,
      provider_name: ctx.provider,
      model: ctx.model,
      parent_session_id: spec.parent ?? null,
      causal_budget_id: spec.parent ?? spec.id,
      runtime_name: "cayu",
      runtime_version: ctx.runtime,
      runtime_build_provenance: {
        availability: "available",
        origin: "wheel_record",
        strength: "structural",
      },
      environment_name: spec.environment ?? "local",
      created_at: iso(started),
      updated_at: iso(last),
      labels: spec.labels ?? {},
    },
    metadata,
    usage: {
      session_id: spec.id,
      model_steps: modelSteps,
      tool_calls: toolCalls,
      provider_names: [ctx.provider],
      models: [ctx.model],
      usage: {
        input_tokens: String(inputTokens),
        output_tokens: String(300 + modelSteps * 220),
        reasoning_output_tokens: "0",
        total_tokens: String(inputTokens + 300 + modelSteps * 220),
        cache: {
          cached_input_tokens: String(Math.floor(inputTokens * 0.6)),
          read_tokens: String(Math.floor(inputTokens * 0.6)),
          uncached_input_tokens: String(Math.ceil(inputTokens * 0.4)),
        },
      } as SeedSession["usage"]["usage"],
    },
    transcript,
    events,
    overview: {
      session_id: spec.id,
      title: spec.title ?? (firstUser ? clip(firstUser, 140) : null),
      summary: spec.summary ?? null,
      summary_tone: spec.summaryTone ?? null,
      state,
      reason: spec.reason ?? null,
      started_at: iso(started),
      ended_at: ended ? iso(ended) : null,
      last_activity_at: iso(last),
      model_steps: modelSteps,
      tool_calls: toolCalls,
      tokens: {
        input: inputTokens,
        output: 300 + modelSteps * 220,
        cached: Math.floor(inputTokens * 0.6),
      },
      cost: {
        amount:
          Math.round((inputTokens * 0.0000025 + (300 + modelSteps * 220) * 0.00001) * 100) / 100,
        currency: "USD",
        complete: true,
      },
      uses: { browser: usesBrowser, subagents: 0, artifacts: 0 },
    },
  }
  return { seed, inbox: spec.waiting ? inboxItem(spec, spec.waiting, last) : null }
}

function inboxItem(spec: SessionSpec, w: Waiting, at: number): InboxItem {
  const base = {
    id: `inbox_${spec.id}`,
    attention_id: `attention_${spec.id}`,
    session_id: spec.id,
    agent_name: spec.agent,
    created_at: iso(at),
  }
  if (w.kind === "approval") {
    return {
      ...base,
      kind: "approval",
      title: `Approve ${w.tool}`,
      detail: w.reason,
      tool_name: w.tool,
      decisions: ["approve", "deny"],
    }
  }
  if (w.kind === "question") {
    return {
      ...base,
      kind: "question",
      title: "Question",
      question: w.question,
      options: w.options ?? [],
      decisions: ["answer"],
    }
  }
  return {
    ...base,
    kind: "recovery",
    title: "Manual recovery required",
    detail: w.detail,
    tool_name: w.tool,
    decisions: [],
  }
}

export function knowledgeItem(
  id: string,
  agent: string,
  sessionId: string,
  text: string,
  at: number,
): InboxItem {
  return {
    id: `inbox_${id}`,
    kind: "knowledge",
    title: "Remember this?",
    detail: text,
    agent_name: agent,
    session_id: sessionId,
    created_at: iso(at),
    decisions: ["keep", "drop"],
    knowledge: { entry_id: id, kind: "fact", revision: 1 },
  }
}

export type TaskSpec = {
  id: string
  title: string
  type: string
  status: string
  agent: string | null
  /** Minutes before the seed's now that it was created. */
  createdAgo: number
  /** Minutes before now it last changed; defaults to createdAgo. */
  updatedAgo?: number
  session?: string
  parent?: string
  reason?: string
  /** Minutes from now until it may start; negative for the past. */
  availableIn?: number
  /** A one-shot schedule (scheduling.py): fires at availableIn, expires a day later. */
  scheduled?: { revision?: number }
  retry?: {
    attempt: number
    max: number
    disposition: string
    predecessor?: string
    successor?: string
    /** Minutes from now until the next attempt may start. */
    nextIn?: number
  }
  graph?: string
  prerequisites?: string[]
}

/** A task as the store holds it, including the graph fields the API doesn't serialize yet. */
export function seedTask(now: number, spec: TaskSpec): TaskItem {
  const min = 60_000
  const created = now - spec.createdAgo * min
  const updated = now - (spec.updatedAgo ?? spec.createdAgo) * min
  const available = spec.availableIn === undefined ? null : now + spec.availableIn * min
  const live = spec.status === "running" || spec.status === "claimed"
  const done = ["completed", "failed", "cancelled", "dependency_skipped"].includes(spec.status)
  return {
    id: spec.id,
    type: spec.type,
    title: spec.title,
    description: null,
    status: spec.status,
    status_reason: spec.reason ?? spec.retry?.disposition ?? null,
    status_payload: null,
    session_id: spec.session ?? null,
    parent_task_id: spec.parent ?? null,
    assigned_agent_name: spec.agent,
    available_at: available === null ? null : iso(available),
    worker_id: live || (done && spec.session) ? "worker-local-1" : null,
    lease_expires_at: live ? iso(now + 90_000) : null,
    created_at: iso(created),
    updated_at: iso(updated),
    completed_at: done ? iso(updated) : null,
    retry_series: spec.retry
      ? {
          series_id: `series_${spec.id.replace(/-\d+$/, "")}`,
          attempt: spec.retry.attempt,
          attempts_remaining: spec.retry.max - spec.retry.attempt,
          disposition: spec.retry.disposition,
          predecessor_task_id: spec.retry.predecessor ?? null,
          successor_task_id: spec.retry.successor ?? null,
          next_eligible_at:
            spec.retry.nextIn === undefined ? null : iso(now + spec.retry.nextIn * min),
          started_at: iso(created),
          causal_budget_id: `budget_${spec.id.replace(/-\d+$/, "")}`,
          cumulative_tokens: String(18_400 * spec.retry.attempt),
          cumulative_estimated_cost: (0.06 * spec.retry.attempt).toFixed(2),
          tokens_remaining: null,
          estimated_cost_remaining: null,
          elapsed_deadline: null,
          policy: {
            max_attempts: spec.retry.max,
            initial_backoff_seconds: 300,
            backoff_multiplier: 2,
            max_backoff_seconds: 3600,
            max_elapsed_seconds: null,
            max_total_tokens: null,
            max_estimated_cost: null,
            cost_currency: "USD",
          },
        }
      : null,
    schedule: spec.scheduled
      ? {
          schema_version: 1,
          revision: spec.scheduled.revision ?? 1,
          admitted_at: spec.status === "pending" ? null : iso(available ?? created),
          policy: {
            expires_at: iso((available ?? created) + 24 * 60 * min),
            misfire_policy: "fire_once",
            misfire_grace_seconds: 900,
          },
        }
      : null,
    graph_id: spec.graph ?? null,
    prerequisite_task_ids: spec.prerequisites ?? [],
  }
}

/** A small deterministic random source, so a seed renders the same every time. */
export function seeded(key: string) {
  let h = 2166136261
  for (const c of key) h = Math.imul(h ^ c.charCodeAt(0), 16777619)
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507) ^ Math.imul(h ^ (h >>> 13), 3266489909)
    h ^= h >>> 16
    return (h >>> 0) / 4294967296
  }
}

const clip = (text: string, n: number) => (text.length > n ? `${text.slice(0, n - 1)}…` : text)
