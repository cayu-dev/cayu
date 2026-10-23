/**
 * The fake server behind today's Usage, Tasks, Environments, Artifacts and System pages.
 * Everything is derived from the current seed's records: environments from the sessions'
 * environment names, usage and cost from each session's usage and events, a task per root
 * session, and a few files the agent produced. Task holds and resumes live in memory.
 */
import type {
  ArtifactRead,
  ArtifactSummary,
  ArtifactsPage,
  EnvironmentSummary,
  EnvironmentsPage,
  OperationalSnapshot,
  SessionsSummary,
  SystemDiagnostics,
  Task,
  UsageRollup,
} from "@/lib/api"
import type {
  AggregateAccuracy,
  AggregateCostSummary,
  AggregateUsageMetrics,
  ApiEventRecord,
  ApiSessionSummaryItem,
  LabelSelectorRequirement,
  SessionAggregateFilter,
  SessionCostSummary,
  SessionStatus,
  SessionStatusCounts,
  SessionUsageSummary,
  TaskAggregateFilter,
  TaskStatusCounts,
  UsageAggregateBreakdown,
  UsageAggregateTotals,
  UsageBreakdownItem,
  UsageCostRollup,
  UsageSessionAggregateBreakdown,
  UsageSessionCostBreakdown,
  UsageSessionCostSummary,
} from "@/lib/generated/server-api"
import type { TaskInfo, TaskItem, TaskScheduleEvent } from "../../api/types"
import type { AppSeed, SeedSession } from "../model"
import { CONTRACT } from "./contract"
import { HttpError, type LegacyContext, type LegacyRequest, type LegacyRoute } from "./server"

const iso = (ms: number) => new Date(ms).toISOString()
const EXACT: AggregateAccuracy = { kind: "exact", reason: null, limit: null }
/** The seed's costs stand in for a price book, so Usage shows estimates. */
const PRICE_BOOK = "prototype-seed-costs"

// ---------------------------------------------------------------------------------------
// Small readers for request bodies and query strings, which arrive untyped.

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value))
    : {}
}

const text = (value: unknown) => (typeof value === "string" && value !== "" ? value : undefined)

function textRecord(value: unknown): Record<string, string> {
  return Object.fromEntries(
    Object.entries(record(value)).filter((e): e is [string, string] => typeof e[1] === "string"),
  )
}

function integer(query: URLSearchParams, key: string, fallback: number): number {
  const value = Number(query.get(key))
  return query.has(key) && Number.isInteger(value) && value >= 0 ? value : fallback
}

/** A deterministic hex string, so derived IDs read the same on every load. */
function hex(key: string, length: number): string {
  let h = 2166136261
  let out = ""
  while (out.length < length) {
    for (const c of `${key}:${out.length}`) h = Math.imul(h ^ c.charCodeAt(0), 16777619)
    out += (h >>> 0).toString(16).padStart(8, "0")
  }
  return out.slice(0, length)
}

const slug = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48) || "output"

// ---------------------------------------------------------------------------------------
// Sessions: status, filters and what the transcript says.

const SESSION_STATUSES: SessionStatus[] = [
  "pending",
  "running",
  "interrupting",
  "completed",
  "failed",
  "interrupted",
]

/** The runtime's status enum; the prototype calls a running session "active". */
function sessionStatus(s: SeedSession): SessionStatus {
  return SESSION_STATUSES.find((status) => status === s.session.status) ?? "running"
}

const LABEL_OPERATORS: LabelSelectorRequirement["operator"][] = [
  "in",
  "not_in",
  "exists",
  "not_exists",
]

function selectorFrom(value: unknown): LabelSelectorRequirement[] {
  const item = record(value)
  const key = text(item.key)
  const operator = LABEL_OPERATORS.find((op) => op === item.operator)
  if (!key || !operator) return []
  const values = Array.isArray(item.values)
    ? item.values.filter((v): v is string => typeof v === "string")
    : []
  return [{ key, operator, values }]
}

function filterFrom(value: unknown): SessionAggregateFilter {
  const body = record(value)
  return {
    agent_name: text(body.agent_name),
    provider_name: text(body.provider_name),
    model: text(body.model),
    environment_name: text(body.environment_name),
    parent_session_id: text(body.parent_session_id),
    causal_budget_id: text(body.causal_budget_id),
    labels: textRecord(body.labels),
    label_selectors: Array.isArray(body.label_selectors)
      ? body.label_selectors.flatMap(selectorFrom)
      : [],
  }
}

/** The same filter from `/sessions/summary`'s query; only exact `key=value` labels. */
function filterFromQuery(query: URLSearchParams): SessionAggregateFilter {
  const labels = query.getAll("label").flatMap((expression) => {
    const at = expression.indexOf("=")
    return at > 0 ? [[expression.slice(0, at), expression.slice(at + 1)]] : []
  })
  return filterFrom({
    ...Object.fromEntries(query.entries()),
    labels: Object.fromEntries(labels),
  })
}

function labelMatches(labels: Record<string, string>, selector: LabelSelectorRequirement) {
  const value = labels[selector.key]
  const values = selector.values ?? []
  if (selector.operator === "exists") return value !== undefined
  if (selector.operator === "not_exists") return value === undefined
  if (selector.operator === "in") return value !== undefined && values.includes(value)
  return value === undefined || !values.includes(value)
}

function sessionMatches(s: SeedSession, filter: SessionAggregateFilter): boolean {
  const session = s.session
  const scalar: [string | null | undefined, string | null][] = [
    [filter.agent_name, session.agent_name],
    [filter.provider_name, session.provider_name],
    [filter.model, session.model],
    [filter.environment_name, session.environment_name],
    [filter.parent_session_id, session.parent_session_id],
    [filter.causal_budget_id, session.causal_budget_id],
  ]
  return (
    scalar.every(([want, got]) => want == null || want === got) &&
    Object.entries(filter.labels ?? {}).every(([key, value]) => session.labels[key] === value) &&
    (filter.label_selectors ?? []).every((selector) => labelMatches(session.labels, selector))
  )
}

function partsOf(s: SeedSession, role: string) {
  return s.transcript.filter((m) => m.role === role).flatMap((m) => m.content)
}

function textsOf(s: SeedSession, role: string): string[] {
  return partsOf(s, role).flatMap((p) =>
    p.type === "text" && typeof p.text === "string" ? [p.text] : [],
  )
}

const firstRequest = (s: SeedSession) => textsOf(s, "user")[0] ?? null
const lastReply = (s: SeedSession) => textsOf(s, "assistant").at(-1) ?? null
const titleOf = (s: SeedSession) => s.overview.title ?? firstRequest(s) ?? s.session.id

/** What the session submitted as structured output, if it did. */
function structuredOutput(s: SeedSession): unknown {
  for (const part of partsOf(s, "tool").reverse()) {
    const structured = record(part.structured)
    if ("output" in structured) return structured.output
  }
  return undefined
}

function toolCalls(s: SeedSession) {
  const results = new Map(
    partsOf(s, "tool").map((p) => [
      String(p.tool_call_id),
      { result: p.content, error: p.is_error },
    ]),
  )
  return partsOf(s, "assistant")
    .filter((p) => p.type === "tool_call" && !String(p.tool_name).startsWith("__cayu"))
    .map((p) => ({
      tool: String(p.tool_name),
      args: record(p.arguments),
      ...(results.get(String(p.tool_call_id)) ?? { result: null, error: false }),
    }))
}

// ---------------------------------------------------------------------------------------
// Tokens, as bigints so sums stay exact like the runtime's string counts.

type Tokens = {
  input: bigint
  output: bigint
  reasoning: bigint
  total: bigint
  cacheRead: bigint
  cached: bigint
  uncached: bigint
}

const NO_TOKENS: Tokens = {
  input: 0n,
  output: 0n,
  reasoning: 0n,
  total: 0n,
  cacheRead: 0n,
  cached: 0n,
  uncached: 0n,
}

const count = (value: string | undefined) => (value && /^\d+$/.test(value) ? BigInt(value) : 0n)

function tokensOf(usage: SessionUsageSummary): Tokens {
  const u = usage.usage
  return {
    input: count(u?.input_tokens),
    output: count(u?.output_tokens),
    reasoning: count(u?.reasoning_output_tokens),
    total: count(u?.total_tokens),
    cacheRead: count(u?.cache.read_tokens),
    cached: count(u?.cache.cached_input_tokens),
    uncached: count(u?.cache.uncached_input_tokens),
  }
}

function addTokens(a: Tokens, b: Tokens): Tokens {
  return {
    input: a.input + b.input,
    output: a.output + b.output,
    reasoning: a.reasoning + b.reasoning,
    total: a.total + b.total,
    cacheRead: a.cacheRead + b.cacheRead,
    cached: a.cached + b.cached,
    uncached: a.uncached + b.uncached,
  }
}

/** A share of a session's tokens: the part that fell inside a time window. */
function scaleTokens(t: Tokens, part: number, whole: number): Tokens {
  if (part === whole) return t
  const p = BigInt(part)
  const w = BigInt(Math.max(whole, 1))
  const input = (t.input * p) / w
  const output = (t.output * p) / w
  return {
    input,
    output,
    reasoning: (t.reasoning * p) / w,
    total: input + output,
    cacheRead: (t.cacheRead * p) / w,
    cached: (t.cached * p) / w,
    uncached: (t.uncached * p) / w,
  }
}

function metrics(t: Tokens): AggregateUsageMetrics {
  return {
    input_tokens: String(t.input),
    output_tokens: String(t.output),
    reasoning_output_tokens: String(t.reasoning),
    total_tokens: String(t.total),
    cache: {
      read_tokens: String(t.cacheRead),
      write_tokens: "0",
      write_5m_tokens: "0",
      write_1h_tokens: "0",
      write_unknown_ttl_tokens: "0",
      cached_input_tokens: String(t.cached),
      uncached_input_tokens: String(t.uncached),
    },
  }
}

function sessionUsage(s: SeedSession): SessionUsageSummary {
  return {
    session_id: s.session.id,
    model_steps: s.usage.model_steps ?? 0,
    tool_calls: s.usage.tool_calls ?? 0,
    unmeasured_model_attempts: 0,
    provider_names: s.usage.provider_names ?? [],
    models: s.usage.models ?? [],
    usage: metrics(tokensOf(s.usage)),
  }
}

// ---------------------------------------------------------------------------------------
// Activity: what each session did, whole or inside a usage window.

type Activity = {
  s: SeedSession
  tokens: Tokens
  steps: number
  tools: number
  /** Cost in millionths, or null when the seed has none for the session. */
  cost: number | null
  currency: string
}

function activityOf(s: SeedSession, window?: { start: number; end: number }): Activity {
  const inWindow = (e: ApiEventRecord) => {
    if (!window) return true
    const at = Date.parse(e.timestamp)
    return at >= window.start && at < window.end
  }
  const steps = s.events.filter((e) => e.type === "model.completed")
  const stepsIn = steps.filter(inWindow).length
  const tools = s.events.filter((e) => e.type === "tool.call.started" && inWindow(e)).length
  const share = steps.length ? stepsIn / steps.length : 0
  const cost = s.overview.cost
  return {
    s,
    tokens: window ? scaleTokens(tokensOf(s.usage), stepsIn, steps.length) : tokensOf(s.usage),
    steps: window ? stepsIn : (s.usage.model_steps ?? steps.length),
    tools: window ? tools : (s.usage.tool_calls ?? tools),
    cost: cost ? Math.round(cost.amount * 1e6 * (window ? share : 1)) : null,
    currency: cost?.currency ?? "USD",
  }
}

const active = (a: Activity) => a.steps > 0 || a.tools > 0
const money = (millionths: number) => (millionths / 1e6).toFixed(6)

function totalsOf(acts: Activity[]): UsageAggregateTotals {
  const steps = acts.reduce((n, a) => n + a.steps, 0)
  return {
    session_count: String(acts.filter(active).length),
    model_steps: String(steps),
    model_steps_with_usage: String(steps),
    tool_calls: String(acts.reduce((n, a) => n + a.tools, 0)),
    usage: metrics(acts.reduce((t, a) => addTokens(t, a.tokens), NO_TOKENS)),
  }
}

/** Groups by a key, largest token total first. */
function groupBy(acts: Activity[], key: (a: Activity) => string) {
  const groups = new Map<string, Activity[]>()
  for (const a of acts) groups.set(key(a), [...(groups.get(key(a)) ?? []), a])
  const total = (list: Activity[]) => list.reduce((t, a) => t + a.tokens.total, 0n)
  return [...groups.values()].sort((a, b) => (total(b) > total(a) ? 1 : -1))
}

function usageBreakdown(
  acts: Activity[],
  limit: number,
  withModel: boolean,
): UsageAggregateBreakdown {
  const groups = groupBy(acts.filter(active), (a) =>
    withModel
      ? `${a.s.session.provider_name}/${a.s.session.model}`
      : `${a.s.session.provider_name}`,
  )
  const rest = groups.slice(limit)
  return {
    accuracy: EXACT,
    groups: groups.slice(0, limit).map((list) => ({
      provider_name: list[0]?.s.session.provider_name ?? null,
      model: withModel ? (list[0]?.s.session.model ?? null) : null,
      totals: totalsOf(list),
    })),
    remainder: rest.length
      ? { group_count: String(rest.length), totals: totalsOf(rest.flat()) }
      : null,
  }
}

function costSummary(acts: Activity[]): UsageSessionCostSummary {
  const priced = acts.filter((a) => a.cost !== null)
  const pricedSteps = priced.reduce((n, a) => n + a.steps, 0)
  const unpriced = acts.reduce((n, a) => n + a.steps, 0) - pricedSteps
  return {
    accuracy: EXACT,
    currencies: groupBy(priced, (a) => a.currency).map((list) => ({
      currency: list[0]?.currency ?? "USD",
      model_steps: String(list.reduce((n, a) => n + a.steps, 0)),
      total_cost: money(list.reduce((n, a) => n + (a.cost ?? 0), 0)),
    })),
    evaluated_model_steps: String(pricedSteps + unpriced),
    priced_model_steps: String(pricedSteps),
    unevaluated_model_steps: "0",
    unpriced_model_steps: String(unpriced),
    unpriced_reasons: unpriced
      ? [
          {
            model_steps: String(unpriced),
            reason: "No price for this model.",
            unpriced_reason: "missing_pricing",
          },
        ]
      : [],
  }
}

function costRollup(acts: Activity[], seed: AppSeed): UsageCostRollup {
  return {
    ...costSummary(acts.filter(active)),
    billing_breakdown: {
      accuracy: EXACT,
      groups: [],
      identified_model_steps: "0",
      remainder: null,
    },
    price_book_version: PRICE_BOOK,
    price_book_generated_at: seed.now,
  }
}

const seedHasCosts = (seed: AppSeed) => seed.sessions.some((s) => s.overview.cost !== null)

// ---------------------------------------------------------------------------------------
// POST /usage/rollup

/**
 * The Usage page anchors its window on the browser's clock. A window that ends about now
 * is moved onto the seed's clock, so "Last 30 days" covers the seed's sessions; a custom
 * window is used as given.
 */
function usageWindow(body: Record<string, unknown>, context: LegacyContext) {
  const start = Date.parse(String(body.start_at))
  const end = Date.parse(String(body.end_at))
  if (Number.isNaN(start) || Number.isNaN(end) || start >= end) {
    throw new HttpError(422, "start_at and end_at must be timestamps, with start before end.")
  }
  const relative = Math.abs(end - Date.now()) < 10 * 60_000
  const shift = relative ? context.now() - Date.now() : 0
  return { start: start + shift, end: end + shift }
}

function sessionBreakdown(acts: Activity[], limit: number): UsageSessionAggregateBreakdown {
  const ranked = acts.filter(active).sort((a, b) => (b.tokens.total > a.tokens.total ? 1 : -1))
  const rest = ranked.slice(limit)
  const isActive = (a: Activity) => sessionStatus(a.s) === "running"
  return {
    accuracy: EXACT,
    groups: ranked.slice(0, limit).map((a) => ({
      session_id: a.s.session.id,
      status: sessionStatus(a.s),
      active: isActive(a),
      totals: totalsOf([a]),
    })),
    remainder: rest.length
      ? {
          group_count: String(rest.length),
          active_session_count: String(rest.filter(isActive).length),
          totals: totalsOf(rest),
        }
      : null,
  }
}

function sessionCostBreakdown(
  acts: Activity[],
  limit: number,
  seed: AppSeed,
): UsageSessionCostBreakdown {
  const ranked = acts.filter(active).sort((a, b) => (b.cost ?? 0) - (a.cost ?? 0))
  const rest = ranked.slice(limit)
  return {
    accuracy: EXACT,
    groups: ranked
      .slice(0, limit)
      .map((a) => ({ session_id: a.s.session.id, cost: costSummary([a]) })),
    remainder: rest.length ? { group_count: String(rest.length), cost: costSummary(rest) } : null,
    price_book_version: PRICE_BOOK,
    price_book_generated_at: seed.now,
  }
}

function usageRollup({ body: raw }: LegacyRequest, context: LegacyContext): UsageRollup {
  const seed = context.records()
  const body = record(raw)
  const window = usageWindow(body, context)
  const filter = filterFrom(body.session_filter)
  const matching = seed.sessions.filter((s) => sessionMatches(s, filter))
  const acts = matching.map((s) => activityOf(s, window))
  const groupLimit = Math.max(1, Number(body.group_limit) || 20)
  const sessionLimit = Number(body.session_group_limit) || 0
  const costs = seedHasCosts(seed)
  return {
    scope: "configured_session_store",
    time_basis: "event.timestamp",
    session_filter_basis: "current_session_attributes",
    as_of: iso(context.now()),
    start_at: iso(window.start),
    end_at: iso(window.end),
    accuracy: EXACT,
    matching_session_count: String(matching.length),
    active_session_count: String(matching.filter((s) => sessionStatus(s) === "running").length),
    includes_active_sessions: true,
    totals: totalsOf(acts),
    provider_breakdown: usageBreakdown(acts, groupLimit, false),
    model_breakdown: usageBreakdown(acts, groupLimit, true),
    cost: costs ? costRollup(acts, seed) : null,
    reported_costs: { records: [], truncated: false },
    session_breakdown: sessionLimit > 0 ? sessionBreakdown(acts, sessionLimit) : null,
    session_cost_breakdown:
      sessionLimit > 0 && costs ? sessionCostBreakdown(acts, sessionLimit, seed) : null,
  }
}

// ---------------------------------------------------------------------------------------
// POST /sessions/summary (Environments lists an environment's latest sessions)

function summaryItem(s: SeedSession): ApiSessionSummaryItem {
  const counts: Record<string, number> = {}
  for (const e of s.events) counts[e.type] = (counts[e.type] ?? 0) + 1
  const latest = s.events.at(-1) ?? null
  const terminal =
    latest && ["session.completed", "session.failed", "session.interrupted"].includes(latest.type)
      ? latest
      : null
  return {
    session: { ...s.session, metadata: s.metadata },
    events: { counts_by_type: counts, latest_event: latest, total_events: s.events.length },
    outcome: {
      session_id: s.session.id,
      status: s.session.status,
      reason: s.overview.reason,
      details: {},
      retry: null,
      latest_retry_event: null,
      terminal_event: terminal,
    },
  }
}

function breakdownItems(acts: Activity[], withModel: boolean): UsageBreakdownItem[] {
  return usageBreakdown(acts, Number.MAX_SAFE_INTEGER, withModel).groups.map((g) => ({
    provider_name: g.provider_name,
    model: g.model,
    session_count: Number(g.totals.session_count),
    model_steps: Number(g.totals.model_steps),
    usage: g.totals.usage,
  }))
}

function sessionCost(a: Activity): SessionCostSummary {
  return {
    session_id: a.s.session.id,
    currency: a.currency,
    model_steps: a.steps,
    priced_model_steps: a.cost === null ? 0 : a.steps,
    unpriced_model_steps: a.cost === null ? a.steps : 0,
    total_cost: money(a.cost ?? 0),
  }
}

function aggregateCost(acts: Activity[]): AggregateCostSummary | null {
  const costs = acts.map(sessionCost)
  if (!acts.some((a) => a.cost !== null)) return null
  return {
    currency: acts[0]?.currency ?? "USD",
    line_items: [],
    session_costs: costs,
    session_count: acts.length,
    session_ids: acts.map((a) => a.s.session.id),
    model_steps: costs.reduce((n, c) => n + c.model_steps, 0),
    priced_model_steps: costs.reduce((n, c) => n + c.priced_model_steps, 0),
    unpriced_model_steps: costs.reduce((n, c) => n + c.unpriced_model_steps, 0),
    total_cost: money(acts.reduce((n, a) => n + (a.cost ?? 0), 0)),
  }
}

const SESSION_ORDERS: Record<string, (s: SeedSession) => string> = {
  created_at: (s) => s.session.created_at,
  updated_at: (s) => s.session.updated_at,
  last_activity_at: (s) => s.overview.last_activity_at,
}

function sessionsSummary({ query }: LegacyRequest, context: LegacyContext): SessionsSummary {
  const filter = filterFromQuery(query)
  const q = query.get("q")?.toLowerCase()
  const status = query.get("status")
  const order = query.get("order_by") ?? "created_at_desc"
  const field = SESSION_ORDERS[order.replace(/_(asc|desc)$/, "")] ?? SESSION_ORDERS.created_at
  const direction = order.endsWith("_asc") ? 1 : -1
  const matching = context
    .records()
    .sessions.filter(
      (s) =>
        sessionMatches(s, filter) &&
        (!status || sessionStatus(s) === status) &&
        (!q || `${s.session.id} ${s.session.agent_name} ${titleOf(s)}`.toLowerCase().includes(q)),
    )
    .sort((a, b) => direction * (field?.(a) ?? "").localeCompare(field?.(b) ?? ""))
  const offset = integer(query, "offset", 0)
  const page = matching.slice(offset, offset + integer(query, "limit", 50))
  const acts = page.map((s) => activityOf(s))
  const total = totalsOf(acts)
  return {
    session_count: page.length,
    total_count: matching.length,
    next_cursor: null,
    sessions: page.map(summaryItem),
    usage: {
      session_ids: page.map((s) => s.session.id),
      session_count: page.length,
      model_steps: Number(total.model_steps),
      unmeasured_model_attempts: 0,
      tool_calls: Number(total.tool_calls),
      provider_names: [...new Set(page.flatMap((s) => s.usage.provider_names ?? []))],
      models: [...new Set(page.flatMap((s) => s.usage.models ?? []))],
      usage: total.usage,
      session_summaries: page.map(sessionUsage),
    },
    provider_breakdown: breakdownItems(acts, false),
    model_breakdown: breakdownItems(acts, true),
    cost: aggregateCost(acts),
  }
}

// ---------------------------------------------------------------------------------------
// Tasks: the seed's own, plus one per root session that has none. Holds live in memory.

const HELD = new Set(["paused", "blocked", "needs_attention"])
const TERMINAL = new Set(["completed", "failed", "cancelled", "dependency_skipped"])

type TaskChange = Pick<Task, "status" | "status_reason" | "status_payload" | "updated_at"> & {
  /** The status before a hold, which resume returns to while the work is still live. */
  before: string
}
const changes = new WeakMap<AppSeed, Map<string, TaskChange>>()

function changesFor(seed: AppSeed) {
  let map = changes.get(seed)
  if (!map) {
    map = new Map()
    changes.set(seed, map)
  }
  return map
}

/** A session's task follows its state; a waiting session's task is still running. */
function taskStatusOf(seed: AppSeed, s: SeedSession): { status: string; reason: string | null } {
  const state = s.overview.state
  if (state === "completed") return { status: "completed", reason: null }
  if (state === "failed") return { status: "failed", reason: s.overview.reason }
  if (state === "interrupted") return { status: "paused", reason: "The session was interrupted." }
  if (state === "needs_recovery") {
    const item = seed.inbox.find((i) => i.session_id === s.session.id && i.kind === "recovery")
    const reason = item?.detail ?? s.overview.reason ?? "The session needs recovery."
    return { status: "needs_attention", reason }
  }
  return { status: "running", reason: null }
}

function taskTypeOf(seed: AppSeed, agent: string) {
  const types = seed.taskTypes ?? []
  return (types.find((t) => t.agent_name === agent) ?? types[0])?.name ?? "session_request"
}

function sessionTask(seed: AppSeed, s: SeedSession, now: number): Task {
  const { status, reason } = taskStatusOf(seed, s)
  const created = Date.parse(s.session.created_at) - 2_000
  const live = status === "running"
  return {
    id: `task_${hex(s.session.id, 20)}`,
    type: taskTypeOf(seed, s.session.agent_name),
    title: titleOf(s),
    description: null,
    status,
    status_reason: reason,
    status_payload: null,
    session_id: s.session.id,
    parent_task_id: null,
    assigned_agent_name: s.session.agent_name,
    available_at: iso(created),
    worker_id: live || TERMINAL.has(status) ? "worker-local-1" : null,
    lease_expires_at: live ? iso(now + 60_000) : null,
    created_at: iso(created),
    updated_at: s.overview.ended_at ?? s.overview.last_activity_at,
    completed_at: TERMINAL.has(status) ? s.overview.ended_at : null,
    retry_series: null,
    schedule: null,
  }
}

/** Every task, the seed's first; the new Tasks page reads these too. */
export function taskList(context: LegacyContext): TaskItem[] {
  const seed = context.records()
  const covered = new Set(seed.tasks.map((t) => t.session_id))
  const derived = seed.sessions
    .filter((s) => !s.session.parent_session_id && !covered.has(s.session.id))
    .map((s) => sessionTask(seed, s, context.now()))
  const held = changesFor(seed)
  return [...seed.tasks, ...derived].map((task) => {
    const change = held.get(task.id)
    if (!change) return task
    const { before: _, ...rest } = change
    return { ...task, ...rest }
  })
}

function findTask(context: LegacyContext, id: string): TaskItem {
  const task = taskList(context).find((t) => t.id === id)
  if (!task) throw new HttpError(404, `Task ${id} was not found.`)
  return task
}

function taskDetail(context: LegacyContext, task: TaskItem): TaskInfo {
  const s = context.records().sessions.find((x) => x.session.id === task.session_id)
  const request = s ? firstRequest(s) : null
  const output = s ? structuredOutput(s) : undefined
  const input = context.records().taskInputs?.[task.id] ?? {}
  return {
    ...task,
    started_at: s?.session.created_at ?? null,
    input: request && Object.keys(input).length === 0 ? { request, ...s?.session.labels } : input,
    result:
      task.status === "completed" && s
        ? {
            summary: s.overview.summary,
            reply: lastReply(s),
            ...(output === undefined ? {} : { output }),
          }
        : null,
    error:
      task.status === "failed" || task.status === "dependency_skipped"
        ? { reason: task.status_reason ?? s?.overview.reason ?? null }
        : null,
    invocation: {
      schema_version: 1,
      source: "task_dispatch",
      origin: { subject: null, tenant: null, trust: "unattributed" },
      root_invocation_id: `inv_${hex(task.id, 20)}`,
      root_session_id: task.session_id,
    },
    metadata: s ? { session_labels: s.session.labels } : {},
  }
}

function taskMatches(task: Task, query: URLSearchParams): boolean {
  const q = query.get("q")?.toLowerCase()
  const exact = ["status", "type", "session_id", "parent_task_id", "assigned_agent_name"] as const
  return (
    exact.every((key) => !query.get(key) || task[key] === query.get(key)) &&
    (!q ||
      [task.id, task.type, task.title, task.status, task.session_id, task.assigned_agent_name]
        .join(" ")
        .toLowerCase()
        .includes(q))
  )
}

function listTasks({ query }: LegacyRequest, context: LegacyContext): TaskItem[] {
  const order = query.get("order_by") ?? "updated_at_desc"
  const field = order.startsWith("created_at") ? "created_at" : "updated_at"
  const direction = order.endsWith("_asc") ? 1 : -1
  const offset = integer(query, "offset", 0)
  return taskList(context)
    .filter((task) => taskMatches(task, query))
    .sort((a, b) => direction * a[field].localeCompare(b[field]))
    .slice(offset, offset + integer(query, "limit", 100))
}

export function taskInfo(context: LegacyContext, id: string): TaskInfo {
  return taskDetail(context, findTask(context, id))
}

/** Pause, block or flag a task; finished tasks and running session tasks can't be held. */
export function holdTask(
  context: LegacyContext,
  id: string,
  status: "paused" | "blocked" | "needs_attention",
  hold: Record<string, unknown> = {},
): TaskInfo {
  const task = findTask(context, id)
  if (TERMINAL.has(task.status) || (task.status === "running" && task.session_id)) {
    throw new HttpError(409, `Task ${task.id} is ${task.status} and can't be held.`)
  }
  const held = changesFor(context.records())
  held.set(task.id, {
    status,
    status_reason: text(hold.reason) ?? null,
    status_payload: hold.payload ? record(hold.payload) : null,
    updated_at: iso(context.now()),
    before: held.get(task.id)?.before ?? task.status,
  })
  return taskInfo(context, task.id)
}

/** Resume returns a held task to its work: running if it was, otherwise pending. */
export function resumeTask(context: LegacyContext, id: string): TaskInfo {
  const task = findTask(context, id)
  if (!HELD.has(task.status)) {
    throw new HttpError(409, `Task ${task.id} is ${task.status}, not held.`)
  }
  const held = changesFor(context.records())
  const before = held.get(task.id)?.before
  held.set(task.id, {
    status: before === "running" || before === "claimed" ? before : "pending",
    status_reason: null,
    status_payload: null,
    updated_at: iso(context.now()),
    before: before ?? task.status,
  })
  return taskInfo(context, task.id)
}

/** A scheduled task's history: admitted when created, then whatever happened since. */
export function scheduleEvents(context: LegacyContext, id: string): TaskScheduleEvent[] {
  const task = findTask(context, id)
  const schedule = task.schedule
  if (!schedule) return []
  const at = task.available_at ?? task.created_at
  const types: TaskScheduleEvent["type"][] = ["task.scheduled"]
  if (schedule.revision > 1) types.push("task.rescheduled")
  if (Date.parse(at) <= context.now() || task.status !== "pending")
    types.push("task.schedule_eligible")
  if (["claimed", "running", "completed", "failed"].includes(task.status))
    types.push("task.schedule_claimed", "task.schedule_started")
  if (task.status === "completed") types.push("task.schedule_completed")
  if (task.status === "failed") types.push("task.schedule_failed")
  if (task.status === "cancelled") types.push("task.schedule_cancelled")
  if (HELD.has(task.status)) types.push("task.schedule_held")
  const times = [Date.parse(task.created_at), Date.parse(at), Date.parse(task.updated_at)]
  return types.map((type, i) => ({
    sequence: i + 1,
    type,
    task_id: task.id,
    revision: type === "task.scheduled" ? 1 : schedule.revision,
    available_at: at,
    occurred_at: iso(
      type === "task.scheduled"
        ? (times[0] ?? 0)
        : type === "task.rescheduled"
          ? (times[0] ?? 0) + 3_600_000
          : type === "task.schedule_eligible"
            ? (times[1] ?? 0)
            : (times[2] ?? 0),
    ),
    operation_id: type === "task.rescheduled" ? `op_${hex(task.id, 12)}` : null,
    invocation_id: `inv_${hex(task.id, 20)}`,
    policy: schedule.policy,
  }))
}

// ---------------------------------------------------------------------------------------
// POST /operations/snapshot

/** How many values equal `key`, as the runtime's string count. */
const tally = (values: string[], key: string) => String(values.filter((v) => v === key).length)

function sessionCounts(statuses: string[]): SessionStatusCounts {
  const [pending, running, interrupting, completed, failed, interrupted] = SESSION_STATUSES.map(
    (key) => tally(statuses, key),
  )
  return {
    pending: pending ?? "0",
    running: running ?? "0",
    interrupting: interrupting ?? "0",
    completed: completed ?? "0",
    failed: failed ?? "0",
    interrupted: interrupted ?? "0",
  }
}

function taskCounts(statuses: string[]): TaskStatusCounts {
  const n = (key: string) => tally(statuses, key)
  return {
    pending: n("pending"),
    waiting_dependencies: n("waiting_dependencies"),
    waiting_group: n("waiting_group"),
    dependency_skipped: n("dependency_skipped"),
    claimed: n("claimed"),
    running: n("running"),
    paused: n("paused"),
    blocked: n("blocked"),
    needs_attention: n("needs_attention"),
    completed: n("completed"),
    failed: n("failed"),
    cancelled: n("cancelled"),
  }
}

function taskFilterFrom(value: unknown): TaskAggregateFilter {
  const body = record(value)
  return {
    type: text(body.type),
    session_id: text(body.session_id),
    parent_task_id: text(body.parent_task_id),
    assigned_agent_name: text(body.assigned_agent_name),
  }
}

function operationsSnapshot(
  { body: raw }: LegacyRequest,
  context: LegacyContext,
): OperationalSnapshot {
  const body = record(raw)
  const asOf = iso(context.now())
  const filter = filterFrom(body.session_filter)
  const sessions = context.records().sessions.filter((s) => sessionMatches(s, filter))
  const includeTasks = body.include_tasks !== false
  const taskFilter = taskFilterFrom(body.task_filter)
  const tasks = taskList(context).filter((task) =>
    (["type", "session_id", "parent_task_id", "assigned_agent_name"] as const).every(
      (key) => taskFilter[key] == null || task[key] === taskFilter[key],
    ),
  )
  const pending = tasks.filter((t) => t.status === "pending")
  const scheduled = pending.filter(
    (t) => t.available_at && Date.parse(t.available_at) > context.now(),
  )
  return {
    scope: "configured_stores",
    cross_store_atomic: false,
    sessions: {
      as_of: asOf,
      total_count: String(sessions.length),
      counts_by_status: sessionCounts(sessions.map(sessionStatus)),
      accuracy: EXACT,
    },
    task_snapshot_status: includeTasks ? "available" : "not_requested",
    tasks: includeTasks
      ? {
          as_of: asOf,
          total_count: String(tasks.length),
          counts_by_status: taskCounts(tasks.map((t) => t.status)),
          claimable_pending_count: String(pending.length - scheduled.length),
          scheduled_pending_count: String(scheduled.length),
          accuracy: EXACT,
        }
      : null,
  }
}

// ---------------------------------------------------------------------------------------
// Environments and system diagnostics, after a real `cayu serve --dev` capture.

function environmentNames(seed: AppSeed): string[] {
  const names = new Set(seed.sessions.flatMap((s) => s.session.environment_name ?? []))
  return names.size ? [...names].sort() : ["local"]
}

const storeOf = (environment: string) =>
  environment === "local" ? "standard-local-artifacts" : `${environment}-artifacts`

function environment(name: string): EnvironmentSummary {
  return {
    name,
    metadata: { profile: "standard-local-v1", runner: "unavailable", network: "unavailable" },
    is_factory: false,
    workspace_id: null,
    workspace_branch_capabilities: {
      isolation: false,
      net_changes: false,
      publication: "unsupported",
      recovery: "unsupported",
      retention: "unsupported",
      lifecycle_inspection: "unsupported",
      detail_code: "workspace_branching_unsupported",
    },
    workspace_branch_lifecycle: { attached_count: 0, statuses: [], truncated: false },
    artifact_store_id: storeOf(name),
    runner_type: null,
    binding_type: null,
    vault_type: null,
    proxy_type: null,
    knowledge_store_type: "SQLiteKnowledgeStore",
    mcp_server_count: 0,
    workspace_checkpoint_policy: null,
    execution_snapshot_capability: {
      fidelity: "unsupported",
      adapter: "unsupported",
      adapter_version: "1",
      snapshot_format: "none",
      compatibility_sha256: null,
      requires_managed_launch: false,
    },
    lifecycle_policy: null,
    workspace_instructions: null,
    bound_workspace: null,
  }
}

function environments(_: LegacyRequest, context: LegacyContext): EnvironmentsPage {
  const names = environmentNames(context.records())
  return { environments: names.map(environment), total_count: names.length }
}

function systemDiagnostics(_: LegacyRequest, context: LegacyContext): SystemDiagnostics {
  const seed = context.records()
  const version = seed.app.runtime_version
  const costs = seedHasCosts(seed)
  const pricing = { enabled: costs, unavailable_reason: costs ? null : ("not_configured" as const) }
  return {
    observed_at: iso(context.now()),
    deployment: {
      name: "development",
      name_status: "available",
      api_access: "open",
      dashboard_access: "open",
      dashboard_enabled: true,
      docs_enabled: true,
    },
    versions: { cayu: version, server_contract: CONTRACT.contract_version ?? "49" },
    capabilities: {
      ...CONTRACT.capabilities,
      cayu_version: version,
      surfaces: {
        ...CONTRACT.capabilities.surfaces,
        pricing: {
          configured: costs,
          read: pricing,
          mutate: { enabled: false, unavailable_reason: "unsupported" },
        },
      },
    },
    artifact_stores: {
      registrations: environmentNames(seed).map((name) => ({
        fingerprint: `sha256:${hex(storeOf(name), 64)}`,
        store_contract_operations: ["list", "read", "write", "delete"],
      })),
      total_count: environmentNames(seed).length,
      truncated: false,
    },
    pricing_catalog: costs
      ? {
          configured: true,
          metadata_status: "available",
          price_book_version: PRICE_BOOK,
          generated_at: seed.now,
        }
      : {
          configured: false,
          metadata_status: "not_configured",
          price_book_version: null,
          generated_at: null,
        },
  }
}

// ---------------------------------------------------------------------------------------
// Artifacts: a report from each finished session and a CSV of its tool results.

type StoredArtifact = { summary: ArtifactSummary; content: string }

const MAX_ARTIFACTS = 8

function csvCell(value: unknown): string {
  const raw = typeof value === "string" ? value : JSON.stringify(value)
  return /[",\n]/.test(raw) ? `"${raw.replaceAll('"', '""')}"` : raw
}

function report(s: SeedSession): string {
  const lines = [`# ${titleOf(s)}`, ""]
  if (s.overview.summary) lines.push(`**${s.overview.summary}**`, "")
  const request = firstRequest(s)
  if (request) lines.push("## Request", "", request, "")
  const reply = lastReply(s)
  if (reply) lines.push("## Findings", "", reply, "")
  const calls = toolCalls(s)
  if (calls.length) {
    lines.push(
      "## Evidence",
      "",
      ...calls.map((c) => `- ${c.tool}: ${String(c.result ?? "no result")}`),
      "",
    )
  }
  lines.push(`Produced by ${s.session.agent_name} in session ${s.session.id}.`, "")
  return lines.join("\n")
}

function evidenceCsv(s: SeedSession): string {
  const rows = toolCalls(s).map((c, i) =>
    [i + 1, c.tool, c.args, c.result ?? "", c.error ? "error" : "ok"].map(csvCell).join(","),
  )
  return ["step,tool,arguments,result,status", ...rows, ""].join("\n")
}

function storedArtifact(
  s: SeedSession,
  filename: string,
  contentType: string,
  content: string,
  createdAt: number,
): StoredArtifact {
  const environmentName = s.session.environment_name ?? "local"
  return {
    content,
    summary: {
      id: `art_${hex(`${s.session.id}/${filename}`, 24)}`,
      artifact_store_id: storeOf(environmentName),
      scope: "session",
      session_id: s.session.id,
      agent_name: s.session.agent_name,
      environment_name: environmentName,
      filename,
      content_type: contentType,
      size_bytes: new TextEncoder().encode(content).length,
      created_at: iso(createdAt),
      metadata: {
        workspace_path: `outputs/${filename}`,
        published_by: "publish_workspace_artifact",
      },
    },
  }
}

/** Newest first, from the most recent finished sessions. */
function allArtifacts(seed: AppSeed): StoredArtifact[] {
  const finished = seed.sessions
    .filter((s) => s.overview.state === "completed" && s.overview.ended_at)
    .sort((a, b) => (b.overview.ended_at ?? "").localeCompare(a.overview.ended_at ?? ""))
  return finished
    .flatMap((s) => {
      const name = slug(titleOf(s))
      const ended = Date.parse(s.overview.ended_at ?? s.overview.last_activity_at)
      const files = [storedArtifact(s, `${name}.md`, "text/markdown", report(s), ended - 1_000)]
      if (toolCalls(s).length >= 2) {
        files.unshift(storedArtifact(s, `${name}-evidence.csv`, "text/csv", evidenceCsv(s), ended))
      }
      return files
    })
    .slice(0, MAX_ARTIFACTS)
}

export function findArtifact(
  { params, query }: Pick<LegacyRequest, "params" | "query">,
  context: LegacyContext,
): StoredArtifact {
  const store = query.get("artifact_store_id")
  const found = allArtifacts(context.records()).find(
    (a) => a.summary.id === params.artifactId && (!store || a.summary.artifact_store_id === store),
  )
  if (!found) throw new HttpError(404, `Artifact ${params.artifactId} was not found.`)
  return found
}

function listArtifacts({ query }: LegacyRequest, context: LegacyContext): ArtifactsPage {
  const keys = [
    "session_id",
    "agent_name",
    "environment_name",
    "artifact_store_id",
    "scope",
  ] as const
  const matching = allArtifacts(context.records())
    .map((a) => a.summary)
    .filter((a) => keys.every((key) => !query.get(key) || a[key] === query.get(key)))
  const limit = integer(query, "limit", 50)
  const offset = integer(query, "offset", 0)
  const more = offset + limit < matching.length
  return {
    artifacts: matching.slice(offset, offset + limit),
    total_count: matching.length,
    limit,
    offset,
    truncated: more,
    next_offset: more ? offset + limit : null,
  }
}

function base64(bytes: Uint8Array): string {
  let binary = ""
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}

function readArtifact(request: LegacyRequest, context: LegacyContext): ArtifactRead {
  const artifact = findArtifact(request, context)
  const bytes = new TextEncoder().encode(artifact.content)
  const preview = bytes.slice(0, integer(request.query, "max_bytes", 65_536))
  return {
    artifact: artifact.summary,
    preview_base64: base64(preview),
    text_preview: new TextDecoder().decode(preview),
    total_bytes: bytes.length,
    truncated: preview.length < bytes.length,
  }
}

function artifactContent(request: LegacyRequest, context: LegacyContext): Response {
  const { summary, content } = findArtifact(request, context)
  const disposition = request.query.get("disposition") === "attachment" ? "attachment" : "inline"
  return new Response(content, {
    headers: {
      "Content-Type": `${summary.content_type}; charset=utf-8`,
      "Content-Disposition": `${disposition}; filename="${summary.filename}"`,
    },
  })
}

// ---------------------------------------------------------------------------------------

export const coreRoutes: LegacyRoute[] = [
  { method: "GET", path: "/system/diagnostics", handle: systemDiagnostics },
  { method: "GET", path: "/environments", handle: environments },
  { method: "GET", path: "/artifacts", handle: listArtifacts },
  { method: "GET", path: "/artifacts/:artifactId", handle: readArtifact },
  { method: "GET", path: "/artifacts/:artifactId/content", handle: artifactContent },
  { method: "POST", path: "/operations/snapshot", handle: operationsSnapshot },
  { method: "POST", path: "/sessions/summary", handle: sessionsSummary },
  { method: "POST", path: "/usage/rollup", handle: usageRollup },
  { method: "GET", path: "/tasks", handle: listTasks },
  {
    method: "GET",
    path: "/tasks/:taskId",
    handle: ({ params }, context) => taskInfo(context, params.taskId ?? ""),
  },
  {
    method: "GET",
    path: "/tasks/:taskId/schedule/events",
    handle: ({ params }, context) => scheduleEvents(context, params.taskId ?? ""),
  },
  ...(
    [
      ["pause", "paused"],
      ["block", "blocked"],
      ["needs-attention", "needs_attention"],
    ] as const
  ).map(
    ([path, status]): LegacyRoute => ({
      method: "POST",
      path: `/tasks/:taskId/${path}`,
      handle: ({ params, body }, context) =>
        holdTask(context, params.taskId ?? "", status, record(body)),
    }),
  ),
  {
    method: "POST",
    path: "/tasks/:taskId/resume",
    handle: ({ params }, context) => resumeTask(context, params.taskId ?? ""),
  },
]
