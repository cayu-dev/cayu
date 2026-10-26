/**
 * The Evals pages' data (backend item 14), derived from the current seed: one or two suites
 * per specialist with cases drawn from its tools, kept knowledge and finished sessions, and
 * two weeks of runs whose latest ones cover every verdict (a regression, an improvement, a
 * tradeoff, a tie, one still running and a suite never run). Trials link to the
 * specialist's real sessions. Runs started here queue, run and complete a few seconds
 * later. A seed without sessions (a new agent) starts with no suites.
 */
import type {
  AddToEvals,
  EvalCase,
  EvalCaseOutcome,
  EvalCaseResult,
  EvalCheck,
  EvalCheckResult,
  EvalRunDetail,
  EvalRunOrigin,
  EvalRunStatus,
  EvalRunSummary,
  EvalSettings,
  EvalSuiteDetail,
  EvalSuiteSummary,
  EvalTrial,
  EvalVerdict,
  NewEvalSuite,
} from "../api/eval-types"
import type { KnowledgeEntry, Specialist, TranscriptMessage } from "../api/types"
import { isEvalTrialMetadata } from "../app/session-model"
import { seeded } from "./builder"
import type { AppSeed, SeedSession } from "./model"
import { NotFound } from "./not-found"

const SECOND = 1000
const MINUTE = 60 * SECOND
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

/** A fresh run waits this long in the queue, then runs until DONE_AFTER. */
const QUEUED_FOR = 2 * SECOND
const DONE_AFTER = 10 * SECOND
/** The seed's running run finishes this long after the store first reads the seed. */
const SEED_RUN_LEFT = 12 * SECOND
const JUDGE_PASS = 0.7
const MAX_TRIALS = 5
const HANDLES = ["j.okafor", "maria.chen", "d.alvarez", "s.nakamura"]

export type EvalContext = { records: () => AppSeed; now: () => number }

type Trials = { per_case: number; min_passes: number }

/** A case plus what its checks look for, so trials pass or fail them consistently. */
type CaseRecord = EvalCase & {
  needle: string | null
  tool: string | null
  maxCalls: number | null
  /** A passing reply for cases no session answers (scope, out-of-scope, knowledge). */
  canned: string | null
}

type SuiteRecord = {
  id: string
  name: string
  description: string | null
  specialist: string
  source: EvalSuiteSummary["source"]
  trials: Trials
  cases: CaseRecord[]
  baseline: { run_id: string; approved_by: string; approved_at: number } | null
}

type CaseRun = { case_id: string; trials: EvalTrial[]; error: boolean }

/** Results are decided when the run is made and revealed as its clock passes. */
type RunRecord = {
  id: string
  suite_id: string
  origin: EvalRunOrigin
  trials: Trials
  queuedAt: number
  startAt: number
  doneAt: number
  results: CaseRun[]
}

type State = {
  seed: AppSeed
  created: number
  suites: SuiteRecord[]
  runs: RunRecord[]
  /** Which session each suite's next trial links to. */
  cursor: Map<string, number>
  made: number
}

type Grade = "pass" | "flaky" | "fail" | "error"
type Scenario =
  | "regression"
  | "improvement"
  | "tradeoff"
  | "tie"
  | "running"
  | "never"
  | "history"
  | "unapproved"
type Flavor = "captured" | "core" | "golden" | "smoke" | "tools"

// ---------------------------------------------------------------- text from sessions

const iso = (ms: number) => new Date(ms).toISOString()
const clip = (text: string, n: number) => (text.length > n ? `${text.slice(0, n - 1)}…` : text)
const round = (n: number, places: number) => Math.round(n * 10 ** places) / 10 ** places

function textOf(m: TranscriptMessage): string {
  return m.content
    .map((p) => (p.type === "text" && typeof p.text === "string" ? p.text : ""))
    .join("")
    .trim()
}

function firstUserText(s: SeedSession): string {
  for (const m of s.transcript) {
    const text = m.role === "user" ? textOf(m) : ""
    if (text) return text
  }
  return ""
}

function lastAssistantText(s: SeedSession): string {
  for (let i = s.transcript.length - 1; i >= 0; i--) {
    const m = s.transcript[i]
    const text = m?.role === "assistant" ? textOf(m) : ""
    if (text) return text
  }
  return ""
}

function toolsUsed(s: SeedSession): string[] {
  return s.transcript.flatMap((m) =>
    m.content.flatMap((p) =>
      p.type === "tool_call" && typeof p.tool_name === "string" && !p.tool_name.startsWith("__cayu")
        ? [p.tool_name]
        : [],
    ),
  )
}

const STOP = new Set([
  "already",
  "another",
  "because",
  "before",
  "between",
  "called",
  "following",
  "however",
  "needed",
  "number",
  "should",
  "something",
  "started",
  "through",
  "without",
])

/** A distinctive word for a "contains" check: a past tense verb if there is one. */
function pickNeedle(text: string): string | null {
  const words = (text.toLowerCase().match(/[a-z][a-z-]{4,}[a-z]/g) ?? []).filter(
    (w) => !STOP.has(w),
  )
  return (
    words.find((w) => w.endsWith("ed")) ??
    [...words].sort((a, b) => b.length - a.length || a.localeCompare(b))[0] ??
    null
  )
}

/** "close-agent" → "Close", "ap-auditor" → "AP auditor". */
function label(name: string): string {
  const words = name
    .replace(/[-_]agent$/, "")
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((w) => (w.length <= 2 ? w.toUpperCase() : w.toLowerCase()))
  const text = words.join(" ")
  return text.charAt(0).toUpperCase() + text.slice(1)
}

const slug = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")

const sentence = (text: string) => text.trim().replace(/[.\s]+$/, "")

function describe(spec: Specialist): string {
  const d = spec.metadata["cayu:description"]
  return typeof d === "string" && d ? d : `Work for ${spec.name}`
}

function hex(random: () => number, n: number) {
  return Array.from({ length: n }, () => Math.floor(random() * 16).toString(16)).join("")
}

function pick<T>(random: () => number, items: T[]): T {
  return items[Math.floor(random() * items.length)] as T
}

function shuffle<T>(random: () => number, items: T[]): T[] {
  const out = [...items]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[out[i], out[j]] = [out[j] as T, out[i] as T]
  }
  return out
}

// ---------------------------------------------------------------- checks and cases

const check = {
  contains: (needle: string): EvalCheck => ({
    kind: "final_output_contains",
    label: `Final output contains “${needle}”`,
    judge: false,
  }),
  calls: (tool: string): EvalCheck => ({
    kind: "tool_called",
    label: `Calls ${tool}`,
    judge: false,
  }),
  maxCalls: (n: number): EvalCheck => ({
    kind: "max_tool_calls",
    label: `No more than ${n} tool calls`,
    judge: false,
  }),
  completes: (): EvalCheck => ({
    kind: "root_status",
    label: "Finishes as completed",
    judge: false,
  }),
  judge: (rubric: string): EvalCheck => ({
    kind: "model_judge",
    label: `Judge: ${rubric}`,
    judge: true,
  }),
}

function makeCase(
  id: string,
  name: string,
  input: string,
  o: {
    needle?: string | null
    tool?: string | null
    maxCalls?: number | null
    completes?: boolean
    judge?: string | null
    source?: string | null
    canned?: string | null
  },
): CaseRecord {
  const checks: EvalCheck[] = []
  if (o.needle) checks.push(check.contains(o.needle))
  if (o.tool) checks.push(check.calls(o.tool))
  if (o.maxCalls) checks.push(check.maxCalls(o.maxCalls))
  if (o.completes) checks.push(check.completes())
  if (o.judge) checks.push(check.judge(o.judge))
  return {
    id,
    name,
    input,
    checks,
    source_session_id: o.source ?? null,
    needle: o.needle ?? null,
    tool: o.tool ?? null,
    maxCalls: o.maxCalls ?? null,
    canned: o.canned ?? null,
  }
}

const SESSION_RUBRICS = [
  "answers with evidence from the records",
  "explains the result with evidence",
  "says what it checked and what it found",
]

type CaseDraft = (id: string) => CaseRecord

function sessionCase(s: SeedSession, captured: boolean, random: () => number): CaseDraft {
  const first = firstUserText(s)
  const name = s.overview.title ?? (first ? clip(first, 70) : `Session ${s.session.id}`)
  const tool = toolsUsed(s)[0] ?? null
  return (id) =>
    makeCase(id, name, first || name, {
      needle: pickNeedle(lastAssistantText(s)),
      tool,
      maxCalls: tool ? Math.max(6, Math.ceil((s.overview.tool_calls + 2) * 1.5)) : null,
      completes: true,
      judge: pick(random, SESSION_RUBRICS),
      source: captured ? s.session.id : null,
    })
}

function toolCase(t: Specialist["tools"][number], random: () => number): CaseDraft {
  const what = sentence(t.description || "")
  const name = what || `Uses ${t.name}`
  return (id) =>
    makeCase(id, name, `${name}. Tell me what you found and anything a person should look at.`, {
      tool: t.name,
      maxCalls: random() < 0.5 ? 8 : 12,
      completes: true,
      judge: random() < 0.6 ? "explains the result in plain words" : null,
    })
}

function knowledgeCase(k: KnowledgeEntry): CaseDraft {
  return (id) =>
    makeCase(
      id,
      `Follows: ${clip(k.text, 60)}`,
      `A request came in that your kept ${k.kind} covers: “${k.text}” Handle it and say which ${k.kind} you followed.`,
      {
        completes: true,
        judge: `follows the kept ${k.kind}`,
        canned: `Handled it the way the kept ${k.kind} says: ${k.text}`,
      },
    )
}

function edgeCases(spec: Specialist): CaseDraft[] {
  const scope = sentence(describe(spec))
  const lower = scope.charAt(0).toLowerCase() + scope.slice(1)
  return [
    (id) =>
      makeCase(id, "Says what it can help with", "What can you help me with?", {
        needle: pickNeedle(scope),
        maxCalls: 2,
        judge: "describes its scope accurately",
        canned: `I can help with ${lower}. Tell me the period or record you have in mind and I'll start there.`,
      }),
    (id) =>
      makeCase(
        id,
        "Declines an out-of-scope request",
        "Book me a flight to Lisbon for next Tuesday.",
        {
          maxCalls: 2,
          completes: true,
          judge: "declines politely and says what it can do",
          canned: `That's outside what I do here. I work on ${lower}, and I'm happy to help with that.`,
        },
      ),
    (id) =>
      makeCase(id, "Asks for a missing detail", "Can you take care of the one from last week?", {
        completes: true,
        judge: "asks one clear question instead of guessing",
        canned:
          "Which one do you mean? I can see a few from last week. Send me the reference and I'll pick it up.",
      }),
  ]
}

// ---------------------------------------------------------------- trials

const PASS_REASONS = [
  "Says which records it used, and the conclusion follows from them.",
  "Clear and specific: names the figures it relied on and where they came from.",
  "Answers what was asked without padding; one small gap in how it cites its source.",
  "Covers the request fully and flags the one item a person should check.",
]
const FAIL_REASONS = [
  "States a conclusion but doesn't show the evidence behind it.",
  "Stops before answering the question that was asked.",
  "Mixes up two periods, so the explanation doesn't hold.",
  "Restates the request instead of resolving it.",
]

/** The seed's sessions that aren't themselves eval trials: what cases and trials draw on. */
const appSessions = (seed: AppSeed) => seed.sessions.filter((s) => !isEvalTrialMetadata(s.metadata))

function sessionsOf(seed: AppSeed, specialist: string): SeedSession[] {
  const all = appSessions(seed).filter((s) => s.session.agent_name === specialist)
  const roots = all.filter((s) => !s.session.parent_session_id)
  return roots.length ? roots : all
}

function passReply(c: CaseRecord): string {
  const how = c.tool ? `I used ${c.tool} to check the figures` : "I checked what I had"
  const what = c.needle ? `the answer is ${c.needle}` : "nothing needs a person to look"
  return `Finished. ${how}, and ${what}. The details are in the session.`
}

function failReply(c: CaseRecord): string {
  const text = c.tool
    ? `I couldn't finish this. ${c.tool} returned nothing for the period I asked for, so I stopped and left a note.`
    : "I couldn't finish this. I didn't have enough to go on, so I stopped and left a note."
  return c.needle && text.toLowerCase().includes(c.needle)
    ? "I stopped early and left a note."
    : text
}

function checkResults(
  c: CaseRecord,
  how: "pass" | "fail" | "error",
  random: () => number,
): EvalCheckResult[] {
  const failing = new Set<string>()
  if (how === "fail") {
    const plain = c.checks.filter((x) =>
      ["final_output_contains", "tool_called", "max_tool_calls"].includes(x.kind),
    )
    const hasJudge = c.checks.some((x) => x.judge)
    if (plain.length) failing.add(pick(random, plain).kind)
    if (hasJudge && (!plain.length || random() < 0.5)) failing.add("model_judge")
    if (!failing.size) failing.add("root_status")
  }
  return c.checks.map((x): EvalCheckResult => {
    const ok = how === "pass" || (how === "fail" && !failing.has(x.kind))
    const base = { ...x, passed: ok, detail: null, score: null, reason: null }
    switch (x.kind) {
      case "final_output_contains":
        return {
          ...base,
          detail:
            how === "error"
              ? "The trial didn't produce a final output."
              : ok
                ? `Found “${c.needle}” in the final output.`
                : `“${c.needle}” isn't in the final output.`,
        }
      case "tool_called": {
        const n = 1 + Math.floor(random() * 3)
        return {
          ...base,
          detail: ok
            ? `Called ${c.tool} ${n === 1 ? "once" : `${n} times`}.`
            : `${c.tool} was never called.`,
        }
      }
      case "max_tool_calls": {
        const limit = c.maxCalls ?? 12
        const made = ok
          ? Math.max(c.tool ? 1 : 0, Math.min(limit, Math.round(random() * limit * 0.8)))
          : limit + 1 + Math.floor(random() * 8)
        return {
          ...base,
          detail:
            how === "error"
              ? `Made ${made} tool calls before the trial timed out.`
              : `Made ${made} tool call${made === 1 ? "" : "s"}.`,
        }
      }
      case "root_status":
        return {
          ...base,
          detail:
            how === "error"
              ? "Finished as failed: timed out after 600s."
              : "Finished as completed.",
        }
      case "model_judge":
        if (how === "error")
          return { ...base, reason: "Not judged: the trial didn't finish.", score: null }
        return {
          ...base,
          score: round(ok ? 0.72 + random() * 0.25 : 0.18 + random() * 0.45, 2),
          reason: pick(random, ok ? PASS_REASONS : FAIL_REASONS),
        }
      default:
        return base
    }
  })
}

// ---------------------------------------------------------------- outcomes and comparison

const passes = (r: CaseRun) => r.trials.filter((t) => t.passed).length

function outcomeOf(r: CaseRun, trials: Trials): EvalCaseOutcome {
  if (r.error) return "error"
  const p = passes(r)
  if (p < trials.min_passes) return "failed"
  return p === r.trials.length ? "passed" : "flaky"
}

const isPassing = (o: EvalCaseOutcome) => o === "passed" || o === "flaky"

function gradeOf(r: CaseRun, trials: Trials): Grade {
  const o = outcomeOf(r, trials)
  return o === "passed" ? "pass" : o === "failed" ? "fail" : o
}

function judgeScores(trials: EvalTrial[]): number[] {
  return trials.flatMap((t) =>
    t.checks.flatMap((x) => (x.judge && x.score != null ? [x.score] : [])),
  )
}

const mean = (xs: number[]) =>
  xs.length ? round(xs.reduce((a, b) => a + b, 0) / xs.length, 2) : null

function verdictOf(improved: number, regressed: number): EvalVerdict {
  if (improved && regressed) return "tradeoff"
  if (improved) return "improvement"
  if (regressed) return "regression"
  return "tie"
}

// ---------------------------------------------------------------- the store

export function createEvalStore(context: EvalContext) {
  const states = new WeakMap<AppSeed, State>()

  const state = (): State => {
    const seed = context.records()
    let s = states.get(seed)
    if (!s) {
      s = build(seed, context.now())
      states.set(seed, s)
    }
    return s
  }

  // ----- trials, given a state

  /**
   * The session a trial links to, cycling through the specialist's finished sessions.
   * When the trial's output should satisfy the case, prefer one whose reply already does.
   */
  function nextSession(st: State, suite: SuiteRecord, c: CaseRecord, satisfied: boolean) {
    const all = sessionsOf(st.seed, suite.specialist)
    const ended = all.filter((s) => s.overview.ended_at)
    const pool = ended.length ? ended : all
    if (!pool.length) return null
    const fits = pool.filter((s) => {
      if (c.tool && !toolsUsed(s).includes(c.tool)) return false
      if (c.needle && !lastAssistantText(s).toLowerCase().includes(c.needle.toLowerCase()))
        return false
      return true
    })
    const from = satisfied && fits.length ? fits : pool
    const n = st.cursor.get(suite.id) ?? 0
    st.cursor.set(suite.id, n + 1)
    return { session: from[n % from.length] as SeedSession, fits: satisfied && fits.length > 0 }
  }

  function makeTrial(
    st: State,
    suite: SuiteRecord,
    c: CaseRecord,
    n: number,
    how: "pass" | "fail" | "error",
    random: () => number,
  ): EvalTrial {
    const checks = checkResults(c, how, random)
    // A failed trial still reads like an answer unless it missed the expected text.
    const answered =
      how !== "error" && !checks.some((x) => x.kind === "final_output_contains" && !x.passed)
    const linked = nextSession(st, suite, c, answered)
    let output = ""
    if (answered) {
      const text = linked?.fits && !c.canned ? lastAssistantText(linked.session) : ""
      output = c.canned ?? (text ? clip(text, 600) : passReply(c))
    } else if (how === "fail") output = failReply(c)
    const tokens = Math.round((how === "error" ? 30_000 : 4_000) + random() * 36_000)
    return {
      n,
      passed: how === "pass",
      session_id: linked?.session.session.id ?? null,
      output,
      checks,
      usage: { tokens, cost: round(tokens * 0.0000045, 4) },
      seconds: how === "error" ? 600 : Math.round(12 + random() * 140),
    }
  }

  function caseRun(
    st: State,
    suite: SuiteRecord,
    c: CaseRecord,
    grade: Grade,
    trials: Trials,
    random: () => number,
  ): CaseRun {
    const k = trials.per_case
    const passCount =
      grade === "pass"
        ? k
        : grade === "flaky"
          ? k > 1
            ? Math.max(trials.min_passes, k - 1)
            : k
          : grade === "fail" && k > 1
            ? Math.floor(random() * trials.min_passes)
            : 0
    const passing = new Set(shuffle(random, [...Array(k).keys()]).slice(0, passCount))
    return {
      case_id: c.id,
      error: grade === "error",
      trials: Array.from({ length: k }, (_, i) =>
        makeTrial(
          st,
          suite,
          c,
          i + 1,
          grade === "error" && i === 0 ? "error" : passing.has(i) ? "pass" : "fail",
          random,
        ),
      ),
    }
  }

  /** Score a session as it ran (captured evaluation): one trial, the session itself. */
  function scoreSession(c: CaseRecord, s: SeedSession, random: () => number): CaseRun {
    const output = clip(lastAssistantText(s), 600)
    const used = toolsUsed(s)
    const completed = s.overview.state === "completed"
    const checks = c.checks.map((x): EvalCheckResult => {
      const base = { ...x, detail: null, score: null, reason: null }
      switch (x.kind) {
        case "root_status":
          return { ...base, passed: completed, detail: `Finished as ${s.overview.state}.` }
        case "final_output_contains": {
          const ok = !!c.needle && output.toLowerCase().includes(c.needle.toLowerCase())
          return {
            ...base,
            passed: ok,
            detail: ok
              ? `Found “${c.needle}” in the final output.`
              : `“${c.needle}” isn't in the final output.`,
          }
        }
        case "tool_called": {
          const n = used.filter((t) => t === c.tool).length
          return {
            ...base,
            passed: n > 0,
            detail: n
              ? `Called ${c.tool} ${n === 1 ? "once" : `${n} times`}.`
              : `${c.tool} was never called.`,
          }
        }
        case "max_tool_calls": {
          const made = s.overview.tool_calls
          return {
            ...base,
            passed: made <= (c.maxCalls ?? Number.POSITIVE_INFINITY),
            detail: `Made ${made} tool call${made === 1 ? "" : "s"}.`,
          }
        }
        case "model_judge": {
          const score = round(completed ? 0.64 + random() * 0.33 : 0.2 + random() * 0.3, 2)
          const ok = score >= JUDGE_PASS
          return {
            ...base,
            passed: ok,
            score,
            reason: pick(random, ok ? PASS_REASONS : FAIL_REASONS),
          }
        }
        default:
          return { ...base, passed: true }
      }
    })
    const tokens = s.overview.tokens.input + s.overview.tokens.output
    const ended = s.overview.ended_at
      ? Date.parse(s.overview.ended_at)
      : Date.parse(s.overview.last_activity_at)
    return {
      case_id: c.id,
      error: false,
      trials: [
        {
          n: 1,
          passed: checks.every((x) => x.passed),
          session_id: s.session.id,
          output,
          checks,
          usage: { tokens, cost: s.overview.cost?.amount ?? null },
          seconds: Math.max(1, Math.round((ended - Date.parse(s.overview.started_at)) / 1000)),
        },
      ],
    }
  }

  // ----- the seed's suites and history

  function build(seed: AppSeed, created: number): State {
    const st: State = { seed, created, suites: [], runs: [], cursor: new Map(), made: 0 }
    if (!seed.sessions.length) return st
    const agents = seed.specialists.agents
    const completedOf = (name: string) =>
      sessionsOf(seed, name).filter((s) => s.overview.state === "completed")
    const finished = (name: string) => {
      const done = appSessions(seed).filter(
        (s) => s.session.agent_name === name && s.overview.state === "completed",
      )
      return done.length ? done : sessionsOf(seed, name)
    }
    const withSessions = agents.filter((a) =>
      seed.sessions.some((s) => s.session.agent_name === a.name),
    )
    const captured = [...withSessions].sort(
      (a, b) => finished(b.name).length - finished(a.name).length,
    )[0]
    const imported = withSessions.filter((a) => a !== captured).at(-1) ?? captured

    const plan: { spec: Specialist; flavor: Flavor }[] = []
    agents.forEach((spec, i) => {
      const two = agents.length <= 3 || spec === captured || spec === imported || i % 2 === 0
      if (spec === captured) {
        plan.push({ spec, flavor: "captured" })
        plan.push({ spec, flavor: spec === imported ? "golden" : "core" })
      } else if (spec === imported) {
        plan.push({ spec, flavor: "golden" })
        if (two) plan.push({ spec, flavor: "smoke" })
      } else {
        plan.push({ spec, flavor: "smoke" })
        if (two) plan.push({ spec, flavor: "tools" })
      }
    })

    const rest: Scenario[] = ["improvement", "tradeoff", "tie", "never"]
    let extra = 0
    let authored = 0
    for (const { spec, flavor } of plan) {
      const id = `suite-${spec.name}-${flavor}`
      const random = seeded(`${seed.id}:${id}`)
      const name = label(spec.name)
      const knowledge = (seed.knowledge ?? []).filter((k) => k.agent_name === spec.name)
      const tools = spec.tools.map((t) => toolCase(t, random))
      const edges = edgeCases(spec)
      const known = knowledge.slice(0, 2).map(knowledgeCase)
      const titled = completedOf(spec.name)
        .filter((s) => s.overview.title)
        .map((s) => sessionCase(s, false, random))

      let drafts: CaseDraft[]
      const take = (pool: CaseDraft[], min: number, max: number) =>
        pool.slice(0, Math.min(pool.length, min + Math.floor(random() * (max - min + 1))))
      if (flavor === "captured") {
        drafts = finished(spec.name)
          .slice(0, 8)
          .map((s) => sessionCase(s, true, random))
        if (drafts.length < 3) drafts.push(...tools.slice(0, 3 - drafts.length))
      } else if (flavor === "core") drafts = take([...tools.slice(0, 3), ...known, ...edges], 3, 6)
      else if (flavor === "golden")
        drafts = take([...titled.slice(0, 3), ...tools, ...known, ...edges], 5, 8)
      else if (flavor === "smoke") drafts = take([...titled.slice(0, 2), ...edges, ...tools], 3, 5)
      else drafts = take([...tools, ...known, ...edges], 3, 6)
      if (drafts.length < 3) drafts.push(...edges.slice(0, 3 - drafts.length))

      const source: SuiteRecord["source"] =
        flavor === "captured" ? "captured" : flavor === "golden" ? "imported" : "authored"
      const trials: Trials =
        source === "captured"
          ? { per_case: 1, min_passes: 1 }
          : source === "imported" || authored++ % 2 === 0
            ? { per_case: 3, min_passes: 2 }
            : { per_case: 1, min_passes: 1 }
      const suite: SuiteRecord = {
        id,
        name: {
          captured: `${name}: captured sessions`,
          core: `${name} core requests`,
          golden: `${name} golden set`,
          smoke: `${name} smoke tests`,
          tools: `${name} tool use`,
        }[flavor],
        description: {
          captured: `Finished ${spec.name} sessions, kept as cases with what they were asked.`,
          core: `The requests ${spec.name} handles most, written in the dashboard.`,
          golden: `Imported from ${slug(name)}-golden.jsonl.`,
          smoke: `Quick checks that ${spec.name} answers, stays in scope and asks when unsure.`,
          tools: `Each of ${spec.name}'s tools, called when it should be.`,
        }[flavor],
        specialist: spec.name,
        source,
        trials,
        cases: drafts.map((d, i) => d(`${id}-case-${i + 1}`)),
        baseline: null,
      }
      st.suites.push(suite)

      const scenario: Scenario =
        source === "captured"
          ? "regression"
          : source === "imported" && spec !== captured
            ? "running"
            : (rest.shift() ?? (extra++ % 2 ? "unapproved" : "history"))
      history(st, suite, scenario, random)
    }
    return st
  }

  function history(st: State, suite: SuiteRecord, scenario: Scenario, random: () => number) {
    if (scenario === "never") return
    const seedNow = Date.parse(st.seed.now)
    const cases = suite.cases
    const draw = (): Grade => {
      const r = random()
      if (r < 0.12) return "fail"
      if (suite.trials.per_case > 1 && r < 0.27) return "flaky"
      return "pass"
    }
    const perturb = (v: Grade[]) => v.map((g) => (random() < 0.15 ? draw() : g))
    const passingAt = (v: Grade[]) =>
      v.flatMap((g, i) => (g === "pass" || g === "flaky" ? [i] : []))
    const failingAt = (v: Grade[]) =>
      v.flatMap((g, i) => (g === "fail" || g === "error" ? [i] : []))

    const reference = cases.map(draw)
    if (scenario === "improvement" || scenario === "tradeoff") {
      const f = failingAt(reference)[0] ?? Math.floor(random() * cases.length)
      reference[f] = "fail"
      if (scenario === "tradeoff" && !passingAt(reference).length)
        reference[(f + 1) % cases.length] = "pass"
    }
    if (scenario === "regression") {
      while (passingAt(reference).length < Math.min(2, cases.length))
        reference[failingAt(reference)[0] ?? 0] = "pass"
    }

    let latest: Grade[]
    if (scenario === "regression") {
      latest = [...reference]
      const broken = shuffle(random, passingAt(reference)).slice(0, 2)
      broken.forEach((i, k) => {
        latest[i] = k === 1 && cases.length >= 4 ? "error" : "fail"
      })
    } else if (scenario === "improvement") {
      latest = reference.map((g) => (g === "fail" || g === "error" ? "pass" : g))
    } else if (scenario === "tradeoff") {
      latest = [...reference]
      latest[pick(random, failingAt(reference))] = "pass"
      latest[pick(random, passingAt(reference))] = "fail"
    } else if (scenario === "tie") {
      latest = [...reference]
      if (suite.trials.per_case > 1) {
        // Same cases pass and fail; one passing case wobbles, which isn't a change.
        const i = pick(random, passingAt(reference))
        latest[i] = reference[i] === "flaky" ? "pass" : "flaky"
      }
    } else latest = perturb(reference)

    const n = 4 + Math.floor(random() * 5)
    const withBaseline = scenario !== "running" && scenario !== "unapproved"
    const b = withBaseline ? Math.max(0, n - 2 - Math.floor(random() * Math.min(3, n - 1))) : -1
    const capturedCase = cases.find((c) => c.source_session_id)
    const fromSession =
      suite.source === "captured" && capturedCase
        ? ([...Array(n - 1).keys()].filter((i) => i !== b && i > 0).at(-1) ?? -1)
        : -1
    const topics = [
      ...new Set([
        ...suite.cases.flatMap((c) => (c.tool ? [`${slug(c.tool)}-retries`] : [])),
        "prompt-tweaks",
        "model-upgrade",
        "tighter-instructions",
      ]),
    ]

    for (let i = 0; i < n; i++) {
      const last = i === n - 1
      const id = `evr_${hex(seeded(`${st.seed.id}:${suite.id}:run:${i}`), 10)}`
      let origin: EvalRunOrigin
      if (random() < 0.55) {
        const onMain = random() < 0.5
        origin = {
          kind: "ci",
          branch: onMain ? "main" : `${random() < 0.5 ? "fix" : "feat"}/${pick(random, topics)}`,
          commit: hex(random, 7),
          pr: onMain ? null : 300 + Math.floor(random() * 240),
        }
      } else origin = { kind: "dashboard", by: pick(random, HANDLES) }

      let startAt = last
        ? seedNow - HOUR * (1 + random() * 7)
        : seedNow - DAY * (13.5 - i * (11 / (n - 2))) + (random() - 0.5) * 10 * HOUR
      let results: CaseRun[]
      let trials = suite.trials
      if (i === fromSession && capturedCase) {
        const s = st.seed.sessions.find((x) => x.session.id === capturedCase.source_session_id)
        if (!s) throw new Error(`Seed session ${capturedCase.source_session_id} is missing.`)
        origin = { kind: "session", session_id: s.session.id }
        trials = { per_case: 1, min_passes: 1 }
        results = [scoreSession(capturedCase, s, random)]
      } else {
        const grades = i === b ? reference : last ? latest : perturb(reference)
        results = cases.map((c, k) => caseRun(st, suite, c, grades[k] ?? "pass", trials, random))
      }
      const busy = results.reduce((sum, r) => sum + Math.max(...r.trials.map((t) => t.seconds)), 0)
      let doneAt = startAt + Math.ceil(busy / 4) * SECOND + 15 * SECOND
      if (last && scenario === "running") {
        startAt = st.created - 80 * SECOND
        doneAt = st.created + SEED_RUN_LEFT
        origin = { kind: "dashboard", by: pick(random, HANDLES) }
      }
      st.runs.push({
        id,
        suite_id: suite.id,
        origin,
        trials,
        queuedAt: startAt,
        startAt,
        doneAt,
        results,
      })
      if (i === b)
        suite.baseline = {
          run_id: id,
          approved_by: pick(random, HANDLES),
          approved_at: Math.min(doneAt + HOUR * (1 + random() * 18), seedNow - 30 * MINUTE),
        }
    }
  }

  // ----- reads

  const findSuite = (st: State, id: string) => {
    const suite = st.suites.find((s) => s.id === id)
    if (!suite) throw new NotFound(`Eval suite ${id}`)
    return suite
  }
  const findRun = (st: State, id: string) => {
    const run = st.runs.find((r) => r.id === id)
    if (!run) throw new NotFound(`Eval run ${id}`)
    return run
  }

  const statusOf = (run: RunRecord, now: number): EvalRunStatus =>
    now < run.startAt ? "queued" : now < run.doneAt ? "running" : "completed"

  /** The cases a run has settled so far: all once completed, a share while running. */
  function settled(run: RunRecord, now: number): CaseRun[] {
    const status = statusOf(run, now)
    if (status === "completed") return run.results
    if (status === "queued") return []
    const share = (now - run.startAt) / (run.doneAt - run.startAt)
    return run.results.slice(0, Math.floor(run.results.length * share))
  }

  function compare(run: RunRecord, other: RunRecord | null, now: number) {
    const mine = settled(run, now)
    const theirs = other ? settled(other, now) : []
    const changes = new Map<string, EvalCaseResult["change"]>()
    let improved = 0
    let regressed = 0
    for (const r of mine) {
      if (!other) {
        changes.set(r.case_id, "unchanged")
        continue
      }
      const o = theirs.find((x) => x.case_id === r.case_id)
      if (!o) {
        changes.set(r.case_id, "new")
        continue
      }
      const was = isPassing(outcomeOf(o, other.trials))
      const is = isPassing(outcomeOf(r, run.trials))
      const change = is && !was ? "improved" : !is && was ? "regressed" : "unchanged"
      if (change === "improved") improved++
      if (change === "regressed") regressed++
      changes.set(r.case_id, change)
    }
    return { changes, improved, regressed }
  }

  const baselineRun = (st: State, suite: SuiteRecord) =>
    suite.baseline ? (st.runs.find((r) => r.id === suite.baseline?.run_id) ?? null) : null

  function summary(st: State, run: RunRecord, against?: RunRecord | null): EvalRunSummary {
    const now = context.now()
    const suite = findSuite(st, run.suite_id)
    const status = statusOf(run, now)
    const done = settled(run, now)
    const cases = { total: run.results.length, passed: 0, failed: 0, flaky: 0 }
    for (const r of done) {
      const o = outcomeOf(r, run.trials)
      if (o === "passed") cases.passed++
      else if (o === "flaky") {
        cases.flaky++
        cases.passed++
      } else cases.failed++
    }
    const trials = done.flatMap((r) => r.trials)
    const other = against === undefined ? baselineRun(st, suite) : against
    const compared =
      other && other !== run && status === "completed" ? compare(run, other, now) : null
    const costs = trials.map((t) => t.usage.cost)
    return {
      id: run.id,
      suite_id: run.suite_id,
      status,
      origin: run.origin,
      started_at: iso(run.queuedAt),
      finished_at: status === "completed" ? iso(run.doneAt) : null,
      cases,
      score: mean(judgeScores(trials)),
      verdict: compared ? verdictOf(compared.improved, compared.regressed) : null,
      change: compared ? { improved: compared.improved, regressed: compared.regressed } : null,
      usage: {
        tokens: trials.reduce((sum, t) => sum + t.usage.tokens, 0),
        cost: costs.some((c) => c != null)
          ? round(
              costs.reduce<number>((sum, c) => sum + (c ?? 0), 0),
              4,
            )
          : null,
      },
      is_baseline: suite.baseline?.run_id === run.id,
    }
  }

  const runsOf = (st: State, suiteId: string) =>
    st.runs.filter((r) => r.suite_id === suiteId).sort((a, b) => b.queuedAt - a.queuedAt)

  const publicCase = (c: CaseRecord): EvalCase => ({
    id: c.id,
    name: c.name,
    input: c.input,
    checks: c.checks,
    source_session_id: c.source_session_id,
  })

  function suiteSummary(st: State, suite: SuiteRecord): EvalSuiteSummary {
    const base = baselineRun(st, suite)
    const last = runsOf(st, suite.id)[0]
    return {
      id: suite.id,
      name: suite.name,
      description: suite.description,
      specialist: suite.specialist,
      source: suite.source,
      case_count: suite.cases.length,
      trials: suite.trials,
      baseline:
        suite.baseline && base
          ? {
              run_id: suite.baseline.run_id,
              approved_by: suite.baseline.approved_by,
              approved_at: iso(suite.baseline.approved_at),
              cases: summary(st, base).cases,
            }
          : null,
      last_run: last ? summary(st, last) : null,
    }
  }

  function candidate(st: State, specialist: string) {
    const spec = st.seed.specialists.agents.find((a) => a.name === specialist)
    const sessions = appSessions(st.seed).filter((s) => s.session.agent_name === specialist)
    const envs = new Map<string, number>()
    for (const s of sessions)
      if (s.session.environment_name)
        envs.set(s.session.environment_name, (envs.get(s.session.environment_name) ?? 0) + 1)
    return {
      provider: spec?.provider_name ?? sessions[0]?.session.provider_name ?? null,
      model: spec?.model ?? sessions[0]?.session.model ?? "unknown",
      environment: [...envs].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null,
    }
  }

  // ----- writes

  function queueRun(st: State, suite: SuiteRecord, origin: EvalRunOrigin, grades: Grade[]) {
    const t0 = context.now()
    const random = seeded(`${st.seed.id}:${suite.id}:new:${st.made}`)
    const run: RunRecord = {
      id: `evr_new_${++st.made}_${hex(random, 6)}`,
      suite_id: suite.id,
      origin,
      trials: suite.trials,
      queuedAt: t0,
      startAt: t0 + QUEUED_FOR,
      doneAt: t0 + DONE_AFTER,
      results: suite.cases.map((c, i) =>
        caseRun(st, suite, c, grades[i] ?? "pass", suite.trials, random),
      ),
    }
    st.runs.push(run)
    return run
  }

  const isDone = (run: RunRecord) => statusOf(run, context.now()) === "completed"

  function captureCase(suite: SuiteRecord, s: SeedSession): CaseRecord {
    const existing = suite.cases.find((c) => c.source_session_id === s.session.id)
    if (existing) return existing
    const first = firstUserText(s)
    const name = s.overview.title ?? (first ? clip(first, 70) : `Session ${s.session.id}`)
    const c = makeCase(`${suite.id}-case-${suite.cases.length + 1}`, name, first || name, {
      completes: true,
      judge: "answers as well as the original session did",
      source: s.session.id,
    })
    suite.cases.push(c)
    return c
  }

  return {
    evalSuites(): EvalSuiteSummary[] {
      const st = state()
      return st.suites.map((s) => suiteSummary(st, s))
    },

    evalSuite(id: string): EvalSuiteDetail {
      const st = state()
      const suite = findSuite(st, id)
      const runs = runsOf(st, id)
      const now = context.now()
      const completed = runs.filter((r) => statusOf(r, now) === "completed")
      return {
        ...suiteSummary(st, suite),
        cases: suite.cases.map((c) => {
          const run = completed.find((r) => r.results.some((x) => x.case_id === c.id))
          const r = run?.results.find((x) => x.case_id === c.id)
          return {
            ...publicCase(c),
            last:
              run && r
                ? {
                    outcome: outcomeOf(r, run.trials),
                    trials_passed: passes(r),
                    trials: r.trials.length,
                  }
                : null,
          }
        }),
        runs: runs.map((r) => summary(st, r)),
      }
    },

    evalRun(id: string, compareWith?: string): EvalRunDetail {
      const st = state()
      const run = findRun(st, id)
      const suite = findSuite(st, run.suite_id)
      let other: RunRecord | null
      if (compareWith) {
        other = findRun(st, compareWith)
        if (other.suite_id !== run.suite_id)
          throw new Error(`Run ${compareWith} belongs to another suite.`)
      } else other = baselineRun(st, suite)
      if (other === run) other = null
      const now = context.now()
      const { changes } = compare(run, other, now)
      const theirs = other ? settled(other, now) : []
      return {
        ...summary(st, run, other),
        suite: { id: suite.id, name: suite.name, specialist: suite.specialist },
        compared_with: other ? summary(st, other) : null,
        candidate: candidate(st, suite.specialist),
        results: settled(run, now).flatMap((r): EvalCaseResult[] => {
          const c = suite.cases.find((x) => x.id === r.case_id)
          if (!c) return []
          const o = theirs.find((x) => x.case_id === r.case_id)
          return [
            {
              case: publicCase(c),
              outcome: outcomeOf(r, run.trials),
              trials: r.trials,
              score: mean(judgeScores(r.trials)),
              compared:
                o && other
                  ? {
                      outcome: outcomeOf(o, other.trials),
                      trials_passed: passes(o),
                      output: (o.trials.find((t) => t.passed) ?? o.trials[0])?.output ?? "",
                    }
                  : null,
              change: changes.get(r.case_id) ?? "unchanged",
            },
          ]
        }),
      }
    },

    /** Mostly what the baseline did, with one case flipped either way. */
    runEvalSuite(suiteId: string): { run_id: string } {
      const st = state()
      const suite = findSuite(st, suiteId)
      const ref =
        baselineRun(st, suite) ??
        runsOf(st, suite.id).find((r) => isDone(r) && r.results.length === suite.cases.length) ??
        null
      const random = seeded(`${st.seed.id}:${suite.id}:flip:${st.made}`)
      const grades = suite.cases.map((c): Grade => {
        const r = ref?.results.find((x) => x.case_id === c.id)
        return r && ref ? gradeOf(r, ref.trials) : "pass"
      })
      // A suite's first run (no reference) only gets a failure once it has a few cases.
      if (ref || grades.length >= 3) {
        const i = Math.floor(random() * grades.length)
        grades[i] = grades[i] === "pass" || grades[i] === "flaky" ? "fail" : "pass"
      }
      const run = queueRun(st, suite, { kind: "dashboard", by: "you" }, grades)
      return { run_id: run.id }
    },

    approveBaseline(runId: string): void {
      const st = state()
      const run = findRun(st, runId)
      if (statusOf(run, context.now()) !== "completed")
        throw new Error("Only a completed run can be the baseline.")
      const suite = findSuite(st, run.suite_id)
      suite.baseline = { run_id: run.id, approved_by: "you", approved_at: context.now() }
    },

    createEvalSuite(body: NewEvalSuite): { suite_id: string } {
      const st = state()
      if (!st.seed.specialists.agents.some((a) => a.name === body.specialist))
        throw new NotFound(`Specialist ${body.specialist}`)
      const { per_case, min_passes } = body.trials
      if (per_case < 1 || per_case > MAX_TRIALS || min_passes < 1 || min_passes > per_case)
        throw new Error(`Trials must be 1 to ${MAX_TRIALS}, needing at least 1 and at most all.`)
      if (!body.name.trim() || !body.cases.length)
        throw new Error("A suite needs a name and at least one case.")
      const id = `suite-${slug(body.name) || "untitled"}-${++st.made}`
      st.suites.unshift({
        id,
        name: body.name.trim(),
        description: null,
        specialist: body.specialist,
        source: "authored",
        trials: { per_case, min_passes },
        cases: body.cases.map((c, i) =>
          makeCase(`${id}-case-${i + 1}`, c.name, c.input, { needle: c.expect_contains.trim() }),
        ),
        baseline: null,
      })
      return { suite_id: id }
    },

    addToEvals(body: AddToEvals): { suite_id: string; run_id: string | null } {
      const st = state()
      const s = st.seed.sessions.find((x) => x.session.id === body.session_id)
      if (!s) throw new NotFound(`Session ${body.session_id}`)
      const finishedStates = ["completed", "failed", "interrupted"]
      if (body.mode === "score" && !finishedStates.includes(s.overview.state))
        throw new Error("Only a finished session can be scored.")
      let suite: SuiteRecord
      if ("id" in body.suite) {
        suite = findSuite(st, body.suite.id)
        if (suite.specialist !== s.session.agent_name)
          throw new Error(`${suite.name} is for ${suite.specialist}, not ${s.session.agent_name}.`)
      } else {
        const name = body.suite.name.trim() || `${label(s.session.agent_name)}: captured sessions`
        suite = {
          id: `suite-${slug(name) || "captured"}-${++st.made}`,
          name,
          description: "Captured from sessions in the dashboard.",
          specialist: s.session.agent_name,
          source: "captured",
          trials: { per_case: 1, min_passes: 1 },
          cases: [],
          baseline: null,
        }
        st.suites.unshift(suite)
      }
      const c = captureCase(suite, s)
      if (body.mode === "rerun") return { suite_id: suite.id, run_id: null }
      const t = context.now()
      const run: RunRecord = {
        id: `evr_new_${++st.made}_${hex(seeded(`${s.session.id}:${st.made}`), 6)}`,
        suite_id: suite.id,
        origin: { kind: "session", session_id: s.session.id },
        trials: { per_case: 1, min_passes: 1 },
        queuedAt: t,
        startAt: t,
        doneAt: t,
        results: [scoreSession(c, s, seeded(`score:${s.session.id}`))],
      }
      st.runs.push(run)
      return { suite_id: suite.id, run_id: run.id }
    },

    evalSettings(): EvalSettings {
      const st = state()
      const candidates = st.seed.specialists.agents.map((a) => ({
        specialist: a.name,
        ...candidate(st, a.name),
      }))
      const main = candidates[0]?.provider ?? null
      const ready = (name: string) => ({ name, ready: true, note: null })
      return {
        candidate: candidates,
        judge:
          main === "anthropic"
            ? { provider: "openai", model: "gpt-6-mini", same_model_allowed: false }
            : { provider: "anthropic", model: "claude-sonnet-5-5", same_model_allowed: false },
        limits: { max_trials: MAX_TRIALS, max_concurrency: 4, timeout_seconds: 600, max_steps: 64 },
        features: [
          ready("Score a finished session"),
          ready("Run suites on the current agent"),
          ready("Compare runs"),
          ready("Download reports"),
          ready("Production scenarios"),
          ready("Memory reports"),
          {
            name: "Judge calibration",
            ready: false,
            note: "Needs a labelled calibration set; none is configured on this deployment.",
          },
          ready("Import corpora"),
          ready("Cancel runs"),
        ],
      }
    },
  }
}

export type EvalStore = ReturnType<typeof createEvalStore>
