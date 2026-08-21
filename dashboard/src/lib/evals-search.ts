import type { EvalStatus } from "./api.ts"

export type EvalsTab = "catalog" | "runs"

export type EvalsSearch = {
  tab?: EvalsTab
  target?: string
  corpus?: string
  suite?: string
  run?: string
  baseline?: string
  status?: EvalStatus
  corpora_cursor?: string
  suites_cursor?: string
  cases_cursor?: string
  runs_cursor?: string
}

const EVAL_STATUSES = new Set<EvalStatus>([
  "queued",
  "running",
  "cancelling",
  "completed",
  "failed",
  "cancelled",
])
const EVAL_CURSOR_MAX_BYTES = 1_024
const EVAL_CURSOR_RE = /^[A-Za-z0-9_-]+$/
const EVAL_CORPUS_REVISION_RE = /^sha256:[0-9a-f]{64}$/
const EVAL_PORTABLE_ID_RE = /^[a-z][a-z0-9._-]{0,127}$/
const EVAL_RUN_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/

function matchingSearchValue(value: unknown, pattern: RegExp): string | undefined {
  if (typeof value !== "string") return undefined
  const normalized = value.trim()
  return pattern.test(normalized) ? normalized : undefined
}

function boundedCursorValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined
  const normalized = value.trim()
  return normalized.length > 0 &&
    normalized.length <= EVAL_CURSOR_MAX_BYTES &&
    EVAL_CURSOR_RE.test(normalized)
    ? normalized
    : undefined
}

export function validateEvalsSearch(search: Record<string, unknown>): EvalsSearch {
  const tab = search.tab === "runs" ? "runs" : search.tab === "catalog" ? "catalog" : undefined
  const status = typeof search.status === "string" ? search.status.trim() : undefined
  const target = matchingSearchValue(search.target, EVAL_PORTABLE_ID_RE)
  const corpus = matchingSearchValue(search.corpus, EVAL_CORPUS_REVISION_RE)
  const suite = matchingSearchValue(search.suite, EVAL_PORTABLE_ID_RE)
  const run = matchingSearchValue(search.run, EVAL_RUN_ID_RE)
  const baseline = matchingSearchValue(search.baseline, EVAL_RUN_ID_RE)
  const corporaCursor = boundedCursorValue(search.corpora_cursor)
  const suitesCursor = boundedCursorValue(search.suites_cursor)
  const casesCursor = boundedCursorValue(search.cases_cursor)
  const runsCursor = boundedCursorValue(search.runs_cursor)
  return {
    ...(tab ? { tab } : {}),
    ...(target ? { target } : {}),
    ...(corpus ? { corpus } : {}),
    ...(suite ? { suite } : {}),
    ...(run ? { run } : {}),
    ...(baseline ? { baseline } : {}),
    ...(status && EVAL_STATUSES.has(status as EvalStatus) ? { status: status as EvalStatus } : {}),
    ...(corporaCursor ? { corpora_cursor: corporaCursor } : {}),
    ...(suitesCursor ? { suites_cursor: suitesCursor } : {}),
    ...(casesCursor ? { cases_cursor: casesCursor } : {}),
    ...(runsCursor ? { runs_cursor: runsCursor } : {}),
  }
}

export function evalRunIdIsValid(value: string): boolean {
  return EVAL_RUN_ID_RE.test(value)
}

export function evalsSearchWithout(
  search: EvalsSearch,
  ...keys: Array<keyof EvalsSearch>
): EvalsSearch {
  const next = { ...search }
  for (const key of keys) delete next[key]
  return next
}
