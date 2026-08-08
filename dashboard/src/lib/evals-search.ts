import type { EvalStatus } from "./api.ts"

export type EvalsTab = "catalog" | "runs"

export type EvalsSearch = {
  tab?: EvalsTab
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

function boundedSearchValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined
  const normalized = value.trim()
  return normalized.length > 0 && normalized.length <= 512 ? normalized : undefined
}

export function validateEvalsSearch(search: Record<string, unknown>): EvalsSearch {
  const tab = search.tab === "runs" ? "runs" : search.tab === "catalog" ? "catalog" : undefined
  const status = boundedSearchValue(search.status)
  return {
    ...(tab ? { tab } : {}),
    ...(boundedSearchValue(search.corpus) ? { corpus: boundedSearchValue(search.corpus) } : {}),
    ...(boundedSearchValue(search.suite) ? { suite: boundedSearchValue(search.suite) } : {}),
    ...(boundedSearchValue(search.run) ? { run: boundedSearchValue(search.run) } : {}),
    ...(boundedSearchValue(search.baseline)
      ? { baseline: boundedSearchValue(search.baseline) }
      : {}),
    ...(status && EVAL_STATUSES.has(status as EvalStatus) ? { status: status as EvalStatus } : {}),
    ...(boundedSearchValue(search.corpora_cursor)
      ? { corpora_cursor: boundedSearchValue(search.corpora_cursor) }
      : {}),
    ...(boundedSearchValue(search.suites_cursor)
      ? { suites_cursor: boundedSearchValue(search.suites_cursor) }
      : {}),
    ...(boundedSearchValue(search.cases_cursor)
      ? { cases_cursor: boundedSearchValue(search.cases_cursor) }
      : {}),
    ...(boundedSearchValue(search.runs_cursor)
      ? { runs_cursor: boundedSearchValue(search.runs_cursor) }
      : {}),
  }
}

export function evalsSearchWithout(
  search: EvalsSearch,
  ...keys: Array<keyof EvalsSearch>
): EvalsSearch {
  const next = { ...search }
  for (const key of keys) delete next[key]
  return next
}
