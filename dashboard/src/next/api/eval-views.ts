/** Contract 65 suite-centred reads. Preserve server authority and unavailable states. */

import type {
  EvalRunViewDetail,
  EvalSuiteViewDetail,
  EvalViewPageEvalSuiteViewSummary,
} from "@/lib/generated/server-api"
import { isApiPayloadTooLarge, queryString, requestJson } from "../../lib/api.ts"
export type EvalCaseFilter = "regressed" | "improved" | "failing" | "flaky"

const VIEW_BYTE_LIMIT = 8 * 1024 * 1024

/** Page sizes are not cursor authority; selectors and every query scope stay fixed. */
async function requestViewPage<T>(
  path: string,
  query: Record<string, string | undefined>,
  initialLimits: Record<string, number>,
  signal?: AbortSignal,
): Promise<T> {
  const scope = { ...query }
  let limits = { ...initialLimits }
  for (;;) {
    try {
      return await requestJson<T>(
        `${path}${queryString({ ...scope, ...limits, max_result_bytes: VIEW_BYTE_LIMIT })}`,
        { signal },
      )
    } catch (error) {
      if (
        signal?.aborted ||
        !isApiPayloadTooLarge(error) ||
        Object.values(limits).every((limit) => limit === 1)
      )
        throw error
      // Suite details include two independent pages in the same byte envelope.
      // Reduce both so either cases or run history can reach its minimum size.
      limits = Object.fromEntries(
        Object.entries(limits).map(([key, limit]) => [key, Math.max(1, Math.floor(limit / 2))]),
      )
    }
  }
}

export function fetchSuiteViews(
  query: { target_key?: string; cursor?: string } = {},
  signal?: AbortSignal,
) {
  return requestViewPage<EvalViewPageEvalSuiteViewSummary>(
    "/evals/suite-views",
    query,
    { limit: 25 },
    signal,
  )
}
export function fetchSuiteView(
  id: string,
  query: { cases_cursor?: string; runs_cursor?: string } = {},
  signal?: AbortSignal,
) {
  return requestViewPage<EvalSuiteViewDetail>(
    `/evals/suite-views/${encodeURIComponent(id)}`,
    query,
    { cases_limit: 25, runs_limit: 20 },
    signal,
  )
}
export function fetchRunView(
  id: string,
  query: {
    compared_with?: string
    placement?: string
    cases_cursor?: string
    case_filter?: EvalCaseFilter
  } = {},
  signal?: AbortSignal,
) {
  return requestViewPage<EvalRunViewDetail>(
    `/evals/run-views/${encodeURIComponent(id)}`,
    query,
    { cases_limit: 25 },
    signal,
  )
}

/** Old corpus links omitted the target. Resolve it from the authoritative catalog. */
export async function resolveLegacyEvalSuite(
  corpus: string,
  suite: string,
  target: string | undefined,
  signal?: AbortSignal,
): Promise<string> {
  let cursor: string | undefined
  const seen = new Set<string>()
  do {
    const page = await fetchSuiteViews({ target_key: target, cursor }, signal)
    const match = page.items.find(
      (item) =>
        item.identity.source === "corpus" &&
        item.identity.revision === corpus &&
        item.identity.suite_id === suite,
    )
    if (match) return match.id
    if (!page.has_more || !page.next_cursor) break
    if (seen.has(page.next_cursor)) throw new Error("The suite catalog returned a repeated cursor.")
    seen.add(page.next_cursor)
    cursor = page.next_cursor
  } while (!signal?.aborted)
  throw new Error("The linked suite is not available in this deployment.")
}
