/** Contract 67 suite-centred reads. Preserve server authority and unavailable states. */

import type {
  EvalReportedRunDetail,
  EvalReportedSuiteDetail,
  EvalRunViewDetail,
  EvalSuiteViewDetail,
  EvalSuiteViewIdentity,
  EvalViewPageUnionEvalSuiteViewSummaryEvalReportedSuiteView,
} from "@/lib/generated/server-api"
import { isApiPayloadTooLarge, queryString, requestJson } from "../../lib/api.ts"
import { suiteSelector } from "../app/eval-view-model.ts"
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
  return requestViewPage<EvalViewPageUnionEvalSuiteViewSummaryEvalReportedSuiteView>(
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
  return requestViewPage<EvalSuiteViewDetail | EvalReportedSuiteDetail>(
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
  return requestViewPage<EvalRunViewDetail | EvalReportedRunDetail>(
    `/evals/run-views/${encodeURIComponent(id)}`,
    query,
    { cases_limit: 25 },
    signal,
  )
}

/** Corpus links may omit the target; catalog visibility is not existence. */
export async function resolveLegacyEvalSuite(
  corpus: string,
  suite: string,
  target: string | undefined,
  signal?: AbortSignal,
): Promise<string> {
  const identity = await requestJson<EvalSuiteViewIdentity>(
    `/evals/suite-views/resolve-corpus${queryString({ corpus, suite, target })}`,
    { signal },
  )
  return suiteSelector(identity)
}
