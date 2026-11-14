/** Contract 67 suite-centred reads. Preserve server authority and unavailable states. */

import type {
  EvalReportedRunDetail,
  EvalReportedSuiteDetail,
  EvalRunViewDetail,
  EvalRunViewPollV1,
  EvalSuiteViewDetail,
  EvalSuiteViewIdentity,
  EvalViewPageUnionEvalSuiteViewSummaryEvalReportedSuiteView,
} from "@/lib/generated/server-api"
import { ApiClientError, isApiPayloadTooLarge, queryString, requestJson } from "../../lib/api.ts"
import { suiteSelector } from "../app/eval-view-model.ts"
export type EvalCaseFilter = "regressed" | "improved" | "failing" | "flaky"

const VIEW_BYTE_LIMIT = 8 * 1024 * 1024

/**
 * One request per view. A view over the byte budget is not retried with smaller
 * pages: each retry costs the server as much as the first. The page says so and
 * names the CLI command that reads it instead.
 */
async function requestViewPage<T>(
  path: string,
  query: Record<string, string | number | undefined>,
  command: string,
  signal?: AbortSignal,
): Promise<T> {
  try {
    return await requestJson<T>(
      `${path}${queryString({ ...query, max_result_bytes: VIEW_BYTE_LIMIT })}`,
      { signal },
    )
  } catch (error) {
    if (!isApiPayloadTooLarge(error)) throw error
    throw new ApiClientError(
      `This view is too large to show here. Read it with \`${command}\`.`,
      error.status,
      error.detail,
    )
  }
}

export function fetchSuiteViews(
  query: { target_key?: string; cursor?: string } = {},
  signal?: AbortSignal,
) {
  return requestViewPage<EvalViewPageUnionEvalSuiteViewSummaryEvalReportedSuiteView>(
    "/evals/suite-views",
    { ...query, limit: 25 },
    "cayu eval suites list",
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
    { ...query, cases_limit: 25, runs_limit: 20 },
    "cayu eval suites show SUITE",
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
    { ...query, cases_limit: 25 },
    "cayu eval runs show RUN_ID",
    signal,
  )
}

/**
 * What a live run page polls: one run row and a version of its view. The response
 * carries an ETag, so an unchanged run answers 304 and keeps the same data.
 */
export function fetchRunViewStatus(id: string, signal?: AbortSignal) {
  return requestJson<EvalRunViewPollV1>(`/evals/run-views/${encodeURIComponent(id)}/status`, {
    signal,
  })
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
