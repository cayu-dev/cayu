import type { EvalAuthoredSuiteRunLaunchRequest } from "./generated/server-api/index.ts"

export type SuiteLaunchAttempt = {
  key: string
  revision: string
  request: EvalAuthoredSuiteRunLaunchRequest
}
type AttemptStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">
const storageKey = (api: string, revision: string) =>
  `cayu.suite-launch.v1:${JSON.stringify([api, revision])}`
const isRevision = (value: unknown): value is string =>
  typeof value === "string" && /^sha256:[0-9a-f]{64}$/.test(value)
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) &&
  value.length > 0 &&
  value.length <= 1000 &&
  value.every((item) => typeof item === "string" && item.length > 0 && item.length <= 256) &&
  new Set(value).size === value.length

/** No draft or evidence: retain only the exact reviewed admission envelope. */
export function saveSuiteLaunchAttempt(
  storage: AttemptStorage,
  api: string,
  attempt: SuiteLaunchAttempt,
) {
  const key = storageKey(api, attempt.revision)
  const text = JSON.stringify(attempt)
  const existing = storage.getItem(key)
  if (existing !== null && existing !== text)
    throw new Error("Resolve the previous launch before starting another one.")
  storage.setItem(key, text)
}

export function readSuiteLaunchAttempt(
  storage: AttemptStorage,
  api: string,
  revision: string,
): SuiteLaunchAttempt | null {
  const text = storage.getItem(storageKey(api, revision))
  if (text === null) return null
  try {
    if (text.length > 1024 * 1024) throw new Error("oversized")
    const value = JSON.parse(text)
    const request = value.request
    if (
      !value ||
      value.revision !== revision ||
      !isRevision(revision) ||
      typeof value.key !== "string" ||
      !/^cayu-dashboard-[0-9a-f-]{36}$/.test(value.key) ||
      !request ||
      !isRevision(request.expected_exposure_revision) ||
      (request.case_ids != null && !strings(request.case_ids)) ||
      !Array.isArray(request.expected_execution_profiles) ||
      request.expected_execution_profiles.length < 1 ||
      request.expected_execution_profiles.length > 1000 ||
      !request.expected_execution_profiles.every(
        (item: { case_ids: unknown; execution_profile_revision: unknown }) =>
          item && strings(item.case_ids) && isRevision(item.execution_profile_revision),
      ) ||
      (request.cost_budget != null &&
        (typeof request.cost_budget !== "object" ||
          typeof request.cost_budget.max_estimated_cost !== "string" ||
          !/^(0|[1-9]\d*)(\.\d+)?$/.test(request.cost_budget.max_estimated_cost) ||
          request.cost_budget.max_estimated_cost.length > 64 ||
          typeof request.cost_budget.currency !== "string" ||
          !/^[A-Z][A-Z0-9._-]{0,15}$/.test(request.cost_budget.currency)))
    )
      throw new Error("invalid")
    return value as SuiteLaunchAttempt
  } catch {
    throw new Error(
      "The unresolved launch cannot be read. Keep this tab open and contact your operator before starting another run.",
    )
  }
}

/** A late callback may only clear its own complete operation. */
export function clearSuiteLaunchAttempt(
  storage: AttemptStorage,
  api: string,
  expected: SuiteLaunchAttempt,
): boolean {
  const key = storageKey(api, expected.revision)
  if (storage.getItem(key) !== JSON.stringify(expected)) return false
  storage.removeItem(key)
  return true
}
