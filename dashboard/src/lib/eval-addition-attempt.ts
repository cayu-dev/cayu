import type { EvalAdditionCommitRequest, EvalAdditionPlan } from "./generated/server-api/index.ts"

export type EvalAdditionAttempt = {
  key: string
  request: EvalAdditionCommitRequest
  plan: EvalAdditionPlan
}
type AttemptStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">
const storageKey = (api: string, session: string) =>
  `cayu.eval-addition.v1:${JSON.stringify([api, session])}`
const revision = /^sha256:[0-9a-f]{64}$/

/** Retain the exact reviewed request before dispatch, including after acknowledgement loss. */
export function saveEvalAdditionAttempt(
  storage: AttemptStorage,
  api: string,
  session: string,
  attempt: EvalAdditionAttempt,
) {
  storage.setItem(storageKey(api, session), JSON.stringify(attempt))
}
/** Only the exact owner may settle or explicitly abandon a saved operation. */
export function clearEvalAdditionAttempt(
  storage: AttemptStorage,
  api: string,
  session: string,
  expected: EvalAdditionAttempt,
): boolean {
  const key = storageKey(api, session)
  if (storage.getItem(key) !== JSON.stringify(expected)) return false
  storage.removeItem(key)
  return true
}
export function readEvalAdditionAttempt(
  storage: AttemptStorage,
  api: string,
  session: string,
): EvalAdditionAttempt | null {
  const text = storage.getItem(storageKey(api, session))
  if (text === null) return null
  try {
    if (text.length > 8 * 1024 * 1024) throw new Error("oversized")
    const value = JSON.parse(text)
    if (
      !value ||
      typeof value !== "object" ||
      typeof value.key !== "string" ||
      !/^cayu-dashboard-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(
        value.key,
      ) ||
      !value.request ||
      !value.plan ||
      !revision.test(value.request.expected_plan_revision) ||
      !revision.test(value.request.expected_source_revision) ||
      value.request.expected_plan_revision !== value.plan.revision ||
      value.request.expected_source_revision !== value.plan.source_revision ||
      !["case", "score"].includes(value.request.mode) ||
      typeof value.request.case_id !== "string" ||
      typeof value.request.case_name !== "string" ||
      !["existing", "new"].includes(value.request.destination?.kind) ||
      value.plan.mode !== value.request.mode ||
      value.plan.case_id !== value.request.case_id ||
      value.plan.case_name !== value.request.case_name ||
      !revision.test(value.plan.source_binding_revision) ||
      !revision.test(value.plan.case_revision) ||
      value.plan.case_definition?.id !== value.plan.case_id ||
      value.plan.case_definition?.revision !== value.plan.case_revision ||
      !Number.isInteger(value.plan.destination_case_count) ||
      value.plan.destination_case_count < 1 ||
      typeof value.plan.membership_changed !== "boolean" ||
      !["authored", "corpus"].includes(value.plan.destination?.source) ||
      !revision.test(value.plan.destination?.revision) ||
      typeof value.plan.destination?.target_key !== "string" ||
      (value.plan.mode === "score"
        ? value.plan.execution_case_count !== 1 ||
          !revision.test(value.plan.captured_result_revision)
        : value.plan.execution_case_count !== null || value.plan.captured_result_revision !== null)
    )
      throw new Error("invalid")
    return value as EvalAdditionAttempt
  } catch {
    throw new Error(
      "The saved addition attempt cannot be read. Keep this browser tab open and contact your operator before submitting another addition.",
    )
  }
}
