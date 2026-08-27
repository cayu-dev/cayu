import type { CapturedEvaluationLaunch, EvalRun, EvalStatus } from "./api.ts"
import { formatCurrencyWithCode } from "./format.ts"
import type {
  CorpusComparisonReason,
  PublishedAssertionResult,
} from "./generated/server-api/index.ts"

export const MAX_EVAL_CORPUS_FILE_BYTES = 8 * 1024 * 1024
export const EVAL_TARGET_QUERY_KEY = ["evals", "targets"] as const
export const EVAL_TARGET_STALE_TIME_MS = 15_000
const EVAL_LAUNCH_REGISTRY_KEY_PREFIX = "cayu.eval-launch-idempotency.v1:"
const EVAL_LAUNCH_REGISTRY_MAX_ENTRIES = 32
const EVAL_LAUNCH_IDENTITY_MAX_CHARS = 1_024
const EVAL_LAUNCH_REGISTRY_MAX_CHARS = 64 * 1_024
const EVAL_LAUNCH_API_SCOPE_MAX_CHARS = 2_048
const EVAL_LAUNCH_KEY_RE =
  /^cayu-dashboard-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const AMBIGUOUS_EVAL_LAUNCH_FAILURE_STATUSES = new Set([408, 425, 429, 499])

const ACTIVE_EVAL_STATUSES = new Set<EvalStatus>(["queued", "running", "cancelling"])

export const EVAL_RESULT_QUERY_RETENTION = Object.freeze({
  gcTime: 0,
  staleTime: Number.POSITIVE_INFINITY,
})

export function evalRunIsActive(run: EvalRun | undefined): boolean {
  return run !== undefined && ACTIVE_EVAL_STATUSES.has(run.status)
}

export function evalRunCanCancel(run: EvalRun | undefined): boolean {
  return run !== undefined && (run.status === "queued" || run.status === "running")
}

export function evalRunHasResult(run: EvalRun | undefined): boolean {
  return run?.status === "completed" && run.result !== null && run.result !== undefined
}

export function evalCancellationNotice(run: EvalRun): string {
  const identity = shortEvalIdentity(run.spec.run_id)
  if (run.status === "cancelled") return `Eval run ${identity} is cancelled.`
  if (run.status === "completed") {
    return `Eval run ${identity} completed before cancellation took effect.`
  }
  if (run.status === "failed") {
    return `Eval run ${identity} failed before cancellation took effect.`
  }
  return `Cancellation requested for ${identity}.`
}

export function evalLaunchNotice(run: EvalRun): string {
  return `Opened eval run ${shortEvalIdentity(run.spec.run_id)} (${run.status}).`
}

export function shortEvalIdentity(value: string, retained = 12): string {
  const separator = value.indexOf(":")
  const digest = separator >= 0 ? value.slice(separator + 1) : value
  return digest.length <= retained ? digest : `${digest.slice(0, retained)}…`
}

export function createEvalIdempotencyKey(): string {
  const crypto = globalThis.crypto
  if (typeof crypto?.randomUUID !== "function") {
    throw new Error("Secure browser randomness is unavailable; the eval run was not submitted.")
  }
  return `cayu-dashboard-${crypto.randomUUID()}`
}

export function evalLaunchFailureIsDefinitive(status: number): boolean {
  return (
    (status >= 400 && status < 500 && !AMBIGUOUS_EVAL_LAUNCH_FAILURE_STATUSES.has(status)) ||
    status === 501
  )
}

export function evalTargetCatalogMayBeStale(error: unknown): boolean {
  return typeof error === "object" && error !== null && "status" in error && error.status === 409
}

export function retryEvalQuery(failureCount: number, error: Error): boolean {
  const status =
    "status" in error && typeof error.status === "number" && Number.isInteger(error.status)
      ? error.status
      : null
  return !(status !== null && evalLaunchFailureIsDefinitive(status)) && failureCount < 3
}

type EvalLaunchStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">

type EvalLaunchRegistryDocument = {
  version: 1
  entries: Array<[string, string]>
}

export class EvalLaunchIdempotencyRegistry {
  readonly #storage: EvalLaunchStorage
  readonly #storageKey: string

  constructor(storage: EvalLaunchStorage, apiScope: string) {
    if (
      apiScope.length === 0 ||
      apiScope.length > EVAL_LAUNCH_API_SCOPE_MAX_CHARS ||
      apiScope.trim() !== apiScope ||
      [...apiScope].some((character) => {
        const codePoint = character.codePointAt(0) ?? 0
        return codePoint <= 31 || codePoint === 127
      })
    ) {
      throw new Error("The eval launch API scope is outside the browser retry-state limit.")
    }
    this.#storage = storage
    this.#storageKey = `${EVAL_LAUNCH_REGISTRY_KEY_PREFIX}${encodeURIComponent(apiScope)}`
  }

  keyFor(requestIdentity: string): string {
    if (requestIdentity.length === 0 || requestIdentity.length > EVAL_LAUNCH_IDENTITY_MAX_CHARS) {
      throw new Error("The eval launch identity is outside the browser retry-state limit.")
    }
    const document = this.#read()
    const retained = document.entries.find(([identity]) => identity === requestIdentity)
    if (retained) return retained[1]
    if (document.entries.length >= EVAL_LAUNCH_REGISTRY_MAX_ENTRIES) {
      throw new Error(
        "Too many eval launches have unresolved responses in this tab; retry or reconcile them before starting another run.",
      )
    }
    const key = createEvalIdempotencyKey()
    document.entries.push([requestIdentity, key])
    this.#write(document)
    return key
  }

  resolve(requestIdentity: string): void {
    const document = this.#read()
    const entries = document.entries.filter(([identity]) => identity !== requestIdentity)
    if (entries.length === document.entries.length) return
    if (entries.length === 0) {
      this.#remove()
    } else {
      this.#write({ version: 1, entries })
    }
  }

  #read(): EvalLaunchRegistryDocument {
    let raw: string | null
    try {
      raw = this.#storage.getItem(this.#storageKey)
    } catch {
      throw new Error("Eval launch retry state is unavailable in this browser.")
    }
    if (raw === null) return { version: 1, entries: [] }
    if (raw.length > EVAL_LAUNCH_REGISTRY_MAX_CHARS) {
      throw new Error("Stored eval launch retry state exceeds its safety limit.")
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      throw new Error("Stored eval launch retry state is invalid.")
    }
    if (!isRecord(parsed) || parsed.version !== 1 || !Array.isArray(parsed.entries)) {
      throw new Error("Stored eval launch retry state is invalid.")
    }
    if (parsed.entries.length > EVAL_LAUNCH_REGISTRY_MAX_ENTRIES) {
      throw new Error("Stored eval launch retry state exceeds its entry limit.")
    }
    const entries: Array<[string, string]> = []
    const identities = new Set<string>()
    const keys = new Set<string>()
    for (const entry of parsed.entries) {
      if (
        !Array.isArray(entry) ||
        entry.length !== 2 ||
        typeof entry[0] !== "string" ||
        entry[0].length === 0 ||
        entry[0].length > EVAL_LAUNCH_IDENTITY_MAX_CHARS ||
        typeof entry[1] !== "string" ||
        !EVAL_LAUNCH_KEY_RE.test(entry[1]) ||
        identities.has(entry[0]) ||
        keys.has(entry[1])
      ) {
        throw new Error("Stored eval launch retry state is invalid.")
      }
      identities.add(entry[0])
      keys.add(entry[1])
      entries.push([entry[0], entry[1]])
    }
    return { version: 1, entries }
  }

  #write(document: EvalLaunchRegistryDocument): void {
    const serialized = JSON.stringify(document)
    if (serialized.length > EVAL_LAUNCH_REGISTRY_MAX_CHARS) {
      throw new Error("Eval launch retry state exceeds its browser safety limit.")
    }
    try {
      this.#storage.setItem(this.#storageKey, serialized)
    } catch {
      throw new Error("Eval launch retry state could not be persisted in this browser.")
    }
  }

  #remove(): void {
    try {
      this.#storage.removeItem(this.#storageKey)
    } catch {
      throw new Error("Completed eval launch retry state could not be cleared.")
    }
  }
}

export function evalLaunchRequestIdentity(
  corpusRevision: string,
  suiteId: string,
  maxConcurrency: number,
  executionProfileRevision: string,
): string {
  return JSON.stringify([
    "corpus-v2",
    corpusRevision,
    suiteId,
    maxConcurrency,
    executionProfileRevision,
  ])
}

export function scenarioEvalLaunchRequestIdentity(
  scenarioRevision: string,
  bindingRevision: string,
  executionProfileRevision: string,
): string {
  return JSON.stringify([
    "scenario-v2",
    scenarioRevision,
    bindingRevision,
    executionProfileRevision,
  ])
}

export function authoredSuiteEvalLaunchRequestIdentity(
  suiteRevision: string,
  selectionRevision: string,
  executionProfiles: ReadonlyArray<{
    case_ids: ReadonlyArray<string>
    execution_profile_revision: string
  }>,
): string {
  return JSON.stringify(["authored-suite-v1", suiteRevision, selectionRevision, executionProfiles])
}

export function capturedEvalLaunchRequestIdentity(
  sessionId: string,
  candidateRevision: string,
  request: Omit<CapturedEvaluationLaunch, "candidate" | "expected_candidate_revision">,
): string {
  return JSON.stringify([
    "captured-v1",
    sessionId,
    candidateRevision,
    request.expected_execution_profile_revision,
    request.trial_request?.trials ?? 1,
    request.trial_request?.timeout_seconds ?? 300,
    request.max_concurrency ?? 1,
    request.max_steps ?? null,
    request.limits?.max_input_tokens ?? null,
    request.limits?.max_output_tokens ?? null,
    request.limits?.max_total_tokens ?? null,
    request.limits?.max_tool_calls ?? null,
    request.limits?.max_elapsed_seconds ?? null,
    request.limits?.scope ?? null,
    request.cost_budget?.max_estimated_cost ?? null,
    request.cost_budget?.currency ?? null,
  ])
}

export type EvalTrialCostSummary = Readonly<{
  display: string
  exact: string
}>

export function evalTrialCostSummary(
  assertions: Array<PublishedAssertionResult>,
  locales?: Intl.LocalesArgument,
): EvalTrialCostSummary {
  for (const assertion of assertions) {
    const detail = assertion.detail
    if (detail.kind !== "max_estimated_cost") continue
    if (detail.estimated_cost !== null && detail.estimated_cost !== undefined) {
      return {
        display: formatCurrencyWithCode(detail.estimated_cost, detail.currency, locales),
        exact: `${detail.estimated_cost} ${detail.currency}`,
      }
    }
    if (detail.unpriced_model_steps) {
      const unavailable = `unavailable · ${detail.unpriced_model_steps} unpriced model steps`
      return { display: unavailable, exact: unavailable }
    }
    return { display: "unavailable", exact: "unavailable" }
  }
  return { display: "not evaluated", exact: "not evaluated" }
}

const COMPARISON_REASON_TEXT: Record<CorpusComparisonReason, string> = {
  target_key_mismatch: "The runs target different attached application keys.",
  corpus_revision_mismatch: "The runs use different corpus revisions.",
  suite_id_mismatch: "The runs execute different suites.",
  suite_revision_mismatch: "The suite contract changed between runs.",
  evidence_policy_revision_mismatch: "The evidence policy changed between runs.",
  pricing_profile_fingerprint_mismatch: "The applicable pricing contract changed between runs.",
  case_contract_mismatch: "At least one case contract changed between runs.",
  assertion_contract_mismatch: "At least one assertion contract changed between runs.",
}

export function evalComparisonReasonText(reason: CorpusComparisonReason): string {
  return COMPARISON_REASON_TEXT[reason]
}

/**
 * Bound browser work and reject obvious non-corpus input before upload.
 * Canonical validation owns the original Blob bytes at the server boundary.
 */
export async function preflightEvalCorpusFile(file: Blob): Promise<void> {
  if (file.size > MAX_EVAL_CORPUS_FILE_BYTES) {
    throw new Error("The eval corpus is larger than the supported 8 MiB limit.")
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(await file.text())
  } catch {
    throw new Error("The selected file is not valid JSON.")
  }

  if (!isRecord(parsed) || parsed.schema_version !== 2) {
    throw new Error("The selected file is not a Cayu eval corpus v2 document.")
  }
  if (
    typeof parsed.revision !== "string" ||
    typeof parsed.target_key !== "string" ||
    !Array.isArray(parsed.suites) ||
    !Array.isArray(parsed.cases) ||
    !isRecord(parsed.evidence_policy)
  ) {
    throw new Error("The selected eval corpus is missing required fields.")
  }
}

export function evalErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}
