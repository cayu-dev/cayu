import type { EvalCorpus, EvalRun, EvalStatus } from "./api.ts"
import type { PublishedAssertionResult } from "./generated/server-api/index.ts"

export const MAX_EVAL_CORPUS_FILE_BYTES = 8 * 1024 * 1024

const ACTIVE_EVAL_STATUSES = new Set<EvalStatus>(["queued", "running", "cancelling"])

export function evalRunIsActive(run: EvalRun | undefined): boolean {
  return run !== undefined && ACTIVE_EVAL_STATUSES.has(run.status)
}

export function evalRunCanCancel(run: EvalRun | undefined): boolean {
  return run !== undefined && (run.status === "queued" || run.status === "running")
}

export function evalRunHasResult(run: EvalRun | undefined): boolean {
  return run?.status === "completed" && run.result !== null && run.result !== undefined
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

export function evalTrialCostSummary(assertions: Array<PublishedAssertionResult>): string {
  for (const assertion of assertions) {
    const detail = assertion.detail
    if (detail.kind !== "max_estimated_cost") continue
    if (detail.estimated_cost !== null && detail.estimated_cost !== undefined) {
      return `${detail.estimated_cost} ${detail.currency}`
    }
    if (detail.unpriced_model_steps) {
      return `unavailable · ${detail.unpriced_model_steps} unpriced model steps`
    }
    return "unavailable"
  }
  return "not evaluated"
}

export async function parseEvalCorpusFile(file: Blob): Promise<EvalCorpus> {
  if (file.size > MAX_EVAL_CORPUS_FILE_BYTES) {
    throw new Error("The eval corpus is larger than the supported 8 MiB limit.")
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(await file.text())
  } catch {
    throw new Error("The selected file is not valid JSON.")
  }

  if (!isRecord(parsed) || parsed.schema_version !== 1) {
    throw new Error("The selected file is not a Cayu eval corpus v1 document.")
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
  return parsed as EvalCorpus
}

export function evalErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}
