import type {
  EvaluationPromotionDraft,
  EvaluationPromotionPreviewResponse,
  PromotionCandidateV1,
} from "./generated/server-api"

export type PromotionAssertion = EvaluationPromotionDraft["case"]["assertions"][number]
export type PromotionAssertionKind = NonNullable<PromotionAssertion["kind"]>

export const PROMOTION_ASSERTION_KINDS = [
  "root_status",
  "child_status",
  "final_output_equals",
  "final_output_contains",
  "tool_called",
  "tools_called_in_order",
  "max_tool_calls",
  "max_model_steps",
  "usage_recorded",
  "max_total_tokens",
  "max_estimated_cost",
] as const satisfies readonly PromotionAssertionKind[]

const PORTABLE_ID_PATTERN = /^[a-z][a-z0-9._-]{0,127}$/
const SHA256_REVISION_PATTERN = /^sha256:[a-f0-9]{64}$/
const CURRENCY_PATTERN = /^[A-Z][A-Z0-9._-]{0,15}$/
const MAX_SAFE_COUNTER = Number.MAX_SAFE_INTEGER

export const PROMOTION_ASSERTION_LABELS: Record<PromotionAssertionKind, string> = {
  root_status: "Root status",
  child_status: "Child status count",
  final_output_equals: "Final output equals",
  final_output_contains: "Final output contains",
  tool_called: "Tool called",
  tools_called_in_order: "Tools called in order",
  max_tool_calls: "Maximum tool calls",
  max_model_steps: "Maximum model steps",
  usage_recorded: "Usage recorded",
  max_total_tokens: "Maximum total tokens",
  max_estimated_cost: "Maximum estimated cost",
}

export type PromotionDraftValidation =
  | { ok: true; draft: EvaluationPromotionDraft }
  | { ok: false; error: string }

export function promotionDraftFromCandidate(
  candidate: PromotionCandidateV1,
): EvaluationPromotionDraft {
  return structuredClone({
    expected_evidence_revision: candidate.evidence.revision,
    suite: {
      id: candidate.suite.id,
      name: candidate.suite.name,
      description: candidate.suite.description ?? null,
      trial_request: candidate.suite.trial_request ?? { trials: 1, timeout_seconds: 300 },
    },
    case: {
      id: candidate.case.id,
      suite_id: candidate.case.suite_id,
      name: candidate.case.name,
      description: candidate.case.description ?? null,
      input: candidate.case.input,
      assertions: candidate.case.assertions,
    },
  })
}

export function previewMatchesDraft(
  preview: EvaluationPromotionPreviewResponse,
  draft: EvaluationPromotionDraft,
): boolean {
  return JSON.stringify(promotionDraftFromCandidate(preview.candidate)) === JSON.stringify(draft)
}

export function validatePromotionDraft(draft: EvaluationPromotionDraft): PromotionDraftValidation {
  try {
    validateDraft(draft)
    return { ok: true, draft }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "The eval candidate is invalid.",
    }
  }
}

function validateDraft(draft: EvaluationPromotionDraft): void {
  if (!SHA256_REVISION_PATTERN.test(draft.expected_evidence_revision)) {
    throw new Error("Reload the promotion preview before editing this candidate.")
  }
  requirePortableId(draft.suite.id, "Suite ID")
  requireBoundedCleanText(draft.suite.name, "Suite name", 256)
  requireOptionalCleanText(draft.suite.description, "Suite description", 2_048)
  requireInteger(draft.suite.trial_request.trials ?? 1, "Trials", 1, 100)
  requireInteger(draft.suite.trial_request.timeout_seconds ?? 300, "Timeout", 1, 3_600)

  requirePortableId(draft.case.id, "Case ID")
  requirePortableId(draft.case.suite_id, "Case suite ID")
  if (draft.case.suite_id !== draft.suite.id) {
    throw new Error("Case suite ID must match the edited suite ID.")
  }
  requireBoundedCleanText(draft.case.name, "Case name", 256)
  requireOptionalCleanText(draft.case.description, "Case description", 2_048)
  const messages = draft.case.input.messages
  if (messages.length < 1 || messages.length > 16) {
    throw new Error("Eval input must contain between 1 and 16 user messages.")
  }
  let totalMessageChars = 0
  for (const [index, message] of messages.entries()) {
    if (message.role !== "user") {
      throw new Error(`Input message ${index + 1} must have the user role.`)
    }
    if (message.text.length > 65_536) {
      throw new Error(`Input message ${index + 1} cannot exceed 65,536 characters.`)
    }
    totalMessageChars += message.text.length
  }
  if (totalMessageChars > 262_144) {
    throw new Error("Eval input cannot exceed 262,144 total characters.")
  }

  const assertions = draft.case.assertions
  if (assertions.length < 1 || assertions.length > 64) {
    throw new Error("An eval case must contain between 1 and 64 assertions.")
  }
  const ids = new Set<string>()
  for (const assertion of assertions) {
    requirePortableId(assertion.id, "Assertion ID")
    if (ids.has(assertion.id)) throw new Error(`Assertion ID ${assertion.id} is duplicated.`)
    ids.add(assertion.id)
    requireOptionalCleanText(assertion.description, "Assertion description", 2_048)
    validateAssertion(assertion)
  }
}

function validateAssertion(assertion: PromotionAssertion): void {
  switch (assertion.kind) {
    case "root_status":
      requireTerminalStatus(assertion.expected)
      return
    case "child_status":
      requireTerminalStatus(assertion.expected)
      requireRange(assertion.min_count ?? 1, assertion.max_count, "Child count", 500)
      return
    case "final_output_equals":
      if (assertion.expected.length > 65_536) {
        throw new Error("Expected final output cannot exceed 65,536 characters.")
      }
      return
    case "final_output_contains":
      if (assertion.expected.length === 0 || assertion.expected.length > 65_536) {
        throw new Error("Expected final-output text must contain 1 to 65,536 characters.")
      }
      return
    case "tool_called":
      requireBoundedCleanText(assertion.tool_name, "Tool name", 256)
      requireRange(assertion.min_count ?? 1, assertion.max_count, "Tool count", 4_096)
      return
    case "tools_called_in_order":
      if (assertion.tool_names.length > 256) {
        throw new Error("Tool order cannot contain more than 256 names.")
      }
      for (const name of assertion.tool_names) {
        requireBoundedCleanText(name, "Tool name", 256)
      }
      return
    case "max_tool_calls":
      requireInteger(assertion.maximum, "Maximum tool calls", 0, 4_096)
      return
    case "max_model_steps":
      requireInteger(assertion.maximum, "Maximum model steps", 0, 4_096)
      return
    case "usage_recorded":
      requireInteger(assertion.min_total_tokens ?? 1, "Minimum total tokens", 0, MAX_SAFE_COUNTER)
      return
    case "max_total_tokens":
      requireInteger(assertion.maximum, "Maximum total tokens", 0, MAX_SAFE_COUNTER)
      return
    case "max_estimated_cost":
      if (!/^(0|[1-9]\d*)(\.\d+)?$/.test(assertion.maximum)) {
        throw new Error("Maximum estimated cost must be a canonical non-negative decimal.")
      }
      if (!CURRENCY_PATTERN.test(assertion.currency ?? "USD")) {
        throw new Error("Cost currency must be a portable uppercase identifier.")
      }
      return
    default:
      throw new Error(`Unsupported assertion kind: ${String(assertion.kind)}`)
  }
}

export function createPromotionAssertion(
  kind: PromotionAssertionKind,
  existing: readonly PromotionAssertion[],
): PromotionAssertion {
  const id = nextAssertionId(kind, existing)
  switch (kind) {
    case "root_status":
      return { id, kind, expected: "completed" }
    case "child_status":
      return { id, kind, expected: "completed", min_count: 1, max_count: null }
    case "final_output_equals":
      return { id, kind, expected: "" }
    case "final_output_contains":
      return { id, kind, expected: "expected text" }
    case "tool_called":
      return { id, kind, tool_name: "tool", min_count: 1, max_count: null }
    case "tools_called_in_order":
      return { id, kind, tool_names: ["tool"] }
    case "max_tool_calls":
      return { id, kind, maximum: 1 }
    case "max_model_steps":
      return { id, kind, maximum: 1 }
    case "usage_recorded":
      return { id, kind, min_total_tokens: 1 }
    case "max_total_tokens":
      return { id, kind, maximum: 1_000 }
    case "max_estimated_cost":
      return { id, kind, maximum: "1", currency: "USD" }
  }
}

function nextAssertionId(
  kind: PromotionAssertionKind,
  assertions: readonly PromotionAssertion[],
): string {
  const used = new Set(assertions.map((assertion) => assertion.id))
  if (!used.has(kind)) return kind
  for (let suffix = 2; suffix <= 64; suffix += 1) {
    const candidate = `${kind}-${suffix}`
    if (!used.has(candidate)) return candidate
  }
  throw new Error("An eval case cannot contain more than 64 assertions.")
}

export function parsePromotionInteger(
  source: string,
  label: string,
  minimum: number,
  maximum: number,
): number {
  if (!/^(0|[1-9]\d*)$/.test(source)) {
    throw new Error(`${label} must be a whole number.`)
  }
  const value = Number(source)
  requireInteger(value, label, minimum, maximum)
  return value
}

function requireRange(
  minimum: number,
  maximum: number | null | undefined,
  label: string,
  limit: number,
): void {
  requireInteger(minimum, `${label} minimum`, 0, limit)
  if (maximum === null || maximum === undefined) return
  requireInteger(maximum, `${label} maximum`, 0, limit)
  if (maximum < minimum) throw new Error(`${label} maximum cannot be below its minimum.`)
}

function requireInteger(value: number, label: string, minimum: number, maximum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be a whole number from ${minimum} to ${maximum}.`)
  }
}

function requireTerminalStatus(value: string): void {
  if (value !== "completed" && value !== "failed") {
    throw new Error("Status assertions must expect completed or failed.")
  }
}

function requirePortableId(value: string, label: string): void {
  if (!PORTABLE_ID_PATTERN.test(value)) {
    throw new Error(
      `${label} must start with a lowercase letter and use lowercase letters, digits, '.', '_', or '-'.`,
    )
  }
}

function requireBoundedCleanText(value: string, label: string, maximum: number): void {
  if (value.length === 0 || value.trim() !== value || value.length > maximum) {
    throw new Error(`${label} must contain 1 to ${maximum} characters without outer whitespace.`)
  }
}

function requireOptionalCleanText(
  value: string | null | undefined,
  label: string,
  maximum: number,
): void {
  if (value === null || value === undefined) return
  requireBoundedCleanText(value, label, maximum)
}
