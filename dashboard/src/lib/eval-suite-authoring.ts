import {
  validateModelJudgeAssertion,
  validateStructuredJudgeAssertion,
} from "./eval-judge-authoring.ts"
import {
  durableTextLength,
  isPythonBlank,
  type PromotionAssertion,
  requireBoundedCleanText,
  requireOptionalCleanText,
  requirePortableId,
  validateDeterministicAssertions,
} from "./evaluation-promotion.ts"
import { EVAL_TEXT_LIMITS } from "./generated/eval-text-limits.ts"
import type {
  EvalCaseDraftV2,
  EvalRunCostBudgetInput,
  EvalScenarioDraftV2,
  EvalSuiteDocumentV1,
  EvalSuiteDocumentV2,
  EvalSuiteDocumentV3,
  EvalSuiteDraftV3,
  EvalTargetCatalogEntry,
  StructuredModelJudgeAssertionDraftV1,
  StructuredModelJudgeAssertionSpec,
} from "./generated/server-api"

const SHA256_REVISION_PATTERN = /^sha256:[a-f0-9]{64}$/

export const EVAL_SUITE_MAX_CASES = 1_000

export function authoredCheckSummary(check: EvalCaseDraftV2["assertions"][number]): string {
  const text = describeAuthoredCheck(check)
  const characters = Array.from(text)
  return characters.length > 180 ? `${characters.slice(0, 180).join("")}…` : text
}

function describeAuthoredCheck(check: EvalCaseDraftV2["assertions"][number]): string {
  const count = (minimum: number, maximum?: number | null) =>
    maximum === minimum
      ? `${minimum} times`
      : maximum == null
        ? `at least ${minimum} times`
        : `${minimum}–${maximum} times`
  switch (check.kind) {
    case "root_status":
      return `Session ${check.expected === "completed" ? "completes" : "fails"}`
    case "child_status":
      return `Child sessions are ${check.expected}, ${count(check.min_count ?? 1, check.max_count)}`
    case "final_output_equals":
      return `Final answer equals ${check.comparison === "json" ? "JSON " : ""}${check.expected}`
    case "final_output_contains":
      return `Final answer contains ${check.expected}`
    case "tool_called":
      return `Calls ${check.tool_name} ${count(check.min_count ?? 1, check.max_count)}`
    case "tool_arguments_contain":
      return `Call ${check.occurrence ?? 1} to ${check.tool_name} has matching arguments`
    case "tool_result_contains":
      return `Call ${check.occurrence ?? 1} to ${check.tool_name} has a matching result`
    case "tools_called_in_order":
      return `Calls ${check.tool_names.join(" → ")}`
    case "process_event":
      return `${check.event.replaceAll("_", " ")}, ${count(check.min_count ?? 1, check.max_count)}`
    case "process_events_in_order":
      return check.events.map((event) => event.replaceAll("_", " ")).join(" → ")
    case "workspace_file":
      return `${check.present === false ? "Does not create" : "Creates"} ${check.path}`
    case "artifact":
      return `Produces ${check.filename ?? check.content_type ?? "an artifact"} in ${check.scope ?? "session"}`
    case "memory_attribution":
      return `Admits at least ${check.min_admitted_items ?? 0} memory items; exposes at least ${check.min_provider_exposures ?? 0} to the provider`
    case "max_tool_calls":
      return `Makes at most ${check.maximum} tool calls`
    case "max_model_steps":
      return `Uses at most ${check.maximum} model steps`
    case "usage_recorded":
      return `Records at least ${check.min_total_tokens ?? 1} tokens`
    case "max_total_tokens":
      return `Uses at most ${check.maximum} tokens`
    case "max_estimated_cost":
      return `Costs at most ${check.maximum} ${check.currency ?? "USD"}`
    case "model_judge":
      return `Judge: ${check.rubric}`
    case "structured_model_judge":
      return `Judge: ${check.rubric.criteria.map((criterion) => criterion.name).join("; ")}`
    default:
      return "Check"
  }
}

export type EvalSuiteDraftValidation =
  | { ok: true; draft: EvalSuiteDraftV3 }
  | { ok: false; error: string }

export function newEvalSuiteDraft(targetKey: string): EvalSuiteDraftV3 {
  return {
    schema_version: 3,
    id: "evaluation-suite",
    target_key: targetKey,
    name: "Evaluation suite",
    description: null,
    trial_request: {
      trials: 1,
      timeout_seconds: 300,
      minimum_passed_trials: 1,
      max_concurrency: 1,
    },
    cases: [newSimpleEvalCase([], "case-1")],
  }
}

export function newSimpleEvalCase(
  existing: readonly EvalCaseDraftV2[],
  preferredId?: string,
): EvalCaseDraftV2 {
  if (preferredId === undefined && existing.length >= EVAL_SUITE_MAX_CASES) {
    throw new Error(`An eval suite cannot contain more than ${EVAL_SUITE_MAX_CASES} cases.`)
  }
  const id =
    preferredId ??
    nextPortableId(
      "case",
      existing.map((item) => item.id),
    )
  return {
    id,
    name: `Case ${existing.length + 1}`,
    description: null,
    source: null,
    stimulus: {
      kind: "simple_input",
      input: {
        messages: [{ role: "user", text: "Describe the behavior to evaluate." }],
      },
    },
    assertions: [{ id: "root_status", kind: "root_status", expected: "completed" }],
  }
}

export function blankEvalScenarioDraft(
  targetKey: string,
  evalCase: EvalCaseDraftV2,
): EvalScenarioDraftV2 {
  return {
    id: portableScenarioId(evalCase.id),
    target_key: targetKey,
    name: `${evalCase.name} scenario`,
    description: evalCase.description ?? null,
    source: evalCase.source ?? null,
    artifact_requirements: [],
    secret_requirements: [],
    events: [
      {
        kind: "initial",
        sequence: 0,
        id: "initial",
        input: {
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Describe the first user input." }],
            },
          ],
        },
      },
    ],
  }
}

export function evalSuiteDraftFromDocument(
  document: EvalSuiteDocumentV1 | EvalSuiteDocumentV2 | EvalSuiteDocumentV3,
): EvalSuiteDraftV3 {
  const storedRequest = document.suite.trial_request
  const trials = storedRequest?.trials ?? 1
  return structuredClone({
    schema_version: 3,
    id: document.suite.id,
    target_key: document.target_key,
    name: document.suite.name,
    description: document.suite.description ?? null,
    trial_request: {
      trials,
      timeout_seconds: storedRequest?.timeout_seconds ?? 300,
      minimum_passed_trials: storedRequest?.trial_policy?.minimum_passed_trials ?? trials,
      max_concurrency: storedRequest?.trial_policy?.max_concurrency ?? 1,
    },
    cases: document.cases.map(({ revision: _revision, ...evalCase }) => ({
      ...evalCase,
      assertions: evalCase.assertions.map((assertion) =>
        assertion.kind === "structured_model_judge"
          ? structuredJudgeDraftFromSpec(assertion)
          : assertion,
      ),
    })),
  })
}

export function duplicateEvalSuiteCase(
  cases: readonly EvalCaseDraftV2[],
  index: number,
): EvalCaseDraftV2 {
  if (cases.length >= EVAL_SUITE_MAX_CASES) {
    throw new Error(`An eval suite cannot contain more than ${EVAL_SUITE_MAX_CASES} cases.`)
  }
  const source = cases[index]
  if (source === undefined) throw new Error("The selected eval case no longer exists.")
  const id = nextPortableId(
    `${source.id}-copy`,
    cases.map((item) => item.id),
  )
  return structuredClone({ ...source, id, name: `${source.name} copy` })
}

export function evalSuitePreviewMatchesDraft(
  document: EvalSuiteDocumentV1 | EvalSuiteDocumentV2 | EvalSuiteDocumentV3,
  draft: EvalSuiteDraftV3,
): boolean {
  return JSON.stringify(evalSuiteDraftFromDocument(document)) === JSON.stringify(draft)
}

export function authoredSuiteRunPreviewIdentity(
  savedSuiteRevision: string | null,
  reviewedSuiteRevision: string | null,
  selection: {
    case_ids?: readonly string[]
    cost_budget?: EvalRunCostBudgetInput | null
  },
): string {
  return JSON.stringify([savedSuiteRevision, reviewedSuiteRevision, selection])
}

export function authoredSuiteCostBudgetContract(
  maximum: string,
  currency: string,
  target: EvalTargetCatalogEntry | undefined,
): EvalRunCostBudgetInput | null | undefined {
  const amount = maximum.trim()
  if (amount === "") return undefined
  const normalizedCurrency = currency.trim().toUpperCase()
  if (
    amount.length > 64 ||
    !/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(amount) ||
    !/[1-9]/.test(amount) ||
    normalizedCurrency.length > 16 ||
    target?.cost_budget_available !== true ||
    !target.cost_budget_currencies.includes(normalizedCurrency)
  ) {
    return null
  }
  return { max_estimated_cost: amount, currency: normalizedCurrency }
}

export function scenarioArtifactBindingsRequireMaterialization(
  artifactReferences: Readonly<Record<string, string>>,
  embedded: boolean,
): boolean {
  return (
    embedded && Object.values(artifactReferences).some((reference) => reference.trim().length > 0)
  )
}

export function validateEvalSuiteDraft(draft: EvalSuiteDraftV3): EvalSuiteDraftValidation {
  try {
    requirePortableId(draft.id, "Suite ID")
    requirePortableId(draft.target_key, "Target key")
    requireBoundedCleanText(draft.name, "Suite name", EVAL_TEXT_LIMITS.name)
    requireOptionalCleanText(draft.description, "Suite description", EVAL_TEXT_LIMITS.description)
    const trials = draft.trial_request?.trials ?? 1
    const minimumPassed = draft.trial_request?.minimum_passed_trials ?? 1
    requireInteger(trials, "Trials", 1, 100)
    requireInteger(minimumPassed, "Required passes", 1, trials)
    requireInteger(draft.trial_request?.max_concurrency ?? 1, "Concurrency", 1, 64)
    if (minimumPassed > trials) {
      throw new Error("Required passes cannot exceed trials.")
    }
    requireInteger(draft.trial_request?.timeout_seconds ?? 300, "Timeout", 1, 3_600)
    if (draft.cases.length < 1 || draft.cases.length > EVAL_SUITE_MAX_CASES) {
      throw new Error(
        `An eval suite must contain between 1 and ${EVAL_SUITE_MAX_CASES.toLocaleString("en-US")} cases.`,
      )
    }
    const caseIds = new Set<string>()
    for (const [index, evalCase] of draft.cases.entries()) {
      const label = `Case ${index + 1}`
      requirePortableId(evalCase.id, `${label} ID`)
      if (caseIds.has(evalCase.id)) throw new Error(`Case ID ${evalCase.id} is duplicated.`)
      caseIds.add(evalCase.id)
      requireBoundedCleanText(evalCase.name, `${label} name`, EVAL_TEXT_LIMITS.name)
      requireOptionalCleanText(
        evalCase.description,
        `${label} description`,
        EVAL_TEXT_LIMITS.description,
      )
      if (evalCase.stimulus.kind === "simple_input") {
        if (evalCase.stimulus.input.opaque_external_case_ref != null) {
          throw new Error(
            `${label} uses runtime-owned opaque external input, which Developer dashboard suite authoring cannot edit.`,
          )
        }
        const messages = evalCase.stimulus.input.messages
        if (messages === undefined) {
          throw new Error(`${label} input must contain user messages.`)
        }
        if (messages.length < 1 || messages.length > 16) {
          throw new Error(`${label} input must contain between 1 and 16 user messages.`)
        }
        let total = 0
        for (const [messageIndex, message] of messages.entries()) {
          if (message.role !== "user") {
            throw new Error(`${label} input message ${messageIndex + 1} must use the user role.`)
          }
          const messageLabel = `${label} input message ${messageIndex + 1}`
          const length = durableTextLength(message.text, messageLabel)
          if (isPythonBlank(message.text) || length > 65_536) {
            throw new Error(`${messageLabel} must contain 1 to 65,536 characters.`)
          }
          total += length
        }
        if (total > 262_144) throw new Error(`${label} input exceeds 262,144 characters.`)
      } else {
        requirePortableId(evalCase.stimulus.scenario_id, `${label} scenario ID`)
        if (!SHA256_REVISION_PATTERN.test(evalCase.stimulus.scenario_revision)) {
          throw new Error(`${label} must reference a saved immutable scenario revision.`)
        }
      }
      const assertionIds = new Set<string>()
      const deterministic: PromotionAssertion[] = []
      for (const assertion of evalCase.assertions) {
        requirePortableId(assertion.id, `${label} assertion ID`)
        if (assertionIds.has(assertion.id)) {
          throw new Error(`${label} assertion ID ${assertion.id} is duplicated.`)
        }
        assertionIds.add(assertion.id)
        if (assertion.kind === "structured_model_judge") {
          validateStructuredJudgeAssertion(assertion)
        } else if (assertion.kind === "model_judge") {
          validateModelJudgeAssertion(assertion)
        } else {
          deterministic.push(assertion as PromotionAssertion)
        }
      }
      if (evalCase.assertions.length < 1 || evalCase.assertions.length > 64) {
        throw new Error(`${label} must contain between 1 and 64 assertions.`)
      }
      if (deterministic.length > 0) validateDeterministicAssertions(deterministic)
    }
    return { ok: true, draft }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "The authored eval suite is invalid.",
    }
  }
}

function structuredJudgeDraftFromSpec(
  assertion: StructuredModelJudgeAssertionSpec,
): StructuredModelJudgeAssertionDraftV1 {
  const { revision: _rubricRevision, ...rubric } = assertion.rubric
  const reference = assertion.reference
  if (reference?.kind === "public_reference") {
    const { revision: _referenceRevision, ...publicReference } = reference
    return { ...assertion, rubric, reference: publicReference }
  }
  return { ...assertion, rubric, reference }
}

function nextPortableId(prefix: string, existing: readonly string[]): string {
  const used = new Set(existing)
  if (!used.has(prefix)) return prefix
  for (let suffix = 2; suffix <= EVAL_SUITE_MAX_CASES; suffix += 1) {
    const candidate = `${prefix}-${suffix}`
    if (!used.has(candidate)) return candidate
  }
  throw new Error("The eval suite has no remaining case identity slots.")
}

function portableScenarioId(caseId: string): string {
  const candidate = `${caseId}-scenario`.slice(0, 128)
  try {
    requirePortableId(candidate, "Scenario ID")
    return candidate
  } catch {
    return "authored-scenario"
  }
}

function requireInteger(value: number, label: string, minimum: number, maximum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be a whole number from ${minimum} to ${maximum}.`)
  }
}
