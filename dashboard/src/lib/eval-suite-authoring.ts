import {
  durableTextLength,
  isPythonBlank,
  type PromotionAssertion,
  requireBoundedCleanText,
  requireOptionalCleanText,
  requirePortableId,
  validateDeterministicAssertions,
} from "./evaluation-promotion.ts"
import type {
  EvalCaseDraftV1,
  EvalScenarioDraftV2,
  EvalSuiteDocumentV1,
  EvalSuiteDraftV1,
} from "./generated/server-api"

const SHA256_REVISION_PATTERN = /^sha256:[a-f0-9]{64}$/

export const EVAL_SUITE_MAX_CASES = 1_000

export type EvalSuiteDraftValidation =
  | { ok: true; draft: EvalSuiteDraftV1 }
  | { ok: false; error: string }

export function newEvalSuiteDraft(targetKey: string): EvalSuiteDraftV1 {
  return {
    id: "evaluation-suite",
    target_key: targetKey,
    name: "Evaluation suite",
    description: null,
    trial_request: { trials: 1, timeout_seconds: 300 },
    cases: [newSimpleEvalCase([], "case-1")],
  }
}

export function newSimpleEvalCase(
  existing: readonly EvalCaseDraftV1[],
  preferredId?: string,
): EvalCaseDraftV1 {
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
  evalCase: EvalCaseDraftV1,
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

export function evalSuiteDraftFromDocument(document: EvalSuiteDocumentV1): EvalSuiteDraftV1 {
  return structuredClone({
    id: document.suite.id,
    target_key: document.target_key,
    name: document.suite.name,
    description: document.suite.description ?? null,
    trial_request: document.suite.trial_request ?? { trials: 1, timeout_seconds: 300 },
    cases: document.cases.map(({ revision: _revision, ...evalCase }) => evalCase),
  })
}

export function duplicateEvalSuiteCase(
  cases: readonly EvalCaseDraftV1[],
  index: number,
): EvalCaseDraftV1 {
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
  document: EvalSuiteDocumentV1,
  draft: EvalSuiteDraftV1,
): boolean {
  return JSON.stringify(evalSuiteDraftFromDocument(document)) === JSON.stringify(draft)
}

export function authoredSuiteRunPreviewIdentity(
  savedSuiteRevision: string | null,
  reviewedSuiteRevision: string | null,
  selection: { case_ids?: readonly string[] },
): string {
  return JSON.stringify([savedSuiteRevision, reviewedSuiteRevision, selection])
}

export function scenarioArtifactBindingsRequireMaterialization(
  artifactReferences: Readonly<Record<string, string>>,
  embedded: boolean,
): boolean {
  return (
    embedded && Object.values(artifactReferences).some((reference) => reference.trim().length > 0)
  )
}

export function validateEvalSuiteDraft(draft: EvalSuiteDraftV1): EvalSuiteDraftValidation {
  try {
    requirePortableId(draft.id, "Suite ID")
    requirePortableId(draft.target_key, "Target key")
    requireBoundedCleanText(draft.name, "Suite name", 256)
    requireOptionalCleanText(draft.description, "Suite description", 2_048)
    if ((draft.trial_request?.trials ?? 1) !== 1) {
      throw new Error("Control Plane suite launch currently supports exactly one trial.")
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
      requireBoundedCleanText(evalCase.name, `${label} name`, 256)
      requireOptionalCleanText(evalCase.description, `${label} description`, 2_048)
      if (evalCase.stimulus.kind === "simple_input") {
        if (evalCase.stimulus.input.opaque_external_case_ref != null) {
          throw new Error(
            `${label} uses runtime-owned opaque external input, which Control Plane suite authoring cannot edit.`,
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
      validateDeterministicAssertions(evalCase.assertions as readonly PromotionAssertion[])
    }
    return { ok: true, draft }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "The authored eval suite is invalid.",
    }
  }
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
