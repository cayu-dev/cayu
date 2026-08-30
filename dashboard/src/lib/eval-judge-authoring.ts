import {
  durableTextLength,
  isPythonBlank,
  requireBoundedCleanText,
  requireOptionalCleanText,
  requirePortableId,
} from "./evaluation-promotion.ts"
import type {
  EvalJudgeCalibrationDraftV1,
  EvalTargetCatalogEntry,
  JudgeProfileIdentityV1,
  StructuredModelJudgeAssertionDraftV1,
  StructuredModelJudgeAssertionSpec,
} from "./generated/server-api"

const SHA256_REVISION_PATTERN = /^sha256:[a-f0-9]{64}$/
const CANONICAL_DECIMAL_PATTERN = /^(?:0|[1-9]\d*)(?:\.\d*[1-9])?$/

export const EVAL_JUDGE_MAX_CRITERIA = 8
export const EVAL_JUDGE_MAX_CALIBRATION_TRIALS = 10

export function newStructuredJudgeAssertion(
  profile: JudgeProfileIdentityV1,
  existingIds: readonly string[],
): StructuredModelJudgeAssertionDraftV1 {
  return {
    id: nextAssertionId("quality", existingIds),
    kind: "structured_model_judge",
    description: "Evaluate the quality of the final answer.",
    judge_profile_key: profile.key,
    judge_profile_revision: profile.revision,
    rubric: {
      schema_version: 1,
      id: "answer-quality",
      criteria: [
        {
          id: "correctness",
          name: "Correctness",
          description: "The answer is factually correct and satisfies the requested task.",
          weight: "1",
        },
      ],
    },
    reference: null,
    threshold: "0.5",
    evidence: { schema_version: 1, include_final_output: true, include_transcript: false },
  }
}

export function newMemoryUseJudgeAssertion(
  profile: JudgeProfileIdentityV1,
  existingIds: readonly string[],
): StructuredModelJudgeAssertionDraftV1 {
  if (
    !profile.allowed_evidence.includes("final_output") ||
    !profile.allowed_evidence.includes("public_reference")
  ) {
    throw new Error(
      "A memory-use judge profile must permit final-output and public-reference evidence.",
    )
  }
  return {
    id: nextAssertionId("memory-use", existingIds),
    kind: "structured_model_judge",
    description: "Evaluate whether the answer applies trusted memory truth correctly.",
    judge_profile_key: profile.key,
    judge_profile_revision: profile.revision,
    rubric: {
      schema_version: 1,
      id: "memory-use",
      criteria: [
        {
          id: "reference-correctness",
          name: "Reference correctness",
          description: "The answer agrees with the trusted memory reference truth.",
          weight: "0.5",
        },
        {
          id: "grounded-use",
          name: "Grounded use",
          description: "The answer applies relevant reference facts to the requested task.",
          weight: "0.3",
        },
        {
          id: "unsupported-memory-avoidance",
          name: "No unsupported memory claims",
          description: "The answer does not invent or contradict memory-derived facts.",
          weight: "0.2",
        },
      ],
    },
    reference: {
      schema_version: 1,
      kind: "public_reference",
      id: "memory-reference",
      expected_answer: null,
      expected_facts: [""],
    },
    threshold: "0.8",
    evidence: { schema_version: 1, include_final_output: true, include_transcript: false },
  }
}

export function validateStructuredJudgeAssertion(
  assertion: StructuredModelJudgeAssertionDraftV1,
): void {
  requirePortableId(assertion.id, "Judge assertion ID")
  requireOptionalCleanText(assertion.description, "Judge assertion description", 2_048)
  requirePortableId(assertion.judge_profile_key, "Judge profile key")
  if (!SHA256_REVISION_PATTERN.test(assertion.judge_profile_revision)) {
    throw new Error("Select a current trusted judge profile.")
  }
  requirePortableId(assertion.rubric.id, "Rubric ID")
  const criteria = assertion.rubric.criteria
  if (criteria.length < 1 || criteria.length > EVAL_JUDGE_MAX_CRITERIA) {
    throw new Error(`A rubric must contain between 1 and ${EVAL_JUDGE_MAX_CRITERIA} criteria.`)
  }
  const ids = new Set<string>()
  for (const [index, criterion] of criteria.entries()) {
    const label = `Criterion ${index + 1}`
    requirePortableId(criterion.id, `${label} ID`)
    if (ids.has(criterion.id)) throw new Error(`Criterion ID ${criterion.id} is duplicated.`)
    ids.add(criterion.id)
    requireBoundedCleanText(criterion.name, `${label} name`, 128)
    const descriptionLength = durableTextLength(criterion.description, `${label} description`)
    if (isPythonBlank(criterion.description) || descriptionLength > 2_048) {
      throw new Error(`${label} description must contain 1 to 2,048 characters.`)
    }
    requireUnitDecimal(criterion.weight, `${label} weight`)
  }
  if (!decimalStringsSumToOne(criteria.map((criterion) => criterion.weight))) {
    throw new Error("Rubric weights must sum exactly to 1.")
  }
  requireUnitDecimal(assertion.threshold ?? "0.5", "Judge threshold")
  const reference = assertion.reference
  if (reference?.kind === "public_reference") {
    requirePortableId(reference.id, "Public reference ID")
    const answer = reference.expected_answer ?? null
    const facts = reference.expected_facts ?? []
    if (answer === null && facts.length === 0) {
      throw new Error("A public reference needs an expected answer or expected facts.")
    }
    if (answer !== null) {
      const answerLength = durableTextLength(answer, "Expected answer")
      if (isPythonBlank(answer) || answerLength > 65_536) {
        throw new Error("Expected answer must contain 1 to 65,536 characters.")
      }
    }
    if (facts.length > 64) throw new Error("A public reference cannot contain more than 64 facts.")
    for (const [index, fact] of facts.entries()) {
      const length = durableTextLength(fact, `Expected fact ${index + 1}`)
      if (isPythonBlank(fact) || length > 2_048) {
        throw new Error(`Expected fact ${index + 1} must contain 1 to 2,048 characters.`)
      }
    }
  } else if (reference?.kind === "private_reference") {
    requirePortableId(reference.key, "Private reference key")
    if (
      !SHA256_REVISION_PATTERN.test(reference.revision) ||
      !SHA256_REVISION_PATTERN.test(reference.privacy_policy_revision)
    ) {
      throw new Error("Select a current private reference.")
    }
  }
}

export function judgeProfileForAssertion(
  target: EvalTargetCatalogEntry | undefined,
  assertion: StructuredModelJudgeAssertionDraftV1,
): JudgeProfileIdentityV1 | undefined {
  return target?.judge_profiles?.find(
    (profile) =>
      profile.key === assertion.judge_profile_key &&
      profile.revision === assertion.judge_profile_revision,
  )
}

export function judgeRouteForAssertion(
  target: EvalTargetCatalogEntry | undefined,
  assertion: StructuredModelJudgeAssertionDraftV1,
): "independent_model" | "same_model" | undefined {
  return target?.judge_profile_routes?.find(
    (route) =>
      route.judge_profile_key === assertion.judge_profile_key &&
      route.judge_profile_revision === assertion.judge_profile_revision,
  )?.candidate_route_relation
}

export function structuredAssertionFromReviewedSuite(
  assertion: StructuredModelJudgeAssertionDraftV1,
  reviewedAssertions: readonly unknown[],
): StructuredModelJudgeAssertionSpec | null {
  const match = reviewedAssertions.find(
    (candidate): candidate is StructuredModelJudgeAssertionSpec =>
      typeof candidate === "object" &&
      candidate !== null &&
      "kind" in candidate &&
      candidate.kind === "structured_model_judge" &&
      "id" in candidate &&
      candidate.id === assertion.id &&
      "rubric" in candidate &&
      typeof candidate.rubric === "object" &&
      candidate.rubric !== null &&
      "revision" in candidate.rubric,
  )
  return match ?? null
}

export function newJudgeCalibrationDraft(
  targetKey: string,
  assertion: StructuredModelJudgeAssertionSpec,
  task: string,
): EvalJudgeCalibrationDraftV1 {
  return {
    schema_version: 1,
    id: `${assertion.id}-calibration`.slice(0, 128),
    target_key: targetKey,
    assertion,
    evidence_source_id: `${assertion.id}-known-evidence`.slice(0, 128),
    task,
    final_output: "Paste a known candidate answer to judge.",
    transcript: assertion.evidence?.include_transcript
      ? "Paste the fixed transcript supplied to the judge."
      : null,
    human_criteria: assertion.rubric.criteria.map((criterion) => ({
      criterion_id: criterion.id,
      score: "1",
    })),
    trials: 3,
  }
}

export function validateJudgeCalibrationDraft(draft: EvalJudgeCalibrationDraftV1): void {
  requirePortableId(draft.id, "Calibration ID")
  requirePortableId(draft.target_key, "Calibration target")
  requirePortableId(draft.evidence_source_id, "Evidence source ID")
  const taskLength = durableTextLength(draft.task, "Calibration task")
  if (isPythonBlank(draft.task) || taskLength > 262_144) {
    throw new Error("Calibration task must contain 1 to 262,144 characters.")
  }
  if (durableTextLength(draft.final_output, "Calibration output") > 262_144) {
    throw new Error("Calibration output cannot exceed 262,144 characters.")
  }
  if (draft.assertion.evidence?.include_transcript) {
    if (draft.transcript == null || isPythonBlank(draft.transcript)) {
      throw new Error("This judge requires a fixed transcript for calibration.")
    }
    if (durableTextLength(draft.transcript, "Calibration transcript") > 262_144) {
      throw new Error("Calibration transcript cannot exceed 262,144 characters.")
    }
  } else if (draft.transcript != null) {
    throw new Error("Remove the transcript or enable transcript evidence for this judge.")
  }
  if (
    !Number.isSafeInteger(draft.trials ?? 1) ||
    (draft.trials ?? 1) < 1 ||
    (draft.trials ?? 1) > EVAL_JUDGE_MAX_CALIBRATION_TRIALS
  ) {
    throw new Error(
      `Calibration trials must be a whole number from 1 to ${EVAL_JUDGE_MAX_CALIBRATION_TRIALS}.`,
    )
  }
  const rubricIds = draft.assertion.rubric.criteria.map((criterion) => criterion.id)
  const humanIds = draft.human_criteria.map((criterion) => criterion.criterion_id)
  if (JSON.stringify(rubricIds) !== JSON.stringify(humanIds)) {
    throw new Error("Human labels must match the reviewed rubric criterion order.")
  }
  for (const [index, label] of draft.human_criteria.entries()) {
    requireUnitDecimal(label.score, `Human score ${index + 1}`)
  }
}

function requireUnitDecimal(value: string, label: string): void {
  if (
    value.length > 20 ||
    !CANONICAL_DECIMAL_PATTERN.test(value) ||
    compareDecimalToOne(value) > 0
  ) {
    throw new Error(`${label} must be a canonical decimal from 0 to 1.`)
  }
}

function decimalStringsSumToOne(values: readonly string[]): boolean {
  const parsed = values.map(parseDecimal)
  const scale = Math.max(...parsed.map((value) => value.scale))
  const sum = parsed.reduce(
    (total, value) => total + value.coefficient * 10n ** BigInt(scale - value.scale),
    0n,
  )
  return sum === 10n ** BigInt(scale)
}

function compareDecimalToOne(value: string): number {
  const parsed = parseDecimal(value)
  const one = 10n ** BigInt(parsed.scale)
  return parsed.coefficient > one ? 1 : parsed.coefficient < one ? -1 : 0
}

function parseDecimal(value: string): { coefficient: bigint; scale: number } {
  const [whole, fraction = ""] = value.split(".")
  return { coefficient: BigInt(`${whole}${fraction}`), scale: fraction.length }
}

function nextAssertionId(prefix: string, existingIds: readonly string[]): string {
  const used = new Set(existingIds)
  if (!used.has(prefix)) return prefix
  for (let suffix = 2; suffix <= 64; suffix += 1) {
    const candidate = `${prefix}-${suffix}`
    if (!used.has(candidate)) return candidate
  }
  throw new Error("The eval case has no remaining assertion identity slots.")
}
