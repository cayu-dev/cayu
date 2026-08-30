import type {
  CapturedEvaluationCandidateV1,
  CapturedEvaluationDraft,
  CapturedEvaluationPreviewResponse,
  EvaluationPromotionDraft,
  EvaluationPromotionPreviewResponse,
  PromotionCandidateV1,
} from "./generated/server-api"

export type PromotionAssertion = EvaluationPromotionDraft["case"]["assertions"][number]
export type CapturedEvaluationAssertion = CapturedEvaluationDraft["case"]["assertions"][number]
// Promotion drafts are candidate-authored. Model judges require target-owned
// execution authority and therefore cannot be created or selected here.
export type PromotionAssertionKind = Exclude<NonNullable<PromotionAssertion["kind"]>, "model_judge">

export const PROMOTION_ASSERTION_KINDS = [
  "root_status",
  "child_status",
  "final_output_equals",
  "final_output_contains",
  "tool_called",
  "tool_arguments_contain",
  "tool_result_contains",
  "tools_called_in_order",
  "process_event",
  "process_events_in_order",
  "workspace_file",
  "artifact",
  "memory_attribution",
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
const MAX_PROCESS_EVENT_ORDER = 256
const MAX_MEMORY_ADMITTED_ITEMS = 1_000
const MAX_MEMORY_PROVIDER_EXPOSURES = 100

export const PROCESS_EVENT_OPTIONS = [
  ["session_started", "Session started"],
  ["session_resumed", "Session resumed"],
  ["session_awaiting_user_input", "Session awaiting user input"],
  ["session_completed", "Session completed"],
  ["session_failed", "Session failed"],
  ["session_interrupted", "Session interrupted"],
  ["session_limit_reached", "Session limit reached"],
  ["tool_call_started", "Tool call started"],
  ["tool_call_completed", "Tool call completed"],
  ["tool_call_failed", "Tool call failed"],
  ["tool_call_blocked", "Tool call blocked"],
  ["tool_approval_requested", "Tool approval requested"],
  ["tool_approved", "Tool approved"],
  ["tool_approval_denied", "Tool approval denied"],
  ["tool_approval_expired", "Tool approval expired"],
  ["structured_output_validated", "Structured output validated"],
  ["structured_output_failed", "Structured output failed"],
  ["budget_limit_reached", "Budget limit reached"],
] as const

export type ProcessEventKind = (typeof PROCESS_EVENT_OPTIONS)[number][0]

const PROCESS_EVENT_KINDS = new Set<string>(PROCESS_EVENT_OPTIONS.map(([kind]) => kind))

type CapturedAssertionEvidence = CapturedEvaluationCandidateV1["evidence"]

function observedProcessOrder(
  evidence: CapturedAssertionEvidence,
): readonly ProcessEventKind[] | null {
  if (evidence.process_event_evidence_state !== "complete") return null

  const counts = new Map<ProcessEventKind, number>()
  for (const event of evidence.process_events) {
    counts.set(event, (counts.get(event) ?? 0) + 1)
  }

  // An exact-order assertion filters by event kind, so a suggestion must retain
  // every occurrence of each selected kind. Greedily select whole kinds in
  // first-observed order; never truncate a kind merely to fit the draft limit.
  const selected = new Set<ProcessEventKind>()
  let retainedCount = 0
  for (const event of evidence.process_events) {
    if (selected.has(event)) continue
    const count = counts.get(event)
    if (count === undefined || retainedCount + count > MAX_PROCESS_EVENT_ORDER) continue
    selected.add(event)
    retainedCount += count
  }
  if (selected.size === 0) return null
  return evidence.process_events.filter((event) => selected.has(event))
}

function isPortableObservedByteCount(value: number | null | undefined): value is number {
  return value != null && Number.isSafeInteger(value) && value >= 0 && value <= MAX_SAFE_COUNTER
}

export function capturedAssertionSuggestionUnavailable(
  kind: PromotionAssertionKind,
  evidence: CapturedAssertionEvidence,
): boolean {
  if (kind === "process_event") {
    return (
      evidence.process_event_evidence_state !== "complete" || evidence.process_events.length === 0
    )
  }
  if (kind === "process_events_in_order") return observedProcessOrder(evidence) === null
  if (kind === "workspace_file") {
    return (
      evidence.workspace_evidence_state !== "complete" ||
      !(evidence.workspace_files ?? []).some(
        (item) =>
          item.state === "missing" ||
          (item.state === "present" && isPortableObservedByteCount(item.total_bytes)),
      )
    )
  }
  if (kind === "artifact") {
    const completeScopes = new Set(
      (evidence.artifact_scopes ?? [])
        .filter((item) => item.state === "complete")
        .map((item) => item.scope),
    )
    return !(evidence.artifacts ?? []).some(
      (item) => completeScopes.has(item.scope) && isPortableObservedByteCount(item.size_bytes),
    )
  }
  if (kind === "memory_attribution") {
    const memory = evidence.memory_attribution
    return memory == null || memory.completeness !== "complete" || memory.has_indeterminate_exposure
  }
  return false
}

export type CapturedAssertionDraft = {
  assertion: PromotionAssertion
  source: "observed" | "expectation"
}

export const PROMOTION_ASSERTION_LABELS: Record<PromotionAssertionKind, string> = {
  root_status: "Root status",
  child_status: "Child status count",
  final_output_equals: "Final output equals",
  final_output_contains: "Final output contains",
  tool_called: "Tool called",
  tool_arguments_contain: "Tool arguments contain JSON",
  tool_result_contains: "Tool result contains JSON",
  tools_called_in_order: "Tools called in order",
  process_event: "Process event count",
  process_events_in_order: "Process events in order",
  workspace_file: "Workspace file structure",
  artifact: "Artifact structure or text",
  memory_attribution: "Memory admission and exposure",
  max_tool_calls: "Maximum tool calls",
  max_model_steps: "Maximum model steps",
  usage_recorded: "Usage recorded",
  max_total_tokens: "Maximum total tokens",
  max_estimated_cost: "Maximum estimated cost",
}

export type PromotionDraftValidation =
  | { ok: true; draft: EvaluationPromotionDraft }
  | { ok: false; error: string }
export type CapturedEvaluationDraftValidation =
  | { ok: true; draft: CapturedEvaluationDraft }
  | { ok: false; error: string }

export function promotionDraftFromCandidate(
  candidate: PromotionCandidateV1,
  baselineRevision: string,
): EvaluationPromotionDraft {
  const input = candidate.case.input
  if (input == null) {
    throw new Error("A runnable promotion candidate must contain eval input.")
  }
  if (input.opaque_external_case_ref != null || input.messages === undefined) {
    throw new Error(
      "A runtime-owned opaque external input cannot be edited as a promotion candidate.",
    )
  }
  return structuredClone({
    expected_baseline_revision: baselineRevision,
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
      input,
      assertions: candidate.case.assertions,
    },
  })
}

export function capturedEvaluationDraftFromCandidate(
  candidate: CapturedEvaluationCandidateV1,
  baselineRevision: string,
): CapturedEvaluationDraft {
  return structuredClone({
    expected_baseline_revision: baselineRevision,
    suite: {
      id: candidate.suite.id,
      name: candidate.suite.name,
      description: candidate.suite.description ?? null,
    },
    case: {
      id: candidate.case.id,
      suite_id: candidate.case.suite_id,
      name: candidate.case.name,
      description: candidate.case.description ?? null,
      assertions: candidate.case.assertions,
    },
  })
}

export function capturedEvaluationPreviewMatchesDraft(
  preview: CapturedEvaluationPreviewResponse,
  draft: CapturedEvaluationDraft,
): boolean {
  return (
    JSON.stringify(
      capturedEvaluationDraftFromCandidate(preview.candidate, preview.baseline_revision),
    ) === JSON.stringify(draft)
  )
}

export function validateCapturedEvaluationDraft(
  draft: CapturedEvaluationDraft,
): CapturedEvaluationDraftValidation {
  try {
    validateCapturedDraft(draft)
    return { ok: true, draft }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "The captured evaluation is invalid.",
    }
  }
}

function validateCapturedDraft(draft: CapturedEvaluationDraft): void {
  if (!SHA256_REVISION_PATTERN.test(draft.expected_baseline_revision)) {
    throw new Error("Reload the captured preview before editing this evaluation.")
  }
  requirePortableId(draft.suite.id, "Suite ID")
  requireBoundedCleanText(draft.suite.name, "Suite name", 256)
  requireOptionalCleanText(draft.suite.description, "Suite description", 2_048)
  requirePortableId(draft.case.id, "Case ID")
  requirePortableId(draft.case.suite_id, "Case suite ID")
  if (draft.case.suite_id !== draft.suite.id) {
    throw new Error("Case suite ID must match the edited suite ID.")
  }
  requireBoundedCleanText(draft.case.name, "Case name", 256)
  requireOptionalCleanText(draft.case.description, "Case description", 2_048)
  validateDeterministicAssertions(draft.case.assertions)
}

export function previewMatchesDraft(
  preview: EvaluationPromotionPreviewResponse,
  draft: EvaluationPromotionDraft,
): boolean {
  return (
    JSON.stringify(promotionDraftFromCandidate(preview.candidate, preview.baseline_revision)) ===
    JSON.stringify(draft)
  )
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
  if (!SHA256_REVISION_PATTERN.test(draft.expected_baseline_revision)) {
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
  if (draft.case.input.opaque_external_case_ref != null) {
    throw new Error("A promotion candidate cannot contain runtime-owned opaque external input.")
  }
  const messages = draft.case.input.messages
  if (messages === undefined) {
    throw new Error("A promotion candidate must contain eval input messages.")
  }
  if (messages.length < 1 || messages.length > 16) {
    throw new Error("Eval input must contain between 1 and 16 user messages.")
  }
  let totalMessageChars = 0
  for (const [index, message] of messages.entries()) {
    if (message.role !== "user") {
      throw new Error(`Input message ${index + 1} must have the user role.`)
    }
    const messageChars = durableTextLength(message.text, `Input message ${index + 1}`)
    if (isPythonBlank(message.text)) {
      throw new Error(`Input message ${index + 1} cannot be blank.`)
    }
    if (messageChars > 65_536) {
      throw new Error(`Input message ${index + 1} cannot exceed 65,536 characters.`)
    }
    totalMessageChars += messageChars
  }
  if (totalMessageChars > 262_144) {
    throw new Error("Eval input cannot exceed 262,144 total characters.")
  }

  validateDeterministicAssertions(draft.case.assertions)
}

export function validateDeterministicAssertions(assertions: readonly PromotionAssertion[]): void {
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
      requireChildTerminalStatus(assertion.expected)
      requireRange(assertion.min_count ?? 1, assertion.max_count, "Child count", 500)
      return
    case "final_output_equals":
      if (durableTextLength(assertion.expected, "Expected final output") > 65_536) {
        throw new Error("Expected final output cannot exceed 65,536 characters.")
      }
      return
    case "final_output_contains":
      if (
        isPythonBlank(assertion.expected) ||
        durableTextLength(assertion.expected, "Expected final-output text") > 65_536
      ) {
        throw new Error("Expected final-output text must contain 1 to 65,536 characters.")
      }
      return
    case "tool_called":
      requireBoundedCleanText(assertion.tool_name, "Tool name", 256)
      requireRange(assertion.min_count ?? 1, assertion.max_count, "Tool count", 4_096)
      return
    case "tool_arguments_contain":
      requireBoundedCleanText(assertion.tool_name, "Tool name", 256)
      requireInteger(assertion.occurrence ?? 1, "Tool occurrence", 1, 256)
      validateToolJsonSubset(assertion.expected_subset, "Expected argument subset")
      return
    case "tool_result_contains": {
      requireBoundedCleanText(assertion.tool_name, "Tool name", 256)
      requireInteger(assertion.occurrence ?? 1, "Tool occurrence", 1, 256)
      const subset = validateToolJsonSubset(assertion.expected_subset, "Expected result subset")
      const keys = Object.keys(subset)
      if (keys.length === 0) throw new Error("Expected result subset cannot be empty.")
      if (keys.some((key) => !["content", "structured", "is_error"].includes(key))) {
        throw new Error("Expected result subset can select only content, structured, and is_error.")
      }
      return
    }
    case "tools_called_in_order":
      if (assertion.tool_names.length > 256) {
        throw new Error("Tool order cannot contain more than 256 names.")
      }
      for (const name of assertion.tool_names) {
        requireBoundedCleanText(name, "Tool name", 256)
      }
      return
    case "process_event":
      requireProcessEvent(assertion.event)
      requireRange(assertion.min_count ?? 1, assertion.max_count, "Process event count", 4_096)
      return
    case "process_events_in_order":
      if (assertion.events.length < 1 || assertion.events.length > 256) {
        throw new Error("Process event order must contain between 1 and 256 events.")
      }
      for (const event of assertion.events) requireProcessEvent(event)
      return
    case "workspace_file":
      requireWorkspacePath(assertion.path)
      requireOptionalByteRange(assertion.minimum_bytes, assertion.maximum_bytes, "Workspace size")
      requireOptionalSha256(assertion.sha256, "Workspace digest")
      if (
        assertion.present === false &&
        (assertion.minimum_bytes != null ||
          assertion.maximum_bytes != null ||
          assertion.sha256 != null)
      ) {
        throw new Error("An absent workspace file cannot require size or digest values.")
      }
      return
    case "artifact":
      if (
        assertion.scope !== undefined &&
        assertion.scope !== "session" &&
        assertion.scope !== "environment"
      ) {
        throw new Error("Artifact scope must be session or environment.")
      }
      requireOptionalBoundedText(assertion.filename, "Artifact filename", 1_024, true)
      requireOptionalPrintableAscii(assertion.content_type, "Artifact content type", 1_024)
      requireOptionalByteRange(assertion.minimum_bytes, assertion.maximum_bytes, "Artifact size")
      requireOptionalSha256(assertion.sha256, "Artifact digest")
      requireOptionalBoundedText(assertion.text_contains, "Artifact text", 32_768, true)
      if (
        assertion.text_contains != null &&
        new TextEncoder().encode(assertion.text_contains).byteLength > 65_536
      ) {
        throw new Error("Artifact text cannot exceed 65,536 UTF-8 bytes.")
      }
      requireRange(assertion.min_count ?? 1, assertion.max_count, "Artifact count", 256)
      return
    case "memory_attribution":
      requireRange(
        assertion.min_admitted_items ?? 1,
        assertion.max_admitted_items,
        "Admitted memory item count",
        MAX_MEMORY_ADMITTED_ITEMS,
      )
      requireRange(
        assertion.min_provider_exposures ?? 1,
        assertion.max_provider_exposures,
        "Memory provider exposure count",
        MAX_MEMORY_PROVIDER_EXPOSURES,
      )
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
      if (
        assertion.maximum.length > 64 ||
        !/^(?:0|[1-9]\d*)(?:\.\d*[1-9])?$/.test(assertion.maximum)
      ) {
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
    case "tool_arguments_contain":
      return { id, kind, tool_name: "tool", occurrence: 1, expected_subset: { key: "value" } }
    case "tool_result_contains":
      return {
        id,
        kind,
        tool_name: "tool",
        occurrence: 1,
        expected_subset: { structured: { status: "ok" } },
      }
    case "tools_called_in_order":
      return { id, kind, tool_names: ["tool"] }
    case "process_event":
      return { id, kind, event: "tool_approval_requested", min_count: 1, max_count: null }
    case "process_events_in_order":
      return { id, kind, events: ["tool_approval_requested", "tool_approved"] }
    case "workspace_file":
      return {
        id,
        kind,
        path: "output.txt",
        present: true,
        minimum_bytes: null,
        maximum_bytes: null,
        sha256: null,
      }
    case "artifact":
      return {
        id,
        kind,
        scope: "session",
        filename: "report.txt",
        content_type: null,
        minimum_bytes: null,
        maximum_bytes: null,
        sha256: null,
        text_contains: null,
        min_count: 1,
        max_count: null,
      }
    case "memory_attribution":
      return {
        id,
        kind,
        min_admitted_items: 1,
        max_admitted_items: null,
        min_provider_exposures: 1,
        max_provider_exposures: null,
      }
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

export function createCapturedEvaluationAssertion(
  kind: PromotionAssertionKind,
  existing: readonly PromotionAssertion[],
  evidence: CapturedEvaluationCandidateV1["evidence"],
): PromotionAssertion {
  const assertion = createPromotionAssertion(kind, existing)
  switch (assertion.kind) {
    case "root_status":
      return evidence.root_status === "failed" ? { ...assertion, expected: "failed" } : assertion
    case "child_status": {
      const expected = evidence.child_statuses.find(
        (status): status is "completed" | "failed" | "interrupted" =>
          status === "completed" || status === "failed" || status === "interrupted",
      )
      if (expected === undefined) return assertion
      const count = evidence.child_statuses.filter((status) => status === expected).length
      return { ...assertion, expected, min_count: count, max_count: count }
    }
    case "final_output_equals":
    case "final_output_contains":
      return evidence.final_output_state === "complete"
        ? { ...assertion, expected: evidence.final_output }
        : assertion
    case "tool_called": {
      const toolName = evidence.started_tool_names[0]
      if (toolName === undefined) return assertion
      const count = evidence.started_tool_names.filter((name) => name === toolName).length
      return { ...assertion, tool_name: toolName, min_count: count, max_count: count }
    }
    case "tool_arguments_contain": {
      const call = evidence.tool_calls.find((item) => item.arguments.state === "available")
      return call?.arguments.value == null
        ? assertion
        : {
            ...assertion,
            tool_name: call.tool_name,
            occurrence: call.occurrence,
            expected_subset: structuredClone(call.arguments.value),
          }
    }
    case "tool_result_contains": {
      const call = evidence.tool_calls.find((item) => item.result.state === "available")
      return call?.result.value == null
        ? assertion
        : {
            ...assertion,
            tool_name: call.tool_name,
            occurrence: call.occurrence,
            expected_subset: structuredClone(call.result.value),
          }
    }
    case "tools_called_in_order":
      return { ...assertion, tool_names: [...evidence.requested_tool_names] }
    case "process_event": {
      if (capturedAssertionSuggestionUnavailable(assertion.kind, evidence)) {
        throw new Error("Complete captured process-event evidence is required for this assertion.")
      }
      const event =
        evidence.process_events.find(
          (item) => item !== "session_started" && item !== "session_completed",
        ) ?? evidence.process_events[0]
      if (event === undefined) {
        throw new Error("Complete captured process-event evidence must contain an observed event.")
      }
      const count = evidence.process_events.filter((item) => item === event).length
      return { ...assertion, event, min_count: count, max_count: count }
    }
    case "process_events_in_order": {
      const events = observedProcessOrder(evidence)
      if (events === null) {
        throw new Error(
          "Complete captured process-event evidence with a representable order is required for this assertion.",
        )
      }
      return { ...assertion, events: [...events] }
    }
    case "workspace_file": {
      const observed = (evidence.workspace_files ?? []).find(
        (item) =>
          item.state === "missing" ||
          (item.state === "present" && isPortableObservedByteCount(item.total_bytes)),
      )
      if (evidence.workspace_evidence_state !== "complete" || observed === undefined) {
        throw new Error("Complete captured workspace structure is required for this assertion.")
      }
      if (observed.state === "missing") {
        return { ...assertion, path: observed.path, present: false }
      }
      return {
        ...assertion,
        path: observed.path,
        present: true,
        minimum_bytes: observed.total_bytes,
        maximum_bytes: observed.total_bytes,
        sha256: observed.digest_state === "complete" ? observed.sha256 : null,
      }
    }
    case "artifact": {
      const completeScopes = new Set(
        (evidence.artifact_scopes ?? [])
          .filter((item) => item.state === "complete")
          .map((item) => item.scope),
      )
      const observed = (evidence.artifacts ?? []).find(
        (item) => completeScopes.has(item.scope) && isPortableObservedByteCount(item.size_bytes),
      )
      if (observed === undefined) {
        throw new Error("Complete captured artifact structure is required for this assertion.")
      }
      const structuralMatches = (evidence.artifacts ?? []).filter(
        (item) =>
          item.scope === observed.scope &&
          item.filename === observed.filename &&
          item.content_type === observed.content_type &&
          item.size_bytes === observed.size_bytes,
      )
      const retainedDigest =
        observed.digest_state === "complete" &&
        observed.sha256 != null &&
        structuralMatches.every((item) => item.digest_state === "complete")
          ? observed.sha256
          : null
      const retainedText =
        observed.text_state === "available" &&
        observed.text != null &&
        !isPythonBlank(observed.text) &&
        durableTextLength(observed.text, "Observed artifact text") <= 32_768 &&
        structuralMatches.every((item) => item.text_state === "available" && item.text != null)
          ? observed.text
          : null
      const exactMatches = structuralMatches.filter(
        (item) =>
          (retainedDigest == null || item.sha256 === retainedDigest) &&
          (retainedText == null || item.text?.includes(retainedText) === true),
      ).length
      return {
        ...assertion,
        scope: observed.scope,
        filename: observed.filename,
        content_type: observed.content_type,
        minimum_bytes: observed.size_bytes,
        maximum_bytes: observed.size_bytes,
        sha256: retainedDigest,
        text_contains: retainedText,
        min_count: exactMatches,
        max_count: exactMatches,
      }
    }
    case "memory_attribution": {
      if (capturedAssertionSuggestionUnavailable(assertion.kind, evidence)) {
        throw new Error(
          "Complete captured memory attribution without indeterminate exposure is required for this assertion.",
        )
      }
      const counts = memoryAttributionCounts(evidence.memory_attribution)
      return {
        ...assertion,
        min_admitted_items: counts.admittedItems,
        max_admitted_items: counts.admittedItems,
        min_provider_exposures: counts.providerExposures,
        max_provider_exposures: counts.providerExposures,
      }
    }
    case "max_tool_calls":
      return evidence.tool_calls_started == null
        ? assertion
        : { ...assertion, maximum: evidence.tool_calls_started }
    case "max_model_steps":
      return evidence.model_steps == null
        ? assertion
        : { ...assertion, maximum: evidence.model_steps }
    case "usage_recorded": {
      const totalTokens = safeEvidenceInteger(evidence.total_tokens)
      return totalTokens == null ? assertion : { ...assertion, min_total_tokens: totalTokens }
    }
    case "max_total_tokens": {
      const totalTokens = safeEvidenceInteger(evidence.total_tokens)
      return totalTokens == null ? assertion : { ...assertion, maximum: totalTokens }
    }
    case "max_estimated_cost": {
      const cost = evidence.costs[0]
      return cost == null
        ? assertion
        : { ...assertion, maximum: cost.total_cost, currency: cost.currency }
    }
  }
  throw new Error(`Unsupported captured assertion kind: ${String(kind)}`)
}

function memoryAttributionCounts(evidence: CapturedAssertionEvidence["memory_attribution"]): {
  admittedItems: number
  providerExposures: number
} {
  let admittedItems = 0
  let providerExposures = 0
  for (const source of evidence.sources ?? []) {
    const attribution = source.attribution
    if (attribution == null) continue
    for (const receipt of attribution.receipts ?? []) admittedItems += receipt.admitted_count
    for (const exposure of attribution.exposures ?? []) {
      if (exposure.provider_exposure_proven) providerExposures += 1
    }
  }
  if (!Number.isSafeInteger(admittedItems) || !Number.isSafeInteger(providerExposures)) {
    throw new Error("Captured memory counts exceed the portable browser integer range.")
  }
  return { admittedItems, providerExposures }
}

function safeEvidenceInteger(value: string | null | undefined): number | null {
  if (value == null || !/^(0|[1-9]\d*)$/.test(value)) return null
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) ? parsed : null
}

function requireProcessEvent(value: string): asserts value is ProcessEventKind {
  if (!PROCESS_EVENT_KINDS.has(value)) {
    throw new Error("Process event must use Cayu's closed portable vocabulary.")
  }
}

export function createCapturedEvaluationAssertionDraft(
  kind: PromotionAssertionKind,
  existing: readonly PromotionAssertion[],
  evidence: CapturedEvaluationCandidateV1["evidence"],
): CapturedAssertionDraft {
  if (capturedAssertionSuggestionUnavailable(kind, evidence)) {
    return {
      assertion: createPromotionAssertion(kind, existing),
      source: "expectation",
    }
  }
  return {
    assertion: createCapturedEvaluationAssertion(kind, existing, evidence),
    source: "observed",
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

function requireWorkspacePath(value: string): void {
  const length = durableTextLength(value, "Workspace path")
  const components = value.split("/")
  if (
    length < 1 ||
    length > 1_024 ||
    hasPythonOuterWhitespace(value) ||
    value.startsWith("/") ||
    value.normalize("NFC") !== value ||
    hasNonportableWorkspaceCharacter(value) ||
    components.some(
      (component) =>
        component.endsWith(" ") ||
        component.endsWith(".") ||
        /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/iu.test(component),
    )
  ) {
    throw new Error("Workspace path must be a canonical relative POSIX path.")
  }
  if (components.some((component) => component === "" || component === "." || component === "..")) {
    throw new Error("Workspace path must be a canonical relative POSIX path.")
  }
}

function hasNonportableWorkspaceCharacter(value: string): boolean {
  for (const character of value) {
    if ((character.codePointAt(0) ?? 0) < 0x20 || '<>:"\\|?*'.includes(character)) return true
  }
  return false
}

function requireOptionalByteRange(
  minimum: number | null | undefined,
  maximum: number | null | undefined,
  label: string,
): void {
  if (minimum != null) requireInteger(minimum, `${label} minimum`, 0, MAX_SAFE_COUNTER)
  if (maximum != null) requireInteger(maximum, `${label} maximum`, 0, MAX_SAFE_COUNTER)
  if (minimum != null && maximum != null && maximum < minimum) {
    throw new Error(`${label} maximum cannot be below its minimum.`)
  }
}

function requireOptionalSha256(value: string | null | undefined, label: string): void {
  if (value != null && !/^[a-f0-9]{64}$/.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256 digest.`)
  }
}

function requireOptionalBoundedText(
  value: string | null | undefined,
  label: string,
  maximum: number,
  rejectBlank: boolean,
): void {
  if (value == null) return
  const length = durableTextLength(value, label)
  if (length < 1 || length > maximum || (rejectBlank && isPythonBlank(value))) {
    throw new Error(`${label} must contain 1 to ${maximum.toLocaleString("en-US")} characters.`)
  }
}

function requireOptionalPrintableAscii(
  value: string | null | undefined,
  label: string,
  maximum: number,
): void {
  requireOptionalBoundedText(value, label, maximum, true)
  if (value != null && !/^[\x20-\x7e]+$/.test(value)) {
    throw new Error(`${label} must contain printable ASCII characters only.`)
  }
}

function requireInteger(value: number, label: string, minimum: number, maximum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be a whole number from ${minimum} to ${maximum}.`)
  }
}

function validateToolJsonSubset(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object.`)
  }
  const pending: Array<{ value: unknown; depth: number }> = [{ value, depth: 0 }]
  let nodes = 0
  while (pending.length > 0) {
    const current = pending.pop()
    if (current === undefined) break
    nodes += 1
    if (nodes > 128) throw new Error(`${label} cannot contain more than 128 JSON values.`)
    if (current.depth >= 12 && typeof current.value === "object" && current.value !== null) {
      throw new Error(`${label} cannot exceed 12 nested JSON containers.`)
    }
    if (typeof current.value === "string") {
      durableTextLength(current.value, label)
    } else if (typeof current.value === "number") {
      if (!Number.isFinite(current.value)) throw new Error(`${label} requires finite numbers.`)
      if (Number.isInteger(current.value) && !Number.isSafeInteger(current.value)) {
        throw new Error(`${label} integers must be exactly representable in the browser.`)
      }
    } else if (Array.isArray(current.value)) {
      for (const item of current.value) pending.push({ value: item, depth: current.depth + 1 })
    } else if (typeof current.value === "object" && current.value !== null) {
      for (const [key, item] of Object.entries(current.value)) {
        durableTextLength(key, label)
        pending.push({ value: item, depth: current.depth + 1 })
      }
    } else if (current.value !== null && typeof current.value !== "boolean") {
      throw new Error(`${label} contains a non-JSON value.`)
    }
  }
  const encoded = JSON.stringify(value)
  if (new TextEncoder().encode(encoded).byteLength > 4_096) {
    throw new Error(`${label} cannot exceed 4,096 encoded JSON bytes.`)
  }
  return value as Record<string, unknown>
}

function requireTerminalStatus(value: string): void {
  if (value !== "completed" && value !== "failed") {
    throw new Error("Status assertions must expect completed or failed.")
  }
}

function requireChildTerminalStatus(value: string): void {
  if (value !== "completed" && value !== "failed" && value !== "interrupted") {
    throw new Error("Child status assertions must expect completed, failed, or interrupted.")
  }
}

export function requirePortableId(value: string, label: string): void {
  if (!PORTABLE_ID_PATTERN.test(value)) {
    throw new Error(
      `${label} must start with a lowercase letter and use lowercase letters, digits, '.', '_', or '-'.`,
    )
  }
}

export function requireBoundedCleanText(value: string, label: string, maximum: number): void {
  const length = durableTextLength(value, label)
  if (length === 0 || hasPythonOuterWhitespace(value) || length > maximum) {
    throw new Error(`${label} must contain 1 to ${maximum} characters without outer whitespace.`)
  }
}

export function requireOptionalCleanText(
  value: string | null | undefined,
  label: string,
  maximum: number,
): void {
  if (value === null || value === undefined) return
  requireBoundedCleanText(value, label, maximum)
}

export function durableTextLength(value: string, label: string): number {
  let length = 0
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index)
    if (codeUnit === 0) throw new Error(`${label} cannot contain NUL characters.`)
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const trailing = value.charCodeAt(index + 1)
      if (!(trailing >= 0xdc00 && trailing <= 0xdfff)) {
        throw new Error(`${label} must contain valid Unicode scalar text.`)
      }
      index += 1
    } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
      throw new Error(`${label} must contain valid Unicode scalar text.`)
    }
    length += 1
  }
  return length
}

export function isPythonBlank(value: string): boolean {
  for (const character of value) {
    if (!isPythonWhitespace(character.codePointAt(0) as number)) return false
  }
  return true
}

function hasPythonOuterWhitespace(value: string): boolean {
  if (value.length === 0) return false
  const lastIndex =
    value.charCodeAt(value.length - 1) >= 0xdc00 && value.charCodeAt(value.length - 1) <= 0xdfff
      ? value.length - 2
      : value.length - 1
  return (
    isPythonWhitespace(value.codePointAt(0) as number) ||
    isPythonWhitespace(value.codePointAt(lastIndex) as number)
  )
}

function isPythonWhitespace(codePoint: number): boolean {
  // Python's str.strip/isspace set is the portable-model authority. ECMAScript
  // trim differs for U+0085 and U+FEFF, so do not use it for corpus validation.
  return (
    (codePoint >= 0x0009 && codePoint <= 0x000d) ||
    (codePoint >= 0x001c && codePoint <= 0x0020) ||
    codePoint === 0x0085 ||
    codePoint === 0x00a0 ||
    codePoint === 0x1680 ||
    (codePoint >= 0x2000 && codePoint <= 0x200a) ||
    codePoint === 0x2028 ||
    codePoint === 0x2029 ||
    codePoint === 0x202f ||
    codePoint === 0x205f ||
    codePoint === 0x3000
  )
}
