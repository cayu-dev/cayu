import assert from "node:assert/strict"
import test from "node:test"

import {
  capturedAssertionSuggestionUnavailable,
  capturedEvaluationDraftFromCandidate,
  capturedEvaluationPreviewMatchesDraft,
  createCapturedEvaluationAssertion,
  createCapturedEvaluationAssertionDraft,
  createPromotionAssertion,
  PROMOTION_ASSERTION_KINDS,
  parsePromotionInteger,
  previewMatchesDraft,
  promotionDraftFromCandidate,
  validateCapturedEvaluationDraft,
  validatePromotionDraft,
} from "../src/lib/evaluation-promotion.ts"

function candidate() {
  return {
    revision: `sha256:${"1".repeat(64)}`,
    evidence: { revision: `sha256:${"2".repeat(64)}` },
    suite: {
      id: "support.regressions",
      revision: `sha256:${"3".repeat(64)}`,
      name: "Support regressions",
      description: "Captured support cases",
      trial_request: { trials: 1, timeout_seconds: 300 },
    },
    case: {
      id: "case-one",
      revision: `sha256:${"4".repeat(64)}`,
      suite_id: "support.regressions",
      name: "Answer a customer",
      description: null,
      source: {},
      input: { messages: [{ role: "user", text: "help me" }] },
      assertions: [{ id: "session-completed", kind: "root_status", expected: "completed" }],
    },
  }
}

function draftFromCandidate(source = candidate()) {
  return promotionDraftFromCandidate(source, source.revision)
}

test("promotion authoring rejects runtime-owned opaque external input", () => {
  const source = candidate()
  source.case.input = {
    opaque_external_case_ref: { id: "private-case", revision: `sha256:${"9".repeat(64)}` },
  }

  assert.throws(
    () => promotionDraftFromCandidate(source, source.revision),
    /runtime-owned opaque external input/,
  )
})

function capturedCandidate() {
  const source = candidate()
  source.case.input = null
  source.evidence = {
    revision: `sha256:${"2".repeat(64)}`,
    policy_revision: `sha256:${"3".repeat(64)}`,
    pricing_profile_fingerprint: null,
    root_evidence_available: true,
    root_status: "failed",
    child_evidence_state: "complete",
    child_statuses: ["completed", "completed"],
    final_output_state: "complete",
    final_output: "Observed answer",
    tool_evidence_state: "complete",
    tool_call_evidence_state: "complete",
    requested_tool_names: ["search", "read"],
    started_tool_names: ["search"],
    tool_calls_started: 2,
    tool_calls: [
      {
        invocation_index: 1,
        invocation_revision: `sha256:${"5".repeat(64)}`,
        tool_name: "search",
        occurrence: 1,
        arguments: { state: "available", value: { query: "refund", limit: 5 } },
        result: {
          state: "available",
          value: { content: "found", structured: { status: "ok" }, is_error: false },
        },
      },
    ],
    process_event_evidence_state: "complete",
    process_events: [
      "session_started",
      "tool_approval_requested",
      "tool_approved",
      "session_completed",
    ],
    workspace_evidence_state: "complete",
    workspace_files: [
      {
        path: "output/report.txt",
        state: "present",
        total_bytes: 42,
        digest_state: "complete",
        sha256: "a".repeat(64),
      },
    ],
    artifact_scopes: [{ scope: "session", state: "complete" }],
    artifacts: [
      {
        observation_index: 1,
        scope: "session",
        filename: "report.txt",
        content_type: "text/plain",
        size_bytes: 42,
        digest_state: "complete",
        sha256: "b".repeat(64),
        text_state: "unsupported",
        text: null,
      },
    ],
    model_step_evidence_state: "complete",
    model_steps: 3,
    usage_evidence_state: "complete",
    total_tokens: "42",
    costs: [
      {
        currency: "USD",
        total_cost: "0.25",
        model_steps: 3,
        priced_model_steps: 3,
        unpriced_model_steps: 0,
      },
    ],
    memory_attribution: {
      completeness: "complete",
      has_indeterminate_exposure: false,
      sources: [
        {
          attribution: {
            receipts: [{ admitted_count: 2 }],
            exposures: [{ provider_exposure_proven: true }],
          },
        },
      ],
    },
  }
  return source
}

test("candidate projection owns a complete authority-free editable draft", () => {
  const source = candidate()
  const draft = draftFromCandidate(source)

  assert.deepEqual(draft, {
    expected_baseline_revision: source.revision,
    suite: {
      id: "support.regressions",
      name: "Support regressions",
      description: "Captured support cases",
      trial_request: { trials: 1, timeout_seconds: 300 },
    },
    case: {
      id: "case-one",
      suite_id: "support.regressions",
      name: "Answer a customer",
      description: null,
      input: { messages: [{ role: "user", text: "help me" }] },
      assertions: [{ id: "session-completed", kind: "root_status", expected: "completed" }],
    },
  })

  draft.case.input.messages[0].text = "edited"
  draft.case.assertions[0].id = "edited-id"
  assert.equal(source.case.input.messages[0].text, "help me")
  assert.equal(source.case.assertions[0].id, "session-completed")
})

test("all portable assertion kinds get valid deterministic editor defaults", () => {
  const assertions = []
  for (const kind of PROMOTION_ASSERTION_KINDS) {
    assertions.push(createPromotionAssertion(kind, assertions))
  }
  assertions.push(createPromotionAssertion("root_status", assertions))
  assert.equal(assertions.at(-1).id, "root_status-2")

  const draft = draftFromCandidate()
  draft.case.assertions = assertions
  assert.deepEqual(validatePromotionDraft(draft), { ok: true, draft })
})

test("captured drafts omit replay input and quick-add assertions use observed facts", () => {
  const source = capturedCandidate()
  const draft = capturedEvaluationDraftFromCandidate(source, source.revision)
  assert.equal("input" in draft.case, false)
  assert.deepEqual(validateCapturedEvaluationDraft(draft), { ok: true, draft })
  assert.equal(
    capturedEvaluationPreviewMatchesDraft(
      { baseline_revision: source.revision, candidate: source },
      draft,
    ),
    true,
  )

  const assertions = []
  for (const kind of PROMOTION_ASSERTION_KINDS) {
    assertions.push(createCapturedEvaluationAssertion(kind, assertions, source.evidence))
  }
  assert.deepEqual(
    assertions.find((item) => item.kind === "root_status"),
    {
      id: "root_status",
      kind: "root_status",
      expected: "failed",
    },
  )
  assert.equal(
    assertions.find((item) => item.kind === "final_output_equals").expected,
    "Observed answer",
  )
  assert.deepEqual(assertions.find((item) => item.kind === "tools_called_in_order").tool_names, [
    "search",
    "read",
  ])
  assert.equal(assertions.find((item) => item.kind === "max_total_tokens").maximum, 42)
  assert.deepEqual(
    assertions.find((item) => item.kind === "max_estimated_cost"),
    {
      id: "max_estimated_cost",
      kind: "max_estimated_cost",
      maximum: "0.25",
      currency: "USD",
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "memory_attribution"),
    {
      id: "memory_attribution",
      kind: "memory_attribution",
      min_admitted_items: 2,
      max_admitted_items: 2,
      min_provider_exposures: 1,
      max_provider_exposures: 1,
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "tool_arguments_contain"),
    {
      id: "tool_arguments_contain",
      kind: "tool_arguments_contain",
      tool_name: "search",
      occurrence: 1,
      expected_subset: { query: "refund", limit: 5 },
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "tool_result_contains"),
    {
      id: "tool_result_contains",
      kind: "tool_result_contains",
      tool_name: "search",
      occurrence: 1,
      expected_subset: { content: "found", structured: { status: "ok" }, is_error: false },
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "process_event"),
    {
      id: "process_event",
      kind: "process_event",
      event: "tool_approval_requested",
      min_count: 1,
      max_count: 1,
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "process_events_in_order"),
    {
      id: "process_events_in_order",
      kind: "process_events_in_order",
      events: ["session_started", "tool_approval_requested", "tool_approved", "session_completed"],
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "workspace_file"),
    {
      id: "workspace_file",
      kind: "workspace_file",
      path: "output/report.txt",
      present: true,
      minimum_bytes: 42,
      maximum_bytes: 42,
      sha256: "a".repeat(64),
    },
  )
  assert.deepEqual(
    assertions.find((item) => item.kind === "artifact"),
    {
      id: "artifact",
      kind: "artifact",
      scope: "session",
      filename: "report.txt",
      content_type: "text/plain",
      minimum_bytes: 42,
      maximum_bytes: 42,
      sha256: "b".repeat(64),
      text_contains: null,
      min_count: 1,
      max_count: 1,
    },
  )
})

test("structural assertions validate canonical paths, bounds, digests, and text", () => {
  const valid = draftFromCandidate()
  valid.case.assertions = [
    {
      id: "workspace",
      kind: "workspace_file",
      path: "output/report.txt",
      present: true,
      minimum_bytes: 1,
      maximum_bytes: 100,
      sha256: "a".repeat(64),
    },
    {
      id: "artifact",
      kind: "artifact",
      scope: "session",
      filename: "report.txt",
      content_type: "text/plain",
      minimum_bytes: 1,
      maximum_bytes: 100,
      sha256: "b".repeat(64),
      text_contains: "ready",
      min_count: 1,
      max_count: 1,
    },
    {
      id: "memory",
      kind: "memory_attribution",
      min_admitted_items: 0,
      max_admitted_items: 1_000,
      min_provider_exposures: 0,
      max_provider_exposures: 100,
    },
  ]
  assert.deepEqual(validatePromotionDraft(valid), { ok: true, draft: valid })

  const traversal = structuredClone(valid)
  traversal.case.assertions[0].path = "../secret"
  assert.match(validatePromotionDraft(traversal).error, /canonical relative POSIX path/)

  const paddedPath = structuredClone(valid)
  paddedPath.case.assertions[0].path = " output/report.txt"
  assert.match(validatePromotionDraft(paddedPath).error, /canonical relative POSIX path/)

  for (const path of ["C:/secret", "report:stream", "CON.txt", "output./report.txt", "café.txt"]) {
    const nonportablePath = structuredClone(valid)
    nonportablePath.case.assertions[0].path = path
    assert.match(validatePromotionDraft(nonportablePath).error, /canonical relative POSIX path/)
  }

  const absentWithSize = structuredClone(valid)
  absentWithSize.case.assertions[0].present = false
  assert.match(validatePromotionDraft(absentWithSize).error, /absent workspace file/)

  const badDigest = structuredClone(valid)
  badDigest.case.assertions[1].sha256 = "ABC"
  assert.match(validatePromotionDraft(badDigest).error, /lowercase SHA-256/)

  const reversedSize = structuredClone(valid)
  reversedSize.case.assertions[1].minimum_bytes = 101
  assert.match(validatePromotionDraft(reversedSize).error, /cannot be below its minimum/)

  const oversizedText = structuredClone(valid)
  oversizedText.case.assertions[1].text_contains = "x".repeat(32_769)
  assert.match(validatePromotionDraft(oversizedText).error, /32,768/)

  const oversizedUtf8Text = structuredClone(valid)
  oversizedUtf8Text.case.assertions[1].text_contains = "😀".repeat(17_000)
  assert.match(validatePromotionDraft(oversizedUtf8Text).error, /65,536 UTF-8 bytes/)

  const oversizedMemoryItems = structuredClone(valid)
  oversizedMemoryItems.case.assertions[2].max_admitted_items = 1_001
  assert.match(validatePromotionDraft(oversizedMemoryItems).error, /from 0 to 1000/)

  const oversizedMemoryExposures = structuredClone(valid)
  oversizedMemoryExposures.case.assertions[2].max_provider_exposures = 101
  assert.match(validatePromotionDraft(oversizedMemoryExposures).error, /from 0 to 100/)
})

test("captured structural suggestions require complete observations", () => {
  const evidence = capturedCandidate().evidence
  assert.equal(capturedAssertionSuggestionUnavailable("workspace_file", evidence), false)
  assert.equal(capturedAssertionSuggestionUnavailable("artifact", evidence), false)
  assert.equal(capturedAssertionSuggestionUnavailable("memory_attribution", evidence), false)

  const incomplete = structuredClone(evidence)
  incomplete.workspace_evidence_state = "limit_exceeded"
  incomplete.artifact_scopes[0].state = "limit_exceeded"
  assert.equal(capturedAssertionSuggestionUnavailable("workspace_file", incomplete), true)
  assert.equal(capturedAssertionSuggestionUnavailable("artifact", incomplete), true)
  assert.equal(
    createCapturedEvaluationAssertionDraft("workspace_file", [], incomplete).source,
    "expectation",
  )
  assert.equal(
    createCapturedEvaluationAssertionDraft("artifact", [], incomplete).source,
    "expectation",
  )

  const incompleteMemory = structuredClone(evidence)
  incompleteMemory.memory_attribution.completeness = "truncated"
  assert.equal(capturedAssertionSuggestionUnavailable("memory_attribution", incompleteMemory), true)
  const memoryExpectation = createCapturedEvaluationAssertionDraft(
    "memory_attribution",
    [],
    incompleteMemory,
  )
  assert.equal(memoryExpectation.source, "expectation")
  assert.deepEqual(memoryExpectation.assertion, {
    id: "memory_attribution",
    kind: "memory_attribution",
    min_admitted_items: 1,
    max_admitted_items: null,
    min_provider_exposures: 1,
    max_provider_exposures: null,
  })

  const indeterminateMemory = structuredClone(evidence)
  indeterminateMemory.memory_attribution.has_indeterminate_exposure = true
  assert.equal(
    capturedAssertionSuggestionUnavailable("memory_attribution", indeterminateMemory),
    true,
  )

  const nonportableWorkspaceSize = structuredClone(evidence)
  nonportableWorkspaceSize.workspace_files[0].total_bytes = Number.MAX_SAFE_INTEGER + 1
  assert.equal(
    capturedAssertionSuggestionUnavailable("workspace_file", nonportableWorkspaceSize),
    true,
  )
  assert.equal(
    createCapturedEvaluationAssertionDraft("workspace_file", [], nonportableWorkspaceSize).source,
    "expectation",
  )

  const nonportableArtifactSize = structuredClone(evidence)
  nonportableArtifactSize.artifacts[0].size_bytes = Number.MAX_SAFE_INTEGER + 1
  assert.equal(capturedAssertionSuggestionUnavailable("artifact", nonportableArtifactSize), true)
  assert.equal(
    createCapturedEvaluationAssertionDraft("artifact", [], nonportableArtifactSize).source,
    "expectation",
  )
})

test("captured artifact suggestions retain only complete bounded public text", () => {
  const evidence = capturedCandidate().evidence
  evidence.artifacts[0].text_state = "available"
  evidence.artifacts[0].text = "public report ready"
  evidence.artifacts.push({
    ...structuredClone(evidence.artifacts[0]),
    observation_index: 2,
    text: "different public report",
  })

  assert.deepEqual(createCapturedEvaluationAssertion("artifact", [], evidence), {
    id: "artifact",
    kind: "artifact",
    scope: "session",
    filename: "report.txt",
    content_type: "text/plain",
    minimum_bytes: 42,
    maximum_bytes: 42,
    sha256: "b".repeat(64),
    text_contains: "public report ready",
    min_count: 1,
    max_count: 1,
  })

  evidence.artifacts[0].text = "x".repeat(32_769)
  const structuralOnly = createCapturedEvaluationAssertion("artifact", [], evidence)
  assert.equal(structuralOnly.text_contains, null)
  assert.equal(structuralOnly.min_count, 2)
  assert.equal(structuralOnly.max_count, 2)
})

test("captured artifact suggestions count only constraints retained in the assertion", () => {
  const evidence = capturedCandidate().evidence
  evidence.artifacts[0].text_state = "available"
  evidence.artifacts[0].text = "public report"
  evidence.artifacts.push({
    ...structuredClone(evidence.artifacts[0]),
    observation_index: 2,
    digest_state: "limit_exceeded",
    sha256: null,
    text_state: "unavailable",
    text: null,
  })

  const assertion = createCapturedEvaluationAssertion("artifact", [], evidence)
  assert.equal(assertion.sha256, null)
  assert.equal(assertion.text_contains, null)
  assert.equal(assertion.min_count, 2)
  assert.equal(assertion.max_count, 2)
})

test("process assertions use a closed vocabulary and bounded exact order", () => {
  const valid = draftFromCandidate()
  valid.case.assertions = [
    {
      id: "approval",
      kind: "process_event",
      event: "tool_approval_requested",
      min_count: 1,
      max_count: 1,
    },
    {
      id: "protocol",
      kind: "process_events_in_order",
      events: ["tool_approval_requested", "tool_approved", "tool_call_started"],
    },
    {
      id: "child-interrupted",
      kind: "child_status",
      expected: "interrupted",
      min_count: 1,
      max_count: null,
    },
  ]
  assert.deepEqual(validatePromotionDraft(valid), { ok: true, draft: valid })

  const raw = structuredClone(valid)
  raw.case.assertions[0].event = "tool.call.approval_requested"
  assert.match(validatePromotionDraft(raw).error, /closed portable vocabulary/)

  const emptyOrder = structuredClone(valid)
  emptyOrder.case.assertions[1].events = []
  assert.match(validatePromotionDraft(emptyOrder).error, /between 1 and 256/)

  const reversedRange = structuredClone(valid)
  reversedRange.case.assertions[0].min_count = 2
  reversedRange.case.assertions[0].max_count = 1
  assert.match(validatePromotionDraft(reversedRange).error, /cannot be below its minimum/)
})

test("captured process suggestions never invent facts when evidence is incomplete or oversized", () => {
  const complete = capturedCandidate().evidence
  complete.process_events = [
    "session_started",
    ...Array(257).fill("tool_call_started"),
    "session_completed",
  ]

  assert.equal(capturedAssertionSuggestionUnavailable("process_events_in_order", complete), false)
  assert.deepEqual(createCapturedEvaluationAssertion("process_events_in_order", [], complete), {
    id: "process_events_in_order",
    kind: "process_events_in_order",
    events: ["session_started", "session_completed"],
  })
  assert.deepEqual(
    createCapturedEvaluationAssertionDraft("process_events_in_order", [], complete),
    {
      assertion: {
        id: "process_events_in_order",
        kind: "process_events_in_order",
        events: ["session_started", "session_completed"],
      },
      source: "observed",
    },
  )

  const unrepresentable = structuredClone(complete)
  unrepresentable.process_events = Array(257).fill("tool_call_started")
  assert.equal(
    capturedAssertionSuggestionUnavailable("process_events_in_order", unrepresentable),
    true,
  )
  assert.deepEqual(
    createCapturedEvaluationAssertionDraft("process_events_in_order", [], unrepresentable),
    {
      assertion: {
        id: "process_events_in_order",
        kind: "process_events_in_order",
        events: ["tool_approval_requested", "tool_approved"],
      },
      source: "expectation",
    },
  )
  assert.throws(
    () => createCapturedEvaluationAssertion("process_events_in_order", [], unrepresentable),
    /Complete captured process-event evidence with a representable order is required/,
  )

  const limited = structuredClone(complete)
  limited.process_event_evidence_state = "limit_exceeded"
  assert.equal(capturedAssertionSuggestionUnavailable("process_event", limited), true)
  assert.equal(capturedAssertionSuggestionUnavailable("process_events_in_order", limited), true)
  assert.deepEqual(createCapturedEvaluationAssertionDraft("process_event", [], limited), {
    assertion: {
      id: "process_event",
      kind: "process_event",
      event: "tool_approval_requested",
      min_count: 1,
      max_count: null,
    },
    source: "expectation",
  })
  assert.throws(
    () => createCapturedEvaluationAssertion("process_event", [], limited),
    /Complete captured process-event evidence is required/,
  )
  assert.throws(
    () => createCapturedEvaluationAssertion("process_events_in_order", [], limited),
    /Complete captured process-event evidence with a representable order is required/,
  )
})

test("tool JSON assertion validation is bounded and result fields are closed", () => {
  const valid = draftFromCandidate()
  valid.case.assertions = [
    {
      id: "arguments",
      kind: "tool_arguments_contain",
      tool_name: "search",
      occurrence: 2,
      expected_subset: { query: "refund", filters: { status: ["open"] } },
    },
    {
      id: "result",
      kind: "tool_result_contains",
      tool_name: "search",
      expected_subset: { structured: { status: "ok" }, is_error: false },
    },
  ]
  assert.deepEqual(validatePromotionDraft(valid), { ok: true, draft: valid })

  const arrayRoot = structuredClone(valid)
  arrayRoot.case.assertions[0].expected_subset = []
  assert.match(validatePromotionDraft(arrayRoot).error, /must be a JSON object/)

  const artifacts = structuredClone(valid)
  artifacts.case.assertions[1].expected_subset = { artifacts: [] }
  assert.match(validatePromotionDraft(artifacts).error, /content, structured, and is_error/)

  const oversized = structuredClone(valid)
  oversized.case.assertions[0].expected_subset = { query: "x".repeat(4_096) }
  assert.match(validatePromotionDraft(oversized).error, /4,096 encoded JSON bytes/)
})

test("draft validation rejects nonportable placement, duplicate assertions, and lossy counters", () => {
  const mismatched = draftFromCandidate()
  mismatched.case.suite_id = "another-suite"
  assert.match(validatePromotionDraft(mismatched).error, /must match/)

  const duplicate = draftFromCandidate()
  duplicate.case.assertions.push({
    id: "session-completed",
    kind: "max_tool_calls",
    maximum: 1,
  })
  assert.match(validatePromotionDraft(duplicate).error, /duplicated/)

  const lossy = draftFromCandidate()
  lossy.case.assertions = [
    { id: "tokens", kind: "max_total_tokens", maximum: Number.MAX_SAFE_INTEGER + 1 },
  ]
  assert.match(validatePromotionDraft(lossy).error, /whole number/)
})

test("promotion drafts reject target-owned model judge authority", () => {
  const draft = draftFromCandidate()
  draft.case.assertions = [
    {
      id: "quality",
      kind: "model_judge",
      evaluator_key: "trusted-quality-judge",
      rubric: "The answer must be correct and concise.",
      rubric_version: "v1",
      threshold: 0.8,
      include_transcript: false,
    },
  ]

  assert.match(validatePromotionDraft(draft).error, /Unsupported assertion kind: model_judge/)
})

test("draft validation follows portable Unicode and text boundaries", () => {
  const unicode = draftFromCandidate()
  unicode.case.input.messages[0].text = "😀".repeat(40_000)
  assert.deepEqual(validatePromotionDraft(unicode), { ok: true, draft: unicode })

  const blank = draftFromCandidate()
  blank.case.input.messages[0].text = "   "
  assert.match(validatePromotionDraft(blank).error, /cannot be blank/)

  blank.case.input.messages[0].text = "\ufeff"
  assert.deepEqual(validatePromotionDraft(blank), { ok: true, draft: blank })

  const surrogate = draftFromCandidate()
  surrogate.case.input.messages[0].text = "\ud800"
  assert.match(validatePromotionDraft(surrogate).error, /Unicode scalar/)

  const nul = draftFromCandidate()
  nul.case.name = "invalid\0name"
  assert.match(validatePromotionDraft(nul).error, /NUL/)
})

test("empty tool order is valid while portable decimal text remains bounded", () => {
  const noTools = draftFromCandidate()
  noTools.case.assertions = [{ id: "no-tools", kind: "tools_called_in_order", tool_names: [] }]
  assert.deepEqual(validatePromotionDraft(noTools), { ok: true, draft: noTools })

  const oversizedDecimal = draftFromCandidate()
  oversizedDecimal.case.assertions = [
    { id: "cost", kind: "max_estimated_cost", maximum: "1".repeat(65), currency: "USD" },
  ]
  assert.match(validatePromotionDraft(oversizedDecimal).error, /canonical non-negative decimal/)

  oversizedDecimal.case.assertions[0].maximum = "1.0"
  assert.match(validatePromotionDraft(oversizedDecimal).error, /canonical non-negative decimal/)
})

test("numeric editor parsing is canonical and bounded", () => {
  assert.equal(parsePromotionInteger("0", "Maximum", 0, 10), 0)
  assert.equal(parsePromotionInteger("10", "Maximum", 0, 10), 10)
  assert.throws(() => parsePromotionInteger("01", "Maximum", 0, 10), /whole number/)
  assert.throws(() => parsePromotionInteger("11", "Maximum", 0, 10), /from 0 to 10/)
})

test("preview equality compares only the complete editable projection", () => {
  const source = candidate()
  const draft = draftFromCandidate(source)
  const preview = {
    baseline_revision: source.revision,
    candidate: source,
    captured_score: {},
  }
  assert.equal(previewMatchesDraft(preview, draft), true)
  preview.baseline_revision = `sha256:${"9".repeat(64)}`
  assert.equal(previewMatchesDraft(preview, draft), false)
  preview.baseline_revision = source.revision
  draft.case.name = "Changed"
  assert.equal(previewMatchesDraft({ candidate: source, captured_score: {} }, draft), false)
})

test("promotion API forwards encoded identity, cancellation, and server filename", async () => {
  const originalWindow = globalThis.window
  const originalFetch = globalThis.fetch
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
  const calls = []
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    if (String(input).endsWith("/preview")) {
      return new Response(
        JSON.stringify({
          baseline_revision: candidate().revision,
          candidate: candidate(),
          captured_score: {},
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      )
    }
    return new Response("corpus", {
      status: 200,
      headers: {
        "content-type": "application/json",
        "content-disposition": 'attachment; filename="support.regressions.eval.json"',
      },
    })
  }
  try {
    const { exportEvaluationPromotion, previewEvaluationPromotion } = await import(
      "../src/lib/api.ts"
    )
    const controller = new AbortController()
    const draft = draftFromCandidate()
    await previewEvaluationPromotion("session/one", draft, controller.signal)
    const exported = await exportEvaluationPromotion(
      "session/one",
      { expected_candidate_revision: candidate().revision, candidate: candidate() },
      controller.signal,
    )

    assert.equal(exported.filename, "support.regressions.eval.json")
    assert.equal(await exported.blob.text(), "corpus")
    assert.equal(calls.length, 2)
    assert.equal(calls[0].input, "/api/evals/promotion/sessions/session%2Fone/preview")
    assert.deepEqual(JSON.parse(calls[0].init.body), { draft })
    assert.equal(calls[0].init.signal, controller.signal)
    assert.equal(calls[1].input, "/api/evals/promotion/sessions/session%2Fone/export")
    assert.equal(calls[1].init.credentials, "same-origin")
  } finally {
    globalThis.fetch = originalFetch
    if (originalWindow === undefined) delete globalThis.window
    else globalThis.window = originalWindow
  }
})
