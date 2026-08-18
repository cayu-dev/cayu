import assert from "node:assert/strict"
import test from "node:test"

import {
  createPromotionAssertion,
  PROMOTION_ASSERTION_KINDS,
  parsePromotionInteger,
  previewMatchesDraft,
  promotionDraftFromCandidate,
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
