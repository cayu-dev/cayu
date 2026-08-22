import assert from "node:assert/strict"
import test from "node:test"

import {
  evalResultRevisionIsValid,
  evalRunIdIsValid,
  evalsSearchWithout,
  validateEvalsSearch,
} from "../src/lib/evals-search.ts"

const CORPUS_REVISION = `sha256:${"a".repeat(64)}`

test("eval search accepts only bounded known values", () => {
  assert.deepEqual(
    validateEvalsSearch({
      tab: "runs",
      target: `eval.${"f".repeat(64)}`,
      corpus: ` ${CORPUS_REVISION} `,
      suite: "suite-1",
      run: "eval-1",
      baseline: CORPUS_REVISION,
      status: "completed",
      corpora_cursor: "next-corpus",
      result: CORPUS_REVISION,
      results_cursor: "next-result",
      unknown: "ignored",
    }),
    {
      tab: "runs",
      target: `eval.${"f".repeat(64)}`,
      corpus: CORPUS_REVISION,
      suite: "suite-1",
      run: "eval-1",
      baseline: CORPUS_REVISION,
      status: "completed",
      corpora_cursor: "next-corpus",
      result: CORPUS_REVISION,
      results_cursor: "next-result",
    },
  )
})

test("eval search drops invalid status, arrays, blanks, and oversized values", () => {
  assert.deepEqual(
    validateEvalsSearch({
      tab: "other",
      status: "succeeded",
      corpus: ["one", "two"],
      suite: "UPPERCASE",
      target: "UPPERCASE",
      run: "x".repeat(129),
    }),
    {},
  )
})

test("eval search accepts only identifiers supported by the durable eval contract", () => {
  assert.deepEqual(
    validateEvalsSearch({
      corpus: "sha256:not-a-digest",
      suite: "suite/one",
      run: "eval/one",
      baseline: "eval one",
    }),
    {},
  )
  assert.equal(evalRunIdIsValid("eval-1234"), true)
  assert.equal(evalRunIdIsValid("eval/1234"), false)
  assert.equal(evalResultRevisionIsValid(CORPUS_REVISION), true)
  assert.equal(evalResultRevisionIsValid("eval-1234"), false)
})

test("eval search preserves the server's complete opaque cursor domain", () => {
  const validRunCursor = "a".repeat(624)
  assert.deepEqual(validateEvalsSearch({ runs_cursor: validRunCursor }), {
    runs_cursor: validRunCursor,
  })
  assert.deepEqual(validateEvalsSearch({ runs_cursor: "a".repeat(1_025) }), {})
  assert.deepEqual(validateEvalsSearch({ runs_cursor: "not+a+base64url+cursor" }), {})
})

test("eval search dependencies can be reset without mutating the current location", () => {
  const current = {
    tab: "catalog",
    corpus: "corpus-1",
    suite: "suite-1",
    cases_cursor: "case-page-2",
  }
  assert.deepEqual(evalsSearchWithout(current, "suite", "cases_cursor"), {
    tab: "catalog",
    corpus: "corpus-1",
  })
  assert.equal(current.suite, "suite-1")
})

test("eval result links retain only bounded result catalog state", () => {
  assert.deepEqual(
    validateEvalsSearch({
      tab: "results",
      target: "eval.target",
      result: CORPUS_REVISION,
      results_cursor: "next-result-page",
    }),
    {
      tab: "results",
      target: "eval.target",
      result: CORPUS_REVISION,
      results_cursor: "next-result-page",
    },
  )
  assert.deepEqual(validateEvalsSearch({ result: "sha256:short", results_cursor: "+" }), {})
})
