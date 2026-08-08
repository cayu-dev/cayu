import assert from "node:assert/strict"
import test from "node:test"

import { evalsSearchWithout, validateEvalsSearch } from "../src/lib/evals-search.ts"

test("eval search accepts only bounded known values", () => {
  assert.deepEqual(
    validateEvalsSearch({
      tab: "runs",
      corpus: " sha256:corpus ",
      suite: "suite-1",
      run: "eval-1",
      baseline: "eval-0",
      status: "completed",
      corpora_cursor: "next-corpus",
      unknown: "ignored",
    }),
    {
      tab: "runs",
      corpus: "sha256:corpus",
      suite: "suite-1",
      run: "eval-1",
      baseline: "eval-0",
      status: "completed",
      corpora_cursor: "next-corpus",
    },
  )
})

test("eval search drops invalid status, arrays, blanks, and oversized values", () => {
  assert.deepEqual(
    validateEvalsSearch({
      tab: "other",
      status: "succeeded",
      corpus: ["one", "two"],
      suite: " ",
      run: "x".repeat(513),
    }),
    {},
  )
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
