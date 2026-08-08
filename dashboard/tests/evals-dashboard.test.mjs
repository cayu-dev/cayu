import assert from "node:assert/strict"
import test from "node:test"

import {
  evalRunCanCancel,
  evalRunHasResult,
  evalRunIsActive,
  MAX_EVAL_CORPUS_FILE_BYTES,
  parseEvalCorpusFile,
  shortEvalIdentity,
} from "../src/lib/evals-dashboard.ts"

function run(status, result = null) {
  return { status, result }
}

test("eval lifecycle helpers distinguish active, cancellable, and published runs", () => {
  assert.equal(evalRunIsActive(run("queued")), true)
  assert.equal(evalRunIsActive(run("cancelling")), true)
  assert.equal(evalRunIsActive(run("cancelled")), false)
  assert.equal(evalRunCanCancel(run("queued")), true)
  assert.equal(evalRunCanCancel(run("running")), true)
  assert.equal(evalRunCanCancel(run("cancelling")), false)
  assert.equal(evalRunHasResult(run("completed", { revision: "sha256:result" })), true)
  assert.equal(evalRunHasResult(run("completed")), false)
  assert.equal(evalRunHasResult(run("failed", { revision: "sha256:result" })), false)
})

test("eval corpus import preflights bytes and the minimum v1 envelope", async () => {
  const corpus = {
    schema_version: 1,
    revision: "sha256:corpus",
    target_key: "support.regressions",
    evidence_policy: { schema_version: 1 },
    suites: [],
    cases: [],
  }
  assert.deepEqual(await parseEvalCorpusFile(new Blob([JSON.stringify(corpus)])), corpus)

  await assert.rejects(parseEvalCorpusFile(new Blob(["not json"])), /not valid JSON/)
  await assert.rejects(
    parseEvalCorpusFile(new Blob([JSON.stringify({ ...corpus, schema_version: 2 })])),
    /corpus v1/,
  )
  await assert.rejects(
    parseEvalCorpusFile(new Blob([new Uint8Array(MAX_EVAL_CORPUS_FILE_BYTES + 1)])),
    /larger than the supported 8 MiB/,
  )
})

test("eval identities are compact without discarding short identifiers", () => {
  assert.equal(shortEvalIdentity("sha256:1234567890abcdef"), "1234567890ab…")
  assert.equal(shortEvalIdentity("suite-1"), "suite-1")
})
