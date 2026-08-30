import assert from "node:assert/strict"
import test from "node:test"
import {
  conclusiveMemoryCounts,
  memoryExposureCertainty,
  parseMemoryExperimentReportFile,
  parseMemoryExperimentReportRequest,
} from "../src/lib/eval-memory-presentation.ts"

function memoryEvidence(overrides = {}) {
  return {
    completeness: "complete",
    has_indeterminate_exposure: false,
    sources: [
      {
        attribution: {
          receipts: [{ admitted_count: 2 }, { admitted_count: 1 }],
          exposures: [{ provider_exposure_proven: true }, { provider_exposure_proven: false }],
        },
      },
    ],
    ...overrides,
  }
}

test("memory presentation exposes counts only for complete determinate evidence", () => {
  assert.deepEqual(conclusiveMemoryCounts(memoryEvidence()), {
    admittedItems: 3,
    providerExposures: 1,
  })
  assert.equal(memoryExposureCertainty(memoryEvidence()), "determinate")

  const truncated = memoryEvidence({ completeness: "truncated" })
  assert.equal(conclusiveMemoryCounts(truncated), null)
  assert.equal(memoryExposureCertainty(truncated), "unavailable")

  const indeterminate = memoryEvidence({ has_indeterminate_exposure: true })
  assert.equal(conclusiveMemoryCounts(indeterminate), null)
  assert.equal(memoryExposureCertainty(indeterminate), "indeterminate")

  const lossy = memoryEvidence({
    sources: [{ attribution: { receipts: [{ admitted_count: Number.MAX_SAFE_INTEGER + 1 }] } }],
  })
  assert.equal(conclusiveMemoryCounts(lossy), null)
  assert.equal(memoryExposureCertainty(lossy), "unavailable")
})

test("memory report preflight bounds untrusted files before rendering them", () => {
  const request = {
    schema_version: 1,
    experiment_id: "memory-campaign",
    cases: [{}],
    repetitions: 1,
    baseline_variant_id: "baseline",
    variants: [{}, {}],
  }
  const source = JSON.stringify(request)
  assert.deepEqual(parseMemoryExperimentReportRequest(source, source.length), request)
  const scalarBounded = { ...request, experiment_id: "🧠".repeat(128) }
  const scalarSource = JSON.stringify(scalarBounded)
  assert.deepEqual(
    parseMemoryExperimentReportRequest(scalarSource, new TextEncoder().encode(scalarSource).length),
    scalarBounded,
  )
  assert.throws(
    () => parseMemoryExperimentReportRequest(source, 32 * 1024 * 1024 + 1),
    /exceeds the 32 MiB limit/,
  )
  assert.throws(
    () =>
      parseMemoryExperimentReportRequest(
        JSON.stringify({ ...request, experiment_id: "x".repeat(129) }),
        1_000,
      ),
    /missing or out-of-range/,
  )
  assert.throws(
    () =>
      parseMemoryExperimentReportRequest(JSON.stringify({ ...request, repetitions: 1.5 }), 1_000),
    /missing or out-of-range/,
  )
})

test("memory report file preflight rejects oversized input before reading it", async () => {
  let read = false
  const oversizedFile = {
    size: 32 * 1024 * 1024 + 1,
    async text() {
      read = true
      return "{}"
    },
  }

  await assert.rejects(parseMemoryExperimentReportFile(oversizedFile), /exceeds the 32 MiB limit/)
  assert.equal(read, false)
})
