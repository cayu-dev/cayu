import { unicodeScalarLength } from "./durable-text.ts"
import type {
  EvalMemoryAttributionEvidenceV1,
  MemoryExperimentReportRequest,
} from "./generated/server-api"

const MEMORY_REPORT_MAX_BYTES = 32 * 1024 * 1024
const MEMORY_REPORT_MAX_CASES = 1_000
const MEMORY_REPORT_MAX_VARIANTS = 128
const MEMORY_REPORT_MAX_REPETITIONS = 1_000
const MEMORY_REPORT_MAX_ID_CHARS = 128

export type ConclusiveMemoryCounts = {
  admittedItems: number
  providerExposures: number
}

export function conclusiveMemoryCounts(
  evidence: EvalMemoryAttributionEvidenceV1,
): ConclusiveMemoryCounts | null {
  if (evidence.completeness !== "complete" || evidence.has_indeterminate_exposure) return null
  const sources = evidence.sources ?? []
  const admittedItems = sources.reduce(
    (total, source) =>
      total +
      (source.attribution?.receipts ?? []).reduce(
        (sourceTotal, receipt) => sourceTotal + receipt.admitted_count,
        0,
      ),
    0,
  )
  const providerExposures = sources.reduce(
    (total, source) =>
      total +
      (source.attribution?.exposures ?? []).filter((item) => item.provider_exposure_proven).length,
    0,
  )
  if (!Number.isSafeInteger(admittedItems) || !Number.isSafeInteger(providerExposures)) return null
  return { admittedItems, providerExposures }
}

export function memoryExposureCertainty(
  evidence: EvalMemoryAttributionEvidenceV1,
): "determinate" | "indeterminate" | "unavailable" {
  if (evidence.has_indeterminate_exposure) return "indeterminate"
  return conclusiveMemoryCounts(evidence) === null ? "unavailable" : "determinate"
}

export function parseMemoryExperimentReportRequest(
  source: string,
  byteSize: number,
): MemoryExperimentReportRequest {
  requireMemoryExperimentReportByteSize(byteSize)
  let value: unknown
  try {
    value = JSON.parse(source)
  } catch {
    throw new Error("The memory experiment request must be valid JSON.")
  }
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The memory experiment request must be a JSON object.")
  }
  const document = value as Record<string, unknown>
  if (document.schema_version !== undefined && document.schema_version !== 1) {
    throw new Error("Only memory experiment request schema version 1 is supported.")
  }
  if (
    typeof document.experiment_id !== "string" ||
    document.experiment_id.length === 0 ||
    unicodeScalarLength(document.experiment_id) > MEMORY_REPORT_MAX_ID_CHARS ||
    !Array.isArray(document.cases) ||
    document.cases.length < 1 ||
    document.cases.length > MEMORY_REPORT_MAX_CASES ||
    !Array.isArray(document.variants) ||
    document.variants.length < 2 ||
    document.variants.length > MEMORY_REPORT_MAX_VARIANTS ||
    typeof document.repetitions !== "number" ||
    !Number.isInteger(document.repetitions) ||
    document.repetitions < 1 ||
    document.repetitions > MEMORY_REPORT_MAX_REPETITIONS ||
    typeof document.baseline_variant_id !== "string" ||
    document.baseline_variant_id.length === 0 ||
    unicodeScalarLength(document.baseline_variant_id) > MEMORY_REPORT_MAX_ID_CHARS
  ) {
    throw new Error("The JSON has missing or out-of-range memory experiment request fields.")
  }
  return value as MemoryExperimentReportRequest
}

export async function parseMemoryExperimentReportFile(
  file: Pick<File, "size" | "text">,
): Promise<MemoryExperimentReportRequest> {
  // `File.text()` materializes the complete file in browser memory. Reject the
  // browser-authoritative byte size before beginning that read so the preflight
  // ceiling is a real resource bound rather than only a post-read parse bound.
  requireMemoryExperimentReportByteSize(file.size)
  return parseMemoryExperimentReportRequest(await file.text(), file.size)
}

function requireMemoryExperimentReportByteSize(byteSize: number): void {
  if (!Number.isSafeInteger(byteSize) || byteSize < 0 || byteSize > MEMORY_REPORT_MAX_BYTES) {
    throw new Error("The memory experiment request exceeds the 32 MiB limit.")
  }
}
