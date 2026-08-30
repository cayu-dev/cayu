import { Download, FileJson, FlaskConical, LoaderCircle, Upload } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  buildMemoryExperimentReport,
  downloadMemoryExperimentReportHtml,
  type EvalMemoryExperimentReport,
  type EvalMemoryExperimentReportRequest,
} from "@/lib/api"
import {
  conclusiveMemoryCounts,
  memoryExposureCertainty,
  parseMemoryExperimentReportFile,
} from "@/lib/eval-memory-presentation"
import { evalErrorMessage, shortEvalIdentity } from "@/lib/evals-dashboard"
import { formatCount } from "@/lib/format"
import type {
  EvalAssertionPresentationV1,
  EvalMemoryAttributionEvidenceV1,
  PublishedMemoryAttributionDetail,
} from "@/lib/generated/server-api"

export function MemoryEvidencePanel({
  evidence,
  assertions,
}: {
  evidence: EvalMemoryAttributionEvidenceV1
  assertions: readonly EvalAssertionPresentationV1[]
}) {
  const sources = evidence.sources ?? []
  const counts = conclusiveMemoryCounts(evidence)
  const limitations = [
    ...new Set([
      ...(evidence.limitations ?? []),
      ...sources.flatMap((source) => source.limitations ?? []),
    ]),
  ].sort()
  const semanticJudges = assertions.filter((assertion) => {
    const judge = assertion.structured_judge
    return (
      judge !== null &&
      judge !== undefined &&
      judge.detail.reference !== null &&
      judge.detail.reference !== undefined &&
      (judge.detail.rubric_id === "memory-use" ||
        assertion.assertion_id === "memory-use" ||
        assertion.assertion_id.startsWith("memory-use-"))
    )
  })
  const scoredSemanticJudges = semanticJudges.filter(
    (assertion) => assertion.outcome === "passed" || assertion.outcome === "failed",
  )

  return (
    <div
      className="space-y-3 rounded-lg border border-border bg-muted/20 p-3"
      data-testid="eval-memory-evidence"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm font-medium">Memory evaluation</div>
          <div className="text-xs text-muted-foreground">
            Structural proof, semantic judgment, and causal evidence are separate claims.
          </div>
        </div>
        <Badge variant="outline">{evidence.completeness}</Badge>
      </div>

      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <MemoryFact
          label="Retained sources"
          value={`${formatCount(evidence.retained_source_count)}/${formatCount(evidence.total_source_count)}`}
        />
        <MemoryFact
          label="Admitted items"
          value={counts === null ? "unavailable" : formatCount(counts.admittedItems)}
        />
        <MemoryFact
          label="Provider exposures"
          value={counts === null ? "unavailable" : formatCount(counts.providerExposures)}
        />
        <MemoryFact label="Exposure certainty" value={memoryExposureCertainty(evidence)} />
      </div>

      {limitations.length > 0 && (
        <div className="text-xs text-amber-700 dark:text-amber-300">
          Limitations: {limitations.map((item) => item.replaceAll("_", " ")).join(", ")}.
        </div>
      )}

      <details className="text-xs">
        <summary className="cursor-pointer text-primary">
          Inspect bounded structural source evidence
        </summary>
        <div className="mt-2 space-y-2">
          {sources.map((source) => {
            const attribution = source.attribution
            const receipts = attribution?.receipts ?? []
            const exposures = attribution?.exposures ?? []
            const exposureStates = [...new Set(exposures.map((item) => item.state))].sort()
            const sourceAdmittedItems =
              counts === null
                ? null
                : receipts.reduce((total, receipt) => total + receipt.admitted_count, 0)
            const sourceProviderExposures =
              counts === null
                ? null
                : exposures.filter((item) => item.provider_exposure_proven).length
            return (
              <div
                key={`${source.source.role}:${source.source.tree_path.join(".")}`}
                className="grid gap-2 rounded-md border border-border p-2 sm:grid-cols-4"
              >
                <MemoryFact
                  label="Source"
                  value={
                    source.source.role === "root"
                      ? "root"
                      : `descendant ${source.source.tree_path.join(".")}`
                  }
                />
                <MemoryFact label="Terminal status" value={source.terminal_status} />
                <MemoryFact label="Attribution" value={attribution?.status ?? "unavailable"} />
                <MemoryFact
                  label="Source alias"
                  value={
                    source.source.session_alias
                      ? shortEvalIdentity(source.source.session_alias.digest)
                      : "not retained"
                  }
                  title={source.source.session_alias?.digest}
                />
                <MemoryFact
                  label="Recall receipts"
                  value={formatRetainedCoverage(receipts.length, source.expected_receipt_count)}
                />
                <MemoryFact
                  label="Exposure records"
                  value={formatRetainedCoverage(exposures.length, source.expected_exposure_count)}
                />
                <MemoryFact
                  label="Admitted items"
                  value={
                    sourceAdmittedItems === null ? "unavailable" : formatCount(sourceAdmittedItems)
                  }
                />
                <MemoryFact
                  label="Proven provider exposures"
                  value={
                    sourceProviderExposures === null
                      ? "unavailable"
                      : formatCount(sourceProviderExposures)
                  }
                />
                <MemoryFact
                  label="Exposure states"
                  value={
                    exposureStates.length === 0
                      ? "none retained"
                      : exposureStates.map((item) => item.replaceAll("_", " ")).join(", ")
                  }
                />
              </div>
            )
          })}
          {sources.length === 0 && (
            <div className="rounded-md border border-dashed border-border p-3 text-muted-foreground">
              No bounded source record is available.
            </div>
          )}
        </div>
      </details>

      <div className="grid gap-3 md:grid-cols-3">
        <MemoryClaim
          title="Structural"
          state={
            evidence.completeness === "complete" && !evidence.has_indeterminate_exposure
              ? "observable"
              : "inconclusive"
          }
          detail="Admission and provider exposure come from runtime evidence."
        />
        <MemoryClaim
          title="Semantic use"
          state={
            scoredSemanticJudges.length > 0
              ? "judged"
              : semanticJudges.length > 0
                ? "inconclusive"
                : "not judged"
          }
          detail={
            scoredSemanticJudges.length > 0
              ? `${scoredSemanticJudges.length} memory-use rubric${scoredSemanticJudges.length === 1 ? "" : "s"} scored against trusted reference truth.`
              : semanticJudges.length > 0
                ? "A reference-backed memory-use judge ran without a conclusive score; inspect its evaluator outcome."
                : "Add a reference-backed memory-use judge to score whether the answer used memory correctly."
          }
        />
        <MemoryClaim
          title="Causal contribution"
          state="not established"
          detail="A normal run cannot show that memory changed the outcome. Use a paired intervention report."
        />
      </div>
    </div>
  )
}

export function MemoryAssertionDetails({
  detail,
  className,
}: {
  detail: PublishedMemoryAttributionDetail
  className?: string
}) {
  return (
    <div
      className={`rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-memory-assertion-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <MemoryFact label="Evidence" value={detail.observation_state.replaceAll("_", " ")} />
        <MemoryFact
          label="Admitted items"
          value={
            detail.admitted_item_count == null
              ? "unavailable"
              : formatCount(detail.admitted_item_count)
          }
        />
        <MemoryFact
          label="Required admitted"
          value={formatRange(detail.min_admitted_items, detail.max_admitted_items)}
        />
        <MemoryFact
          label="Provider exposures"
          value={
            detail.provider_exposure_count == null
              ? "unavailable"
              : formatCount(detail.provider_exposure_count)
          }
        />
        <MemoryFact
          label="Required exposures"
          value={formatRange(detail.min_provider_exposures, detail.max_provider_exposures)}
        />
        <MemoryFact
          label="Evidence revision"
          value={shortEvalIdentity(detail.evidence_revision)}
          title={detail.evidence_revision}
        />
        <MemoryFact
          label="Limitations"
          value={
            (detail.limitations ?? []).length === 0
              ? "none"
              : (detail.limitations ?? []).map((item) => item.replaceAll("_", " ")).join(", ")
          }
        />
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        This deterministic assertion checks structural admission and exposure only; it does not
        judge correct use or causal impact.
      </p>
    </div>
  )
}

export function MemoryExperimentReportAction({ disabled }: { disabled: boolean }) {
  const [open, setOpen] = useState(false)
  const [request, setRequest] = useState<EvalMemoryExperimentReportRequest | null>(null)
  const [sourceName, setSourceName] = useState<string | null>(null)
  const [report, setReport] = useState<EvalMemoryExperimentReport | null>(null)
  const [pending, setPending] = useState<"build" | "html" | null>(null)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => () => controllerRef.current?.abort(), [])

  const close = () => {
    if (pending !== null) return
    setOpen(false)
    setRequest(null)
    setSourceName(null)
    setReport(null)
    setError(null)
  }

  const selectFile = async (file: File | undefined) => {
    if (!file || pending !== null) return
    setError(null)
    setReport(null)
    try {
      const parsed = await preflightMemoryExperimentRequestFile(file)
      setRequest(parsed)
      setSourceName(file.name)
    } catch (fileError) {
      setRequest(null)
      setSourceName(null)
      setError(
        fileError instanceof Error
          ? fileError.message
          : "The memory experiment request is invalid.",
      )
    }
  }

  const run = async (kind: "build" | "html") => {
    if (!request || pending !== null) return
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setPending(kind)
    setError(null)
    try {
      if (kind === "build") {
        setReport(await buildMemoryExperimentReport(request, controller.signal))
      } else {
        const file = await downloadMemoryExperimentReportHtml(request, controller.signal)
        downloadBlob(file.blob, file.filename)
      }
    } catch (actionError) {
      if (!controller.signal.aborted) {
        setError(evalErrorMessage(actionError, "The paired memory report could not be built."))
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setPending(null)
      }
    }
  }

  return (
    <>
      <Button type="button" variant="outline" disabled={disabled} onClick={() => setOpen(true)}>
        <FlaskConical /> Paired memory report
      </Button>
      <Sheet open={open} onOpenChange={(next) => (next ? setOpen(true) : close())}>
        <SheetContent className="w-[min(96vw,52rem)] overflow-y-auto sm:max-w-none">
          <SheetHeader>
            <SheetTitle>Paired memory experiment report</SheetTitle>
            <SheetDescription>
              Validate an exact campaign request against stored eval results, compare paired
              variants, and render the causal report. This does not run or invent a campaign.
            </SheetDescription>
          </SheetHeader>

          <div className="space-y-4 px-4">
            <input
              ref={inputRef}
              type="file"
              accept="application/json,.json"
              className="sr-only"
              onChange={(event) => {
                const file = event.target.files?.[0]
                event.target.value = ""
                void selectFile(file)
              }}
            />
            <div className="rounded-lg border border-dashed border-border p-4">
              <Button
                type="button"
                variant="outline"
                disabled={pending !== null}
                onClick={() => inputRef.current?.click()}
              >
                <Upload /> Select campaign request JSON
              </Button>
              <div className="mt-2 text-xs text-muted-foreground">
                {sourceName ?? "Maximum 32 MiB. Server validation remains authoritative."}
              </div>
            </div>

            {request && (
              <div className="grid gap-3 rounded-lg border border-border bg-muted/20 p-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
                <MemoryFact label="Experiment" value={request.experiment_id} />
                <MemoryFact label="Cases" value={formatCount(request.cases.length)} />
                <MemoryFact label="Variants" value={formatCount(request.variants.length)} />
                <MemoryFact label="Repetitions" value={formatCount(request.repetitions)} />
              </div>
            )}

            {error && (
              <div
                className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
                role="alert"
              >
                {error}
              </div>
            )}

            {report && <MemoryExperimentReportSummary report={report} />}
          </div>

          <SheetFooter>
            <Button type="button" variant="outline" disabled={pending !== null} onClick={close}>
              Close
            </Button>
            {report && (
              <Button
                type="button"
                variant="outline"
                disabled={pending !== null}
                onClick={() =>
                  downloadBlob(
                    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
                    `${safeFilename(report.experiment_id)}.memory-report.json`,
                  )
                }
              >
                <FileJson /> JSON
              </Button>
            )}
            {report && (
              <Button
                type="button"
                variant="outline"
                disabled={pending !== null}
                onClick={() => void run("html")}
              >
                {pending === "html" ? <LoaderCircle className="animate-spin" /> : <Download />}
                HTML
              </Button>
            )}
            <Button
              type="button"
              disabled={!request || pending !== null}
              onClick={() => void run("build")}
            >
              {pending === "build" ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}
              {pending === "build" ? "Building..." : "Build report"}
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </>
  )
}

function MemoryExperimentReportSummary({ report }: { report: EvalMemoryExperimentReport }) {
  return (
    <div
      className="space-y-3 rounded-lg border border-border p-3"
      data-testid="memory-report-summary"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm font-medium">Validated report</div>
          <div className="text-xs text-muted-foreground" title={report.revision}>
            {shortEvalIdentity(report.revision)}
          </div>
        </div>
        <Badge variant="secondary">Selected {report.selected_variant_id}</Badge>
      </div>
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <MemoryFact label="Baseline" value={report.baseline_variant_id} />
        <MemoryFact label="Rows" value={formatCount(report.rows.length)} />
        <MemoryFact label="Comparisons" value={formatCount(report.comparisons.length)} />
        <MemoryFact label="Dispositions" value={formatCount(report.dispositions.length)} />
      </div>
      <div className="space-y-2">
        {report.dispositions.map((item) => (
          <div
            key={item.variant_id}
            className="grid gap-2 rounded-md border border-border p-2 text-xs sm:grid-cols-[minmax(0,1fr)_auto_auto_auto]"
          >
            <div>
              <div className="font-medium">{item.variant_id}</div>
              <div className="text-muted-foreground">{item.status.replaceAll("_", " ")}</div>
            </div>
            <MemoryFact label="Comparable" value={formatCount(item.comparable_pair_count)} />
            <MemoryFact label="Incomparable" value={formatCount(item.incomparable_pair_count)} />
            <MemoryFact label="Unavailable" value={formatCount(item.unavailable_pair_count)} />
          </div>
        ))}
      </div>
    </div>
  )
}

async function preflightMemoryExperimentRequestFile(
  file: File,
): Promise<EvalMemoryExperimentReportRequest> {
  return parseMemoryExperimentReportFile(file)
}

function MemoryFact({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-muted-foreground">{label}</div>
      <div className="truncate font-medium text-foreground" title={title ?? value}>
        {value}
      </div>
    </div>
  )
}

function MemoryClaim({ title, state, detail }: { title: string; state: string; detail: string }) {
  return (
    <div className="rounded-md border border-border bg-background p-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div className="font-medium">{title}</div>
        <Badge variant="outline">{state}</Badge>
      </div>
      <p className="mt-2 text-muted-foreground">{detail}</p>
    </div>
  )
}

function formatRange(minimum: number, maximum: number | null | undefined): string {
  if (maximum === minimum) return formatCount(minimum)
  if (maximum == null) return `at least ${formatCount(minimum)}`
  return `${formatCount(minimum)} to ${formatCount(maximum)}`
}

function formatRetainedCoverage(retained: number, expected: number | null | undefined): string {
  if (expected == null || !Number.isSafeInteger(expected) || expected < retained) {
    return `${formatCount(retained)} retained`
  }
  return `${formatCount(retained)}/${formatCount(expected)} retained`
}

function safeFilename(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 128) || "cayu-memory-experiment"
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.hidden = true
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}
