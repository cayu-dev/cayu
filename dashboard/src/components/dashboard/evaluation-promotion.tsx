import { useQueryClient } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FlaskConical,
  LoaderCircle,
  Plus,
  RotateCcw,
  Save,
  Trash2,
  XCircle,
} from "lucide-react"
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Textarea } from "@/components/ui/textarea"
import {
  ApiClientError,
  type CapturedEvaluationCandidateDraft as EvaluationPromotionCandidateDraft,
  type CapturedEvaluationPreview as EvaluationPromotionPreview,
  exportCapturedEvaluation as exportEvaluationPromotion,
  fetchEvalResultDetail,
  previewCapturedEvaluation as previewEvaluationPromotion,
  saveCapturedEvaluation,
  selectEvalBaseline,
} from "@/lib/api"
import { shortEvalIdentity } from "@/lib/evals-dashboard"
import { evalsReadinessReasonText } from "@/lib/evals-readiness"
import {
  createCapturedEvaluationAssertion,
  createPromotionAssertion,
  PROMOTION_ASSERTION_KINDS,
  PROMOTION_ASSERTION_LABELS,
  type PromotionAssertion,
  type PromotionAssertionKind,
  capturedEvaluationPreviewMatchesDraft as previewMatchesDraft,
  capturedEvaluationDraftFromCandidate as promotionDraftFromCandidate,
  validateCapturedEvaluationDraft as validatePromotionDraft,
} from "@/lib/evaluation-promotion"
import { useServerContract } from "./server-contract"

const SELECT_CLASS =
  "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
const LABEL_CLASS = "mb-1 block text-xs font-medium text-muted-foreground"
const ELIGIBLE_STATUSES = new Set(["completed", "failed"])

export function EvaluationPromotionAction({
  sessionId,
  status,
}: {
  sessionId: string
  status: string
}) {
  const readiness = useServerContract().capabilities.evals_readiness
  const capturedReadiness = readiness.captured_evaluation
  const persistenceReadiness = readiness.captured_result_persistence
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [preview, setPreview] = useState<EvaluationPromotionPreview | null>(null)
  const [draft, setDraft] = useState<EvaluationPromotionCandidateDraft | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [savedRevision, setSavedRevision] = useState<string | null>(null)
  const [savedCorpusRevision, setSavedCorpusRevision] = useState<string | null>(null)
  const [savedTargetKey, setSavedTargetKey] = useState<string | null>(null)
  const [baselineGeneration, setBaselineGeneration] = useState<number | null>(null)
  const [baselining, setBaselining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const previewRequestRef = useRef<{ generation: number; controller: AbortController } | null>(null)
  const exportControllerRef = useRef<AbortController | null>(null)
  const generationRef = useRef(0)

  const cancelRequests = useCallback(() => {
    generationRef.current += 1
    previewRequestRef.current?.controller.abort()
    previewRequestRef.current = null
    exportControllerRef.current?.abort()
    exportControllerRef.current = null
  }, [])

  useEffect(() => cancelRequests, [cancelRequests])

  const loadPreview = useCallback(
    async (nextDraft?: EvaluationPromotionCandidateDraft) => {
      if (nextDraft !== undefined) {
        const validation = validatePromotionDraft(nextDraft)
        if (!validation.ok) {
          setError(validation.error)
          return
        }
      }
      previewRequestRef.current?.controller.abort()
      const controller = new AbortController()
      const generation = generationRef.current + 1
      generationRef.current = generation
      previewRequestRef.current = { generation, controller }
      setPreviewing(true)
      setSavedRevision(null)
      setError(null)
      try {
        const response = await previewEvaluationPromotion(sessionId, nextDraft, controller.signal)
        if (generationRef.current !== generation || controller.signal.aborted) return
        setPreview(response)
        setDraft(promotionDraftFromCandidate(response.candidate, response.baseline_revision))
      } catch (previewError) {
        if (controller.signal.aborted || generationRef.current !== generation) return
        if (isPromotionConflict(previewError)) {
          setPreview(null)
          setDraft(null)
        }
        setError(promotionErrorMessage(previewError))
      } finally {
        if (generationRef.current === generation) {
          previewRequestRef.current = null
          setPreviewing(false)
        }
      }
    },
    [sessionId],
  )

  const openPromotion = () => {
    cancelRequests()
    setPreview(null)
    setDraft(null)
    setError(null)
    setPreviewing(false)
    setExporting(false)
    setSaving(false)
    setBaselining(false)
    setSavedRevision(null)
    setSavedCorpusRevision(null)
    setSavedTargetKey(null)
    setBaselineGeneration(null)
    setOpen(true)
    void loadPreview()
  }

  const changeOpen = (nextOpen: boolean) => {
    setOpen(nextOpen)
    if (!nextOpen) {
      cancelRequests()
      setPreviewing(false)
      setExporting(false)
      setSaving(false)
      setBaselining(false)
    }
  }

  const editDraft = (edit: (next: EvaluationPromotionCandidateDraft) => void) => {
    if (draft === null) return
    const next = structuredClone(draft)
    edit(next)
    setDraft(next)
    setSavedRevision(null)
    setError(null)
  }

  const validation = useMemo(() => (draft ? validatePromotionDraft(draft) : null), [draft])
  const previewIsCurrent =
    preview !== null &&
    draft !== null &&
    validation?.ok === true &&
    previewMatchesDraft(preview, draft)
  const persistenceUnavailable =
    persistenceReadiness.state === "ready" ? null : evalsReadinessReasonText(persistenceReadiness)

  const exportPreviewedCandidate = (signal: AbortSignal) => {
    if (!previewIsCurrent || preview === null) return null
    return exportEvaluationPromotion(
      sessionId,
      {
        candidate: preview.candidate,
        expected_candidate_revision: preview.candidate.revision,
      },
      signal,
    )
  }

  const exportCandidate = async () => {
    if (!previewIsCurrent || preview === null) return
    exportControllerRef.current?.abort()
    const controller = new AbortController()
    exportControllerRef.current = controller
    setExporting(true)
    setError(null)
    try {
      const exported = await exportPreviewedCandidate(controller.signal)
      if (exported === null) return
      if (controller.signal.aborted) return
      downloadBlob(exported.blob, exported.filename)
    } catch (exportError) {
      if (controller.signal.aborted) return
      if (isPromotionConflict(exportError)) {
        setPreview(null)
        setDraft(null)
      }
      setError(promotionErrorMessage(exportError))
    } finally {
      if (exportControllerRef.current === controller) {
        exportControllerRef.current = null
        setExporting(false)
      }
    }
  }

  const saveCandidate = async () => {
    if (!previewIsCurrent || preview === null) return
    exportControllerRef.current?.abort()
    const controller = new AbortController()
    exportControllerRef.current = controller
    setSaving(true)
    setSavedRevision(null)
    setError(null)
    try {
      const saved = await saveCapturedEvaluation(
        sessionId,
        {
          candidate: preview.candidate,
          expected_candidate_revision: preview.candidate.revision,
        },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setSavedRevision(saved.record.revision)
      setSavedCorpusRevision(saved.record.corpus_revision)
      setSavedTargetKey(saved.record.target.target_key)
      setBaselineGeneration(null)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["evals", "results"] }),
        queryClient.invalidateQueries({ queryKey: ["evals", "corpora"] }),
      ])
    } catch (saveError) {
      if (controller.signal.aborted) return
      if (isPromotionConflict(saveError)) {
        setPreview(null)
        setDraft(null)
      }
      setError(promotionErrorMessage(saveError))
    } finally {
      if (exportControllerRef.current === controller) {
        exportControllerRef.current = null
        setSaving(false)
      }
    }
  }

  const approveBaseline = async () => {
    if (savedRevision === null) return
    setBaselining(true)
    setError(null)
    try {
      const current = await fetchEvalResultDetail(savedRevision)
      if (current.baseline?.result_revision === savedRevision) {
        setBaselineGeneration(current.baseline.generation)
        return
      }
      const selected = await selectEvalBaseline(savedRevision, {
        result_revision: savedRevision,
        expected_generation: current.baseline?.generation ?? 0,
        operation_id: randomOperationId(),
      })
      setBaselineGeneration(selected.baseline.generation)
      await queryClient.invalidateQueries({ queryKey: ["evals", "results"] })
    } catch (baselineError) {
      setError(promotionErrorMessage(baselineError))
    } finally {
      setBaselining(false)
    }
  }

  if (capturedReadiness.state !== "ready" || !ELIGIBLE_STATUSES.has(status)) return null

  return (
    <>
      <Button size="sm" variant="outline" data-testid="evaluate-session" onClick={openPromotion}>
        <FlaskConical className="h-4 w-4" />
        Evaluate
      </Button>
      <Sheet open={open} onOpenChange={changeOpen}>
        <SheetContent
          className="w-full! gap-0 sm:max-w-4xl!"
          data-testid="promotion-sheet"
          aria-busy={previewing || exporting || saving || baselining}
        >
          <SheetHeader className="border-b border-border pr-12">
            <SheetTitle>Evaluate captured session</SheetTitle>
            <SheetDescription>
              Review retained evidence, define expectations, then save or export the exact previewed
              evaluation. No application workload runs here.
            </SheetDescription>
          </SheetHeader>

          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {previewing && draft === null ? (
              <div className="flex min-h-56 items-center justify-center gap-2 text-muted-foreground">
                <LoaderCircle className="h-4 w-4 animate-spin" />
                Reconstructing captured evidence...
              </div>
            ) : draft === null ? (
              <div className="flex min-h-56 flex-col items-center justify-center gap-3 text-center">
                <p className="max-w-md text-sm text-muted-foreground">
                  The captured evidence could not be loaded. No evaluation has been saved.
                </p>
                <Button variant="outline" onClick={() => void loadPreview()} disabled={previewing}>
                  <RotateCcw /> Retry preview
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {preview && (
                  <PromotionEvidenceSummary preview={preview} current={previewIsCurrent} />
                )}
                <fieldset
                  className="contents"
                  disabled={previewing || exporting || saving || baselining}
                >
                  <PromotionIdentityEditor draft={draft} editDraft={editDraft} />
                  <PromotionAssertionsEditor
                    draft={draft}
                    evidence={preview?.candidate.evidence}
                    editDraft={editDraft}
                  />
                </fieldset>
                {preview && <PromotionScore preview={preview} current={previewIsCurrent} />}
              </div>
            )}

            {(error || (validation && !validation.ok)) && (
              <div
                className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
                role="alert"
                data-testid="promotion-error"
              >
                {error ?? (validation && !validation.ok ? validation.error : null)}
              </div>
            )}
            {savedRevision && (
              <div
                className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-300"
                role="status"
              >
                Saved result {shortEvalIdentity(savedRevision)} to Evals.{" "}
                <Link
                  to="/evals"
                  search={{
                    tab: "results",
                    result: savedRevision,
                    corpus: savedCorpusRevision ?? undefined,
                    target: savedTargetKey ?? undefined,
                  }}
                  className="font-medium underline"
                  onClick={() => changeOpen(false)}
                >
                  Open Evals
                </Link>
                {baselineGeneration === null ? (
                  <Button
                    size="xs"
                    variant="outline"
                    className="ml-2"
                    disabled={baselining}
                    onClick={() => void approveBaseline()}
                  >
                    {baselining ? <LoaderCircle className="animate-spin" /> : <CheckCircle2 />}
                    {baselining ? "Approving..." : "Approve baseline"}
                  </Button>
                ) : (
                  <span className="ml-2 font-medium">Baseline approved</span>
                )}
              </div>
            )}
          </div>

          <SheetFooter className="border-t border-border bg-background sm:flex-row sm:items-center sm:justify-end">
            {persistenceUnavailable && (
              <span className="mr-auto text-xs text-muted-foreground">
                {persistenceUnavailable}
              </span>
            )}
            <Button variant="ghost" onClick={() => changeOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="outline"
              data-testid="promotion-preview"
              disabled={
                draft === null || previewing || exporting || saving || validation?.ok !== true
              }
              onClick={() => draft && void loadPreview(draft)}
            >
              {previewing ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}
              {previewing ? "Scoring..." : "Preview score"}
            </Button>
            <Button
              data-testid="promotion-export"
              disabled={!previewIsCurrent || previewing || exporting || saving || baselining}
              onClick={() => void exportCandidate()}
            >
              {exporting ? <LoaderCircle className="animate-spin" /> : <Download />}
              {exporting ? "Exporting..." : "Export eval JSON"}
            </Button>
            <Button
              data-testid="promotion-save"
              disabled={
                !previewIsCurrent ||
                persistenceReadiness.state !== "ready" ||
                previewing ||
                exporting ||
                saving ||
                baselining
              }
              title={persistenceUnavailable ?? undefined}
              onClick={() => void saveCandidate()}
            >
              {saving ? <LoaderCircle className="animate-spin" /> : <Save />}
              {saving ? "Saving..." : "Save evaluation"}
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </>
  )
}

function PromotionEvidenceSummary({
  preview,
  current,
}: {
  preview: EvaluationPromotionPreview
  current: boolean
}) {
  const { candidate } = preview
  const evidence = candidate.evidence
  return (
    <Card size="sm">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Captured evidence</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            {candidate.source.source_agent_name} · release {candidate.source.application_release_id}
          </p>
        </div>
        <Badge variant={current ? "secondary" : "outline"}>
          {current ? "Preview current" : "Preview out of date"}
        </Badge>
      </CardHeader>
      <CardContent className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <EvidenceFact label="Root status" value={evidence.root_status ?? "unavailable"} />
        <EvidenceFact label="Final output" value={evidence.final_output_state} />
        <EvidenceFact
          label="Tool calls"
          value={
            evidence.tool_calls_started == null
              ? evidence.tool_evidence_state
              : String(evidence.tool_calls_started)
          }
        />
        <EvidenceFact
          label="Total tokens"
          value={evidence.total_tokens ?? evidence.usage_evidence_state}
        />
      </CardContent>
      {candidate.warnings.length > 0 && (
        <div className="mx-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-2.5 text-xs text-amber-700 dark:text-amber-300">
          <AlertTriangle className="mr-1.5 inline h-3.5 w-3.5" />
          {candidate.warnings.map(promotionWarning).join(" ")}
        </div>
      )}
      <div className="mx-3 mt-3 rounded-lg border border-border bg-muted/20 p-2.5 text-xs text-muted-foreground">
        {preview.runnable_conversion.available
          ? "This captured session can also be converted into runnable input for a fresh evaluation."
          : `Captured scoring and saving are available. Fresh execution needs authored runnable input${preview.runnable_conversion.reason_code ? ` (${preview.runnable_conversion.reason_code.replaceAll("_", " ")})` : ""}.`}
      </div>
    </Card>
  )
}

function EvidenceFact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate font-medium text-foreground">{value}</div>
    </div>
  )
}

function PromotionIdentityEditor({ draft, editDraft }: PromotionEditorProps) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>Suite and case</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-2">
        <PromotionField label="Suite ID" id="promotion-suite-id">
          <Input
            id="promotion-suite-id"
            value={draft.suite.id}
            onChange={(event) =>
              editDraft((next) => {
                next.suite.id = event.target.value
                next.case.suite_id = event.target.value
              })
            }
          />
        </PromotionField>
        <PromotionField label="Suite name" id="promotion-suite-name">
          <Input
            id="promotion-suite-name"
            value={draft.suite.name}
            onChange={(event) => editDraft((next) => (next.suite.name = event.target.value))}
          />
        </PromotionField>
        <PromotionField label="Suite description" id="promotion-suite-description" wide>
          <Textarea
            id="promotion-suite-description"
            value={draft.suite.description ?? ""}
            onChange={(event) =>
              editDraft((next) => (next.suite.description = event.target.value || null))
            }
          />
        </PromotionField>
        <PromotionField label="Case ID" id="promotion-case-id">
          <Input
            id="promotion-case-id"
            value={draft.case.id}
            onChange={(event) => editDraft((next) => (next.case.id = event.target.value))}
          />
        </PromotionField>
        <PromotionField label="Case name" id="promotion-case-name">
          <Input
            id="promotion-case-name"
            value={draft.case.name}
            onChange={(event) => editDraft((next) => (next.case.name = event.target.value))}
          />
        </PromotionField>
        <PromotionField label="Case description" id="promotion-case-description" wide>
          <Textarea
            id="promotion-case-description"
            value={draft.case.description ?? ""}
            onChange={(event) =>
              editDraft((next) => (next.case.description = event.target.value || null))
            }
          />
        </PromotionField>
      </CardContent>
    </Card>
  )
}

function PromotionAssertionsEditor({
  draft,
  evidence,
  editDraft,
}: PromotionEditorProps & {
  evidence: EvaluationPromotionPreview["candidate"]["evidence"] | undefined
}) {
  const [quickKind, setQuickKind] = useState<PromotionAssertionKind>("root_status")
  const addAssertion = () =>
    editDraft((next) => {
      next.case.assertions.push(
        evidence
          ? createCapturedEvaluationAssertion(quickKind, next.case.assertions, evidence)
          : createPromotionAssertion(quickKind, next.case.assertions),
      )
    })
  return (
    <Card size="sm">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Assertions</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Assertions are rescored against the captured run before export.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            className={SELECT_CLASS}
            value={quickKind}
            aria-label="Assertion quick-add type"
            onChange={(event) => setQuickKind(event.target.value as PromotionAssertionKind)}
          >
            {PROMOTION_ASSERTION_KINDS.map((kind) => (
              <option key={kind} value={kind}>
                {PROMOTION_ASSERTION_LABELS[kind]}
              </option>
            ))}
          </select>
          <Button
            size="xs"
            variant="outline"
            disabled={draft.case.assertions.length >= 64}
            onClick={addAssertion}
          >
            <Plus /> Add observed
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {draft.case.assertions.map((assertion, index) => (
          <AssertionEditor
            // biome-ignore lint/suspicious/noArrayIndexKey: rows never reorder and own no local state.
            key={index}
            assertion={assertion}
            index={index}
            assertions={draft.case.assertions}
            evidence={evidence}
            update={(updated) =>
              editDraft((next) => {
                next.case.assertions[index] = updated
              })
            }
            remove={() => editDraft((next) => next.case.assertions.splice(index, 1))}
          />
        ))}
      </CardContent>
    </Card>
  )
}

function AssertionEditor({
  assertion,
  index,
  assertions,
  evidence,
  update,
  remove,
}: {
  assertion: PromotionAssertion
  index: number
  assertions: PromotionAssertion[]
  evidence: EvaluationPromotionPreview["candidate"]["evidence"] | undefined
  update: (assertion: PromotionAssertion) => void
  remove: () => void
}) {
  const replaceKind = (kind: PromotionAssertionKind) => {
    const remaining = assertions.filter((_, candidateIndex) => candidateIndex !== index)
    const replacement = evidence
      ? createCapturedEvaluationAssertion(kind, remaining, evidence)
      : createPromotionAssertion(kind, remaining)
    replacement.id = assertion.id
    replacement.description = assertion.description ?? null
    update(replacement)
  }
  return (
    <div className="rounded-lg border border-border p-3" data-testid="promotion-assertion">
      <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
        <PromotionField label="Assertion ID" id={`promotion-assertion-${index}-id`}>
          <Input
            id={`promotion-assertion-${index}-id`}
            value={assertion.id}
            onChange={(event) => update({ ...assertion, id: event.target.value })}
          />
        </PromotionField>
        <PromotionField label="Type" id={`promotion-assertion-${index}-kind`}>
          <select
            id={`promotion-assertion-${index}-kind`}
            className={SELECT_CLASS}
            value={assertion.kind}
            onChange={(event) => replaceKind(event.target.value as PromotionAssertionKind)}
          >
            {PROMOTION_ASSERTION_KINDS.map((kind) => (
              <option key={kind} value={kind}>
                {PROMOTION_ASSERTION_LABELS[kind]}
              </option>
            ))}
          </select>
        </PromotionField>
        <Button
          className="mt-5"
          size="icon-sm"
          variant="ghost"
          aria-label={`Remove assertion ${assertion.id}`}
          disabled={assertions.length === 1}
          onClick={remove}
        >
          <Trash2 />
        </Button>
      </div>
      <div className="mt-3">
        <AssertionFields assertion={assertion} update={update} index={index} />
      </div>
      <div className="mt-3">
        <label className={LABEL_CLASS} htmlFor={`promotion-assertion-${index}-description`}>
          Description
        </label>
        <Input
          id={`promotion-assertion-${index}-description`}
          value={assertion.description ?? ""}
          onChange={(event) => update({ ...assertion, description: event.target.value || null })}
        />
      </div>
    </div>
  )
}

function AssertionFields({
  assertion,
  update,
  index,
}: {
  assertion: PromotionAssertion
  update: (assertion: PromotionAssertion) => void
  index: number
}) {
  const id = (name: string) => `promotion-assertion-${index}-${name}`
  switch (assertion.kind) {
    case "root_status":
      return (
        <PromotionField label="Expected status" id={id("expected")}>
          <TerminalStatusSelect
            id={id("expected")}
            value={assertion.expected}
            onChange={(expected) => update({ ...assertion, expected })}
          />
        </PromotionField>
      )
    case "child_status":
      return (
        <div className="grid gap-3 sm:grid-cols-3">
          <PromotionField label="Expected child status" id={id("expected")}>
            <TerminalStatusSelect
              id={id("expected")}
              value={assertion.expected}
              onChange={(expected) => update({ ...assertion, expected })}
            />
          </PromotionField>
          <IntegerField
            label="Minimum count"
            id={id("minimum")}
            value={assertion.min_count ?? 1}
            onChange={(min_count) => update({ ...assertion, min_count })}
          />
          <NullableIntegerField
            label="Maximum count"
            id={id("maximum")}
            value={assertion.max_count}
            onChange={(max_count) => update({ ...assertion, max_count })}
          />
        </div>
      )
    case "final_output_equals":
    case "final_output_contains":
      return (
        <PromotionField label="Expected output text" id={id("expected")}>
          <Textarea
            id={id("expected")}
            className="min-h-20 font-mono"
            value={assertion.expected}
            onChange={(event) => update({ ...assertion, expected: event.target.value })}
          />
        </PromotionField>
      )
    case "tool_called":
      return (
        <div className="grid gap-3 sm:grid-cols-3">
          <PromotionField label="Tool name" id={id("tool-name")}>
            <Input
              id={id("tool-name")}
              value={assertion.tool_name}
              onChange={(event) => update({ ...assertion, tool_name: event.target.value })}
            />
          </PromotionField>
          <IntegerField
            label="Minimum count"
            id={id("minimum")}
            value={assertion.min_count ?? 1}
            onChange={(min_count) => update({ ...assertion, min_count })}
          />
          <NullableIntegerField
            label="Maximum count"
            id={id("maximum")}
            value={assertion.max_count}
            onChange={(max_count) => update({ ...assertion, max_count })}
          />
        </div>
      )
    case "tools_called_in_order":
      return (
        <div>
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className={LABEL_CLASS}>Ordered tool names</span>
            <Button
              size="xs"
              variant="outline"
              disabled={assertion.tool_names.length >= 256}
              onClick={() =>
                update({ ...assertion, tool_names: [...assertion.tool_names, "tool"] })
              }
            >
              <Plus /> Add tool
            </Button>
          </div>
          {assertion.tool_names.length === 0 ? (
            <p className="rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground">
              The expected tool-call sequence is empty.
            </p>
          ) : (
            <div className="space-y-2">
              {assertion.tool_names.map((toolName, toolIndex) => (
                <div
                  // biome-ignore lint/suspicious/noArrayIndexKey: ordered names own no local state.
                  key={toolIndex}
                  className="flex items-start gap-2"
                >
                  <Textarea
                    aria-label={`Expected tool ${toolIndex + 1}`}
                    className="min-h-16 font-mono"
                    value={toolName}
                    onChange={(event) => {
                      const tool_names = [...assertion.tool_names]
                      tool_names[toolIndex] = event.target.value
                      update({ ...assertion, tool_names })
                    }}
                  />
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    aria-label={`Remove expected tool ${toolIndex + 1}`}
                    onClick={() =>
                      update({
                        ...assertion,
                        tool_names: assertion.tool_names.filter(
                          (_, candidateIndex) => candidateIndex !== toolIndex,
                        ),
                      })
                    }
                  >
                    <Trash2 />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      )
    case "max_tool_calls":
      return (
        <IntegerField
          label="Maximum tool calls"
          id={id("maximum")}
          value={assertion.maximum}
          onChange={(maximum) => update({ ...assertion, maximum })}
        />
      )
    case "max_model_steps":
      return (
        <IntegerField
          label="Maximum model steps"
          id={id("maximum")}
          value={assertion.maximum}
          onChange={(maximum) => update({ ...assertion, maximum })}
        />
      )
    case "usage_recorded":
      return (
        <IntegerField
          label="Minimum total tokens"
          id={id("minimum")}
          value={assertion.min_total_tokens ?? 1}
          onChange={(min_total_tokens) => update({ ...assertion, min_total_tokens })}
        />
      )
    case "max_total_tokens":
      return (
        <IntegerField
          label="Maximum total tokens"
          id={id("maximum")}
          value={assertion.maximum}
          onChange={(maximum) => update({ ...assertion, maximum })}
        />
      )
    case "max_estimated_cost":
      return (
        <div className="grid gap-3 sm:grid-cols-2">
          <PromotionField label="Maximum cost" id={id("maximum")}>
            <Input
              id={id("maximum")}
              inputMode="decimal"
              value={assertion.maximum}
              onChange={(event) => update({ ...assertion, maximum: event.target.value })}
            />
          </PromotionField>
          <PromotionField label="Currency" id={id("currency")}>
            <Input
              id={id("currency")}
              value={assertion.currency ?? "USD"}
              onChange={(event) => update({ ...assertion, currency: event.target.value })}
            />
          </PromotionField>
        </div>
      )
  }
}

function PromotionScore({
  preview,
  current,
}: {
  preview: EvaluationPromotionPreview
  current: boolean
}) {
  const score = preview.captured_score
  const variant =
    score.status === "passed" ? "secondary" : score.status === "failed" ? "destructive" : "outline"
  return (
    <Card size="sm" data-testid="promotion-score">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Captured-run score</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            {current
              ? "This score matches the current edits."
              : "Edit detected. Preview again before export."}
          </p>
        </div>
        <Badge variant={variant}>
          {score.status}
          {score.score != null ? ` · ${Math.round(score.score * 100)}%` : ""}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-2">
        {score.assertions.map((result) => (
          <div
            key={result.assertion_id}
            className="grid gap-1 rounded-lg border border-border px-3 py-2 sm:grid-cols-[auto_minmax(0,1fr)_auto] sm:items-center"
          >
            {result.outcome === "passed" ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            ) : result.outcome === "failed" ? (
              <XCircle className="h-4 w-4 text-destructive" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-amber-600" />
            )}
            <div className="min-w-0">
              <div className="truncate font-mono text-xs font-medium">{result.assertion_id}</div>
              <div className="text-xs text-muted-foreground">{result.message}</div>
            </div>
            <Badge variant="outline">{result.outcome}</Badge>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function PromotionField({
  label,
  id,
  wide = false,
  children,
}: {
  label: string
  id: string
  wide?: boolean
  children: ReactNode
}) {
  return (
    <div className={wide ? "sm:col-span-2" : undefined}>
      <label className={LABEL_CLASS} htmlFor={id}>
        {label}
      </label>
      {children}
    </div>
  )
}

function IntegerField({
  label,
  id,
  value,
  onChange,
}: {
  label: string
  id: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <PromotionField label={label} id={id}>
      <Input
        id={id}
        type="number"
        min={0}
        step={1}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </PromotionField>
  )
}

function NullableIntegerField({
  label,
  id,
  value,
  onChange,
}: {
  label: string
  id: string
  value: number | null | undefined
  onChange: (value: number | null) => void
}) {
  return (
    <PromotionField label={label} id={id}>
      <Input
        id={id}
        type="number"
        min={0}
        step={1}
        placeholder="No maximum"
        value={value ?? ""}
        onChange={(event) =>
          onChange(event.target.value === "" ? null : Number(event.target.value))
        }
      />
    </PromotionField>
  )
}

function TerminalStatusSelect({
  id,
  value,
  onChange,
}: {
  id: string
  value: "completed" | "failed"
  onChange: (value: "completed" | "failed") => void
}) {
  return (
    <select
      id={id}
      className={SELECT_CLASS}
      value={value}
      onChange={(event) => onChange(event.target.value as "completed" | "failed")}
    >
      <option value="completed">Completed</option>
      <option value="failed">Failed</option>
    </select>
  )
}

type PromotionEditorProps = {
  draft: EvaluationPromotionCandidateDraft
  editDraft: (edit: (next: EvaluationPromotionCandidateDraft) => void) => void
}

function promotionWarning(warning: string): string {
  if (warning === "source_run_failed")
    return "The source run failed; review its expectations carefully."
  return warning
}

function promotionErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError && error.detail && typeof error.detail === "object") {
    const message = (error.detail as Record<string, unknown>).message
    if (typeof message === "string" && message.trim()) return message
  }
  return error instanceof Error ? error.message : "The captured evaluation request failed."
}

function isPromotionConflict(error: unknown): boolean {
  return error instanceof ApiClientError && error.status === 409
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

function randomOperationId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32))
  const digest = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
  return `sha256:${digest}`
}
