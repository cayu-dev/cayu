import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FlaskConical,
  LoaderCircle,
  Play,
  Plus,
  RotateCcw,
  Save,
  Trash2,
  XCircle,
} from "lucide-react"
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ScenarioAuthoring } from "@/components/dashboard/scenario-authoring"
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
  type CapturedEvaluationLaunch,
  type CapturedEvaluationLaunched,
  type EvalTarget,
  type CapturedEvaluationCandidateDraft as EvaluationPromotionCandidateDraft,
  type CapturedEvaluationPreview as EvaluationPromotionPreview,
  exportCapturedEvaluation as exportEvaluationPromotion,
  fetchEvalResultDetail,
  fetchEvalTargets,
  launchCapturedEvaluation,
  previewCapturedEvaluation as previewEvaluationPromotion,
  saveCapturedEvaluation,
  selectEvalBaseline,
} from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import {
  capturedEvalLaunchRequestIdentity,
  EVAL_TARGET_QUERY_KEY,
  EVAL_TARGET_STALE_TIME_MS,
  EvalLaunchIdempotencyRegistry,
  evalLaunchFailureIsDefinitive,
  evalTargetCatalogMayBeStale,
  shortEvalIdentity,
} from "@/lib/evals-dashboard"
import { evalsReadinessReasonText } from "@/lib/evals-readiness"
import {
  capturedAssertionSuggestionUnavailable,
  createCapturedEvaluationAssertionDraft,
  createPromotionAssertion,
  PROCESS_EVENT_OPTIONS,
  PROMOTION_ASSERTION_KINDS,
  PROMOTION_ASSERTION_LABELS,
  type ProcessEventKind,
  type PromotionAssertion,
  type PromotionAssertionKind,
  capturedEvaluationPreviewMatchesDraft as previewMatchesDraft,
  capturedEvaluationDraftFromCandidate as promotionDraftFromCandidate,
  validateCapturedEvaluationDraft as validatePromotionDraft,
} from "@/lib/evaluation-promotion"
import type { EvaluationEvidencePolicySpec } from "@/lib/generated/server-api"
import { useServerContract } from "./server-contract"

const SELECT_CLASS =
  "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
const LABEL_CLASS = "mb-1 block text-xs font-medium text-muted-foreground"
const ELIGIBLE_STATUSES = new Set(["completed", "failed"])
const MAX_SAFE_RUNTIME_LIMIT = Number.MAX_SAFE_INTEGER

type FreshExecutionDraft = {
  trials: string
  maxConcurrency: string
  timeoutSeconds: string
  maxSteps: string
  maxTotalTokens: string
  maxToolCalls: string
  maxElapsedSeconds: string
  maxEstimatedCost: string
  currency: string
}

const DEFAULT_FRESH_EXECUTION: FreshExecutionDraft = Object.freeze({
  trials: "1",
  maxConcurrency: "1",
  timeoutSeconds: "300",
  maxSteps: "",
  maxTotalTokens: "",
  maxToolCalls: "",
  maxElapsedSeconds: "",
  maxEstimatedCost: "",
  currency: "USD",
})

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
  const freshReadiness = readiness.fresh_launch
  const scenarioCatalogReady = readiness.catalog_write.state === "ready"
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [preview, setPreview] = useState<EvaluationPromotionPreview | null>(null)
  const [draft, setDraft] = useState<EvaluationPromotionCandidateDraft | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [freshExecution, setFreshExecution] = useState<FreshExecutionDraft>({
    ...DEFAULT_FRESH_EXECUTION,
  })
  const [savedRevision, setSavedRevision] = useState<string | null>(null)
  const [savedCorpusRevision, setSavedCorpusRevision] = useState<string | null>(null)
  const [savedTargetKey, setSavedTargetKey] = useState<string | null>(null)
  const [launchedRunId, setLaunchedRunId] = useState<string | null>(null)
  const [baselineGeneration, setBaselineGeneration] = useState<number | null>(null)
  const [baselining, setBaselining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const previewRequestRef = useRef<{ generation: number; controller: AbortController } | null>(null)
  const exportControllerRef = useRef<AbortController | null>(null)
  const generationRef = useRef(0)
  const launchRegistryRef = useRef<EvalLaunchIdempotencyRegistry | null>(null)
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    enabled: open && freshReadiness.state === "ready",
    staleTime: EVAL_TARGET_STALE_TIME_MS,
  })

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
      setLaunchedRunId(null)
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
    setLaunching(false)
    setFreshExecution({ ...DEFAULT_FRESH_EXECUTION })
    setBaselining(false)
    setSavedRevision(null)
    setSavedCorpusRevision(null)
    setSavedTargetKey(null)
    setLaunchedRunId(null)
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
      setLaunching(false)
      setBaselining(false)
    }
  }

  const editDraft = (edit: (next: EvaluationPromotionCandidateDraft) => void) => {
    if (draft === null) return
    const next = structuredClone(draft)
    edit(next)
    setDraft(next)
    setSavedRevision(null)
    setLaunchedRunId(null)
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
  const freshExecutionUnavailable =
    freshReadiness.state === "ready" ? null : evalsReadinessReasonText(freshReadiness)
  const selectedTarget = targets.data?.items.find(
    (target) => target.target_key === preview?.candidate.target_key,
  )
  const freshLaunch = useMemo(
    () => validateFreshExecution(freshExecution, selectedTarget),
    [freshExecution, selectedTarget],
  )

  useEffect(() => {
    if (selectedTarget === undefined) return
    setFreshExecution((current) => {
      const timeout = Number(current.timeoutSeconds)
      const timeoutSeconds =
        !Number.isInteger(timeout) || timeout <= selectedTarget.max_timeout_seconds
          ? current.timeoutSeconds
          : String(selectedTarget.max_timeout_seconds)
      const currency = selectedTarget.cost_budget_currencies.includes(current.currency)
        ? current.currency
        : selectedTarget.cost_budget_currencies.includes("USD")
          ? "USD"
          : (selectedTarget.cost_budget_currencies[0] ?? current.currency)
      if (timeoutSeconds === current.timeoutSeconds && currency === current.currency) {
        return current
      }
      return { ...current, timeoutSeconds, currency }
    })
  }, [selectedTarget])

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
    setLaunchedRunId(null)
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

  const launchFreshTrial = async () => {
    if (
      !previewIsCurrent ||
      preview === null ||
      !preview.runnable_conversion.available ||
      freshReadiness.state !== "ready" ||
      !freshLaunch.ok ||
      selectedTarget?.execution_profile === null ||
      selectedTarget?.execution_profile === undefined
    ) {
      return
    }
    exportControllerRef.current?.abort()
    const controller = new AbortController()
    exportControllerRef.current = controller
    const request: CapturedEvaluationLaunch = {
      candidate: preview.candidate,
      expected_candidate_revision: preview.candidate.revision,
      expected_execution_profile_revision: selectedTarget.execution_profile.revision,
      ...freshLaunch.request,
    }
    const requestIdentity = capturedEvalLaunchRequestIdentity(
      sessionId,
      preview.candidate.revision,
      request,
    )
    setLaunching(true)
    setSavedRevision(null)
    setLaunchedRunId(null)
    setError(null)
    try {
      const registry =
        launchRegistryRef.current ??
        new EvalLaunchIdempotencyRegistry(window.sessionStorage, dashboardConfig.apiBaseUrl)
      launchRegistryRef.current = registry
      const idempotencyKey = registry.keyFor(requestIdentity)
      let launched: CapturedEvaluationLaunched
      try {
        launched = await launchCapturedEvaluation(
          sessionId,
          request,
          idempotencyKey,
          controller.signal,
        )
      } catch (launchError) {
        if (evalTargetCatalogMayBeStale(launchError)) {
          await queryClient.invalidateQueries({ queryKey: EVAL_TARGET_QUERY_KEY })
        }
        if (
          launchError instanceof ApiClientError &&
          evalLaunchFailureIsDefinitive(launchError.status)
        ) {
          registry.resolve(requestIdentity)
        }
        throw launchError
      }
      registry.resolve(requestIdentity)
      if (controller.signal.aborted) return
      setSavedRevision(launched.captured.record.revision)
      setSavedCorpusRevision(launched.run.spec.corpus_revision)
      setSavedTargetKey(launched.run.spec.target_key)
      setLaunchedRunId(launched.run.spec.run_id)
      setBaselineGeneration(null)
      queryClient.setQueryData(["evals", "run", launched.run.spec.run_id], launched.run)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["evals", "results"] }),
        queryClient.invalidateQueries({ queryKey: ["evals", "corpora"] }),
        queryClient.invalidateQueries({ queryKey: ["evals", "runs"] }),
      ])
    } catch (launchError) {
      if (controller.signal.aborted) return
      if (isPromotionConflict(launchError)) {
        setPreview(null)
        setDraft(null)
      }
      setError(promotionErrorMessage(launchError))
    } finally {
      if (exportControllerRef.current === controller) {
        exportControllerRef.current = null
        setLaunching(false)
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
          aria-busy={previewing || exporting || saving || launching || baselining}
        >
          <SheetHeader className="border-b border-border pr-12">
            <SheetTitle>Evaluate captured session</SheetTitle>
            <SheetDescription>
              Review retained evidence, define expectations, then save or export the exact previewed
              evaluation. Preview is evidence-only; Run on current app explicitly starts new work
              through the mounted application target.
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
                {preview?.scenario_conversion.available && preview.scenario_conversion.scenario && (
                  <ScenarioAuthoring
                    captured={preview.scenario_conversion.scenario}
                    disabled={
                      !scenarioCatalogReady ||
                      previewing ||
                      exporting ||
                      saving ||
                      launching ||
                      baselining
                    }
                  />
                )}
                <fieldset
                  className="contents"
                  disabled={previewing || exporting || saving || launching || baselining}
                >
                  <PromotionIdentityEditor draft={draft} editDraft={editDraft} />
                  <ExpectedBehaviorEditor
                    assertions={draft.case.assertions}
                    evidence={preview?.candidate.evidence}
                    evidencePolicy={preview?.candidate.evidence_policy}
                    onChange={(assertions) =>
                      editDraft((next) => {
                        next.case.assertions = assertions
                      })
                    }
                  />
                  {preview?.runnable_conversion.available && (
                    <FreshExecutionEditor
                      target={selectedTarget}
                      loadingTarget={targets.isLoading}
                      draft={freshExecution}
                      validation={freshLaunch}
                      unavailable={freshExecutionUnavailable}
                      edit={setFreshExecution}
                    />
                  )}
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
                {launchedRunId === null
                  ? `Saved result ${shortEvalIdentity(savedRevision)} to Evals. `
                  : `Started current-app eval run ${shortEvalIdentity(launchedRunId)}. `}
                <Link
                  to="/evals"
                  search={{
                    tab: launchedRunId === null ? "results" : "runs",
                    result: launchedRunId === null ? savedRevision : undefined,
                    run: launchedRunId ?? undefined,
                    corpus: savedCorpusRevision ?? undefined,
                    target: savedTargetKey ?? undefined,
                  }}
                  className="font-medium underline"
                  onClick={() => changeOpen(false)}
                >
                  {launchedRunId === null ? "Open Evals" : "Open run"}
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
                draft === null ||
                previewing ||
                exporting ||
                saving ||
                launching ||
                validation?.ok !== true
              }
              onClick={() => draft && void loadPreview(draft)}
            >
              {previewing ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}
              {previewing ? "Scoring..." : "Preview score"}
            </Button>
            <Button
              data-testid="promotion-export"
              disabled={
                !previewIsCurrent || previewing || exporting || saving || launching || baselining
              }
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
                launching ||
                baselining
              }
              title={persistenceUnavailable ?? undefined}
              onClick={() => void saveCandidate()}
            >
              {saving ? <LoaderCircle className="animate-spin" /> : <Save />}
              {saving ? "Saving..." : "Save evaluation"}
            </Button>
            <Button
              data-testid="promotion-launch"
              disabled={
                !previewIsCurrent ||
                preview?.runnable_conversion.available !== true ||
                persistenceReadiness.state !== "ready" ||
                freshReadiness.state !== "ready" ||
                !freshLaunch.ok ||
                previewing ||
                exporting ||
                saving ||
                launching ||
                baselining
              }
              title={freshExecutionUnavailable ?? persistenceUnavailable ?? undefined}
              onClick={() => void launchFreshTrial()}
            >
              {launching ? <LoaderCircle className="animate-spin" /> : <Play />}
              {launching ? "Starting..." : "Run on current app"}
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
  const scenarioDiagnostics = preview.scenario_conversion.diagnostics ?? []
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
      <CardContent className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-3">
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
        <EvidenceFact
          label="Tool arguments"
          value={toolValueEvidenceSummary(evidence.tool_calls, "arguments")}
        />
        <EvidenceFact
          label="Tool results"
          value={toolValueEvidenceSummary(evidence.tool_calls, "result")}
        />
        <EvidenceFact
          label="Process events"
          value={
            evidence.process_event_evidence_state === "complete"
              ? String(evidence.process_events.length)
              : evidence.process_event_evidence_state
          }
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
          ? "This captured session can also be converted into runnable input for a current-app evaluation."
          : `Captured scoring and saving are available. Current-app execution needs authored runnable input${preview.runnable_conversion.reason_code ? ` (${preview.runnable_conversion.reason_code.replaceAll("_", " ")})` : ""}.`}
      </div>
      <div
        className="mx-3 mt-3 rounded-lg border border-border bg-muted/20 p-2.5 text-xs text-muted-foreground"
        data-testid="scenario-conversion-status"
      >
        {preview.scenario_conversion.available && preview.scenario_conversion.scenario ? (
          <>
            <CheckCircle2 className="mr-1.5 inline h-3.5 w-3.5 text-emerald-600" />
            Production scenario captured with {preview.scenario_conversion.scenario.events.length}{" "}
            ordered
            {preview.scenario_conversion.scenario.events.length === 1 ? " stage" : " stages"}.
          </>
        ) : (
          <div className="space-y-1.5">
            <div className="font-medium text-foreground">Scenario conversion needs attention</div>
            {scenarioDiagnostics.slice(0, 3).map((diagnostic) => (
              <div
                key={`${diagnostic.code}:${diagnostic.event_sequence ?? "none"}:${diagnostic.transcript_index ?? "none"}:${diagnostic.artifact_requirement_id ?? "none"}`}
              >
                {diagnostic.message} {diagnostic.remediation}
              </div>
            ))}
            {scenarioDiagnostics.length > 3 && (
              <div>{scenarioDiagnostics.length - 3} more findings.</div>
            )}
          </div>
        )}
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

function toolValueEvidenceSummary(
  calls: EvaluationPromotionPreview["candidate"]["evidence"]["tool_calls"],
  field: "arguments" | "result",
): string {
  const available = calls.filter((call) => call[field].state === "available").length
  if (available > 0) return `${available} available`
  if (calls.some((call) => call[field].state === "truncated")) return "truncated"
  return calls.length > 0 ? "not retained" : "unavailable"
}

type FreshLaunchRequest = Omit<
  CapturedEvaluationLaunch,
  "candidate" | "expected_candidate_revision" | "expected_execution_profile_revision"
>
type FreshLaunchValidation =
  | { ok: true; request: FreshLaunchRequest }
  | { ok: false; error: string }

function FreshExecutionEditor({
  target,
  loadingTarget,
  draft,
  validation,
  unavailable,
  edit,
}: {
  target: EvalTarget | undefined
  loadingTarget: boolean
  draft: FreshExecutionDraft
  validation: FreshLaunchValidation
  unavailable: string | null
  edit: (next: FreshExecutionDraft) => void
}) {
  const update = (field: keyof FreshExecutionDraft, value: string) =>
    edit({ ...draft, [field]: value })
  const targetSummary = target
    ? `${target.label} · up to ${target.max_trials} trial${target.max_trials === 1 ? "" : "s"}, concurrency ${target.max_concurrency}`
    : loadingTarget
      ? "Loading server-owned execution limits..."
      : "Server-owned execution limits are unavailable."

  return (
    <Card size="sm" data-testid="promotion-fresh-execution">
      <CardHeader>
        <CardTitle>Current-app execution</CardTitle>
        <p className="mt-1 text-xs text-muted-foreground">
          Starts new application work with the target&apos;s existing tools, environment, approvals,
          and operator policy. {targetSummary}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <FreshExecutionField
            label="Trials"
            id="promotion-trials"
            value={draft.trials}
            maximum={target?.max_trials}
            onChange={(value) => update("trials", value)}
          />
          <FreshExecutionField
            label="Parallel trials"
            id="promotion-concurrency"
            value={draft.maxConcurrency}
            maximum={target?.max_concurrency}
            onChange={(value) => update("maxConcurrency", value)}
          />
          <FreshExecutionField
            label="Trial timeout (seconds)"
            id="promotion-timeout"
            value={draft.timeoutSeconds}
            maximum={target?.max_timeout_seconds}
            onChange={(value) => update("timeoutSeconds", value)}
          />
          <FreshExecutionField
            label="Max model steps per trial"
            id="promotion-max-steps"
            value={draft.maxSteps}
            maximum={target?.max_steps}
            optional
            onChange={(value) => update("maxSteps", value)}
          />
          <FreshExecutionField
            label="Max total tokens per trial"
            id="promotion-max-total-tokens"
            value={draft.maxTotalTokens}
            optional
            onChange={(value) => update("maxTotalTokens", value)}
          />
          <FreshExecutionField
            label="Max tool calls per trial"
            id="promotion-max-tool-calls"
            value={draft.maxToolCalls}
            optional
            onChange={(value) => update("maxToolCalls", value)}
          />
          <FreshExecutionField
            label="Max run time per trial (seconds)"
            id="promotion-max-elapsed"
            value={draft.maxElapsedSeconds}
            optional
            onChange={(value) => update("maxElapsedSeconds", value)}
          />
          <PromotionField label="Max estimated cost per trial" id="promotion-max-cost">
            <div className="grid grid-cols-[minmax(0,1fr)_5rem] gap-2">
              <Input
                id="promotion-max-cost"
                inputMode="decimal"
                placeholder={target?.cost_budget_available ? "Optional" : "No pricing"}
                value={draft.maxEstimatedCost}
                disabled={target?.cost_budget_available !== true}
                onChange={(event) => update("maxEstimatedCost", event.target.value)}
              />
              <select
                aria-label="Cost budget currency"
                className={SELECT_CLASS}
                value={draft.currency}
                disabled={target?.cost_budget_available !== true}
                onChange={(event) => update("currency", event.target.value.toUpperCase())}
              >
                {(target?.cost_budget_currencies ?? []).map((currency) => (
                  <option key={currency} value={currency}>
                    {currency}
                  </option>
                ))}
              </select>
            </div>
          </PromotionField>
        </div>
        {(unavailable || !validation.ok) && (
          <div className="text-xs text-amber-700 dark:text-amber-300">
            {unavailable ?? (!validation.ok ? validation.error : null)}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function FreshExecutionField({
  label,
  id,
  value,
  maximum,
  optional = false,
  onChange,
}: {
  label: string
  id: string
  value: string
  maximum?: number
  optional?: boolean
  onChange: (value: string) => void
}) {
  return (
    <PromotionField label={label} id={id}>
      <Input
        id={id}
        type="number"
        inputMode="numeric"
        min={1}
        max={maximum}
        step={1}
        placeholder={optional ? "Inherit target" : undefined}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </PromotionField>
  )
}

function validateFreshExecution(
  draft: FreshExecutionDraft,
  target: EvalTarget | undefined,
): FreshLaunchValidation {
  if (target === undefined) {
    return { ok: false, error: "Execution target details have not loaded." }
  }
  if (!target.execution_profile_ready || target.execution_profile == null) {
    return {
      ok: false,
      error:
        target.execution_profile_diagnostics?.[0]?.message ??
        "The current execution profile is unavailable.",
    }
  }
  try {
    const trials = freshExecutionInteger(draft.trials, "Trials", target.max_trials)
    const maxConcurrency = freshExecutionInteger(
      draft.maxConcurrency,
      "Parallel trials",
      target.max_concurrency,
    )
    const timeoutSeconds = freshExecutionInteger(
      draft.timeoutSeconds,
      "Trial timeout",
      target.max_timeout_seconds,
    )
    const maxSteps = optionalFreshExecutionInteger(
      draft.maxSteps,
      "Max model steps per trial",
      target.max_steps,
    )
    const maxTotalTokens = optionalFreshExecutionInteger(
      draft.maxTotalTokens,
      "Max total tokens per trial",
      MAX_SAFE_RUNTIME_LIMIT,
    )
    const maxToolCalls = optionalFreshExecutionInteger(
      draft.maxToolCalls,
      "Max tool calls per trial",
      MAX_SAFE_RUNTIME_LIMIT,
    )
    const maxElapsedSeconds = optionalFreshExecutionInteger(
      draft.maxElapsedSeconds,
      "Max run time per trial",
      MAX_SAFE_RUNTIME_LIMIT,
    )
    const limits =
      maxTotalTokens === undefined && maxToolCalls === undefined && maxElapsedSeconds === undefined
        ? undefined
        : {
            scope: "run" as const,
            max_total_tokens: maxTotalTokens,
            max_tool_calls: maxToolCalls,
            max_elapsed_seconds: maxElapsedSeconds,
          }
    const cost = draft.maxEstimatedCost.trim()
    let costBudget: FreshLaunchRequest["cost_budget"]
    if (cost !== "") {
      if (!target.cost_budget_available) {
        throw new Error("This target has no server-owned pricing for a cost budget.")
      }
      if (!/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(cost) || Number(cost) <= 0) {
        throw new Error("Max estimated cost per trial must be a positive decimal.")
      }
      const currency = draft.currency.trim().toUpperCase()
      if (!target.cost_budget_currencies.includes(currency)) {
        throw new Error("Cost currency is not compatible with this target's current pricing.")
      }
      costBudget = { max_estimated_cost: cost, currency }
    }
    return {
      ok: true,
      request: {
        trial_request: { trials, timeout_seconds: timeoutSeconds },
        max_concurrency: maxConcurrency,
        max_steps: maxSteps,
        limits,
        cost_budget: costBudget,
      },
    }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Current-app execution settings are invalid.",
    }
  }
}

function freshExecutionInteger(source: string, label: string, maximum: number): number {
  if (!/^[1-9]\d*$/.test(source)) {
    throw new Error(`${label} must be a positive whole number.`)
  }
  const value = Number(source)
  if (!Number.isSafeInteger(value) || value > maximum) {
    throw new Error(`${label} must be from 1 to ${maximum}.`)
  }
  return value
}

function optionalFreshExecutionInteger(
  source: string,
  label: string,
  maximum: number,
): number | undefined {
  return source === "" ? undefined : freshExecutionInteger(source, label, maximum)
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

export function ExpectedBehaviorEditor({
  assertions,
  evidence,
  evidencePolicy,
  onChange,
}: {
  assertions: PromotionAssertion[]
  evidence: EvaluationPromotionPreview["candidate"]["evidence"] | undefined
  evidencePolicy?: EvaluationEvidencePolicySpec
  onChange: (assertions: PromotionAssertion[]) => void
}) {
  const [quickKind, setQuickKind] = useState<PromotionAssertionKind>("root_status")
  const editAssertions = (edit: (next: PromotionAssertion[]) => void) => {
    const next = structuredClone(assertions)
    edit(next)
    onChange(next)
  }
  const addAssertion = () =>
    editAssertions((next) => {
      next.push(
        evidence
          ? createCapturedEvaluationAssertionDraft(quickKind, next, evidence).assertion
          : createPromotionAssertion(quickKind, next),
      )
    })
  const quickKindUnavailable = assertionKindUnavailable(quickKind, evidencePolicy)
  const quickAddsObserved =
    evidence !== undefined && !capturedAssertionSuggestionUnavailable(quickKind, evidence)
  return (
    <Card size="sm">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Assertions</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            {evidence
              ? "Assertions are rescored against the captured run before export."
              : "Define deterministic behavior that every fresh trial must satisfy."}
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
              <option
                key={kind}
                value={kind}
                disabled={assertionKindUnavailable(kind, evidencePolicy)}
              >
                {PROMOTION_ASSERTION_LABELS[kind]}
                {assertionKindUnavailable(kind, evidencePolicy) ? " (target unavailable)" : ""}
              </option>
            ))}
          </select>
          <Button
            size="xs"
            variant="outline"
            disabled={assertions.length >= 64 || quickKindUnavailable}
            onClick={addAssertion}
          >
            <Plus /> {quickAddsObserved ? "Add observed" : "Add expectation"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {assertions.map((assertion, index) => (
          <AssertionEditor
            // biome-ignore lint/suspicious/noArrayIndexKey: rows never reorder and own no local state.
            key={index}
            assertion={assertion}
            index={index}
            assertions={assertions}
            evidence={evidence}
            evidencePolicy={evidencePolicy}
            update={(updated) =>
              editAssertions((next) => {
                next[index] = updated
              })
            }
            remove={() => editAssertions((next) => next.splice(index, 1))}
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
  evidencePolicy,
  update,
  remove,
}: {
  assertion: PromotionAssertion
  index: number
  assertions: PromotionAssertion[]
  evidence: EvaluationPromotionPreview["candidate"]["evidence"] | undefined
  evidencePolicy?: EvaluationEvidencePolicySpec
  update: (assertion: PromotionAssertion) => void
  remove: () => void
}) {
  const replaceKind = (kind: PromotionAssertionKind) => {
    const remaining = assertions.filter((_, candidateIndex) => candidateIndex !== index)
    const replacement = evidence
      ? createCapturedEvaluationAssertionDraft(kind, remaining, evidence).assertion
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
              <option
                key={kind}
                value={kind}
                disabled={assertionKindUnavailable(kind, evidencePolicy)}
              >
                {PROMOTION_ASSERTION_LABELS[kind]}
                {assertionKindUnavailable(kind, evidencePolicy) ? " (target unavailable)" : ""}
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
        <AssertionFields
          assertion={assertion}
          update={update}
          index={index}
          evidencePolicy={evidencePolicy}
        />
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
  evidencePolicy,
}: {
  assertion: PromotionAssertion
  update: (assertion: PromotionAssertion) => void
  index: number
  evidencePolicy?: EvaluationEvidencePolicySpec
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
            <ChildTerminalStatusSelect
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
    case "tool_arguments_contain":
    case "tool_result_contains":
      return (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <PromotionField label="Tool name" id={id("tool-name")}>
              <Input
                id={id("tool-name")}
                value={assertion.tool_name}
                onChange={(event) => update({ ...assertion, tool_name: event.target.value })}
              />
            </PromotionField>
            <IntegerField
              label="Occurrence"
              id={id("occurrence")}
              value={assertion.occurrence ?? 1}
              onChange={(occurrence) => update({ ...assertion, occurrence })}
            />
          </div>
          <PromotionField
            label={
              assertion.kind === "tool_arguments_contain"
                ? "Expected argument JSON subset"
                : "Expected result JSON subset"
            }
            id={id("expected-subset")}
          >
            <Textarea
              id={id("expected-subset")}
              className="min-h-32 font-mono"
              spellCheck={false}
              value={jsonSubsetEditorText(assertion.expected_subset)}
              onChange={(event) =>
                update({
                  ...assertion,
                  expected_subset: editedJsonSubset(event.target.value),
                })
              }
            />
          </PromotionField>
          <p className="text-xs text-muted-foreground">
            Object keys are matched recursively. Arrays use exact length and order. Values are
            bounded to 4,096 encoded bytes.
          </p>
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
    case "process_event":
      return (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <PromotionField label="Process event" id={id("event")}>
              <ProcessEventSelect
                id={id("event")}
                value={assertion.event}
                onChange={(event) => update({ ...assertion, event })}
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
          <p className="text-xs text-muted-foreground">
            Set both counts to zero to forbid this event. Event payloads and approval identity are
            never retained in portable assertions.
          </p>
        </div>
      )
    case "process_events_in_order":
      return (
        <div>
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className={LABEL_CLASS}>Exact selected process order</span>
            <Button
              size="xs"
              variant="outline"
              disabled={assertion.events.length >= 256}
              onClick={() =>
                update({
                  ...assertion,
                  events: [...assertion.events, "tool_call_completed"],
                })
              }
            >
              <Plus /> Add event
            </Button>
          </div>
          <div className="space-y-2">
            {assertion.events.map((processEvent, eventIndex) => (
              <div
                // biome-ignore lint/suspicious/noArrayIndexKey: ordered values own no local state.
                key={eventIndex}
                className="flex items-start gap-2"
              >
                <ProcessEventSelect
                  ariaLabel={`Expected process event ${eventIndex + 1}`}
                  value={processEvent}
                  onChange={(event) => {
                    const events = [...assertion.events]
                    events[eventIndex] = event
                    update({ ...assertion, events })
                  }}
                />
                <Button
                  size="icon-xs"
                  variant="ghost"
                  aria-label={`Remove expected process event ${eventIndex + 1}`}
                  disabled={assertion.events.length === 1}
                  onClick={() =>
                    update({
                      ...assertion,
                      events: assertion.events.filter(
                        (_, candidateIndex) => candidateIndex !== eventIndex,
                      ),
                    })
                  }
                >
                  <Trash2 />
                </Button>
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Advanced protocol check: Cayu filters the trace to the event kinds selected here, then
            requires the complete filtered sequence and multiplicity to match exactly.
          </p>
        </div>
      )
    case "workspace_file":
      return (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <PromotionField label="Relative workspace path" id={id("path")}>
              <Input
                id={id("path")}
                className="font-mono"
                value={assertion.path}
                onChange={(event) => update({ ...assertion, path: event.target.value })}
              />
            </PromotionField>
            <label className="mt-5 flex h-8 items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={assertion.present ?? true}
                onChange={(event) =>
                  update({
                    ...assertion,
                    present: event.target.checked,
                    minimum_bytes: event.target.checked ? assertion.minimum_bytes : null,
                    maximum_bytes: event.target.checked ? assertion.maximum_bytes : null,
                    sha256: event.target.checked ? assertion.sha256 : null,
                  })
                }
              />
              File must be present
            </label>
          </div>
          {(assertion.present ?? true) && (
            <>
              <div className="grid gap-3 sm:grid-cols-2">
                <NullableIntegerField
                  label="Minimum bytes"
                  id={id("minimum-bytes")}
                  placeholder="No minimum"
                  value={assertion.minimum_bytes}
                  onChange={(minimum_bytes) => update({ ...assertion, minimum_bytes })}
                />
                <NullableIntegerField
                  label="Maximum bytes"
                  id={id("maximum-bytes")}
                  value={assertion.maximum_bytes}
                  onChange={(maximum_bytes) => update({ ...assertion, maximum_bytes })}
                />
              </div>
              <PromotionField label="Whole-file SHA-256 (optional)" id={id("sha256")}>
                <Input
                  id={id("sha256")}
                  className="font-mono"
                  value={assertion.sha256 ?? ""}
                  onChange={(event) => update({ ...assertion, sha256: event.target.value || null })}
                />
              </PromotionField>
            </>
          )}
          <p className="text-xs text-muted-foreground">
            Cayu retains only presence, byte size, and an optional whole-file digest. Workspace
            contents never enter portable eval evidence.
          </p>
        </div>
      )
    case "artifact": {
      const artifactTextEnabled = evidencePolicy?.include_artifact_text === true
      const artifactTextLocked = !artifactTextEnabled && assertion.text_contains == null
      return (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <PromotionField label="Artifact scope" id={id("scope")}>
              <select
                id={id("scope")}
                className={SELECT_CLASS}
                value={assertion.scope ?? "session"}
                onChange={(event) =>
                  update({
                    ...assertion,
                    scope: event.target.value as "session" | "environment",
                  })
                }
              >
                <option value="session">Current session</option>
                <option value="environment">Current environment</option>
              </select>
            </PromotionField>
            <PromotionField label="Filename (optional)" id={id("filename")}>
              <Input
                id={id("filename")}
                value={assertion.filename ?? ""}
                onChange={(event) => update({ ...assertion, filename: event.target.value || null })}
              />
            </PromotionField>
            <PromotionField label="Content type (optional)" id={id("content-type")}>
              <Input
                id={id("content-type")}
                className="font-mono"
                placeholder="text/plain"
                value={assertion.content_type ?? ""}
                onChange={(event) =>
                  update({ ...assertion, content_type: event.target.value || null })
                }
              />
            </PromotionField>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <NullableIntegerField
              label="Minimum bytes"
              id={id("minimum-bytes")}
              placeholder="No minimum"
              value={assertion.minimum_bytes}
              onChange={(minimum_bytes) => update({ ...assertion, minimum_bytes })}
            />
            <NullableIntegerField
              label="Maximum bytes"
              id={id("maximum-bytes")}
              value={assertion.maximum_bytes}
              onChange={(maximum_bytes) => update({ ...assertion, maximum_bytes })}
            />
            <IntegerField
              label="Minimum count"
              id={id("minimum-count")}
              value={assertion.min_count ?? 1}
              onChange={(min_count) => update({ ...assertion, min_count })}
            />
            <NullableIntegerField
              label="Maximum count"
              id={id("maximum-count")}
              value={assertion.max_count}
              onChange={(max_count) => update({ ...assertion, max_count })}
            />
          </div>
          <PromotionField label="Whole-artifact SHA-256 (optional)" id={id("sha256")}>
            <Input
              id={id("sha256")}
              className="font-mono"
              value={assertion.sha256 ?? ""}
              onChange={(event) => update({ ...assertion, sha256: event.target.value || null })}
            />
          </PromotionField>
          <PromotionField label="Public artifact text contains (optional)" id={id("text")}>
            <Textarea
              id={id("text")}
              className="min-h-20 font-mono"
              disabled={artifactTextLocked}
              placeholder={
                artifactTextEnabled
                  ? "Expected bounded text"
                  : "The selected target does not retain public artifact text"
              }
              value={assertion.text_contains ?? ""}
              onChange={(event) =>
                update({ ...assertion, text_contains: event.target.value || null })
              }
            />
          </PromotionField>
          <p className="text-xs text-muted-foreground">
            Metadata is structural. Text matching is available only when the trusted target profile
            explicitly retains bounded, application-redacted public artifact text.
          </p>
        </div>
      )
    }
    case "memory_attribution":
      return (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <IntegerField
              label="Minimum admitted items"
              id={id("minimum-admitted-items")}
              value={assertion.min_admitted_items ?? 1}
              onChange={(value) => update({ ...assertion, min_admitted_items: value })}
            />
            <NullableIntegerField
              label="Maximum admitted items"
              id={id("maximum-admitted-items")}
              value={assertion.max_admitted_items}
              onChange={(value) => update({ ...assertion, max_admitted_items: value })}
            />
            <IntegerField
              label="Minimum provider exposures"
              id={id("minimum-provider-exposures")}
              value={assertion.min_provider_exposures ?? 1}
              onChange={(value) => update({ ...assertion, min_provider_exposures: value })}
            />
            <NullableIntegerField
              label="Maximum provider exposures"
              id={id("maximum-provider-exposures")}
              value={assertion.max_provider_exposures}
              onChange={(value) => update({ ...assertion, max_provider_exposures: value })}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            This proves bounded memory admission and provider exposure only. Complete evidence is
            required; correct use needs a reference-backed judge, and causal impact needs a paired
            experiment.
          </p>
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

function assertionKindUnavailable(
  kind: PromotionAssertionKind,
  policy: EvaluationEvidencePolicySpec | undefined,
): boolean {
  if (kind === "tool_arguments_contain") return policy?.include_tool_arguments !== true
  if (kind === "tool_result_contains") return policy?.include_tool_results !== true
  return false
}

function jsonSubsetEditorText(value: unknown): string {
  return typeof value === "object" && value !== null
    ? JSON.stringify(value, null, 2)
    : String(value ?? "")
}

function editedJsonSubset(source: string): Record<string, unknown> {
  try {
    return JSON.parse(source) as Record<string, unknown>
  } catch {
    // Preserve transient invalid text in editor state. Shared draft validation
    // blocks preview/save until the text becomes one bounded JSON object.
    return source as unknown as Record<string, unknown>
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
  placeholder = "No maximum",
}: {
  label: string
  id: string
  value: number | null | undefined
  onChange: (value: number | null) => void
  placeholder?: string
}) {
  return (
    <PromotionField label={label} id={id}>
      <Input
        id={id}
        type="number"
        min={0}
        step={1}
        placeholder={placeholder}
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

function ChildTerminalStatusSelect({
  id,
  value,
  onChange,
}: {
  id: string
  value: "completed" | "failed" | "interrupted"
  onChange: (value: "completed" | "failed" | "interrupted") => void
}) {
  return (
    <select
      id={id}
      className={SELECT_CLASS}
      value={value}
      onChange={(event) => onChange(event.target.value as "completed" | "failed" | "interrupted")}
    >
      <option value="completed">Completed</option>
      <option value="failed">Failed</option>
      <option value="interrupted">Interrupted</option>
    </select>
  )
}

function ProcessEventSelect({
  id,
  ariaLabel,
  value,
  onChange,
}: {
  id?: string
  ariaLabel?: string
  value: ProcessEventKind
  onChange: (value: ProcessEventKind) => void
}) {
  return (
    <select
      id={id}
      aria-label={ariaLabel}
      className={SELECT_CLASS}
      value={value}
      onChange={(event) => onChange(event.target.value as ProcessEventKind)}
    >
      {PROCESS_EVENT_OPTIONS.map(([event, label]) => (
        <option key={event} value={event}>
          {label}
        </option>
      ))}
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
