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
  type EvaluationPromotionCandidateDraft,
  type EvaluationPromotionPreview,
  exportEvaluationPromotion,
  importEvalCorpus,
  previewEvaluationPromotion,
} from "@/lib/api"
import { dashboardCapabilityUnavailableText } from "@/lib/dashboard-capabilities"
import { parseEvalCorpusFile, shortEvalIdentity } from "@/lib/evals-dashboard"
import {
  createPromotionAssertion,
  PROMOTION_ASSERTION_KINDS,
  PROMOTION_ASSERTION_LABELS,
  type PromotionAssertion,
  type PromotionAssertionKind,
  previewMatchesDraft,
  promotionDraftFromCandidate,
  validatePromotionDraft,
} from "@/lib/evaluation-promotion"
import { useDashboardCapability } from "./server-contract"

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
  const readCapability = useDashboardCapability({
    kind: "surface",
    surface: "evaluation_promotion",
  })
  const mutateCapability = useDashboardCapability({
    kind: "surface",
    surface: "evaluation_promotion",
    operation: "mutate",
  })
  const evalsMutateCapability = useDashboardCapability({
    kind: "surface",
    surface: "evals",
    operation: "mutate",
  })
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [preview, setPreview] = useState<EvaluationPromotionPreview | null>(null)
  const [draft, setDraft] = useState<EvaluationPromotionCandidateDraft | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [savedRevision, setSavedRevision] = useState<string | null>(null)
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
    setSavedRevision(null)
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
  const mutationUnavailable = dashboardCapabilityUnavailableText(mutateCapability)
  const evalsMutationUnavailable = dashboardCapabilityUnavailableText(evalsMutateCapability)

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
    let exportCompleted = false
    try {
      const exported = await exportPreviewedCandidate(controller.signal)
      if (exported === null || controller.signal.aborted) return
      exportCompleted = true
      const corpus = await parseEvalCorpusFile(exported.blob)
      const imported = await importEvalCorpus(corpus, controller.signal)
      if (controller.signal.aborted) return
      setSavedRevision(imported.revision)
      await queryClient.invalidateQueries({ queryKey: ["evals", "corpora"] })
    } catch (saveError) {
      if (controller.signal.aborted) return
      if (!exportCompleted && isPromotionConflict(saveError)) {
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

  if (!readCapability.enabled || !ELIGIBLE_STATUSES.has(status)) return null

  return (
    <>
      <Button size="sm" variant="outline" data-testid="promote-to-eval" onClick={openPromotion}>
        <FlaskConical className="h-4 w-4" />
        Promote to eval
      </Button>
      <Sheet open={open} onOpenChange={changeOpen}>
        <SheetContent
          className="w-full! gap-0 sm:max-w-4xl!"
          data-testid="promotion-sheet"
          aria-busy={previewing || exporting || saving}
        >
          <SheetHeader className="border-b border-border pr-12">
            <SheetTitle>Promote captured run to an eval</SheetTitle>
            <SheetDescription>
              Edit a portable candidate, score it against this run, then save or export the exact
              previewed version.
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
                  The promotion candidate could not be loaded. No eval has been exported.
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
                <fieldset className="contents" disabled={previewing || exporting || saving}>
                  <PromotionIdentityEditor draft={draft} editDraft={editDraft} />
                  <PromotionInputEditor draft={draft} editDraft={editDraft} />
                  <PromotionAssertionsEditor draft={draft} editDraft={editDraft} />
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
                Saved corpus {shortEvalIdentity(savedRevision)} to Evals.{" "}
                <Link
                  to="/evals"
                  search={{ tab: "catalog", corpus: savedRevision }}
                  className="font-medium underline"
                  onClick={() => changeOpen(false)}
                >
                  Open Evals
                </Link>
              </div>
            )}
          </div>

          <SheetFooter className="border-t border-border bg-background sm:flex-row sm:items-center sm:justify-end">
            {mutationUnavailable && (
              <span className="mr-auto text-xs text-muted-foreground">{mutationUnavailable}</span>
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
              disabled={
                !previewIsCurrent || !mutateCapability.enabled || previewing || exporting || saving
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
                !mutateCapability.enabled ||
                !evalsMutateCapability.enabled ||
                previewing ||
                exporting ||
                saving
              }
              title={evalsMutationUnavailable ?? undefined}
              onClick={() => void saveCandidate()}
            >
              {saving ? <LoaderCircle className="animate-spin" /> : <Save />}
              {saving ? "Saving..." : "Save to Evals"}
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
        <PromotionField label="Trials" id="promotion-trials">
          <Input
            id="promotion-trials"
            type="number"
            min={1}
            max={100}
            value={draft.suite.trial_request.trials ?? 1}
            onChange={(event) =>
              editDraft((next) => (next.suite.trial_request.trials = Number(event.target.value)))
            }
          />
        </PromotionField>
        <PromotionField label="Timeout seconds" id="promotion-timeout">
          <Input
            id="promotion-timeout"
            type="number"
            min={1}
            max={3600}
            value={draft.suite.trial_request.timeout_seconds ?? 300}
            onChange={(event) =>
              editDraft(
                (next) => (next.suite.trial_request.timeout_seconds = Number(event.target.value)),
              )
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

function PromotionInputEditor({ draft, editDraft }: PromotionEditorProps) {
  return (
    <Card size="sm">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Eval input</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Portable user messages captured before the run began.
          </p>
        </div>
        <Button
          size="xs"
          variant="outline"
          disabled={draft.case.input.messages.length >= 16}
          onClick={() =>
            editDraft((next) => next.case.input.messages.push({ role: "user", text: "" }))
          }
        >
          <Plus /> Add message
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {draft.case.input.messages.map((message, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: portable input messages have no identity field.
          <div key={index} className="rounded-lg border border-border p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium">User message {index + 1}</span>
              <Button
                size="icon-xs"
                variant="ghost"
                aria-label={`Remove user message ${index + 1}`}
                disabled={draft.case.input.messages.length === 1}
                onClick={() => editDraft((next) => next.case.input.messages.splice(index, 1))}
              >
                <Trash2 />
              </Button>
            </div>
            <Textarea
              aria-label={`User message ${index + 1}`}
              className="min-h-24 font-mono"
              value={message.text}
              onChange={(event) => {
                const text = event.target.value
                editDraft((next) => {
                  const target = next.case.input.messages[index]
                  if (target !== undefined) target.text = text
                })
              }}
            />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function PromotionAssertionsEditor({ draft, editDraft }: PromotionEditorProps) {
  const addAssertion = () =>
    editDraft((next) => {
      next.case.assertions.push(createPromotionAssertion("root_status", next.case.assertions))
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
        <Button
          size="xs"
          variant="outline"
          disabled={draft.case.assertions.length >= 64}
          onClick={addAssertion}
        >
          <Plus /> Add assertion
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {draft.case.assertions.map((assertion, index) => (
          <AssertionEditor
            // biome-ignore lint/suspicious/noArrayIndexKey: rows never reorder and own no local state.
            key={index}
            assertion={assertion}
            index={index}
            assertions={draft.case.assertions}
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
  update,
  remove,
}: {
  assertion: PromotionAssertion
  index: number
  assertions: PromotionAssertion[]
  update: (assertion: PromotionAssertion) => void
  remove: () => void
}) {
  const replaceKind = (kind: PromotionAssertionKind) => {
    const replacement = createPromotionAssertion(
      kind,
      assertions.filter((_, candidateIndex) => candidateIndex !== index),
    )
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
  if (warning === "input_redacted") return "Sensitive input was redacted before promotion."
  if (warning === "source_run_failed")
    return "The source run failed; review its expectations carefully."
  return warning
}

function promotionErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError && error.detail && typeof error.detail === "object") {
    const message = (error.detail as Record<string, unknown>).message
    if (typeof message === "string" && message.trim()) return message
  }
  return error instanceof Error ? error.message : "The eval promotion request failed."
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
