import { ArrowDown, ArrowUp, CheckCircle2, FlaskConical, Plus, Trash2 } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  type EvalJudgeCalibrationPreview,
  type EvalJudgeCalibrationReport,
  previewEvalJudgeCalibration,
  runEvalJudgeCalibration,
} from "@/lib/api"
import {
  EVAL_JUDGE_MAX_CRITERIA,
  judgeProfileForAssertion,
  judgeRouteForAssertion,
  newJudgeCalibrationDraft,
  validateJudgeCalibrationDraft,
} from "@/lib/eval-judge-authoring"
import { evalErrorMessage, shortEvalIdentity } from "@/lib/evals-dashboard"
import type {
  EvalJudgeCalibrationDraftV1,
  EvalTargetCatalogEntry,
  StructuredModelJudgeAssertionDraftV1,
  StructuredModelJudgeAssertionSpec,
} from "@/lib/generated/server-api"

const FIELD_LABEL = "mb-1 block text-xs font-medium text-muted-foreground"
const SELECT_CLASS =
  "h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"

export function StructuredJudgeEditor({
  assertion,
  target,
  reviewedAssertion,
  defaultTask,
  disabled,
  onChange,
  onRemove,
}: {
  assertion: StructuredModelJudgeAssertionDraftV1
  target: EvalTargetCatalogEntry | undefined
  reviewedAssertion: StructuredModelJudgeAssertionSpec | null
  defaultTask: string
  disabled: boolean
  onChange: (assertion: StructuredModelJudgeAssertionDraftV1) => void
  onRemove: () => void
}) {
  const profiles = target?.judge_profiles ?? []
  const profile = judgeProfileForAssertion(target, assertion)
  const candidateRouteRelation = judgeRouteForAssertion(target, assertion)
  const privateReferences = (target?.judge_private_references ?? []).filter(
    (entry) =>
      entry.judge_profile_key === assertion.judge_profile_key &&
      entry.judge_profile_revision === assertion.judge_profile_revision,
  )
  const sameModel = candidateRouteRelation === "same_model"

  const edit = (mutate: (next: StructuredModelJudgeAssertionDraftV1) => void) => {
    const next = structuredClone(assertion)
    mutate(next)
    onChange(next)
  }

  const referenceMode = assertion.reference?.kind ?? "none"

  return (
    <Card size="sm" data-testid="structured-judge-editor">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>AI judge rubric</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            A trusted server-owned model scores fixed criteria. Cayu computes the weighted result.
          </p>
        </div>
        <Button type="button" size="icon-xs" variant="ghost" disabled={disabled} onClick={onRemove}>
          <Trash2 />
          <span className="sr-only">Remove AI judge</span>
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Assertion ID" id={`${assertion.id}-judge-id`}>
            <Input
              id={`${assertion.id}-judge-id`}
              value={assertion.id}
              disabled={disabled}
              onChange={(event) => edit((next) => (next.id = event.target.value))}
            />
          </Field>
          <Field label="Trusted judge profile" id={`${assertion.id}-judge-profile`}>
            <select
              id={`${assertion.id}-judge-profile`}
              className={SELECT_CLASS}
              value={`${assertion.judge_profile_key}:${assertion.judge_profile_revision}`}
              disabled={disabled || profiles.length === 0}
              onChange={(event) => {
                const selected = profiles.find(
                  (item) => `${item.key}:${item.revision}` === event.target.value,
                )
                if (!selected) return
                edit((next) => {
                  next.judge_profile_key = selected.key
                  next.judge_profile_revision = selected.revision
                  next.reference = null
                  next.evidence = {
                    schema_version: 1,
                    include_final_output: true,
                    include_transcript: false,
                  }
                })
              }}
            >
              {profiles.length === 0 && <option value="">No judge profiles configured</option>}
              {profiles.map((item) => (
                <option key={item.revision} value={`${item.key}:${item.revision}`}>
                  {item.label} · {item.provider_name}/{item.model}
                  {target?.judge_profile_routes?.find(
                    (route) =>
                      route.judge_profile_key === item.key &&
                      route.judge_profile_revision === item.revision,
                  )?.candidate_route_relation === "same_model"
                    ? " · same model as candidate"
                    : ""}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Description" id={`${assertion.id}-judge-description`} wide>
            <Input
              id={`${assertion.id}-judge-description`}
              value={assertion.description ?? ""}
              disabled={disabled}
              onChange={(event) => edit((next) => (next.description = event.target.value || null))}
            />
          </Field>
          <Field label="Rubric ID" id={`${assertion.id}-rubric-id`}>
            <Input
              id={`${assertion.id}-rubric-id`}
              value={assertion.rubric.id}
              disabled={disabled}
              onChange={(event) => edit((next) => (next.rubric.id = event.target.value))}
            />
          </Field>
          <Field label="Pass threshold (0–1)" id={`${assertion.id}-threshold`}>
            <Input
              id={`${assertion.id}-threshold`}
              inputMode="decimal"
              value={assertion.threshold ?? "0.5"}
              disabled={disabled}
              onChange={(event) => edit((next) => (next.threshold = event.target.value))}
            />
          </Field>
        </div>

        {profile ? (
          <div className="flex flex-wrap gap-1.5 text-xs" data-testid="judge-profile-summary">
            <Badge variant="outline">
              {profile.provider_name}/{profile.model}
            </Badge>
            <Badge variant="outline">privacy {profile.privacy_policy_key}</Badge>
            <Badge variant="outline">≤ {profile.max_total_tokens.toLocaleString()} tokens</Badge>
            {profile.max_estimated_cost && (
              <Badge variant="outline">
                ≤ {profile.max_estimated_cost} {profile.cost_currency}
              </Badge>
            )}
            {sameModel && (
              <Badge variant="secondary">
                Same model as candidate · explicitly selected and labeled
              </Badge>
            )}
          </div>
        ) : (
          <div className="text-xs text-amber-700 dark:text-amber-300">
            Select a current trusted judge profile before checking the suite.
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div>
              <div className="text-sm font-medium">Criteria</div>
              <div className="text-xs text-muted-foreground">
                Stable ordered IDs; canonical weights must sum exactly to 1.
              </div>
            </div>
            <Button
              type="button"
              size="xs"
              variant="outline"
              disabled={disabled || assertion.rubric.criteria.length >= EVAL_JUDGE_MAX_CRITERIA}
              onClick={() =>
                edit((next) =>
                  next.rubric.criteria.push({
                    id: `criterion-${next.rubric.criteria.length + 1}`,
                    name: `Criterion ${next.rubric.criteria.length + 1}`,
                    description: "Describe the behavior this criterion measures.",
                    weight: "0",
                  }),
                )
              }
            >
              <Plus /> Add criterion
            </Button>
          </div>
          {assertion.rubric.criteria.map((criterion, index) => (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: criterion rows do not own local state, and their editable IDs may temporarily collide.
              key={`${index}:${assertion.rubric.criteria.length}`}
              className="grid gap-2 rounded-md border border-border p-3 sm:grid-cols-[minmax(8rem,0.7fr)_minmax(10rem,1fr)_7rem_auto]"
            >
              <Input
                aria-label={`Criterion ${index + 1} ID`}
                value={criterion.id}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    const current = next.rubric.criteria[index]
                    if (current) current.id = event.target.value
                  })
                }
              />
              <Input
                aria-label={`Criterion ${index + 1} name`}
                value={criterion.name}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    const current = next.rubric.criteria[index]
                    if (current) current.name = event.target.value
                  })
                }
              />
              <Input
                aria-label={`Criterion ${index + 1} weight`}
                inputMode="decimal"
                value={criterion.weight}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    const current = next.rubric.criteria[index]
                    if (current) current.weight = event.target.value
                  })
                }
              />
              <div className="flex justify-end gap-1">
                <Button
                  type="button"
                  size="icon-xs"
                  variant="ghost"
                  aria-label={`Move criterion ${index + 1} up`}
                  disabled={disabled || index === 0}
                  onClick={() =>
                    edit((next) => {
                      const [moved] = next.rubric.criteria.splice(index, 1)
                      if (moved) next.rubric.criteria.splice(index - 1, 0, moved)
                    })
                  }
                >
                  <ArrowUp />
                </Button>
                <Button
                  type="button"
                  size="icon-xs"
                  variant="ghost"
                  aria-label={`Move criterion ${index + 1} down`}
                  disabled={disabled || index === assertion.rubric.criteria.length - 1}
                  onClick={() =>
                    edit((next) => {
                      const [moved] = next.rubric.criteria.splice(index, 1)
                      if (moved) next.rubric.criteria.splice(index + 1, 0, moved)
                    })
                  }
                >
                  <ArrowDown />
                </Button>
                <Button
                  type="button"
                  size="icon-xs"
                  variant="ghost"
                  aria-label={`Remove criterion ${index + 1}`}
                  disabled={disabled || assertion.rubric.criteria.length === 1}
                  onClick={() => edit((next) => next.rubric.criteria.splice(index, 1))}
                >
                  <Trash2 />
                </Button>
              </div>
              <Textarea
                className="sm:col-span-4"
                rows={2}
                aria-label={`Criterion ${index + 1} description`}
                value={criterion.description}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    const current = next.rubric.criteria[index]
                    if (current) current.description = event.target.value
                  })
                }
              />
            </div>
          ))}
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Judge evidence" id={`${assertion.id}-evidence`}>
            <select
              id={`${assertion.id}-evidence`}
              className={SELECT_CLASS}
              value={assertion.evidence?.include_transcript ? "transcript" : "final_output"}
              disabled={disabled || !profile?.allowed_evidence.includes("transcript")}
              onChange={(event) =>
                edit((next) => {
                  next.evidence = {
                    schema_version: 1,
                    include_final_output: true,
                    include_transcript: event.target.value === "transcript",
                  }
                })
              }
            >
              <option value="final_output">Final output only</option>
              {profile?.allowed_evidence.includes("transcript") && (
                <option value="transcript">Final output and transcript</option>
              )}
            </select>
          </Field>
          <Field label="Reference truth" id={`${assertion.id}-reference-mode`}>
            <select
              id={`${assertion.id}-reference-mode`}
              className={SELECT_CLASS}
              value={referenceMode}
              disabled={disabled}
              onChange={(event) => {
                const mode = event.target.value
                edit((next) => {
                  if (mode === "public_reference") {
                    next.reference = {
                      schema_version: 1,
                      kind: "public_reference",
                      id: "expected-answer",
                      expected_answer: "Describe the expected answer.",
                      expected_facts: [],
                    }
                  } else if (mode === "private_reference") {
                    const selected = privateReferences[0]
                    next.reference = selected
                      ? { ...selected.reference, kind: "private_reference" }
                      : null
                  } else {
                    next.reference = null
                  }
                })
              }}
            >
              <option value="none">No reference</option>
              {profile?.allowed_evidence.includes("public_reference") && (
                <option value="public_reference">Public expected answer/facts</option>
              )}
              {profile?.allowed_evidence.includes("private_reference") &&
                privateReferences.length > 0 && (
                  <option value="private_reference">Server-held private reference</option>
                )}
            </select>
          </Field>
        </div>

        {assertion.reference?.kind === "public_reference" && (
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Public reference ID" id={`${assertion.id}-public-reference-id`}>
              <Input
                id={`${assertion.id}-public-reference-id`}
                value={assertion.reference.id}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    if (next.reference?.kind === "public_reference") {
                      next.reference.id = event.target.value
                    }
                  })
                }
              />
            </Field>
            <Field label="Expected answer" id={`${assertion.id}-expected-answer`}>
              <Textarea
                id={`${assertion.id}-expected-answer`}
                rows={3}
                value={assertion.reference.expected_answer ?? ""}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    if (next.reference?.kind === "public_reference") {
                      next.reference.expected_answer = event.target.value || null
                    }
                  })
                }
              />
            </Field>
            <Field label="Expected facts (one per line)" id={`${assertion.id}-expected-facts`} wide>
              <Textarea
                id={`${assertion.id}-expected-facts`}
                rows={3}
                value={(assertion.reference.expected_facts ?? []).join("\n")}
                disabled={disabled}
                onChange={(event) =>
                  edit((next) => {
                    if (next.reference?.kind === "public_reference") {
                      next.reference.expected_facts = event.target.value
                        ? event.target.value.split("\n")
                        : []
                    }
                  })
                }
              />
            </Field>
          </div>
        )}

        {assertion.reference?.kind === "private_reference" && (
          <Field label="Private reference" id={`${assertion.id}-private-reference`}>
            <select
              id={`${assertion.id}-private-reference`}
              className={SELECT_CLASS}
              value={`${assertion.reference.key}:${assertion.reference.revision}`}
              disabled={disabled}
              onChange={(event) => {
                const selected = privateReferences.find(
                  (item) =>
                    `${item.reference.key}:${item.reference.revision}` === event.target.value,
                )
                if (selected) {
                  edit(
                    (next) =>
                      (next.reference = { ...selected.reference, kind: "private_reference" }),
                  )
                }
              }}
            >
              {privateReferences.map((item) => (
                <option
                  key={`${item.reference.key}:${item.reference.revision}`}
                  value={`${item.reference.key}:${item.reference.revision}`}
                >
                  {item.reference.key} · {shortEvalIdentity(item.reference.revision)}
                </option>
              ))}
            </select>
          </Field>
        )}

        <JudgeCalibrationPanel
          targetKey={target?.target_key ?? ""}
          assertion={reviewedAssertion}
          defaultTask={defaultTask}
          disabled={disabled}
        />
      </CardContent>
    </Card>
  )
}

function JudgeCalibrationPanel({
  targetKey,
  assertion,
  defaultTask,
  disabled,
}: {
  targetKey: string
  assertion: StructuredModelJudgeAssertionSpec | null
  defaultTask: string
  disabled: boolean
}) {
  const [draft, setDraft] = useState<EvalJudgeCalibrationDraftV1 | null>(null)
  const [preview, setPreview] = useState<EvalJudgeCalibrationPreview | null>(null)
  const [previewIdentity, setPreviewIdentity] = useState<string | null>(null)
  const [report, setReport] = useState<EvalJudgeCalibrationReport | null>(null)
  const [runId, setRunId] = useState("")
  const [pending, setPending] = useState<"preview" | "run" | null>(null)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    controllerRef.current?.abort()
    setDraft(assertion ? newJudgeCalibrationDraft(targetKey, assertion, defaultTask) : null)
    setPreview(null)
    setPreviewIdentity(null)
    setReport(null)
    setRunId("")
    setPending(null)
    setError(null)
  }, [assertion, defaultTask, targetKey])
  useEffect(() => () => controllerRef.current?.abort(), [])

  const identity = useMemo(() => (draft ? JSON.stringify(draft) : null), [draft])
  const previewCurrent = preview !== null && previewIdentity === identity

  const edit = (mutate: (next: EvalJudgeCalibrationDraftV1) => void) => {
    if (!draft) return
    const next = structuredClone(draft)
    mutate(next)
    setDraft(next)
    setReport(null)
    setError(null)
  }

  const runAction = async (
    name: "preview" | "run",
    action: (signal: AbortSignal) => Promise<void>,
  ) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setPending(name)
    setError(null)
    try {
      await action(controller.signal)
    } catch (actionError) {
      if (!controller.signal.aborted) {
        setError(evalErrorMessage(actionError, "The judge calibration operation failed."))
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setPending(null)
      }
    }
  }

  const check = () => {
    if (!draft || pending) return
    try {
      validateJudgeCalibrationDraft(draft)
    } catch (validationError) {
      setError(validationError instanceof Error ? validationError.message : "Invalid calibration.")
      return
    }
    const current = identity
    void runAction("preview", async (signal) => {
      const result = await previewEvalJudgeCalibration({ draft }, signal)
      if (signal.aborted) return
      setPreview(result)
      setPreviewIdentity(current)
      setReport(null)
      setRunId(randomCalibrationRunId())
    })
  }

  const run = () => {
    if (!previewCurrent || !preview?.ready || !runId || pending) return
    void runAction("run", async (signal) => {
      const result = await runEvalJudgeCalibration(
        {
          run_id: runId,
          definition: preview.definition,
          expected_definition_revision: preview.definition.revision,
        },
        signal,
      )
      if (!signal.aborted) setReport(result.report)
    })
  }

  if (!assertion || !draft) {
    return (
      <div className="rounded-md border border-dashed border-border p-3 text-xs text-muted-foreground">
        Check the current suite first. Calibration is bound to the exact compiled rubric and judge
        profile reviewed by the server.
      </div>
    )
  }

  return (
    <div
      className="space-y-3 rounded-lg border border-border bg-muted/20 p-3"
      data-testid="judge-calibration"
    >
      <div>
        <div className="text-sm font-medium">Calibrate on fixed evidence</div>
        <div className="text-xs text-muted-foreground">
          Compare repeated judge scores with your labels. This never runs the candidate agent.
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Calibration ID" id={`${assertion.id}-calibration-id`}>
          <Input
            id={`${assertion.id}-calibration-id`}
            value={draft.id}
            disabled={disabled || pending !== null}
            onChange={(event) => edit((next) => (next.id = event.target.value))}
          />
        </Field>
        <Field label="Repeated judge calls" id={`${assertion.id}-calibration-trials`}>
          <Input
            id={`${assertion.id}-calibration-trials`}
            type="number"
            min={1}
            max={10}
            value={draft.trials ?? 1}
            disabled={disabled || pending !== null}
            onChange={(event) => edit((next) => (next.trials = Number(event.target.value)))}
          />
        </Field>
        <Field label="Evidence source ID" id={`${assertion.id}-calibration-source`} wide>
          <Input
            id={`${assertion.id}-calibration-source`}
            aria-label="Evidence source ID"
            aria-describedby={`${assertion.id}-calibration-source-help`}
            value={draft.evidence_source_id}
            disabled={disabled || pending !== null}
            onChange={(event) => edit((next) => (next.evidence_source_id = event.target.value))}
          />
          <div
            id={`${assertion.id}-calibration-source-help`}
            className="mt-1 text-xs text-muted-foreground"
          >
            An operator-declared link to where this fixed evidence came from; Cayu does not infer or
            verify the source.
          </div>
        </Field>
        <Field label="Fixed task" id={`${assertion.id}-calibration-task`} wide>
          <Textarea
            id={`${assertion.id}-calibration-task`}
            aria-label="Fixed task"
            rows={3}
            value={draft.task}
            disabled={disabled || pending !== null}
            onChange={(event) => edit((next) => (next.task = event.target.value))}
          />
        </Field>
        <Field label="Known candidate output" id={`${assertion.id}-calibration-output`} wide>
          <Textarea
            id={`${assertion.id}-calibration-output`}
            aria-label="Known candidate output"
            aria-describedby={`${assertion.id}-calibration-output-help`}
            rows={4}
            value={draft.final_output}
            disabled={disabled || pending !== null}
            onChange={(event) => edit((next) => (next.final_output = event.target.value))}
          />
          <div
            id={`${assertion.id}-calibration-output-help`}
            className="mt-1 text-xs text-muted-foreground"
          >
            Untrusted candidate evidence only. It is delimited from evaluator instructions and
            cannot add judge authority.
          </div>
        </Field>
        {assertion.evidence?.include_transcript && (
          <Field label="Fixed transcript" id={`${assertion.id}-calibration-transcript`} wide>
            <Textarea
              id={`${assertion.id}-calibration-transcript`}
              aria-label="Fixed transcript"
              rows={4}
              value={draft.transcript ?? ""}
              disabled={disabled || pending !== null}
              onChange={(event) => edit((next) => (next.transcript = event.target.value || null))}
            />
          </Field>
        )}
      </div>
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground">Human ground truth</div>
        {draft.human_criteria.map((label, index) => (
          <label
            key={label.criterion_id}
            htmlFor={`${assertion.id}-human-${label.criterion_id}`}
            className="grid items-center gap-2 text-xs sm:grid-cols-[minmax(8rem,1fr)_8rem]"
          >
            <span>
              {assertion.rubric.criteria[index]?.name ?? label.criterion_id} ({label.criterion_id})
            </span>
            <Input
              id={`${assertion.id}-human-${label.criterion_id}`}
              aria-label={`Human score for ${label.criterion_id}`}
              inputMode="decimal"
              value={label.score}
              disabled={disabled || pending !== null}
              onChange={(event) =>
                edit((next) => {
                  const current = next.human_criteria[index]
                  if (current) current.score = event.target.value
                })
              }
            />
          </label>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={disabled || pending !== null}
          onClick={check}
        >
          <CheckCircle2 /> {pending === "preview" ? "Checking..." : "Check calibration"}
        </Button>
        <Button
          type="button"
          size="sm"
          disabled={disabled || pending !== null || !previewCurrent || !preview?.ready}
          onClick={run}
        >
          <FlaskConical /> {pending === "run" ? "Running judge..." : "Run calibration"}
        </Button>
      </div>
      {previewCurrent && preview && (
        <div className="rounded-md border border-border p-2 text-xs">
          <div className="font-medium">
            {preview.ready ? "Calibration is ready" : "Calibration needs attention"}
          </div>
          {preview.diagnostics?.map((diagnostic) => (
            <div key={`${diagnostic.code}:${diagnostic.message}`} className="mt-1">
              {diagnostic.message}
            </div>
          ))}
          {preview.work && (
            <div className="mt-1 text-muted-foreground">
              {preview.work.judge_calls} judge calls · ≤{" "}
              {preview.work.max_total_tokens.toLocaleString()} tokens
              {preview.work.max_estimated_cost
                ? ` · ≤ ${preview.work.max_estimated_cost} ${preview.work.cost_currency}`
                : ""}
            </div>
          )}
          {preview.candidate_route_relation === "same_model" && (
            <div className="mt-1 text-amber-700 dark:text-amber-300">
              Same-model judge: results are explicitly labeled and should not be treated as an
              independent evaluation.
            </div>
          )}
        </div>
      )}
      {report && (
        <div className="space-y-2 rounded-md border border-emerald-500/30 bg-emerald-500/5 p-2 text-xs">
          <div className="font-medium">
            Calibration complete · human aggregate {report.definition.human_label.aggregate_score}
          </div>
          {report.trials.map((trial) => {
            const detail = trial.judgment.detail
            const aggregate =
              detail.kind === "structured_model_judge" ? detail.aggregate_score : null
            const humanScores = new Map(
              report.definition.human_label.criteria.map((label) => [
                label.criterion_id,
                label.score,
              ]),
            )
            return (
              <div
                key={trial.revision}
                className="rounded border border-border/70 bg-background/60 p-2"
              >
                Trial {trial.sequence}: {trial.judgment.outcome} · score{" "}
                {aggregate ?? "unavailable"}
                {trial.aggregate_absolute_error != null
                  ? ` · absolute error ${trial.aggregate_absolute_error}`
                  : ""}
                {trial.pass_agreement != null
                  ? ` · pass agreement ${trial.pass_agreement ? "yes" : "no"}`
                  : ""}
                {detail.kind === "structured_model_judge" && (
                  <div className="mt-2 space-y-1 border-t border-border/70 pt-2">
                    <div className="text-muted-foreground">
                      Evaluator: {detail.diagnostic.replaceAll("_", " ")}
                    </div>
                    {(detail.criteria ?? []).map((criterion) => (
                      <div key={criterion.criterion_id}>
                        <span className="font-medium">{criterion.criterion_id}</span>: judge{" "}
                        {criterion.score} · human {humanScores.get(criterion.criterion_id) ?? "—"} ·
                        weight {criterion.weight}
                        {criterion.explanation
                          ? ` · ${criterion.explanation}`
                          : ` · explanation ${criterion.explanation_state}`}
                      </div>
                    ))}
                    {detail.usage && (
                      <div className="text-muted-foreground">
                        Usage: {detail.usage.model_steps} model step
                        {detail.usage.model_steps === 1 ? "" : "s"} · {detail.usage.input_tokens}
                        {" input / "}
                        {detail.usage.output_tokens} output / {detail.usage.total_tokens} total
                        tokens
                      </div>
                    )}
                    {detail.cost && (
                      <div className="text-muted-foreground">
                        Cost:{" "}
                        {detail.cost.availability === "priced"
                          ? `${detail.cost.estimated_cost} ${detail.cost.currency}`
                          : "unavailable"}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
          <div className="font-mono text-[11px] text-muted-foreground">
            {shortEvalIdentity(report.revision)}
          </div>
        </div>
      )}
      {error && (
        <div className="text-xs text-destructive" role="alert">
          {error}
        </div>
      )}
    </div>
  )
}

function randomCalibrationRunId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  const token = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
  return `calibration-${token}`
}

function Field({
  label,
  id,
  wide = false,
  children,
}: {
  label: string
  id: string
  wide?: boolean
  children: React.ReactNode
}) {
  return (
    <label htmlFor={id} className={wide ? "sm:col-span-2" : undefined}>
      <span className={FIELD_LABEL}>{label}</span>
      {children}
    </label>
  )
}
