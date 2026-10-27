import { Link } from "@tanstack/react-router"
import { Download, FileJson, LoaderCircle, RotateCcw } from "lucide-react"
import { useState } from "react"
import { MemoryAssertionDetails, MemoryEvidencePanel } from "@/components/dashboard/evals-memory"
import { DataCard, PayloadViewer, StateMessage } from "@/components/dashboard/layout"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { EvalResult, EvalResultDetail, EvalRun, EvalTarget } from "@/lib/api"
import { evalRunIsActive, evalTrialCostSummary, shortEvalIdentity } from "@/lib/evals-dashboard"

import { PROCESS_EVENT_OPTIONS } from "@/lib/evaluation-promotion"
import { formatBytes, formatCount, formatDateTime } from "@/lib/format"
import type {
  EvalAssertionPresentationV1,
  EvalCaseReliabilityV1,
  EvalMaximumCostExposureV1,
  EvalResultOutcomeDimensionsV1,
  EvalResultPresentationV2,
  EvalStructuredJudgePresentationV1,
  EvalSuiteRunExposureV1,
  PublishedModelJudgeDetail,
  PublishedProcessEventsInOrderDetail,
} from "@/lib/generated/server-api"

export function EvalExecutionProfileSummary({
  target,
  refreshing,
  refresh,
}: {
  target: EvalTarget
  refreshing: boolean
  refresh: () => void
}) {
  const profile = target.execution_profile
  return (
    <DataCard
      title="Current execution profile"
      description="Server-published candidate identity and authority that fresh Evals will freeze at launch."
      actions={
        <Button type="button" size="sm" variant="outline" disabled={refreshing} onClick={refresh}>
          <RotateCcw className={refreshing ? "animate-spin" : undefined} />
          Refresh profile
        </Button>
      }
      contentClassName="p-4"
    >
      {profile ? (
        <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <ProfileFact label="Candidate" value={profile.candidate.agent_name} />
          <ProfileFact
            label="Provider / model"
            value={`${profile.candidate.provider_name} / ${profile.candidate.model}`}
          />
          <ProfileFact
            label="Environment"
            value={profile.candidate.environment_name ?? "Application default"}
          />
          <ProfileFact
            label="Fixture / reset"
            value={`${executionProfileLabel(profile.fixture_strategy)} / ${executionProfileLabel(profile.reset_strategy)}`}
          />
          <ProfileFact label="Effects" value={executionProfileLabel(profile.effect_posture)} />
          <ProfileFact
            label="Maximum scale"
            value={`${profile.ceilings.max_cases} case${profile.ceilings.max_cases === 1 ? "" : "s"}, ${profile.ceilings.max_trials} trial${profile.ceilings.max_trials === 1 ? "" : "s"}, concurrency ${profile.ceilings.max_concurrency}`}
          />
          <ProfileFact
            label="Execution ceilings"
            value={`${profile.ceilings.max_timeout_seconds}s timeout · ${profile.ceilings.max_steps == null ? "no step cap" : `${profile.ceilings.max_steps} steps`} · ${profile.ceilings.max_compiled_input_chars.toLocaleString()} compiled chars`}
          />
          <ProfileFact
            label="Evidence"
            value={`Public runtime projection · output up to ${(profile.evidence_policy.max_final_output_chars ?? 65_536).toLocaleString()} chars`}
          />
          <ProfileFact
            label="Runtime identity"
            value={profile.candidate.runtime_execution_profile_fingerprint.slice(0, 12)}
            mono
          />
          <ProfileFact
            label="Target material"
            value={`${profile.target_material.fingerprint.slice(0, 12)} · ${profile.target_material.kind === "structural_sha256" ? "structural" : "process-local"}`}
            mono
          />
          <ProfileFact label="Profile revision" value={shortEvalIdentity(profile.revision)} mono />
        </div>
      ) : (
        <div className="text-sm text-amber-700 dark:text-amber-300">
          {target.execution_profile_diagnostics?.[0]?.message ??
            "The current runtime execution profile is unavailable."}
        </div>
      )}
    </DataCard>
  )
}

function ProfileFact({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="min-w-0 rounded-md border border-border bg-muted/20 p-3">
      <div className="text-muted-foreground">{label}</div>
      <div className={`mt-1 truncate text-foreground ${mono ? "font-mono" : "font-medium"}`}>
        {value}
      </div>
    </div>
  )
}

function executionProfileLabel(value: string): string {
  return value.replaceAll("_", " ")
}

export function RunLifecycleCard({ run }: { run: EvalRun }) {
  const scenarioInvocation = run.spec.invocation?.scenario
  return (
    <DataCard
      title={
        <span className="flex items-center gap-2">
          Run {shortEvalIdentity(run.spec.run_id)} <EvalStatusBadge run={run} />
        </span>
      }
      description={run.spec.run_id}
    >
      <div className="grid gap-4 p-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
        <RunFact label="Suite" value={run.spec.suite_id} />
        <RunFact label="Corpus" value={shortEvalIdentity(run.spec.corpus_revision)} />
        <RunFact label="Concurrency" value={formatCount(run.spec.max_concurrency)} />
        <RunFact label="Created" value={formatDateTime(run.created_at)} />
        <RunFact label="Started" value={formatDateTime(run.started_at)} />
        <RunFact label="Finished" value={formatDateTime(run.finished_at)} />
        {run.spec.invocation?.recovery_policy && (
          <>
            <RunFact label="Ownership claims" value={String(run.attempt_count)} />
            <RunFact
              label="Execution attempt ceiling"
              value={String(run.spec.invocation.recovery_policy.max_execution_attempts)}
            />
          </>
        )}
        {run.spec.invocation?.retry_of && (
          <>
            <RunFact label="Retry of run" value={run.spec.invocation.retry_of.run_id} />
            <RunFact
              label="Original trial"
              value={`${run.spec.invocation.retry_of.case_id} · ${run.spec.invocation.retry_of.trial_number}`}
            />
            <RunFact label="Retry attempt" value={String(run.spec.invocation.retry_of.attempt)} />
            <RunFact
              label="Replay decision"
              value={run.spec.invocation.retry_of.replay_decision.replaceAll("_", " ")}
            />
          </>
        )}
        {scenarioInvocation && (
          <>
            <RunFact
              label="Scenario"
              value={shortEvalIdentity(scenarioInvocation.scenario_revision)}
            />
            <RunFact label="Trials" value={formatCount(scenarioInvocation.trials)} />
          </>
        )}
      </div>
      {run.cancel_requested_at && (
        <div className="border-t border-border px-4 py-3 text-sm text-amber-700 dark:text-amber-300">
          Cancellation requested {formatDateTime(run.cancel_requested_at)}.
        </div>
      )}
      {run.failure_code && (
        <div className="border-t border-border px-4 py-3 text-sm text-destructive" role="alert">
          Run failed safely: {run.failure_code.replaceAll("_", " ")}.
        </div>
      )}
      {run.result && (
        <div className="grid gap-4 border-t border-border p-4 text-sm sm:grid-cols-3">
          <RunFact label="Result" value={run.result.status} />
          <RunFact label="Score" value={formatScore(run.result.score)} />
          <RunFact label="Duration" value={formatDuration(run.result.duration_ms)} />
        </div>
      )}
      {evalRunIsActive(run) && (
        <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
          <LoaderCircle className="mr-1.5 inline h-3.5 w-3.5 animate-spin" aria-hidden="true" />
          Following durable status.
        </div>
      )}
    </DataCard>
  )
}

export function ResultInspector({
  result,
  pendingAction,
  download,
}: {
  result: EvalResult
  pendingAction: string | null
  download: (format: "json" | "html") => void
}) {
  const published = result.result.run
  const [selection, setSelection] = useState({
    revision: published.revision,
    caseIndex: 0,
    trialIndex: 0,
  })
  const currentSelection =
    selection.revision === published.revision
      ? selection
      : { revision: published.revision, caseIndex: 0, trialIndex: 0 }
  const caseIndex = Math.min(currentSelection.caseIndex, published.cases.length - 1)
  const selectedCase = published.cases[caseIndex]
  const trialIndex = selectedCase
    ? Math.min(currentSelection.trialIndex, selectedCase.trials.length - 1)
    : 0
  const selectedTrial = selectedCase?.trials[trialIndex]
  const presentedCase = result.presentation.cases[caseIndex]
  const presentedTrial = presentedCase?.trials[trialIndex]
  const selectedTrialCost = selectedTrial ? evalTrialCostSummary(selectedTrial.assertions) : null
  const evidenceLink = result.trial_evidence_links?.find(
    (link) =>
      link.case_id === selectedCase?.case_id &&
      link.trial_number === selectedTrial?.trial_number &&
      link.source_trial_revision === selectedTrial?.source_trial_revision,
  )

  return (
    <DataCard
      title={
        <span className="flex items-center gap-2">
          Published result <OutcomeBadge outcome={published.status} />
        </span>
      }
      description={`Release ${result.result.target.application_release_id} · result ${shortEvalIdentity(published.revision)}`}
      actions={
        <>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={pendingAction !== null}
            onClick={() => download("json")}
          >
            {pendingAction === "download-result-json" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <FileJson />
            )}
            JSON
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={pendingAction !== null}
            onClick={() => download("html")}
          >
            {pendingAction === "download-result-html" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <Download />
            )}
            HTML
          </Button>
        </>
      }
    >
      <div className="grid gap-4 p-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Score" value={formatScore(published.score)} />
        <RunFact label="Duration" value={formatDuration(published.duration_ms)} />
        <RunFact label="Cases" value={formatCount(published.cases.length)} />
        <RunFact
          label="App manifest"
          value={shortEvalIdentity(result.result.target.app_manifest.fingerprint)}
        />
      </div>
      <div className="border-t border-border p-4">
        <div className="mb-4 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <RunFact
            label="Corpus"
            value={shortEvalIdentity(result.presentation.corpus_revision)}
            title={result.presentation.corpus_revision}
          />
          <RunFact
            label="Suite revision"
            value={shortEvalIdentity(result.presentation.suite_revision)}
            title={result.presentation.suite_revision}
          />
          <RunFact
            label="Evidence policy"
            value={shortEvalIdentity(result.presentation.evidence_policy_revision)}
            title={result.presentation.evidence_policy_revision}
          />
          <RunFact
            label="Pricing profile"
            value={
              result.presentation.pricing_profile_fingerprint
                ? shortEvalIdentity(result.presentation.pricing_profile_fingerprint)
                : "not used"
            }
            title={result.presentation.pricing_profile_fingerprint ?? undefined}
          />
          <RunFact
            label="Trial policy"
            value={`${result.presentation.trial_policy.minimum_passed_trials}/${result.presentation.trial_policy.trial_count} passes · concurrency ${result.presentation.trial_policy.max_concurrency}`}
            title={result.presentation.trial_policy.revision}
          />
        </div>
        {result.presentation.accepted_exposure && (
          <AcceptedExposureSummary exposure={result.presentation.accepted_exposure} />
        )}
        <OutcomeDimensionsGrid dimensions={result.presentation.dimensions} />
      </div>

      {selectedCase && selectedTrial && presentedCase && presentedTrial && (
        <div className="space-y-4 border-t border-border p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="min-w-0 text-xs font-medium text-muted-foreground">
              Case
              <select
                className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
                value={caseIndex}
                onChange={(event) =>
                  setSelection({
                    revision: published.revision,
                    caseIndex: Number(event.target.value),
                    trialIndex: 0,
                  })
                }
              >
                {published.cases.map((evalCase, index) => (
                  <option key={evalCase.case_id} value={index}>
                    {evalCase.case_id} — {evalCase.status}
                  </option>
                ))}
              </select>
            </label>
            <label className="min-w-0 text-xs font-medium text-muted-foreground">
              Trial
              <select
                className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
                value={trialIndex}
                onChange={(event) =>
                  setSelection({
                    revision: published.revision,
                    caseIndex,
                    trialIndex: Number(event.target.value),
                  })
                }
              >
                {selectedCase.trials.map((trial, index) => (
                  <option key={trial.trial_number} value={index}>
                    Trial {trial.trial_number} — {trial.status}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid gap-4 rounded-lg border border-border bg-muted/20 p-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <RunFact label="Case outcome" value={selectedCase.status} />
            <RunFact label="Case score" value={formatScore(selectedCase.score)} />
            <ReliabilityFacts reliability={presentedCase.reliability} />
            <RunFact label="Trial outcome" value={selectedTrial.status} />
            <RunFact label="Trial score" value={formatScore(selectedTrial.score)} />
            <RunFact label="Trial duration" value={formatDuration(selectedTrial.duration_ms)} />
            <RunFact
              label="Evidence"
              value={selectedTrial.evidence_complete ? "complete" : "incomplete"}
            />
            <RunFact label="Diagnostic" value={selectedTrial.code.replaceAll("_", " ")} />
            {selectedTrial.execution_failure_category && (
              <RunFact
                label="Execution failure"
                value={selectedTrial.execution_failure_category.replaceAll("_", " ")}
              />
            )}
            <div>
              <div className="text-xs text-muted-foreground">Execution evidence</div>
              {evidenceLink ? (
                <Link
                  className="text-primary underline"
                  to="/sessions/$sessionId"
                  params={{ sessionId: evidenceLink.session_id }}
                >
                  Open trial session
                </Link>
              ) : (
                <span>Exact session link unavailable</span>
              )}
            </div>
            <RunFact
              label="Usage"
              value={
                selectedTrial.usage && selectedTrial.usage_evidence_state !== "unavailable"
                  ? `${formatCount(selectedTrial.usage.total_tokens)} tokens · ${formatCount(selectedTrial.usage.model_steps)} model steps · ${formatCount(selectedTrial.usage.tool_calls)} tools${selectedTrial.usage_evidence_state === "partial" ? " · partial observed usage" : ""}`
                  : "unavailable"
              }
            />
            <RunFact
              label="Estimated cost"
              value={selectedTrialCost?.display ?? "not evaluated"}
              title={selectedTrialCost?.exact}
            />
          </div>
          <OutcomeDimensionsGrid dimensions={presentedTrial.dimensions} />
          <MemoryEvidencePanel
            evidence={selectedTrial.memory_attribution}
            assertions={presentedTrial.assertions}
          />

          <div className="text-sm">
            <div className="text-xs font-medium text-muted-foreground">Trial diagnostic</div>
            <div className="mt-1">{selectedTrial.message}</div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="text-sm font-medium">Output preview</div>
              <Badge variant="outline">{selectedTrial.output.evidence_state}</Badge>
            </div>
            {selectedTrial.output.evidence_state !== "unavailable" &&
            selectedTrial.output.text !== undefined ? (
              <PayloadViewer value={selectedTrial.output.text} maxHeight="max-h-72" />
            ) : (
              <StateMessage className="rounded-md border border-border py-6">
                No output preview is available for this trial.
              </StateMessage>
            )}
            {selectedTrial.output.preview_truncated && (
              <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
                The preview is truncated. Its retained SHA-256 fingerprint is{" "}
                {selectedTrial.output.retained_sha256 ?? "unavailable"}.
              </div>
            )}
          </div>

          <div>
            <div className="mb-2 text-sm font-medium">
              Assertions ({formatCount(selectedTrial.assertions.length)})
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Assertion</TableHead>
                  <TableHead>Outcome</TableHead>
                  <TableHead>Observation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {selectedTrial.assertions.map((assertion, assertionIndex) => {
                  const presentedAssertion = presentedTrial.assertions[assertionIndex]
                  return (
                    <TableRow key={assertion.assertion_id}>
                      <TableCell>
                        <div
                          className="max-w-52 truncate font-medium"
                          title={assertion.assertion_id}
                        >
                          {assertion.assertion_id}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {assertion.detail.kind?.replaceAll("_", " ") ?? "assertion"}
                        </div>
                      </TableCell>
                      <TableCell>
                        <OutcomeBadge outcome={assertion.outcome} />
                        <div className="mt-1 text-xs text-muted-foreground">
                          {formatScore(assertion.score)}
                        </div>
                      </TableCell>
                      <TableCell className="min-w-64 whitespace-normal">
                        <div>{assertion.message}</div>
                        {presentedAssertion?.model_judge ? (
                          <ModelJudgeDetails
                            className="mt-3"
                            detail={presentedAssertion.model_judge}
                          />
                        ) : presentedAssertion?.structured_judge ? (
                          <StructuredJudgeDetails
                            className="mt-3"
                            judgment={presentedAssertion.structured_judge}
                          />
                        ) : presentedAssertion?.tool_json ? (
                          <ToolJsonDetails className="mt-3" detail={presentedAssertion.tool_json} />
                        ) : presentedAssertion?.process ? (
                          <ProcessAssertionDetails
                            className="mt-3"
                            detail={presentedAssertion.process}
                          />
                        ) : presentedAssertion?.structure ? (
                          <StructureAssertionDetails
                            className="mt-3"
                            detail={presentedAssertion.structure}
                          />
                        ) : presentedAssertion?.memory ? (
                          <MemoryAssertionDetails
                            className="mt-3"
                            detail={presentedAssertion.memory}
                          />
                        ) : (
                          <details className="mt-2 text-xs">
                            <summary className="cursor-pointer text-primary">
                              Inspect evidence
                            </summary>
                            <PayloadViewer
                              value={assertion.detail}
                              className="mt-2"
                              maxHeight="max-h-40"
                            />
                          </details>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </DataCard>
  )
}

export function ResultPresentationInspector({
  presentation,
  result,
}: {
  presentation: EvalResultPresentationV2
  result: EvalResultDetail["result"]
}) {
  const [selection, setSelection] = useState({
    revision: presentation.result_revision,
    caseIndex: 0,
    trialIndex: 0,
  })
  const current =
    selection.revision === presentation.result_revision
      ? selection
      : { revision: presentation.result_revision, caseIndex: 0, trialIndex: 0 }
  const caseIndex = Math.min(current.caseIndex, presentation.cases.length - 1)
  const selectedCase = presentation.cases[caseIndex]
  const trialIndex = selectedCase ? Math.min(current.trialIndex, selectedCase.trials.length - 1) : 0
  const selectedTrial = selectedCase?.trials[trialIndex]
  const selectedMemory =
    "score" in result
      ? result.score.memory_attribution
      : result.run.cases[caseIndex]?.trials[trialIndex]?.memory_attribution

  if (!selectedCase || !selectedTrial) {
    return <StateMessage>No retained case and trial presentation is available.</StateMessage>
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Release" value={presentation.application_release_id} />
        <RunFact
          label={presentation.origin === "fresh_execution" ? "Published run" : "Captured score"}
          value={shortEvalIdentity(presentation.evaluation_revision)}
          title={presentation.evaluation_revision}
        />
        <RunFact
          label="App manifest"
          value={shortEvalIdentity(presentation.app_manifest_fingerprint)}
          title={presentation.app_manifest_fingerprint}
        />
        <RunFact
          label="Evidence policy"
          value={shortEvalIdentity(presentation.evidence_policy_revision)}
          title={presentation.evidence_policy_revision}
        />
        <RunFact
          label="Pricing profile"
          value={
            presentation.pricing_profile_fingerprint
              ? shortEvalIdentity(presentation.pricing_profile_fingerprint)
              : "not used"
          }
          title={presentation.pricing_profile_fingerprint ?? undefined}
        />
        <RunFact
          label="Trial policy"
          value={`${presentation.trial_policy.minimum_passed_trials}/${presentation.trial_policy.trial_count} passes · concurrency ${presentation.trial_policy.max_concurrency}`}
          title={presentation.trial_policy.revision}
        />
      </div>
      {presentation.accepted_exposure && (
        <AcceptedExposureSummary exposure={presentation.accepted_exposure} />
      )}
      <OutcomeDimensionsGrid dimensions={presentation.dimensions} />
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="min-w-0 text-xs font-medium text-muted-foreground">
          Case
          <select
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
            value={caseIndex}
            onChange={(event) =>
              setSelection({
                revision: presentation.result_revision,
                caseIndex: Number(event.target.value),
                trialIndex: 0,
              })
            }
          >
            {presentation.cases.map((evalCase, index) => (
              <option key={evalCase.case_id} value={index}>
                {evalCase.case_id} — {evalCase.status}
              </option>
            ))}
          </select>
        </label>
        <label className="min-w-0 text-xs font-medium text-muted-foreground">
          Trial
          <select
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
            value={trialIndex}
            onChange={(event) =>
              setSelection({
                revision: presentation.result_revision,
                caseIndex,
                trialIndex: Number(event.target.value),
              })
            }
          >
            {selectedCase.trials.map((trial, index) => (
              <option key={trial.trial_number ?? `captured-${index}`} value={index}>
                {trial.trial_number ? `Trial ${trial.trial_number}` : "Captured evidence"} —{" "}
                {trial.status}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="grid gap-3 rounded-lg border border-border bg-muted/20 p-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Case outcome" value={selectedCase.status} />
        <RunFact label="Case score" value={formatScore(selectedCase.score)} />
        <ReliabilityFacts reliability={selectedCase.reliability} />
        <RunFact label="Trial outcome" value={selectedTrial.status} />
        <RunFact label="Trial score" value={formatScore(selectedTrial.score)} />
      </div>
      <OutcomeDimensionsGrid dimensions={selectedTrial.dimensions} />
      {selectedMemory && (
        <MemoryEvidencePanel evidence={selectedMemory} assertions={selectedTrial.assertions} />
      )}
      <PresentedAssertions assertions={selectedTrial.assertions} />
    </div>
  )
}

export function OutcomeDimensionsGrid({
  dimensions,
}: {
  dimensions: EvalResultOutcomeDimensionsV1
}) {
  const items = [
    ["Candidate", dimensions.candidate],
    ["Deterministic assertions", dimensions.deterministic_assertions],
    ["Semantic quality", dimensions.semantic_quality],
    ["Evaluator health", dimensions.evaluator_health],
    ["Runtime", dimensions.runtime],
    ["Evidence", dimensions.evidence],
  ] as const
  return (
    <div className="grid gap-3 rounded-lg border border-border bg-muted/20 p-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
      {items.map(([label, value]) => (
        <RunFact key={label} label={label} value={value.replaceAll("_", " ")} />
      ))}
    </div>
  )
}

export function PresentedAssertions({
  assertions,
}: {
  assertions: Array<EvalAssertionPresentationV1>
}) {
  return (
    <div className="space-y-3">
      <div className="text-sm font-medium">Assertions ({formatCount(assertions.length)})</div>
      {assertions.map((assertion) => (
        <div key={assertion.assertion_id} className="rounded-lg border border-border p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="font-medium">{assertion.assertion_id}</div>
              <div className="text-xs text-muted-foreground">
                {assertion.kind.replaceAll("_", " ")} · {assertion.category}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <OutcomeBadge outcome={assertion.outcome} />
              <span className="text-xs text-muted-foreground">{formatScore(assertion.score)}</span>
            </div>
          </div>
          {assertion.model_judge && (
            <ModelJudgeDetails className="mt-3" detail={assertion.model_judge} />
          )}
          {assertion.structured_judge && (
            <StructuredJudgeDetails className="mt-3" judgment={assertion.structured_judge} />
          )}
          {assertion.tool_json && <ToolJsonDetails className="mt-3" detail={assertion.tool_json} />}
          {assertion.process && (
            <ProcessAssertionDetails className="mt-3" detail={assertion.process} />
          )}
          {assertion.structure && (
            <StructureAssertionDetails className="mt-3" detail={assertion.structure} />
          )}
          {assertion.memory && (
            <MemoryAssertionDetails className="mt-3" detail={assertion.memory} />
          )}
        </div>
      ))}
    </div>
  )
}

function ToolJsonDetails({
  detail,
  className,
}: {
  detail: NonNullable<EvalAssertionPresentationV1["tool_json"]>
  className?: string
}) {
  return (
    <div
      className={`space-y-3 rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-tool-json-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Tool" value={detail.tool_name} />
        <RunFact label="Occurrence" value={String(detail.occurrence)} />
        <RunFact label="Evidence" value={detail.observation_state.replaceAll("_", " ")} />
        <RunFact
          label="Match"
          value={detail.matched == null ? "unavailable" : detail.matched ? "matched" : "mismatch"}
        />
        <RunFact
          label="Invocation"
          value={
            detail.invocation_index == null || detail.invocation_revision == null
              ? "unavailable"
              : `#${detail.invocation_index} · ${shortEvalIdentity(detail.invocation_revision)}`
          }
          title={detail.invocation_revision ?? undefined}
        />
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Expected subset</div>
          <PayloadViewer value={detail.expected_subset} maxHeight="max-h-40" />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Observed safe value</div>
          {detail.actual == null ? (
            <StateMessage className="rounded-md border border-border py-4">
              No safe observed value is available.
            </StateMessage>
          ) : (
            <PayloadViewer value={detail.actual} maxHeight="max-h-40" />
          )}
        </div>
      </div>
    </div>
  )
}

const PROCESS_EVENT_LABELS = new Map(PROCESS_EVENT_OPTIONS)

function ProcessAssertionDetails({
  detail,
  className,
}: {
  detail: NonNullable<EvalAssertionPresentationV1["process"]>
  className?: string
}) {
  let facts: Array<[string, string]>
  if (detail.kind === "child_status") {
    facts = [
      ["Child status", detail.expected],
      ["Observed", detail.matching_count == null ? "unavailable" : String(detail.matching_count)],
      ["Required", formatCountRange(detail.min_count, detail.max_count)],
    ]
  } else if (detail.kind === "process_event") {
    facts = [
      ["Process event", PROCESS_EVENT_LABELS.get(detail.event) ?? detail.event],
      ["Observed", detail.matching_count == null ? "unavailable" : String(detail.matching_count)],
      ["Required", formatCountRange(detail.min_count, detail.max_count)],
    ]
  } else {
    const order = detail as PublishedProcessEventsInOrderDetail
    facts = [
      ["Expected selected events", String(order.expected.length)],
      [
        "Observed selected events",
        order.actual_count == null ? "unavailable" : String(order.actual_count),
      ],
      [
        "Exact order",
        order.matched == null ? "unavailable" : order.matched ? "matched" : "mismatch",
      ],
    ]
  }
  return (
    <div
      className={`rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-process-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-3">
        {facts.map(([label, value]) => (
          <RunFact key={label} label={label} value={value} />
        ))}
      </div>
      {detail.kind === "process_events_in_order" && (
        <div className="mt-3 space-y-2 text-xs text-muted-foreground">
          <p className="break-words font-mono" data-testid="eval-process-expected-order">
            {detail.expected.map((event) => PROCESS_EVENT_LABELS.get(event) ?? event).join(" → ")}
          </p>
          <p>
            Only the event kinds selected by the assertion participate; their complete filtered
            sequence and multiplicity must match exactly.
          </p>
        </div>
      )}
    </div>
  )
}

function formatCountRange(minimum: number, maximum: number | null | undefined): string {
  if (maximum === minimum) return String(minimum)
  if (maximum == null) return `at least ${minimum}`
  return `${minimum} to ${maximum}`
}

function StructureAssertionDetails({
  detail,
  className,
}: {
  detail: NonNullable<EvalAssertionPresentationV1["structure"]>
  className?: string
}) {
  const workspaceDetail = "path" in detail
  const facts: Array<[string, string]> = workspaceDetail
    ? [
        ["Path", detail.path],
        ["Evidence", detail.observation_state.replaceAll("_", " ")],
        ["Expected", detail.expected_present ? "present" : "absent"],
        [
          "Observed",
          detail.actual_present == null
            ? "unavailable"
            : detail.actual_present
              ? "present"
              : "absent",
        ],
        ["Required size", formatOptionalByteRange(detail.minimum_bytes, detail.maximum_bytes)],
        [
          "Observed size",
          detail.actual_size_bytes == null ? "unavailable" : formatBytes(detail.actual_size_bytes),
        ],
        [
          "Digest",
          !detail.digest_required
            ? "not required"
            : detail.digest_matched == null
              ? "unavailable"
              : detail.digest_matched
                ? "matched"
                : "mismatch",
        ],
      ]
    : [
        ["Scope", detail.scope],
        ["Evidence", detail.observation_state.replaceAll("_", " ")],
        ["Filename", detail.filename ?? "any"],
        ["Content type", detail.content_type ?? "any"],
        ["Required size", formatOptionalByteRange(detail.minimum_bytes, detail.maximum_bytes)],
        ["Required count", formatCountRange(detail.min_count, detail.max_count)],
        [
          "Matching artifacts",
          detail.matching_count == null ? "unavailable" : String(detail.matching_count),
        ],
        ["Digest", detail.digest_required ? "required" : "not required"],
        ["Public text", detail.text_required ? "required" : "not required"],
      ]
  return (
    <div
      className={`rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-structure-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        {facts.map(([label, value]) => (
          <RunFact key={label} label={label} value={value} />
        ))}
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        {workspaceDetail
          ? "Portable workspace evidence contains structure only; file bytes are never retained."
          : "Published artifact detail excludes artifact IDs, store identity, arbitrary metadata, and content."}
      </p>
    </div>
  )
}

function formatOptionalByteRange(
  minimum: number | null | undefined,
  maximum: number | null | undefined,
): string {
  if (minimum == null && maximum == null) return "any"
  if (minimum != null && maximum === minimum) return formatBytes(minimum)
  if (minimum == null) return `at most ${formatBytes(maximum ?? 0)}`
  if (maximum == null) return `at least ${formatBytes(minimum)}`
  return `${formatBytes(minimum)} to ${formatBytes(maximum)}`
}

function StructuredJudgeDetails({
  judgment,
  className,
}: {
  judgment: EvalStructuredJudgePresentationV1
  className?: string
}) {
  const detail = judgment.detail
  const profile = detail.judge_profile
  const reference = detail.reference
  const criteria = judgment.criteria ?? []
  const thresholdState =
    judgment.threshold_passed === null
      ? "unavailable"
      : judgment.threshold_passed
        ? "passed"
        : "failed"
  const usage = structuredJudgeUsageText(judgment)
  const cost = structuredJudgeCostText(judgment)
  return (
    <div className={`space-y-3 rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}>
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-3">
        <RunFact
          label="Judge profile"
          value={`${profile.label} · ${shortEvalIdentity(profile.revision)}`}
          title={`${profile.key}@${profile.revision}`}
        />
        <RunFact label="Provider / model" value={`${profile.provider_name} / ${profile.model}`} />
        <RunFact
          label="Candidate route"
          value={
            detail.candidate_route_relation === "same_model"
              ? "same model · explicitly allowed"
              : "independent model"
          }
        />
        <RunFact
          label="Rubric"
          value={`${detail.rubric_id} · ${shortEvalIdentity(detail.rubric_revision)}`}
          title={detail.rubric_revision}
        />
        <RunFact label="Evaluator" value={detail.diagnostic.replaceAll("_", " ")} />
        <RunFact
          label="Aggregate / threshold"
          value={`${detail.aggregate_score ?? "unavailable"} / ${detail.threshold} · ${thresholdState}`}
        />
        <RunFact label="Observed usage" value={usage} />
        <RunFact label="Observed cost" value={cost} />
        <RunFact
          label="Evidence"
          value={
            [
              detail.evidence.include_final_output ? "final output" : null,
              detail.evidence.include_transcript ? "transcript" : null,
            ]
              .filter(Boolean)
              .join(" + ") || "none"
          }
        />
        <RunFact
          label="Reference"
          value={reference ? `${reference.kind.replaceAll("_", " ")} · available` : "none"}
          title={reference ? `${reference.key}@${reference.revision}` : undefined}
        />
        <RunFact
          label="Evaluator implementation"
          value={shortEvalIdentity(profile.implementation_revision)}
          title={profile.implementation_revision}
        />
        <RunFact
          label="Evidence privacy policy"
          value={`${profile.privacy_policy_key} · ${shortEvalIdentity(profile.privacy_policy_revision)}`}
          title={profile.privacy_policy_revision}
        />
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Criterion</TableHead>
            <TableHead>Weight</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Contribution</TableHead>
            <TableHead>Explanation</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {criteria.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5}>No criterion scores were recorded.</TableCell>
            </TableRow>
          ) : (
            criteria.map((criterion) => (
              <TableRow key={criterion.criterion_id}>
                <TableCell className="font-medium">{criterion.criterion_id}</TableCell>
                <TableCell>{criterion.weight}</TableCell>
                <TableCell>{criterion.score}</TableCell>
                <TableCell>{criterion.weighted_contribution}</TableCell>
                <TableCell className="min-w-64 whitespace-normal">
                  <div>{criterion.explanation ?? "Unavailable"}</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {criterion.explanation_state}
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  )
}

function ModelJudgeDetails({
  detail,
  className,
}: {
  detail: PublishedModelJudgeDetail
  className?: string
}) {
  const profile = detail.judge_profile
  return (
    <div
      className={`rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-model-judge-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact
          label="Judge profile"
          value={`${profile.label} · ${shortEvalIdentity(profile.revision)}`}
          title={`${profile.key}@${profile.revision}`}
        />
        <RunFact label="Provider / model" value={`${profile.provider_name} / ${profile.model}`} />
        <RunFact
          label="Candidate route"
          value={
            detail.candidate_route_relation === "same_model"
              ? "same model · explicitly allowed"
              : "independent model"
          }
        />
        <RunFact label="Evaluator" value={detail.diagnostic.replaceAll("_", " ")} />
        <RunFact
          label="Rubric"
          value={`${detail.rubric_version} · threshold ${detail.threshold}`}
        />
        <RunFact label="Observed usage" value={modelJudgeUsageText(detail)} />
        <RunFact label="Observed cost" value={modelJudgeCostText(detail)} />
        <RunFact
          label="Evaluator implementation"
          value={shortEvalIdentity(detail.evaluator_implementation_revision)}
          title={detail.evaluator_implementation_revision}
        />
      </div>
    </div>
  )
}

export function structuredJudgeUsageText(judgment: EvalStructuredJudgePresentationV1): string {
  const usage = judgment.detail.usage
  return usage
    ? `${formatCount(usage.total_tokens)} tokens · ${formatCount(usage.model_steps)} model steps`
    : "unavailable"
}

function modelJudgeUsageText(detail: PublishedModelJudgeDetail): string {
  const usage = detail.usage
  return usage
    ? `${formatCount(usage.total_tokens)} tokens · ${formatCount(usage.model_steps)} model steps`
    : "unavailable"
}

function modelJudgeCostText(detail: PublishedModelJudgeDetail): string {
  const cost = detail.cost
  if (!cost) return "unavailable · not observed"
  return cost.availability === "priced"
    ? `${cost.estimated_cost} ${cost.currency}`
    : "unavailable · unpriced"
}

export function structuredJudgeCostText(judgment: EvalStructuredJudgePresentationV1): string {
  const cost = judgment.detail.cost
  if (!cost) return "unavailable · not observed"
  return cost.availability === "priced"
    ? `${cost.estimated_cost} ${cost.currency}`
    : "unavailable · unpriced"
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  const variant =
    outcome === "failed" || outcome === "error"
      ? "destructive"
      : outcome === "passed" || outcome === "completed"
        ? "secondary"
        : "outline"
  return <Badge variant={variant}>{outcome}</Badge>
}

function EvalStatusBadge({ run }: { run: EvalRun }) {
  const status = run.status
  const variant =
    status === "failed" ? "destructive" : status === "completed" ? "secondary" : "outline"
  return <Badge variant={variant}>{status}</Badge>
}

function RunFact({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 truncate" title={title ?? value}>
        {value}
      </div>
    </div>
  )
}

function ReliabilityFacts({ reliability }: { reliability: EvalCaseReliabilityV1 }) {
  return (
    <>
      <RunFact
        label="Reliability"
        value={`${reliability.passed_trials}/${reliability.total_trials} passed`}
      />
      <RunFact label="Variability" value={reliability.variability.replaceAll("_", " ")} />
      <RunFact
        label="Candidate failures"
        value={formatCount(reliability.candidate_failed_trials)}
      />
      <RunFact label="Runtime errors" value={formatCount(reliability.runtime_error_trials)} />
      <RunFact label="Evaluator errors" value={formatCount(reliability.evaluator_error_trials)} />
      <RunFact
        label="Unavailable / cancelled"
        value={`${formatCount(reliability.unavailable_trials)} / ${formatCount(reliability.cancelled_trials)}`}
      />
      <RunFact label="Score min / mean / max" value={formatReliabilityScores(reliability)} />
    </>
  )
}

function AcceptedExposureSummary({ exposure }: { exposure: EvalSuiteRunExposureV1 }) {
  return (
    <div className="mb-4 grid gap-3 rounded-lg border border-border bg-muted/20 p-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
      <RunFact
        label="Accepted candidate maximum"
        value={`${formatCount(exposure.candidate_trials)} trials · ${exposure.maximum_candidate_model_steps == null ? "no model-step cap" : `${formatCount(exposure.maximum_candidate_model_steps)} model steps`} · ${exposure.maximum_candidate_total_tokens == null ? "token ceiling unavailable" : `${formatCount(exposure.maximum_candidate_total_tokens)} tokens`}`}
      />
      <RunFact
        label="Accepted judge work"
        value={`${formatCount(exposure.judge_evaluations)} evaluations · ${exposure.maximum_judge_total_tokens == null ? "token ceilings unavailable" : `${formatCount(exposure.maximum_judge_input_tokens ?? 0)} input / ${formatCount(exposure.maximum_judge_output_tokens ?? 0)} output / ${formatCount(exposure.maximum_judge_total_tokens)} total tokens`}`}
      />
      <RunFact
        label="Maximum priced cost"
        value={`Candidate ${formatMaximumExposureCost(exposure.candidate_cost)} · judge ${formatMaximumExposureCost(exposure.judge_cost)}`}
      />
      <RunFact
        label="Accepted exposure"
        value={shortEvalIdentity(exposure.revision)}
        title={exposure.revision}
      />
    </div>
  )
}

function formatMaximumExposureCost(exposure: EvalMaximumCostExposureV1): string {
  if (exposure.state === "not_applicable") return "not applicable"
  if (exposure.state === "unavailable") {
    return exposure.unavailable_reason?.replaceAll("_", " ") ?? "unavailable"
  }
  return (exposure.totals ?? []).map((item) => `${item.amount} ${item.currency}`).join(" + ")
}

function formatReliabilityScores(reliability: EvalCaseReliabilityV1): string {
  if (reliability.scored_trials === 0) return "No comparable scores"
  return `${formatScore(reliability.minimum_score)} / ${formatScore(reliability.mean_score)} / ${formatScore(reliability.maximum_score)} · ${formatCount(reliability.scored_trials)} scored`
}

function formatScore(score: number | null | undefined): string {
  return score == null ? "No score" : `${(score * 100).toFixed(1)}%`
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) return `${formatCount(milliseconds)} ms`
  return `${(milliseconds / 1_000).toFixed(milliseconds < 10_000 ? 1 : 0)} s`
}

export function downloadBlob(blob: Blob, filename: string): void {
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
