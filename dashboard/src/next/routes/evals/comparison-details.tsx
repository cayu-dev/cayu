import { PayloadViewer } from "@/components/dashboard/layout"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { EvalResultComparison, EvalResultSummary } from "@/lib/api"
import { evalComparisonReasonText, shortEvalIdentity } from "@/lib/evals-dashboard"
import { formatCount, formatDateTime } from "@/lib/format"
import type {
  CorpusReliabilityDistributionV1,
  EvalStructuredJudgeComparisonV1,
  EvalToolJsonAssertionComparisonV1,
} from "@/lib/generated/server-api"
import { structuredJudgeCostText, structuredJudgeUsageText } from "./technical-details"

export function ComparisonSummary({ comparison }: { comparison: EvalResultComparison }) {
  const { comparison: resultComparison, baseline, current } = comparison
  const { compatibility } = resultComparison
  const regressions = resultComparison.regressions ?? []
  const caseComparisons = resultComparison.cases ?? []
  const structuredJudgments = resultComparison.structured_judgments ?? []
  const structuredRegressions = structuredJudgments.filter((item) => item.regressed)
  const toolJsonAssertions = resultComparison.tool_json_assertions ?? []
  const toolJsonRegressions = toolJsonAssertions.filter((item) => item.regressed)
  const regressionCount =
    regressions.length + structuredRegressions.length + toolJsonRegressions.length
  const reasons = compatibility.reasons ?? []
  return (
    <div className="space-y-4 p-4">
      <div
        className={
          compatibility.comparable
            ? "rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-300"
            : "rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-300"
        }
        role="status"
      >
        <div className="font-medium">
          {compatibility.comparable
            ? "These results are comparable."
            : "These results are not comparable as one regression contract."}
        </div>
        {!compatibility.comparable && reasons.length > 0 && (
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {reasons.map((reason) => (
              <li key={reason}>{evalComparisonReasonText(reason)}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <ComparisonResultSummary
          label="Baseline"
          record={baseline}
          result={resultComparison.baseline}
        />
        <ComparisonResultSummary
          label="Current"
          record={current}
          result={resultComparison.current}
        />
      </div>
      {compatibility.comparable && (
        <div
          className={
            regressionCount === 0
              ? "rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm"
              : "rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm"
          }
          data-testid="eval-comparison-regressions"
        >
          <div className="font-medium">
            {regressionCount === 0
              ? "No compatible-result regressions."
              : `${regressionCount} compatible-result regression${regressionCount === 1 ? "" : "s"}.`}
          </div>
          {regressions.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {regressions.map((regression) => (
                <li key={`${regression.scope}:${regression.case_id ?? "run"}:${regression.kind}`}>
                  {regression.scope === "case" ? `Case ${regression.case_id}: ` : "Run: "}
                  {regression.kind === "status"
                    ? `status ${regression.baseline_status} → ${regression.current_status}`
                    : regression.kind === "score"
                      ? `score ${formatScore(regression.baseline_score)} → ${formatScore(regression.current_score)}`
                      : `reliability ${formatComparisonReliability(regression.baseline_reliability)} → ${formatComparisonReliability(regression.current_reliability)}`}
                </li>
              ))}
            </ul>
          )}
          {structuredRegressions.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {structuredRegressions.map((item) => (
                <li key={`${item.case_id}:${item.trial_number ?? "captured"}:${item.assertion_id}`}>
                  Case {item.case_id},{" "}
                  {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"},
                  assertion {item.assertion_id}: evaluator {item.evaluator_change}, aggregate{" "}
                  {item.aggregate_change}
                  {item.aggregate_delta === null ? "" : ` (${item.aggregate_delta})`}.
                </li>
              ))}
            </ul>
          )}
          {toolJsonRegressions.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {toolJsonRegressions.map((item) => (
                <li
                  key={[item.case_id, item.trial_number ?? "captured", item.assertion_id].join(":")}
                >
                  Case {item.case_id},{" "}
                  {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"},
                  assertion {item.assertion_id}: outcome {item.baseline_outcome} →{" "}
                  {item.current_outcome}; evidence {item.baseline.observation_state} →{" "}
                  {item.current.observation_state}; observed value {item.observed_value_change}.
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {compatibility.comparable && caseComparisons.length > 0 && (
        <div className="space-y-2" data-testid="eval-comparison-reliability">
          <div className="font-medium">Case reliability</div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Case</TableHead>
                <TableHead>Baseline trials</TableHead>
                <TableHead>Current trials</TableHead>
                <TableHead>Change</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {caseComparisons.map((item) => (
                <TableRow key={item.case_id}>
                  <TableCell className="font-medium">{item.case_id}</TableCell>
                  <TableCell>{formatComparisonReliability(item.baseline_reliability)}</TableCell>
                  <TableCell>{formatComparisonReliability(item.current_reliability)}</TableCell>
                  <TableCell>{item.reliability_change}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
      <StructuredJudgeComparison comparison={resultComparison} />
      <ToolJsonComparison comparison={resultComparison} />
      <div className="grid gap-4 rounded-lg border border-border bg-muted/20 p-3 text-xs sm:grid-cols-2">
        <RunFact
          label="Baseline result revision"
          value={shortEvalIdentity(compatibility.baseline_result_revision)}
        />
        <RunFact
          label="Current result revision"
          value={shortEvalIdentity(compatibility.current_result_revision)}
        />
      </div>
    </div>
  )
}

function ToolJsonComparison({ comparison }: { comparison: EvalResultComparison["comparison"] }) {
  const state = comparison.tool_json_comparison_state
  const stateText = {
    compared: "Exact bounded tool-JSON observations were compared.",
    contract_incompatible:
      "Tool-JSON observations were not diffed because the immutable evaluation contracts differ.",
    no_tool_json_assertions: "Neither result contains tool-JSON assertions.",
    observation_identity_mismatch:
      "Tool-JSON observations were not paired because retained observation identities differ.",
    source_detail_unavailable:
      "One compact result projection does not retain tool-JSON observation detail.",
  }[state]
  const mismatches = comparison.tool_json_observation_mismatches ?? []
  const assertions = comparison.tool_json_assertions ?? []
  return (
    <div className="space-y-3" data-testid="eval-tool-json-comparison">
      <div>
        <div className="font-medium">Tool JSON assertion comparison</div>
        <div className="mt-1 text-xs text-muted-foreground">
          {stateText} State: {state.replaceAll("_", " ")}.
        </div>
      </div>
      {mismatches.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm">
          <div className="font-medium">Unmatched retained observations</div>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {mismatches.map((item) => (
              <li
                key={[
                  item.availability,
                  item.case_id,
                  item.trial_number ?? "captured",
                  item.assertion_id,
                ].join(":")}
              >
                {item.case_id} ·{" "}
                {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
                {item.assertion_id}: {item.availability.replaceAll("_", " ")}
              </li>
            ))}
          </ul>
        </div>
      )}
      {assertions.map((item) => (
        <ToolJsonComparisonDetails
          key={[item.case_id, item.trial_number ?? "captured", item.assertion_id].join(":")}
          item={item}
        />
      ))}
    </div>
  )
}

function ToolJsonComparisonDetails({ item }: { item: EvalToolJsonAssertionComparisonV1 }) {
  return (
    <div
      className={
        item.regressed
          ? "rounded-lg border border-destructive/30 bg-destructive/5 p-3"
          : "rounded-lg border border-border p-3"
      }
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium">
            {item.case_id} ·{" "}
            {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
            {item.assertion_id}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {(item.baseline.kind ?? "tool JSON").replaceAll("_", " ")} · {item.baseline.tool_name} ·
            occurrence {item.baseline.occurrence}
          </div>
        </div>
        <Badge variant={item.regressed ? "destructive" : "outline"}>
          {item.regressed ? "regressed" : item.observed_value_change}
        </Badge>
      </div>
      <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Outcome" value={`${item.baseline_outcome} → ${item.current_outcome}`} />
        <RunFact
          label="Evidence"
          value={`${item.baseline.observation_state} → ${item.current.observation_state}`}
        />
        <RunFact label="Observed value" value={item.observed_value_change} />
        <RunFact
          label="Evidence state changed"
          value={item.evidence_state_changed ? "yes" : "no"}
        />
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Expected subset</div>
          <PayloadViewer value={item.baseline.expected_subset} maxHeight="max-h-40" />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Baseline safe value</div>
          <PayloadViewer value={item.baseline.actual ?? "unavailable"} maxHeight="max-h-40" />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Current safe value</div>
          <PayloadViewer value={item.current.actual ?? "unavailable"} maxHeight="max-h-40" />
        </div>
      </div>
    </div>
  )
}

function StructuredJudgeComparison({
  comparison,
}: {
  comparison: EvalResultComparison["comparison"]
}) {
  const state = comparison.structured_judge_comparison_state
  const stateText = {
    compared: "Exact retained case, trial, and assertion identities were paired.",
    contract_incompatible:
      "Judged outcomes were not diffed because the immutable evaluation contracts differ.",
    no_structured_judges: "Neither result contains structured AI-judge observations.",
    observation_identity_mismatch:
      "Judged outcomes were not paired because retained observation identities differ.",
    source_detail_unavailable:
      "One compact result projection does not retain structured judgment detail.",
  }[state]
  const mismatches = comparison.structured_judge_observation_mismatches ?? []
  const judgments = comparison.structured_judgments ?? []
  return (
    <div className="space-y-3">
      <div>
        <div className="font-medium">Structured judge comparison</div>
        <div className="mt-1 text-xs text-muted-foreground">
          {stateText} State: {state.replaceAll("_", " ")}.
        </div>
      </div>
      {mismatches.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm">
          <div className="font-medium">Unmatched retained observations</div>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {mismatches.map((item) => (
              <li
                key={`${item.availability}:${item.case_id}:${item.trial_number ?? "captured"}:${item.assertion_id}`}
              >
                {item.case_id} ·{" "}
                {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
                {item.assertion_id}: {item.availability.replaceAll("_", " ")}
              </li>
            ))}
          </ul>
        </div>
      )}
      {judgments.map((item) => (
        <StructuredJudgeComparisonDetails
          key={`${item.case_id}:${item.trial_number ?? "captured"}:${item.assertion_id}`}
          item={item}
        />
      ))}
    </div>
  )
}

function StructuredJudgeComparisonDetails({ item }: { item: EvalStructuredJudgeComparisonV1 }) {
  const criteria = item.criteria ?? []
  const baselineCriteria = new Map(
    (item.baseline.criteria ?? []).map((criterion) => [criterion.criterion_id, criterion]),
  )
  const currentCriteria = new Map(
    (item.current.criteria ?? []).map((criterion) => [criterion.criterion_id, criterion]),
  )
  const profile = item.baseline.detail.judge_profile
  return (
    <div
      className={
        item.regressed
          ? "rounded-lg border border-destructive/30 bg-destructive/5 p-3"
          : "rounded-lg border border-border p-3"
      }
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium">
            {item.case_id} ·{" "}
            {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
            {item.assertion_id}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {profile.label} · {profile.provider_name}/{profile.model} ·{" "}
            {item.baseline.detail.candidate_route_relation.replaceAll("_", " ")}
          </div>
        </div>
        <Badge variant={item.regressed ? "destructive" : "outline"}>
          {item.regressed ? "regressed" : item.aggregate_change}
        </Badge>
      </div>
      <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Outcome" value={`${item.baseline_outcome} → ${item.current_outcome}`} />
        <RunFact label="Evaluator" value={item.evaluator_change} />
        <RunFact
          label="Aggregate"
          value={`${item.baseline.detail.aggregate_score ?? "unavailable"} → ${item.current.detail.aggregate_score ?? "unavailable"}`}
        />
        <RunFact
          label="Aggregate delta"
          value={`${item.aggregate_delta ?? "unavailable"} · ${item.aggregate_change}`}
        />
        <RunFact
          label="Rubric"
          value={`${item.baseline.detail.rubric_id} · ${shortEvalIdentity(item.baseline.detail.rubric_revision)}`}
          title={item.baseline.detail.rubric_revision}
        />
        <RunFact
          label="Judge profile"
          value={`${profile.key} · ${shortEvalIdentity(profile.revision)}`}
          title={profile.revision}
        />
        <RunFact
          label="Evaluator diagnostic"
          value={`${item.baseline.detail.diagnostic.replaceAll("_", " ")} → ${item.current.detail.diagnostic.replaceAll("_", " ")}`}
        />
        <RunFact
          label="Reference"
          value={
            item.baseline.detail.reference
              ? `${item.baseline.detail.reference.kind.replaceAll("_", " ")} · available`
              : "none"
          }
        />
        <RunFact
          label="Observed usage"
          value={`${structuredJudgeUsageText(item.baseline)} → ${structuredJudgeUsageText(item.current)}`}
        />
        <RunFact
          label="Observed cost"
          value={`${structuredJudgeCostText(item.baseline)} → ${structuredJudgeCostText(item.current)}`}
        />
      </div>
      <Table className="mt-3">
        <TableHeader>
          <TableRow>
            <TableHead>Criterion</TableHead>
            <TableHead>Weight</TableHead>
            <TableHead>Baseline</TableHead>
            <TableHead>Current</TableHead>
            <TableHead>Delta</TableHead>
            <TableHead>Explanations</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {criteria.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6}>
                Criterion deltas are unavailable because one or both judgments were not recorded.
              </TableCell>
            </TableRow>
          ) : (
            criteria.map((criterion) => {
              const baseline = baselineCriteria.get(criterion.criterion_id)
              const current = currentCriteria.get(criterion.criterion_id)
              return (
                <TableRow key={criterion.criterion_id}>
                  <TableCell className="font-medium">{criterion.criterion_id}</TableCell>
                  <TableCell>{criterion.weight}</TableCell>
                  <TableCell>{criterion.baseline_score}</TableCell>
                  <TableCell>{criterion.current_score}</TableCell>
                  <TableCell>{criterion.score_delta}</TableCell>
                  <TableCell className="min-w-72 whitespace-normal">
                    <div>
                      <span className="font-medium">Baseline:</span>{" "}
                      {baseline?.explanation ?? "Unavailable"}
                    </div>
                    <div className="mt-1">
                      <span className="font-medium">Current:</span>{" "}
                      {current?.explanation ?? "Unavailable"}
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {criterion.baseline_explanation_state} → {criterion.current_explanation_state}
                    </div>
                  </TableCell>
                </TableRow>
              )
            })
          )}
        </TableBody>
      </Table>
    </div>
  )
}

function ComparisonResultSummary({
  label,
  record,
  result,
}: {
  label: string
  record: EvalResultSummary
  result: EvalResultComparison["comparison"]["baseline"]
}) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="font-medium">{label}</div>
        <OutcomeBadge outcome={result.status} />
      </div>
      <div className="grid gap-3 text-sm sm:grid-cols-2">
        <RunFact label="Result" value={shortEvalIdentity(record.revision)} />
        <RunFact
          label="Origin"
          value={
            record.origin === "captured_session" ? "Captured session" : "Current-app execution"
          }
        />
        <RunFact label="Suite" value={record.suite_id} />
        <RunFact label="Target" value={result.target_key} />
        <RunFact label="Release" value={result.application_release_id} />
        <RunFact label="App manifest" value={shortEvalIdentity(result.app_manifest_fingerprint)} />
        <RunFact label="Score" value={formatScore(result.score)} />
        <RunFact
          label="Corpus"
          value={shortEvalIdentity(result.corpus_revision)}
          title={result.corpus_revision}
        />
        <RunFact
          label="Suite revision"
          value={shortEvalIdentity(result.suite_revision)}
          title={result.suite_revision}
        />
        <RunFact
          label="Evidence policy"
          value={shortEvalIdentity(result.evidence_policy_revision)}
          title={result.evidence_policy_revision}
        />
        <RunFact
          label="Pricing profile"
          value={
            result.pricing_profile_fingerprint
              ? shortEvalIdentity(result.pricing_profile_fingerprint)
              : "not used"
          }
          title={result.pricing_profile_fingerprint ?? undefined}
        />
        <RunFact
          label="Trial policy"
          value={shortEvalIdentity(result.trial_policy_revision)}
          title={result.trial_policy_revision}
        />
        <RunFact
          label="Accepted exposure"
          value={
            result.accepted_exposure_revision
              ? shortEvalIdentity(result.accepted_exposure_revision)
              : "not applicable"
          }
          title={result.accepted_exposure_revision ?? undefined}
        />
        <RunFact
          label="Exposure comparison contract"
          value={
            result.accepted_exposure_comparison_revision
              ? shortEvalIdentity(result.accepted_exposure_comparison_revision)
              : "not applicable"
          }
          title={result.accepted_exposure_comparison_revision ?? undefined}
        />
        <RunFact label="Created" value={formatDateTime(record.created_at)} />
      </div>
    </div>
  )
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

function formatComparisonReliability(
  reliability: CorpusReliabilityDistributionV1 | null | undefined,
): string {
  if (!reliability) return "Unavailable"
  return [
    `${formatCount(reliability.passed_trials)} passed`,
    `${formatCount(reliability.candidate_failed_trials)} candidate failed`,
    `${formatCount(reliability.runtime_error_trials)} runtime error`,
    `${formatCount(reliability.evaluator_error_trials)} evaluator error`,
    `${formatCount(reliability.unavailable_trials)} unavailable`,
    `${formatCount(reliability.cancelled_trials)} cancelled`,
  ].join(" · ")
}

function formatScore(score: number | null | undefined): string {
  return score == null ? "No score" : `${(score * 100).toFixed(1)}%`
}
