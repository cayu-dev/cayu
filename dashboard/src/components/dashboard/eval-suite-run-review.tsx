import { CheckCircle2, FlaskConical, LoaderCircle } from "lucide-react"
import { type ReactNode, useId } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import type { EvalAuthoredSuiteRunPreview } from "@/lib/api"
import { shortEvalIdentity } from "@/lib/evals-dashboard"
import type { EvalSuiteRunExposureV1, EvalTargetCatalogEntry } from "@/lib/generated/server-api"

export function ReadinessSummary({
  title,
  ready,
  diagnostics,
  identity,
}: {
  title: string
  ready: boolean
  diagnostics: Array<{ case_id?: string | null; message: string }>
  identity: string
}) {
  return (
    <div
      className={`mt-4 rounded-lg border p-3 text-sm ${
        ready ? "border-emerald-500/30 bg-emerald-500/5" : "border-amber-500/30 bg-amber-500/5"
      }`}
      data-testid="authored-suite-readiness"
    >
      <div className="font-medium">{title}</div>
      {diagnostics.map((diagnostic, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: diagnostics are immutable preview rows.
        <div key={`${diagnostic.case_id ?? "suite"}:${index}`} className="mt-1 text-xs">
          {diagnostic.case_id ? `${diagnostic.case_id}: ` : ""}
          {diagnostic.message}
        </div>
      ))}
      <div className="mt-2 font-mono text-[11px] text-muted-foreground">
        {shortEvalIdentity(identity)}
      </div>
    </div>
  )
}

export function RunExposureSummary({ exposure }: { exposure: EvalSuiteRunExposureV1 }) {
  return (
    <div
      className="mt-3 grid gap-2 rounded-lg border border-border bg-muted/20 p-3 text-xs @min-[28rem]/eval-form:grid-cols-2"
      data-testid="authored-suite-exposure"
    >
      <div>
        <span className="text-muted-foreground">Candidate work</span>
        <div className="font-medium">
          {exposure.candidate_trials.toLocaleString()} trials ·{" "}
          {exposure.maximum_candidate_model_steps == null
            ? "no model-step cap"
            : `up to ${exposure.maximum_candidate_model_steps.toLocaleString()} model steps`}{" "}
          ·{" "}
          {exposure.maximum_candidate_total_tokens == null
            ? "token ceiling unavailable"
            : `${exposure.maximum_candidate_total_tokens.toLocaleString()} tokens`}
        </div>
      </div>
      <div>
        <span className="text-muted-foreground">Judge work</span>
        <div className="font-medium">
          {exposure.judge_evaluations.toLocaleString()} evaluations ·{" "}
          {exposure.maximum_judge_total_tokens == null
            ? "token ceilings unavailable"
            : `up to ${exposure.maximum_judge_input_tokens?.toLocaleString()} input / ${exposure.maximum_judge_output_tokens?.toLocaleString()} output / ${exposure.maximum_judge_total_tokens.toLocaleString()} total tokens`}
        </div>
      </div>
      <div>
        <span className="text-muted-foreground">Maximum concurrency</span>
        <div className="font-medium">{exposure.max_concurrency.toLocaleString()}</div>
      </div>
      <div>
        <span className="text-muted-foreground">Maximum priced cost</span>
        <div className="font-medium">
          Candidate {formatCandidateExposureCost(exposure)} · judge{" "}
          {formatExposureCost(exposure.judge_cost)}
        </div>
      </div>
      <div className="font-mono text-[11px] text-muted-foreground @min-[28rem]/eval-form:col-span-2">
        Accepted exposure {shortEvalIdentity(exposure.revision)}
      </div>
    </div>
  )
}

function formatCandidateExposureCost(exposure: EvalSuiteRunExposureV1): string {
  const thresholds = [
    ...new Set(
      exposure.execution_profiles.flatMap((profile) => {
        const budget = profile.candidate_cost_budget
        return budget ? [`${budget.amount} ${budget.currency}`] : []
      }),
    ),
  ]
  const observedCeiling = formatExposureCost(exposure.candidate_cost)
  if (thresholds.length === 0) return observedCeiling
  return `${thresholds.join(" + ")} per-trial interruption threshold · ${observedCeiling}`
}

function formatExposureCost(exposure: EvalSuiteRunExposureV1["candidate_cost"]): string {
  if (exposure.state === "not_applicable") return "not applicable"
  if (exposure.state === "unavailable") {
    return exposure.unavailable_reason?.replaceAll("_", " ") ?? "unavailable"
  }
  return (exposure.totals ?? []).map((item) => `${item.amount} ${item.currency}`).join(" + ")
}

export function EvalSuiteRunReview({
  preview,
  current,
  maximumCost,
  currency,
  budgetInvalid,
  disabled,
  canCheck,
  canLaunch,
  pending,
  target,
  onMaximumCostChange,
  onCurrencyChange,
  onPreview,
  onLaunch,
}: {
  preview: EvalAuthoredSuiteRunPreview | null
  current: boolean
  maximumCost: string
  currency: string
  budgetInvalid: boolean
  disabled: boolean
  canCheck: boolean
  canLaunch: boolean
  pending: string | null
  target: EvalTargetCatalogEntry | undefined
  onMaximumCostChange: (value: string) => void
  onCurrencyChange: (value: string) => void
  onPreview: () => void
  onLaunch: () => void
}) {
  const id = useId()
  return (
    <section className="mt-4 rounded-lg border border-border p-3" aria-label="Review run">
      <h2 className="text-base font-semibold">Review run</h2>
      <p className="text-sm text-muted-foreground">
        Choose cases in the list, then check the work and cost before starting a run.
      </p>
      <Card size="sm" className="mt-4" data-testid="authored-suite-cost-budget">
        <CardHeader>
          <div>
            <CardTitle>Candidate cost ceiling</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              Optional per-trial runtime interruption budget. It narrows this launch and is reviewed
              again before admission; it is not stored in the immutable suite.
            </p>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 @min-[28rem]/eval-form:grid-cols-2">
          <Field label="Maximum cost per trial" id={`${id}-maximum-cost`}>
            <Input
              id={`${id}-maximum-cost`}
              data-testid="authored-suite-maximum-cost"
              inputMode="decimal"
              maxLength={64}
              placeholder="No cost ceiling"
              value={maximumCost}
              disabled={disabled}
              onChange={(event) => onMaximumCostChange(event.target.value)}
            />
          </Field>
          <Field label="Cost currency" id={`${id}-cost-currency`}>
            <Input
              id={`${id}-cost-currency`}
              data-testid="authored-suite-cost-currency"
              maxLength={16}
              value={currency}
              disabled={disabled}
              onChange={(event) => onCurrencyChange(event.target.value.toUpperCase())}
            />
          </Field>
          <p
            className={`text-xs @min-[28rem]/eval-form:col-span-2 ${
              budgetInvalid ? "text-destructive" : "text-muted-foreground"
            }`}
          >
            {target?.cost_budget_available === true
              ? budgetInvalid
                ? "Use a positive canonical decimal and one of: " +
                  target.cost_budget_currencies.join(", ")
                : `Published currencies: ${target.cost_budget_currencies.join(", ")}.`
              : "This target has no compatible server-owned price book, so a monetary ceiling is unavailable."}
          </p>
        </CardContent>
      </Card>

      {preview && current && (
        <>
          <ReadinessSummary
            title={
              preview.ready
                ? `${preview.launches.length} durable ${preview.launches.length === 1 ? "run" : "runs"} ready`
                : "Launch needs attention"
            }
            ready={preview.ready}
            diagnostics={preview.diagnostics ?? []}
            identity={preview.selection.revision}
          />
          {preview.exposure && <RunExposureSummary exposure={preview.exposure} />}
        </>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button
          type="button"
          variant="outline"
          data-testid="authored-suite-run-preview"
          disabled={!canCheck}
          onClick={onPreview}
        >
          <CheckCircle2 /> Check launch
        </Button>
        <Button
          type="button"
          data-testid="authored-suite-launch"
          disabled={!canLaunch}
          onClick={onLaunch}
        >
          {pending === "launch" ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}
          {pending === "launch" ? "Starting..." : "Run selected"}
        </Button>
      </div>
    </section>
  )
}

function Field({ label, id, children }: { label: string; id: string; children: ReactNode }) {
  return (
    <label htmlFor={id} className="min-w-0">
      <span className="mb-1 block text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  )
}
