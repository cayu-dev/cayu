import { Plus, Trash2 } from "lucide-react"
import { useCallback, useId, useLayoutEffect, useState } from "react"
import { AssertionFields } from "@/components/dashboard/evaluation-promotion"
import { ModelJudgeEditor } from "@/components/dashboard/model-judge-editor"
import {
  type CalibrationState,
  StructuredJudgeEditor,
} from "@/components/dashboard/structured-judge-editor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  newStructuredJudgeAssertion,
  structuredAssertionFromReviewedSuite,
} from "@/lib/eval-judge-authoring"
import { authoredCheckSummary } from "@/lib/eval-suite-authoring"
import {
  createPromotionAssertion,
  PROMOTION_ASSERTION_LABELS,
  type PromotionAssertion,
} from "@/lib/evaluation-promotion"
import type {
  EvalCaseDraftV2,
  EvalTargetCatalogEntry,
  JudgeProfileIdentityV1,
} from "@/lib/generated/server-api"

type Check = EvalCaseDraftV2["assertions"][number]
type Kind = NonNullable<Check["kind"]>
const GROUPS: { label: string; kinds: Kind[] }[] = [
  {
    label: "Outcome",
    kinds: ["root_status", "child_status", "final_output_equals", "final_output_contains"],
  },
  {
    label: "Tools and process",
    kinds: [
      "tool_called",
      "tool_arguments_contain",
      "tool_result_contains",
      "tools_called_in_order",
      "process_event",
      "process_events_in_order",
    ],
  },
  { label: "Files", kinds: ["workspace_file", "artifact"] },
  { label: "Memory", kinds: ["memory_attribution"] },
  {
    label: "Limits",
    kinds: [
      "max_tool_calls",
      "max_model_steps",
      "usage_recorded",
      "max_total_tokens",
      "max_estimated_cost",
    ],
  },
  { label: "Judges", kinds: ["structured_model_judge", "model_judge"] },
]

/** Mount with a stable case-row key. Check rows own calibration state, never suite publication. */
export function EvalSuiteChecks({
  assertions,
  targetKey,
  target,
  defaultJudgeProfile,
  reviewedAssertions,
  defaultTask,
  disabled,
  calibrationDisabled,
  onCalibrationStateChange,
  onChange,
}: {
  assertions: Check[]
  targetKey: string
  target: EvalTargetCatalogEntry | undefined
  defaultJudgeProfile: JudgeProfileIdentityV1 | undefined
  reviewedAssertions: Parameters<typeof structuredAssertionFromReviewedSuite>[1]
  defaultTask: string
  disabled: boolean
  calibrationDisabled: boolean
  onCalibrationStateChange: (state: CalibrationState) => void
  onChange: (assertions: Check[]) => void
}) {
  const id = useId()
  const [calibrations, setCalibrations] = useState<Record<string, CalibrationState>>({})
  const reportCalibration = useCallback((key: string, state: CalibrationState) => {
    setCalibrations((current) =>
      current[key]?.dirty === state.dirty && current[key]?.locked === state.locked
        ? current
        : { ...current, [key]: state },
    )
  }, [])
  const dirty = Object.values(calibrations).some((state) => state.dirty)
  const locked = Object.values(calibrations).some((state) => state.locked)
  useLayoutEffect(
    () => onCalibrationStateChange({ dirty, locked }),
    [dirty, locked, onCalibrationStateChange],
  )
  const [rows, setRows] = useState(() => ({
    keys: assertions.map((_, index) => `check-${index}`),
    next: assertions.length,
  }))
  // Shared memory presets may append a check outside this list. Allocate once without
  // reusing an earlier removed row's calibration owner.
  const rowKeys = rows.keys.slice(0, assertions.length)
  let nextKey = rows.next
  while (rowKeys.length < assertions.length) rowKeys.push(`check-${nextKey++}`)
  if (rows.keys.length !== assertions.length) setRows({ keys: rowKeys, next: nextKey })
  const [kind, setKind] = useState<Kind>("final_output_contains")
  const policy = target?.execution_profile?.evidence_policy
  const unavailable = (candidate: Kind) =>
    (candidate === "structured_model_judge" && defaultJudgeProfile === undefined) ||
    (candidate === "tool_arguments_contain" && policy?.include_tool_arguments !== true) ||
    (candidate === "tool_result_contains" && policy?.include_tool_results !== true)
  const replace = (index: number, value: Check) =>
    onChange(assertions.map((check, i) => (i === index ? value : check)))
  const remove = (index: number) => {
    if (disabled || assertions.length <= 1) return
    setRows({ keys: rowKeys.filter((_, i) => i !== index), next: nextKey })
    onChange(assertions.filter((_, i) => i !== index))
  }
  const add = () => {
    if (disabled || assertions.length >= 64 || unavailable(kind)) return
    const ids = assertions.map((check) => check.id)
    let next: Check
    if (kind === "structured_model_judge") {
      if (!defaultJudgeProfile) return
      next = newStructuredJudgeAssertion(defaultJudgeProfile, ids)
    } else if (kind === "model_judge") {
      let suffix = 1
      while (ids.includes(`judge-${suffix}`)) suffix += 1
      next = {
        id: `judge-${suffix}`,
        kind,
        evaluator_key: "quality",
        rubric: "Explain the answer with evidence.",
        rubric_version: "v1",
        threshold: 0.5,
        include_transcript: false,
      }
    } else {
      next = createPromotionAssertion(kind, assertions)
    }
    setRows({ keys: [...rowKeys, `check-${nextKey}`], next: nextKey + 1 })
    onChange([...assertions, next])
  }
  return (
    <section className="@container space-y-3" aria-label="Case checks">
      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1 space-y-1 text-sm">
          <label htmlFor={`${id}-kind`}>Check type</label>
          <select
            id={`${id}-kind`}
            className="h-9 w-full rounded-md border border-input bg-background px-2"
            value={kind}
            disabled={disabled}
            onChange={(event) => setKind(event.target.value as Kind)}
          >
            {GROUPS.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.kinds.map((candidate) => (
                  <option key={candidate} value={candidate} disabled={unavailable(candidate)}>
                    {candidate === "structured_model_judge"
                      ? "AI judge with weighted criteria"
                      : candidate === "model_judge"
                        ? "Configured evaluator with a rubric"
                        : PROMOTION_ASSERTION_LABELS[candidate]}
                    {unavailable(candidate) ? " (unavailable)" : ""}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <Button
          type="button"
          variant="outline"
          disabled={disabled || assertions.length >= 64 || unavailable(kind)}
          onClick={add}
        >
          <Plus /> Add check
        </Button>
      </div>
      {assertions.map((assertion, index) => (
        <details
          key={rowKeys[index]}
          className="min-w-0 rounded-lg border border-border"
          data-testid="authored-check"
        >
          <summary className="cursor-pointer break-words px-3 py-3 text-sm font-medium">
            {authoredCheckSummary(assertion)}
          </summary>
          <div className="space-y-3 border-t border-border p-3">
            {assertion.kind === "structured_model_judge" ? (
              <StructuredJudgeEditor
                targetKey={targetKey}
                assertion={assertion}
                target={target}
                reviewedAssertion={structuredAssertionFromReviewedSuite(
                  assertion,
                  reviewedAssertions,
                )}
                defaultTask={defaultTask}
                disabled={disabled}
                calibrationDisabled={calibrationDisabled}
                onCalibrationStateChange={(state) => reportCalibration(rowKeys[index]!, state)}
                onChange={(next) => replace(index, next)}
                onRemove={() => remove(index)}
                canRemove={assertions.length > 1}
              />
            ) : (
              <fieldset disabled={disabled} className="@container/assertion space-y-3">
                {assertion.kind === "model_judge" ? (
                  <ModelJudgeEditor
                    assertion={assertion}
                    disabled={disabled}
                    onChange={(next) => replace(index, next)}
                  />
                ) : (
                  <AssertionFields
                    assertion={assertion as PromotionAssertion}
                    index={index}
                    evidencePolicy={policy}
                    update={(next) => replace(index, next)}
                  />
                )}
                <details className="text-sm">
                  <summary className="cursor-pointer text-muted-foreground">Check details</summary>
                  <div className="mt-2 grid gap-3 @sm:grid-cols-2">
                    <label className="space-y-1" htmlFor={`${id}-${rowKeys[index]}-id`}>
                      <span>Check ID</span>
                      <Input
                        id={`${id}-${rowKeys[index]}-id`}
                        value={assertion.id}
                        onChange={(event) =>
                          replace(index, { ...assertion, id: event.target.value })
                        }
                      />
                    </label>
                    <label className="space-y-1" htmlFor={`${id}-${rowKeys[index]}-description`}>
                      <span>Description</span>
                      <Input
                        id={`${id}-${rowKeys[index]}-description`}
                        value={assertion.description ?? ""}
                        onChange={(event) =>
                          replace(index, { ...assertion, description: event.target.value || null })
                        }
                      />
                    </label>
                  </div>
                </details>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={disabled || assertions.length <= 1}
                  onClick={() => remove(index)}
                >
                  <Trash2 /> Remove check
                </Button>
              </fieldset>
            )}
          </div>
        </details>
      ))}
    </section>
  )
}
