import { useId } from "react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { ModelJudgeAssertionSpec } from "@/lib/generated/server-api"

/** An authored evaluator reference; only server preview can establish readiness. */
export function ModelJudgeEditor({
  assertion,
  disabled,
  onChange,
}: {
  assertion: ModelJudgeAssertionSpec
  disabled: boolean
  onChange: (assertion: ModelJudgeAssertionSpec) => void
}) {
  const id = useId()
  return (
    <fieldset disabled={disabled} className="space-y-3" data-testid="model-judge-editor">
      <p className="text-xs text-muted-foreground">
        The specialist must provide this evaluator. Check the suite to verify its configuration
        before saving or running it.
      </p>
      <div className="grid gap-3 @sm:grid-cols-2">
        <label className="space-y-1 text-sm" htmlFor={`${id}-key`}>
          <span>Evaluator key</span>
          <Input
            id={`${id}-key`}
            value={assertion.evaluator_key}
            onChange={(event) => onChange({ ...assertion, evaluator_key: event.target.value })}
          />
        </label>
        <label className="space-y-1 text-sm" htmlFor={`${id}-version`}>
          <span>Rubric version</span>
          <Input
            id={`${id}-version`}
            value={assertion.rubric_version}
            onChange={(event) => onChange({ ...assertion, rubric_version: event.target.value })}
          />
        </label>
      </div>
      <label className="block space-y-1 text-sm" htmlFor={`${id}-rubric`}>
        <span>Rubric</span>
        <Textarea
          id={`${id}-rubric`}
          className="min-h-28"
          value={assertion.rubric}
          onChange={(event) => onChange({ ...assertion, rubric: event.target.value })}
        />
      </label>
      <label className="block space-y-1 text-sm" htmlFor={`${id}-threshold`}>
        <span>Minimum passing score</span>
        <Input
          id={`${id}-threshold`}
          type="number"
          min={0}
          max={1}
          step="any"
          value={Number.isNaN(assertion.threshold) ? "" : (assertion.threshold ?? 0.5)}
          onChange={(event) => onChange({ ...assertion, threshold: event.target.valueAsNumber })}
        />
      </label>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={assertion.include_transcript ?? false}
          onChange={(event) => onChange({ ...assertion, include_transcript: event.target.checked })}
        />
        Include the transcript as judge evidence
      </label>
    </fieldset>
  )
}
