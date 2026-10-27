import { useId } from "react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { EvalSuiteDraftV3, EvalTargetCatalogEntry } from "@/lib/generated/server-api"

export function EvalSuiteSettings({
  draft,
  target,
  saved,
  edit,
}: {
  draft: EvalSuiteDraftV3
  target: EvalTargetCatalogEntry | undefined
  saved: boolean
  edit: (change: (draft: EvalSuiteDraftV3) => void) => void
}) {
  const id = useId()
  const request = draft.trial_request ?? {}
  const fields = [
    ["trials", "Trials per case", target?.max_trials ?? 100, request.trials ?? 1],
    [
      "minimum_passed_trials",
      "Required passes",
      request.trials ?? 1,
      request.minimum_passed_trials ?? 1,
    ],
    [
      "max_concurrency",
      "Maximum concurrency",
      target?.max_concurrency ?? 64,
      request.max_concurrency ?? 1,
    ],
    ["timeout_seconds", "Timeout per trial (seconds)", 3_600, request.timeout_seconds ?? 300],
  ] as const
  return (
    <div className="space-y-3">
      <label className="block space-y-1 text-sm" htmlFor={`${id}-name`}>
        <span>Suite name</span>
        <Input
          id={`${id}-name`}
          data-testid="authored-suite-name"
          value={draft.name}
          onChange={(event) =>
            edit((next) => {
              next.name = event.target.value
            })
          }
        />
      </label>
      <details className="rounded-lg border border-border p-3 text-sm">
        <summary className="cursor-pointer">
          Suite settings · {request.trials ?? 1} trials per case,{" "}
          {request.minimum_passed_trials ?? 1} must pass
        </summary>
        <div className="mt-3 grid gap-3 @min-[28rem]/eval-form:grid-cols-2">
          {fields.map(([key, label, maximum, value]) => (
            <label key={key} className="space-y-1" htmlFor={`${id}-${key}`}>
              <span>{label}</span>
              <Input
                id={`${id}-${key}`}
                type="number"
                min={1}
                max={maximum}
                value={Number.isNaN(value) ? "" : value}
                onChange={(event) =>
                  edit((next) => {
                    const number = event.target.valueAsNumber
                    next.trial_request = { ...next.trial_request, [key]: number }
                    if (key === "trials" && Number.isFinite(number)) {
                      next.trial_request.minimum_passed_trials = Math.min(
                        next.trial_request.minimum_passed_trials ?? 1,
                        number,
                      )
                    }
                  })
                }
              />
            </label>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          Every admitted trial runs. Execution and evidence errors fail the case even when the
          required pass count is met.
        </p>
      </details>
      <details className="rounded-lg border border-border p-3 text-sm">
        <summary className="cursor-pointer text-muted-foreground">Suite details</summary>
        <div className="mt-3 grid gap-3 @min-[28rem]/eval-form:grid-cols-2">
          <label className="space-y-1" htmlFor={`${id}-identity`}>
            <span>Suite ID</span>
            <Input
              id={`${id}-identity`}
              data-testid="authored-suite-id"
              disabled={saved}
              value={draft.id}
              onChange={(event) =>
                edit((next) => {
                  next.id = event.target.value
                })
              }
            />
            {saved && (
              <span className="block text-xs text-muted-foreground">
                Kept across saved revisions.
              </span>
            )}
          </label>
          <label className="space-y-1" htmlFor={`${id}-description`}>
            <span>Description</span>
            <Textarea
              id={`${id}-description`}
              value={draft.description ?? ""}
              onChange={(event) =>
                edit((next) => {
                  next.description = event.target.value || null
                })
              }
            />
          </label>
        </div>
      </details>
    </div>
  )
}
