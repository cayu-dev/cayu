import { Tabs } from "@base-ui/react/tabs"
import { Copy, Trash2 } from "lucide-react"
import { type ReactNode, useId } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { EvalCaseDraftV2 } from "@/lib/generated/server-api"

export function EvalSuiteCase({
  evalCase,
  disabled,
  editsLocked,
  scenarioDirty,
  scenario,
  input,
  checks,
  onChange,
  onStimulusChange,
  onDuplicate,
  onRemove,
  canDuplicate,
  canRemove,
}: {
  evalCase: EvalCaseDraftV2
  disabled: boolean
  editsLocked: boolean
  scenarioDirty: boolean
  scenario: boolean
  input: ReactNode
  checks: ReactNode
  onChange: (evalCase: EvalCaseDraftV2) => void
  onStimulusChange: (kind: "simple_input" | "scenario") => void
  onDuplicate: () => void
  onRemove: () => void
  canDuplicate: boolean
  canRemove: boolean
}) {
  const id = useId()
  return (
    <fieldset
      disabled={disabled}
      className="@container space-y-4"
      data-testid="authored-case-editor"
    >
      <fieldset disabled={editsLocked} className="flex flex-wrap items-end gap-2">
        <label className="min-w-0 flex-1 space-y-1 text-sm" htmlFor={`${id}-name`}>
          <span>Case name</span>
          <Input
            id={`${id}-name`}
            data-testid="authored-case-name"
            value={evalCase.name}
            onChange={(event) => onChange({ ...evalCase, name: event.target.value })}
          />
        </label>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!canDuplicate}
          onClick={onDuplicate}
        >
          <Copy /> Duplicate
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={`Remove case ${evalCase.name}`}
          disabled={!canRemove}
          onClick={onRemove}
        >
          <Trash2 />
        </Button>
      </fieldset>
      <fieldset disabled={editsLocked}>
        <details className="rounded-lg border border-border p-3 text-sm">
          <summary className="cursor-pointer text-muted-foreground">Case details</summary>
          <div className="mt-3 grid gap-3 @sm:grid-cols-2">
            <label className="space-y-1" htmlFor={`${id}-identity`}>
              <span>Case ID</span>
              <Input
                id={`${id}-identity`}
                data-testid="authored-case-id"
                value={evalCase.id}
                disabled={scenarioDirty}
                onChange={(event) => onChange({ ...evalCase, id: event.target.value })}
              />
            </label>
            <label className="space-y-1" htmlFor={`${id}-description`}>
              <span>Description</span>
              <Textarea
                id={`${id}-description`}
                value={evalCase.description ?? ""}
                onChange={(event) =>
                  onChange({ ...evalCase, description: event.target.value || null })
                }
              />
            </label>
          </div>
        </details>
      </fieldset>
      <Tabs.Root defaultValue="input" className="min-w-0">
        <Tabs.List aria-label="Case editor" className="mb-4 flex gap-1 border-b border-border">
          <Tabs.Tab
            value="input"
            className="rounded-t-md px-3 py-2 text-sm data-[active]:border-b-2 data-[active]:border-primary"
          >
            Input{scenarioDirty ? " · unsaved" : ""}
          </Tabs.Tab>
          <Tabs.Tab
            value="checks"
            className="rounded-t-md px-3 py-2 text-sm data-[active]:border-b-2 data-[active]:border-primary"
          >
            Checks ({evalCase.assertions.length})
          </Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="input" keepMounted className="space-y-3">
          <fieldset disabled={editsLocked} className="space-y-3">
            <fieldset className="flex flex-wrap gap-2" aria-label="Input type">
              <Button
                type="button"
                size="sm"
                variant={!scenario ? "default" : "outline"}
                aria-pressed={!scenario}
                disabled={scenarioDirty}
                onClick={() => onStimulusChange("simple_input")}
              >
                Simple input
              </Button>
              <Button
                type="button"
                size="sm"
                variant={scenario ? "default" : "outline"}
                aria-pressed={scenario}
                disabled={scenarioDirty}
                onClick={() => onStimulusChange("scenario")}
              >
                Multi-stage scenario
              </Button>
            </fieldset>
            {input}
          </fieldset>
        </Tabs.Panel>
        <Tabs.Panel value="checks" keepMounted className="space-y-3">
          {checks}
        </Tabs.Panel>
      </Tabs.Root>
    </fieldset>
  )
}
