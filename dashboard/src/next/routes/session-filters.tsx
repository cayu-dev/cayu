import { Search, SlidersHorizontal } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  multilineSessionFilters,
  type SESSION_INDEX_DEBUG_STATES,
  type SESSION_INDEX_ORDERS,
  type SessionIndexFilterPatch,
  type SessionIndexSearch,
} from "@/lib/session-index-search"

type AdvancedFilterDraft = {
  agentName: string
  providerName: string
  model: string
  environmentName: string
  parentSessionId: string
  causalBudgetId: string
  labels: string
  labelSelectors: string
}

export const selectClassName =
  "h-8 min-w-32 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

const advancedInputClassName = "mt-1.5"

export const ORDER_LABELS: Record<(typeof SESSION_INDEX_ORDERS)[number], string> = {
  updated_at_desc: "Updated newest",
  updated_at_asc: "Updated oldest",
  created_at_desc: "Created newest",
  created_at_asc: "Created oldest",
}

export const DEBUG_LABELS: Record<(typeof SESSION_INDEX_DEBUG_STATES)[number], string> = {
  needs_attention: "Needs attention",
  session_failure: "Session failures",
  tool_issue: "Tool issues",
  interruption: "Interruptions",
}

function optionalFilter(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function advancedFilterDraft(search: SessionIndexSearch): AdvancedFilterDraft {
  return {
    agentName: search.agent_name ?? "",
    providerName: search.provider_name ?? "",
    model: search.model ?? "",
    environmentName: search.environment_name ?? "",
    parentSessionId: search.parent_session_id ?? "",
    causalBudgetId: search.causal_budget_id ?? "",
    labels: search.label?.join("\n") ?? "",
    labelSelectors: search.label_selector?.join("\n") ?? "",
  }
}

function advancedFilterDraftKey(draft: AdvancedFilterDraft): string {
  return JSON.stringify(Object.values(draft).map((value) => value.trim()))
}

function advancedFilterPatch(draft: AdvancedFilterDraft): SessionIndexFilterPatch {
  return {
    agent_name: optionalFilter(draft.agentName),
    provider_name: optionalFilter(draft.providerName),
    model: optionalFilter(draft.model),
    environment_name: optionalFilter(draft.environmentName),
    parent_session_id: optionalFilter(draft.parentSessionId),
    causal_budget_id: optionalFilter(draft.causalBudgetId),
    label: multilineSessionFilters(draft.labels),
    label_selector: multilineSessionFilters(draft.labelSelectors),
  }
}

function activeAdvancedFilterCount(search: SessionIndexSearch): number {
  return (
    [
      search.agent_name,
      search.provider_name,
      search.model,
      search.environment_name,
      search.parent_session_id,
      search.causal_budget_id,
    ].filter((value) => value !== undefined).length +
    (search.label?.length ?? 0) +
    (search.label_selector?.length ?? 0)
  )
}

export function SessionSearchInput({
  value,
  resetVersion,
  onCommit,
}: {
  value: string
  resetVersion: number
  onCommit: (value: string | undefined) => void
}) {
  const [input, setInput] = useState({ committed: value, draft: value, resetVersion })
  const pendingCommit = useRef<string | null>(null)
  // Acknowledge our navigation without discarding characters typed while it settled.
  // Back/forward and clear instead replace the draft, without replacing the input node.
  if (input.committed !== value || input.resetVersion !== resetVersion) {
    setInput({
      committed: value,
      resetVersion,
      draft:
        input.resetVersion === resetVersion && pendingCommit.current === value
          ? input.draft
          : value,
    })
    pendingCommit.current = null
  }

  useEffect(() => {
    if (input.committed !== value || input.resetVersion !== resetVersion) return
    const normalized = optionalFilter(input.draft)
    if (normalized === optionalFilter(value)) return
    const timer = window.setTimeout(() => {
      pendingCommit.current = normalized ?? ""
      onCommit(normalized)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [input, onCommit, value, resetVersion])

  return (
    <div className="relative min-w-56 flex-[2_1_18rem]">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        value={input.draft}
        onChange={(event) =>
          setInput({ committed: value, draft: event.target.value, resetVersion })
        }
        placeholder="Search by title, ID or label"
        className="pl-8"
        aria-label="Search sessions"
      />
    </div>
  )
}

export function AdvancedFilters({
  search,
  onApply,
}: {
  search: SessionIndexSearch
  onApply: (patch: SessionIndexFilterPatch) => void
}) {
  const confirmedDraft = advancedFilterDraft(search)
  const [draft, setDraft] = useState(confirmedDraft)
  const [open, setOpen] = useState(activeAdvancedFilterCount(search) > 0)
  const confirmedKey = advancedFilterDraftKey(confirmedDraft)
  const dirty = advancedFilterDraftKey(draft) !== confirmedKey
  const activeCount = activeAdvancedFilterCount(search)

  function setField<Key extends keyof AdvancedFilterDraft>(
    field: Key,
    value: AdvancedFilterDraft[Key],
  ) {
    setDraft((current) => ({ ...current, [field]: value }))
  }

  return (
    <details
      className="min-w-full border-t border-border pt-3"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium text-foreground">
        <SlidersHorizontal className="h-4 w-4" />
        Advanced filters
        {activeCount > 0 && <Badge variant="secondary">{activeCount}</Badge>}
      </summary>
      <form
        className="mt-4 space-y-4"
        onSubmit={(event) => {
          event.preventDefault()
          onApply(advancedFilterPatch(draft))
        }}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <label htmlFor="session-agent-filter" className="text-sm text-muted-foreground">
            Specialist
            <Input
              id="session-agent-filter"
              value={draft.agentName}
              onChange={(event) => setField("agentName", event.target.value)}
              placeholder="specialist name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-environment-filter" className="text-sm text-muted-foreground">
            Environment
            <Input
              id="session-environment-filter"
              value={draft.environmentName}
              onChange={(event) => setField("environmentName", event.target.value)}
              placeholder="environment name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-provider-filter" className="text-sm text-muted-foreground">
            Provider
            <Input
              id="session-provider-filter"
              value={draft.providerName}
              onChange={(event) => setField("providerName", event.target.value)}
              placeholder="provider name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-model-filter" className="text-sm text-muted-foreground">
            Model
            <Input
              id="session-model-filter"
              value={draft.model}
              onChange={(event) => setField("model", event.target.value)}
              placeholder="model name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-parent-filter" className="text-sm text-muted-foreground">
            Parent session
            <Input
              id="session-parent-filter"
              value={draft.parentSessionId}
              onChange={(event) => setField("parentSessionId", event.target.value)}
              placeholder="parent session ID"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-budget-filter" className="text-sm text-muted-foreground">
            Causal budget
            <Input
              id="session-budget-filter"
              value={draft.causalBudgetId}
              onChange={(event) => setField("causalBudgetId", event.target.value)}
              placeholder="causal budget ID"
              className={advancedInputClassName}
            />
          </label>
        </div>
        <div className="grid gap-3 lg:grid-cols-2">
          <label htmlFor="session-label-filter" className="text-sm text-muted-foreground">
            Exact labels
            <Textarea
              id="session-label-filter"
              value={draft.labels}
              onChange={(event) => setField("labels", event.target.value)}
              placeholder={"tenant=acme\ntier=critical"}
              className={advancedInputClassName}
              rows={3}
            />
            <span className="mt-1 block text-xs">One key=value label per line.</span>
          </label>
          <label htmlFor="session-selector-filter" className="text-sm text-muted-foreground">
            Label selectors
            <Textarea
              id="session-selector-filter"
              value={draft.labelSelectors}
              onChange={(event) => setField("labelSelectors", event.target.value)}
              placeholder={"region in (us,eu)\n!archived"}
              className={advancedInputClassName}
              rows={3}
            />
            <span className="mt-1 block text-xs">One selector expression per line.</span>
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button type="submit" size="sm" disabled={!dirty}>
            Apply filters
          </Button>
          {dirty && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setDraft(confirmedDraft)}
            >
              Discard changes
            </Button>
          )}
        </div>
      </form>
    </details>
  )
}
