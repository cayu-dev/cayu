import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router"
import {
  AlertTriangle,
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  CircleDollarSign,
  Clock,
  ExternalLink,
  GitBranch,
  LoaderCircle,
  RefreshCw,
  SlidersHorizontal,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { useDashboardCapability } from "../components/dashboard/server-contract"
import { Badge } from "../components/ui/badge"
import { Button, buttonVariants } from "../components/ui/button"
import { Input } from "../components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import { retryAggregateRequest } from "../lib/aggregate-query.ts"
import { fetchSessionTopology, fetchUsageRollup, type UsageRollup } from "../lib/api.ts"
import { dashboardConfig } from "../lib/config.ts"
import { dashboardCapabilityUnavailableText } from "../lib/dashboard-capabilities.ts"
import { formatCount, formatDateTime, formatDecimal } from "../lib/format.ts"
import type {
  AggregateAccuracy,
  ApiSessionTopologyNode,
  ApiTaskTopologyNode,
  UsageSessionCostSummary,
} from "../lib/generated/server-api"
import { dashboardPath } from "../lib/links.ts"
import { usageTimestampFromUtcInput, usageTimestampInputValue } from "../lib/usage-rollup.ts"
import {
  DEFAULT_USAGE_RANGE,
  USAGE_RANGE_OPTIONS,
  type UsageRange,
} from "../lib/usage-rollup-search.ts"
import { cn } from "../lib/utils.ts"
import {
  validateWorkflowSearch,
  WORKFLOW_SESSION_EXPANSION_LIMIT,
  WORKFLOW_STATUS_FILTERS,
  WORKFLOW_TASK_EXPANSION_LIMIT,
  type WorkflowNodeTypeFilter,
  type WorkflowSearch,
  type WorkflowStatusFilter,
  workflowSearchForUrl,
} from "../lib/workflow-search.ts"
import {
  buildWorkflowTopologyRequest,
  buildWorkflowUsageRequest,
  LatestWorkflowRequestCoordinator,
  mergeWorkflowTopologyResponse,
  type WorkflowContinuation,
  type WorkflowRequestTicket,
  type WorkflowTopologyState,
  workflowTopologyContainsActiveNodes,
  workflowTopologyContainsMixedSnapshots,
} from "../lib/workflow-topology.ts"
import {
  buildWorkflowRenderIndex,
  type WorkflowRenderIndex,
  WorkflowTerminalUsageReconciler,
  workflowCausalBudgetRelationship,
  workflowControlsKey,
  workflowFilterCount,
  workflowNodeKey,
  workflowNodeVisibility,
  workflowRefreshDelay,
  workflowSearchForNewFocus,
  workflowSearchWithExpansion,
  workflowSearchWithFocusCollapsed,
  workflowSearchWithoutFilters,
  workflowSessionChildrenForParent,
  workflowSessionSideChildrenForPathParent,
  workflowShouldAutoRefresh,
  workflowTaskCausalBudgetRelationship,
  workflowTaskChildrenForParent,
  workflowTasksForSession,
  workflowTopologyError,
  workflowTopologyShapeKey,
  workflowUnattachedExpandedSessions,
  workflowUnattachedExpandedTasks,
} from "../lib/workflow-view.ts"

const RANGE_LABELS: Record<UsageRange, string> = {
  "24h": "Last 24 hours",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "90d": "Last 90 days",
  "365d": "Last 365 days",
  custom: "Custom window",
}
const WORKFLOW_REFRESH_LABEL =
  "Active Workflow topology and available usage snapshots refresh independently while visible."
const WORKFLOW_RENDERED_COST_DETAIL_LIMIT = 20
const selectClassName =
  "h-9 min-w-36 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

type TopologyPendingRead = "initial" | "refresh" | WorkflowContinuation["kind"]

type WorkflowFilterDraft = {
  statuses: WorkflowStatusFilter[]
  nodeTypes: WorkflowNodeTypeFilter[]
  agentName: string
  environmentName: string
  range: UsageRange
  startAt: string
  endAt: string
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError"
}

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "completed") return "default"
  if (status === "running" || status === "claimed") return "secondary"
  if (status === "failed" || status === "interrupted" || status === "cancelled") {
    return "destructive"
  }
  return "outline"
}

function optionalDraftValue(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function workflowFilterDraft(search: WorkflowSearch): WorkflowFilterDraft {
  return {
    statuses: [...(search.status ?? [])],
    nodeTypes: [...(search.node_type ?? [])],
    agentName: search.agent_name ?? "",
    environmentName: search.environment_name ?? "",
    range: search.range ?? DEFAULT_USAGE_RANGE,
    startAt: usageTimestampInputValue(search.start_at),
    endAt: usageTimestampInputValue(search.end_at),
  }
}

function toggleListValue<T extends string>(values: readonly T[], value: T): T[] {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value]
}

function WorkflowControls({
  search,
  resolvedWindow,
  usageEnabled,
  onApply,
  onClearFilters,
}: {
  search: WorkflowSearch
  resolvedWindow: { startAt: string; endAt: string } | null
  usageEnabled: boolean
  onApply: (search: WorkflowSearch) => void
  onClearFilters: () => void
}) {
  const confirmed = workflowFilterDraft(search)
  const [draft, setDraft] = useState(confirmed)
  const [error, setError] = useState<string | null>(null)
  const filterCount = workflowFilterCount(search)
  const [filtersOpen, setFiltersOpen] = useState(() => filterCount > 0)

  function submit() {
    try {
      const next = validateWorkflowSearch({
        ...search,
        status: draft.statuses,
        node_type: draft.nodeTypes,
        agent_name: optionalDraftValue(draft.agentName),
        environment_name: optionalDraftValue(draft.environmentName),
        range: draft.range,
        start_at:
          draft.range === "custom"
            ? usageTimestampFromUtcInput(draft.startAt, "Custom start", {
                originalValue: search.start_at,
                originalInputValue: confirmed.startAt,
              })
            : undefined,
        end_at:
          draft.range === "custom"
            ? usageTimestampFromUtcInput(draft.endAt, "Custom end", {
                originalValue: search.end_at,
                originalInputValue: confirmed.endAt,
              })
            : undefined,
      })
      if (next.invalid) throw new Error("The Workflow filters exceed their safe URL bounds.")
      onApply(next)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The Workflow filters are invalid.")
    }
  }

  return (
    <DataCard>
      <form
        className="space-y-4 p-4"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <div className="flex flex-wrap items-end gap-3">
          {usageEnabled && (
            <>
              <label htmlFor="workflow-range" className="text-sm text-muted-foreground">
                Usage window
                <select
                  id="workflow-range"
                  className={cn(selectClassName, "mt-1.5 block")}
                  value={draft.range}
                  onChange={(event) => {
                    const range = event.target.value as UsageRange
                    setDraft((current) => {
                      if (
                        range !== "custom" ||
                        current.startAt ||
                        current.endAt ||
                        !resolvedWindow
                      ) {
                        return { ...current, range }
                      }
                      return {
                        ...current,
                        range,
                        startAt: usageTimestampInputValue(resolvedWindow.startAt),
                        endAt: usageTimestampInputValue(resolvedWindow.endAt),
                      }
                    })
                    setError(null)
                  }}
                >
                  {USAGE_RANGE_OPTIONS.map((range) => (
                    <option key={range} value={range}>
                      {RANGE_LABELS[range]}
                    </option>
                  ))}
                </select>
              </label>
              {draft.range === "custom" && (
                <>
                  <label htmlFor="workflow-start-at" className="text-sm text-muted-foreground">
                    Start (UTC)
                    <Input
                      id="workflow-start-at"
                      type="datetime-local"
                      step="0.001"
                      className="mt-1.5"
                      value={draft.startAt}
                      onChange={(event) => {
                        setDraft((current) => ({ ...current, startAt: event.target.value }))
                        setError(null)
                      }}
                    />
                  </label>
                  <label htmlFor="workflow-end-at" className="text-sm text-muted-foreground">
                    End (UTC)
                    <Input
                      id="workflow-end-at"
                      type="datetime-local"
                      step="0.001"
                      className="mt-1.5"
                      value={draft.endAt}
                      onChange={(event) => {
                        setDraft((current) => ({ ...current, endAt: event.target.value }))
                        setError(null)
                      }}
                    />
                  </label>
                </>
              )}
            </>
          )}
          <Button type="submit">Apply view</Button>
          {filterCount > 0 && (
            <Button type="button" variant="outline" onClick={onClearFilters}>
              Clear loaded-node filters
            </Button>
          )}
        </div>

        <details
          open={filtersOpen}
          onToggle={(event) => setFiltersOpen(event.currentTarget.open)}
          className="rounded-md border border-border"
        >
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
            <span className="inline-flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4" />
              Loaded-node filters{filterCount > 0 ? ` (${filterCount})` : ""}
            </span>
          </summary>
          <div className="space-y-4 border-t border-border p-3">
            <p className="text-xs text-muted-foreground">
              Filters apply only to nodes already loaded below. Context ancestors remain visible and
              are labelled; unloaded branches are not searched.
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <label htmlFor="workflow-agent" className="text-sm text-muted-foreground">
                Agent or assignee
                <Input
                  id="workflow-agent"
                  className="mt-1.5"
                  value={draft.agentName}
                  onChange={(event) => {
                    setDraft((current) => ({ ...current, agentName: event.target.value }))
                    setError(null)
                  }}
                  placeholder="exact loaded name"
                />
              </label>
              <label htmlFor="workflow-environment" className="text-sm text-muted-foreground">
                Session environment
                <Input
                  id="workflow-environment"
                  className="mt-1.5"
                  value={draft.environmentName}
                  onChange={(event) => {
                    setDraft((current) => ({ ...current, environmentName: event.target.value }))
                    setError(null)
                  }}
                  placeholder="exact loaded environment"
                />
              </label>
            </div>
            <fieldset>
              <legend className="text-sm font-medium">Node type</legend>
              <div className="mt-2 flex flex-wrap gap-4">
                {(["session", "task"] as const).map((kind) => (
                  <label key={kind} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={draft.nodeTypes.includes(kind)}
                      onChange={() =>
                        setDraft((current) => ({
                          ...current,
                          nodeTypes: toggleListValue(current.nodeTypes, kind),
                        }))
                      }
                    />
                    {kind}
                  </label>
                ))}
              </div>
            </fieldset>
            <fieldset>
              <legend className="text-sm font-medium">Lifecycle status</legend>
              <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                {WORKFLOW_STATUS_FILTERS.map((status) => (
                  <label key={status} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={draft.statuses.includes(status)}
                      onChange={() =>
                        setDraft((current) => ({
                          ...current,
                          statuses: toggleListValue(current.statuses, status),
                        }))
                      }
                    />
                    {status}
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
        </details>
        {error && (
          <p className="text-sm text-destructive" role="alert">
            {error}
          </p>
        )}
      </form>
    </DataCard>
  )
}

function ContextBadge({ direct }: { direct: boolean }) {
  return direct ? null : <Badge variant="outline">context for a loaded match</Badge>
}

function SessionBudgetRelationshipBadge({
  node,
  focus,
}: {
  node: ApiSessionTopologyNode
  focus: ApiSessionTopologyNode
}) {
  const relationship = workflowCausalBudgetRelationship(node, focus)
  const label =
    relationship === "focus"
      ? "focus causal budget"
      : relationship === "shared"
        ? "shares focus causal budget"
        : "different causal budget"
  return <Badge variant={relationship === "different" ? "outline" : "secondary"}>{label}</Badge>
}

function TaskBudgetRelationshipBadge({
  node,
  state,
  index,
}: {
  node: ApiTaskTopologyNode
  state: WorkflowTopologyState
  index: WorkflowRenderIndex
}) {
  const relationship = workflowTaskCausalBudgetRelationship(node, state, index)
  const label =
    relationship === "shared"
      ? "via shared causal budget"
      : relationship === "different"
        ? "via different causal budget"
        : relationship === "unlinked"
          ? "no linked session budget"
          : "linked session budget not loaded"
  return <Badge variant={relationship === "shared" ? "secondary" : "outline"}>{label}</Badge>
}

function SessionOperationalBadges({
  node,
  focus,
  direct,
  kindLabel,
}: {
  node: ApiSessionTopologyNode
  focus: ApiSessionTopologyNode
  direct: boolean
  kindLabel: string
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge variant="outline">{kindLabel}</Badge>
      <Badge variant={statusVariant(node.status)}>{node.status}</Badge>
      <SessionBudgetRelationshipBadge node={node} focus={focus} />
      <ContextBadge direct={direct} />
    </div>
  )
}

function SessionOperationalMetadata({ node }: { node: ApiSessionTopologyNode }) {
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="break-all">agent {node.agent_name}</span>
      <span className="break-all">environment {node.environment_name ?? "none"}</span>
      <span className="break-all">
        model {node.provider_name}/{node.model}
      </span>
      <span>created {formatDateTime(node.created_at)}</span>
      <span>updated {formatDateTime(node.updated_at)}</span>
      <span className="break-all font-mono">causal budget {node.causal_budget_id}</span>
    </div>
  )
}

function BranchSnapshotBadge({ mixed }: { mixed: boolean }) {
  return mixed ? <Badge variant="outline">mixed observations</Badge> : null
}

function BranchLoadMore({
  label,
  disabled,
  onClick,
}: {
  label: string
  disabled: boolean
  onClick: () => void
}) {
  return (
    <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={onClick}>
      {disabled ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
      {label}
    </Button>
  )
}

type WorkflowTreeBaseProps = {
  state: WorkflowTopologyState
  index: WorkflowRenderIndex
  search: WorkflowSearch
  visibleNodes: ReadonlySet<string>
  directMatches: ReadonlySet<string>
  pendingRead: TopologyPendingRead | null
  onTaskExpanded: (id: string, expanded: boolean) => void
  onLoadMore: (continuation: WorkflowContinuation) => void
}

type WorkflowSessionTreeProps = WorkflowTreeBaseProps & {
  onSessionExpanded: (id: string, expanded: boolean) => void
  onFocusCollapsed: (collapsed: boolean) => void
}

function TaskTreeNode({
  node,
  state,
  index,
  search,
  visibleNodes,
  directMatches,
  pendingRead,
  onTaskExpanded,
  onLoadMore,
  path,
  relationship,
}: WorkflowTreeBaseProps & {
  node: ApiTaskTopologyNode
  path: ReadonlySet<string>
  relationship:
    | "session_link"
    | "task_child"
    | "restored_session_link"
    | "restored_task_child"
    | "unattached_restored"
}) {
  const key = workflowNodeKey("task", node.id)
  if (!visibleNodes.has(key)) return null
  const expanded = search.expanded_task_id?.includes(node.id) ?? false
  const branch = state.taskChildBranches.get(node.id)
  const childPlacements = workflowTaskChildrenForParent(index, node.id)
  const visibleChildren = childPlacements.filter((placement) =>
    visibleNodes.has(workflowNodeKey("task", placement.node.id)),
  )
  const restoredChildCount = childPlacements.filter((placement) => placement.restored).length
  const nextPath = new Set(path)
  const cycle = nextPath.has(node.id)
  nextPath.add(node.id)
  return (
    <li className="min-w-0">
      <div className="rounded-md border border-border bg-background p-3">
        <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2">
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-expanded={expanded}
              aria-label={`${expanded ? "Collapse" : "Expand"} child tasks for ${node.id}`}
              disabled={
                !expanded && (search.expanded_task_id?.length ?? 0) >= WORKFLOW_TASK_EXPANSION_LIMIT
              }
              onClick={() => onTaskExpanded(node.id, !expanded)}
            >
              {expanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">
                  {relationship === "session_link"
                    ? "linked to session"
                    : relationship === "task_child"
                      ? "child task"
                      : relationship === "restored_session_link"
                        ? "restored session-linked task"
                        : relationship === "restored_task_child"
                          ? "restored child task"
                          : "unattached restored task"}
                </Badge>
                <Badge variant={statusVariant(node.status)}>{node.status}</Badge>
                <TaskBudgetRelationshipBadge node={node} state={state} index={index} />
                <ContextBadge direct={directMatches.has(key)} />
                {node.truncated_fields.length > 0 && (
                  <Badge variant="outline">bounded fields</Badge>
                )}
              </div>
              <div className="mt-1 truncate text-sm font-medium">
                {node.title ?? node.type ?? "Task"}
              </div>
              <div className="mt-1 break-all font-mono text-xs text-muted-foreground">
                {node.id}
              </div>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span>assignee {node.assigned_agent_name ?? "unassigned"}</span>
                <span>created {formatDateTime(node.created_at)}</span>
                <span>updated {formatDateTime(node.updated_at)}</span>
                {node.session_id && <span>session {node.session_id}</span>}
              </div>
              {node.status_reason && (
                <p className="mt-2 break-words text-sm text-destructive">{node.status_reason}</p>
              )}
            </div>
          </div>
          <a
            href={dashboardPath("/tasks", { q: node.id })}
            className={buttonVariants({ variant: "outline", size: "sm" })}
          >
            Task details <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
      {expanded && (
        <div className="ml-5 border-l border-border pl-4 pt-3">
          {cycle ? (
            <p className="text-sm text-destructive" role="alert">
              The loaded task branch contains a cycle and cannot be rendered.
            </p>
          ) : branch === undefined ? (
            <p className="text-sm text-muted-foreground" role="status">
              {pendingRead ? "Loading child tasks..." : "Child-task branch is unavailable."}
            </p>
          ) : (
            <>
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span>{branch.nodes.length} loaded child tasks</span>
                {restoredChildCount > 0 && (
                  <span>{restoredChildCount} restored outside the current child page</span>
                )}
                <BranchSnapshotBadge mixed={branch.mixedSnapshot} />
              </div>
              {visibleChildren && visibleChildren.length > 0 ? (
                <ul className="space-y-3" aria-label={`Child tasks for ${node.id}`}>
                  {visibleChildren.map((placement) => (
                    <TaskTreeNode
                      key={placement.node.id}
                      node={placement.node}
                      state={state}
                      index={index}
                      search={search}
                      visibleNodes={visibleNodes}
                      directMatches={directMatches}
                      pendingRead={pendingRead}
                      onTaskExpanded={onTaskExpanded}
                      onLoadMore={onLoadMore}
                      path={nextPath}
                      relationship={placement.restored ? "restored_task_child" : "task_child"}
                    />
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {branch.nodes.length > 0
                    ? "No loaded child tasks match the current filters."
                    : "No child tasks are attached."}
                </p>
              )}
              {branch.hasMore && branch.nextCursor && (
                <div className="mt-3">
                  <BranchLoadMore
                    label="Load more child tasks"
                    disabled={pendingRead !== null}
                    onClick={() =>
                      onLoadMore({
                        kind: "task_children",
                        scopeId: node.id,
                        cursor: branch.nextCursor!,
                      })
                    }
                  />
                </div>
              )}
            </>
          )}
        </div>
      )}
    </li>
  )
}

function LinkedTasks({
  sessionId,
  state,
  index,
  search,
  visibleNodes,
  directMatches,
  pendingRead,
  onTaskExpanded,
  onLoadMore,
}: WorkflowTreeBaseProps & { sessionId: string }) {
  if (state.taskStatus !== "available") return null
  const branch = state.linkedTaskBranches.get(sessionId)
  if (branch === undefined) return null
  const taskPlacements = workflowTasksForSession(index, sessionId)
  const tasks = taskPlacements.filter((placement) =>
    visibleNodes.has(workflowNodeKey("task", placement.node.id)),
  )
  const sessionRootTaskIds = new Set(taskPlacements.map((placement) => placement.node.id))
  const nestedTaskCount = branch.nodes.filter((node) => !sessionRootTaskIds.has(node.id)).length
  const restoredTaskCount = taskPlacements.filter((placement) => placement.restored).length
  return (
    <div className="mt-3 rounded-md border border-dashed border-border bg-muted/20 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>{branch.nodes.length} loaded linked tasks</span>
        {restoredTaskCount > 0 && (
          <span>{restoredTaskCount} restored outside the current linked-task page</span>
        )}
        {nestedTaskCount > 0 && <span>{nestedTaskCount} placed beneath expanded parent tasks</span>}
        <BranchSnapshotBadge mixed={branch.mixedSnapshot} />
      </div>
      {tasks.length > 0 ? (
        <ul className="space-y-3" aria-label={`Tasks linked to session ${sessionId}`}>
          {tasks.map((placement) => (
            <TaskTreeNode
              key={placement.node.id}
              node={placement.node}
              state={state}
              index={index}
              search={search}
              visibleNodes={visibleNodes}
              directMatches={directMatches}
              pendingRead={pendingRead}
              onTaskExpanded={onTaskExpanded}
              onLoadMore={onLoadMore}
              path={new Set()}
              relationship={placement.restored ? "restored_session_link" : "session_link"}
            />
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">
          {taskPlacements.length > 0
            ? "No loaded linked tasks match the current filters."
            : nestedTaskCount > 0
              ? "Loaded session-linked tasks are placed beneath their expanded parent tasks."
              : "No tasks are linked to this loaded session branch."}
        </p>
      )}
      {branch.hasMore && branch.nextCursor && (
        <div className="mt-3">
          <BranchLoadMore
            label="Load more linked tasks"
            disabled={pendingRead !== null}
            onClick={() =>
              onLoadMore({ kind: "task_session", scopeId: sessionId, cursor: branch.nextCursor! })
            }
          />
        </div>
      )}
    </div>
  )
}

function AncestorPath({
  state,
  index,
  search,
  visibleNodes,
  directMatches,
  pendingRead,
  onSessionExpanded,
  onTaskExpanded,
  onFocusCollapsed,
  onLoadMore,
}: WorkflowSessionTreeProps) {
  const ancestors = state.ancestors.filter((node) =>
    visibleNodes.has(workflowNodeKey("session", node.id)),
  )
  if (ancestors.length === 0) return null

  return (
    <div className="mb-4 rounded-md border border-border bg-muted/20 p-3">
      <div className="mb-2 text-xs font-medium uppercase text-muted-foreground">Ancestor path</div>
      <ol className="space-y-2" aria-label="Loaded ancestor path to the focus session">
        {ancestors.map((node) => {
          const key = workflowNodeKey("session", node.id)
          const expanded = search.expanded_session_id?.includes(node.id) ?? false
          const branch = state.sessionBranches.get(node.id)
          const sideChildPlacements = workflowSessionSideChildrenForPathParent(index, node.id)
          const visibleSideChildren = sideChildPlacements.filter((placement) =>
            visibleNodes.has(workflowNodeKey("session", placement.node.id)),
          )
          const pathChildCount =
            workflowSessionChildrenForParent(index, node.id).length - sideChildPlacements.length
          return (
            <li key={node.id}>
              <div className="rounded-md border border-border bg-background p-3">
                <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
                  <div className="flex min-w-0 items-start gap-2">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-expanded={expanded}
                      aria-label={`${expanded ? "Collapse" : "Expand"} session ${node.id}`}
                      disabled={
                        !expanded &&
                        (search.expanded_session_id?.length ?? 0) >=
                          WORKFLOW_SESSION_EXPANSION_LIMIT
                      }
                      onClick={() => onSessionExpanded(node.id, !expanded)}
                    >
                      {expanded ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </Button>
                    <div className="min-w-0">
                      <SessionOperationalBadges
                        node={node}
                        focus={state.focus}
                        direct={directMatches.has(key)}
                        kindLabel="ancestor session"
                      />
                      <div className="mt-1 break-all font-mono text-sm font-medium">{node.id}</div>
                      <SessionOperationalMetadata node={node} />
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Link
                      to="/sessions/$sessionId/workflow"
                      params={{ sessionId: node.id }}
                      search={workflowSearchForUrl(workflowSearchForNewFocus(search))}
                      className={buttonVariants({ variant: "outline", size: "sm" })}
                    >
                      Use as focus
                    </Link>
                    <Link
                      to="/sessions/$sessionId"
                      params={{ sessionId: node.id }}
                      search={{}}
                      className={buttonVariants({ variant: "outline", size: "sm" })}
                    >
                      Session details <ExternalLink className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                </div>
                {expanded && (
                  <div className="ml-5 border-l border-border pl-4 pt-3">
                    <LinkedTasks
                      sessionId={node.id}
                      state={state}
                      index={index}
                      search={search}
                      visibleNodes={visibleNodes}
                      directMatches={directMatches}
                      pendingRead={pendingRead}
                      onTaskExpanded={onTaskExpanded}
                      onLoadMore={onLoadMore}
                    />
                    {branch === undefined ? (
                      <p className="mt-3 text-sm text-muted-foreground" role="status">
                        {pendingRead
                          ? "Loading child sessions..."
                          : "Child-session branch is unavailable."}
                      </p>
                    ) : (
                      <div className="mt-3">
                        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                          <span>{branch.nodes.length} loaded child sessions</span>
                          {pathChildCount > 0 && <span>{pathChildCount} on the focus path</span>}
                          <BranchSnapshotBadge mixed={branch.mixedSnapshot} />
                        </div>
                        {visibleSideChildren.length > 0 ? (
                          <ul
                            className="space-y-3"
                            aria-label={`Off-path child sessions for ancestor ${node.id}`}
                          >
                            {visibleSideChildren.map((placement) => (
                              <SessionTreeNode
                                key={placement.node.id}
                                node={placement.node}
                                state={state}
                                index={index}
                                search={search}
                                visibleNodes={visibleNodes}
                                directMatches={directMatches}
                                pendingRead={pendingRead}
                                onSessionExpanded={onSessionExpanded}
                                onTaskExpanded={onTaskExpanded}
                                onFocusCollapsed={onFocusCollapsed}
                                onLoadMore={onLoadMore}
                                path={index.focusPathSessionIds}
                                placement={placement.restored ? "restored_child" : "page_child"}
                              />
                            ))}
                          </ul>
                        ) : (
                          <p className="text-sm text-muted-foreground">
                            {sideChildPlacements.length > 0
                              ? "No loaded off-path child sessions match the current filters."
                              : pathChildCount > 0
                                ? "Loaded focus-path children are shown once in the ancestor path."
                                : "No child sessions are attached."}
                          </p>
                        )}
                        {branch.hasMore && branch.nextCursor && (
                          <div className="mt-3">
                            <BranchLoadMore
                              label="Load more child sessions"
                              disabled={pendingRead !== null}
                              onClick={() =>
                                onLoadMore({
                                  kind: "session_children",
                                  scopeId: node.id,
                                  cursor: branch.nextCursor!,
                                })
                              }
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div
                className="flex items-center gap-2 px-3 py-1 text-xs text-muted-foreground"
                aria-hidden="true"
              >
                <ChevronDown className="h-3.5 w-3.5" /> parent of the next session
              </div>
            </li>
          )
        })}
        <li className="flex min-w-0 flex-wrap items-center gap-2 rounded-md border border-dashed border-border bg-primary/5 px-3 py-2">
          <Badge variant="secondary">focus session</Badge>
          <span className="min-w-0 break-all font-mono text-sm font-medium">{state.focus.id}</span>
        </li>
      </ol>
    </div>
  )
}

function SessionTreeNode({
  node,
  state,
  index,
  search,
  visibleNodes,
  directMatches,
  pendingRead,
  onSessionExpanded,
  onTaskExpanded,
  onFocusCollapsed,
  onLoadMore,
  path,
  placement,
}: WorkflowSessionTreeProps & {
  node: ApiSessionTopologyNode
  path: ReadonlySet<string>
  placement: "focus" | "page_child" | "restored_child" | "unattached_restored"
}) {
  const key = workflowNodeKey("session", node.id)
  if (!visibleNodes.has(key)) return null
  const focus = node.id === state.focus.id
  const expanded = focus
    ? !search.focus_collapsed
    : (search.expanded_session_id?.includes(node.id) ?? false)
  const branch = state.sessionBranches.get(node.id)
  const childPlacements = workflowSessionChildrenForParent(index, node.id)
  const visibleChildren = childPlacements.filter((childPlacement) =>
    visibleNodes.has(workflowNodeKey("session", childPlacement.node.id)),
  )
  const restoredChildCount = childPlacements.filter(
    (childPlacement) => childPlacement.restored,
  ).length
  const nextPath = new Set(path)
  const cycle = nextPath.has(node.id)
  nextPath.add(node.id)
  return (
    <li className="min-w-0">
      <div className={cn("rounded-md border border-border p-3", focus && "bg-primary/5")}>
        <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2">
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-expanded={expanded}
              aria-label={`${expanded ? "Collapse" : "Expand"} session ${node.id}`}
              disabled={
                !focus &&
                !expanded &&
                (search.expanded_session_id?.length ?? 0) >= WORKFLOW_SESSION_EXPANSION_LIMIT
              }
              onClick={() =>
                focus ? onFocusCollapsed(expanded) : onSessionExpanded(node.id, !expanded)
              }
            >
              {expanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
            <div className="min-w-0">
              <SessionOperationalBadges
                node={node}
                focus={state.focus}
                direct={directMatches.has(key)}
                kindLabel={
                  placement === "focus"
                    ? "focus session"
                    : placement === "page_child"
                      ? "child session"
                      : placement === "restored_child"
                        ? "restored child session"
                        : "unattached restored session"
                }
              />
              <div className="mt-1 break-all font-mono text-sm font-medium">{node.id}</div>
              <SessionOperationalMetadata node={node} />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {!focus && (
              <Link
                to="/sessions/$sessionId/workflow"
                params={{ sessionId: node.id }}
                search={workflowSearchForUrl(workflowSearchForNewFocus(search))}
                className={buttonVariants({ variant: "outline", size: "sm" })}
              >
                Use as focus
              </Link>
            )}
            <Link
              to="/sessions/$sessionId"
              params={{ sessionId: node.id }}
              search={{}}
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Session details <ExternalLink className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </div>
      {expanded && (
        <div className="ml-5 border-l border-border pl-4 pt-3">
          {cycle ? (
            <p className="text-sm text-destructive" role="alert">
              The loaded session branch contains a cycle and cannot be rendered.
            </p>
          ) : (
            <>
              <LinkedTasks
                sessionId={node.id}
                state={state}
                index={index}
                search={search}
                visibleNodes={visibleNodes}
                directMatches={directMatches}
                pendingRead={pendingRead}
                onTaskExpanded={onTaskExpanded}
                onLoadMore={onLoadMore}
              />
              {branch === undefined ? (
                <p className="mt-3 text-sm text-muted-foreground" role="status">
                  {pendingRead
                    ? "Loading child sessions..."
                    : "Child-session branch is unavailable."}
                </p>
              ) : (
                <div className="mt-3">
                  <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span>{branch.nodes.length} loaded child sessions</span>
                    {restoredChildCount > 0 && (
                      <span>{restoredChildCount} restored outside the current child page</span>
                    )}
                    <BranchSnapshotBadge mixed={branch.mixedSnapshot} />
                  </div>
                  {visibleChildren && visibleChildren.length > 0 ? (
                    <ul className="space-y-3" aria-label={`Child sessions for ${node.id}`}>
                      {visibleChildren.map((childPlacement) => (
                        <SessionTreeNode
                          key={childPlacement.node.id}
                          node={childPlacement.node}
                          state={state}
                          index={index}
                          search={search}
                          visibleNodes={visibleNodes}
                          directMatches={directMatches}
                          pendingRead={pendingRead}
                          onSessionExpanded={onSessionExpanded}
                          onTaskExpanded={onTaskExpanded}
                          onFocusCollapsed={onFocusCollapsed}
                          onLoadMore={onLoadMore}
                          path={nextPath}
                          placement={childPlacement.restored ? "restored_child" : "page_child"}
                        />
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      {branch.nodes.length > 0
                        ? "No loaded child sessions match the current filters."
                        : "No child sessions are attached."}
                    </p>
                  )}
                  {branch.hasMore && branch.nextCursor && (
                    <div className="mt-3">
                      <BranchLoadMore
                        label="Load more child sessions"
                        disabled={pendingRead !== null}
                        onClick={() =>
                          onLoadMore({
                            kind: "session_children",
                            scopeId: node.id,
                            cursor: branch.nextCursor!,
                          })
                        }
                      />
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </li>
  )
}

function UnattachedRestoredExpansions({
  state,
  index,
  search,
  visibleNodes,
  directMatches,
  pendingRead,
  onSessionExpanded,
  onTaskExpanded,
  onFocusCollapsed,
  onLoadMore,
}: WorkflowSessionTreeProps) {
  const sessions = workflowUnattachedExpandedSessions(index).filter((node) =>
    visibleNodes.has(workflowNodeKey("session", node.id)),
  )
  const tasks = workflowUnattachedExpandedTasks(index).filter((node) =>
    visibleNodes.has(workflowNodeKey("task", node.id)),
  )
  if (sessions.length === 0 && tasks.length === 0) return null

  return (
    <div className="mt-4 rounded-md border border-chart-1/30 bg-chart-1/5 p-3">
      <div className="text-sm font-medium">Restored expansions outside the loaded focus path</div>
      <p className="mt-1 text-xs text-muted-foreground">
        The URL requested these bounded branches, but they do not connect through the currently
        expanded focus tree. They are shown once as separate roots rather than silently hidden or
        presented under a collapsed or unrelated branch.
      </p>
      {sessions.length > 0 && (
        <ul className="mt-3 space-y-3" aria-label="Unattached restored session expansions">
          {sessions.map((node) => (
            <SessionTreeNode
              key={node.id}
              node={node}
              state={state}
              index={index}
              search={search}
              visibleNodes={visibleNodes}
              directMatches={directMatches}
              pendingRead={pendingRead}
              onSessionExpanded={onSessionExpanded}
              onTaskExpanded={onTaskExpanded}
              onFocusCollapsed={onFocusCollapsed}
              onLoadMore={onLoadMore}
              path={new Set()}
              placement="unattached_restored"
            />
          ))}
        </ul>
      )}
      {tasks.length > 0 && (
        <ul className="mt-3 space-y-3" aria-label="Unattached restored task expansions">
          {tasks.map((node) => (
            <TaskTreeNode
              key={node.id}
              node={node}
              state={state}
              index={index}
              search={search}
              visibleNodes={visibleNodes}
              directMatches={directMatches}
              pendingRead={pendingRead}
              onTaskExpanded={onTaskExpanded}
              onLoadMore={onLoadMore}
              path={new Set()}
              relationship="unattached_restored"
            />
          ))}
        </ul>
      )}
    </div>
  )
}

function costText(cost: UsageSessionCostSummary | undefined): string {
  if (cost === undefined) return "Unavailable"
  if (cost.accuracy.kind === "truncated") return "Not evaluated"
  if (cost.currencies.length === 0) return "No priced cost"
  const currencies = cost.currencies.slice(0, 3)
  const value = currencies
    .map((item) => `${formatDecimal(item.total_cost)} ${item.currency}`)
    .join(" · ")
  const omitted = cost.currencies.length - currencies.length
  return omitted > 0 ? `${value} · ${formatCount(omitted)} more currencies` : value
}

function WorkflowAccuracyBadge({
  label,
  accuracy,
}: {
  label: string
  accuracy: AggregateAccuracy
}) {
  return (
    <Badge variant={accuracy.kind === "exact" ? "secondary" : "outline"}>
      {label} {accuracy.kind}
    </Badge>
  )
}

function workflowAccuracyExplanation(accuracy: AggregateAccuracy, fallback: string): string {
  const detail = accuracy.reason ?? fallback
  return accuracy.limit ? `${detail} Limit: ${formatCount(accuracy.limit)}.` : detail
}

function SessionCostCell({ cost }: { cost: UsageSessionCostSummary | undefined }) {
  if (cost === undefined) return <span className="text-muted-foreground">Unavailable</span>
  return (
    <div className="space-y-1">
      <div>{costText(cost)}</div>
      <div className="flex flex-wrap justify-end gap-1 text-xs text-muted-foreground">
        <WorkflowAccuracyBadge label="cost" accuracy={cost.accuracy} />
        <span>{formatCount(cost.priced_model_steps)} priced</span>
        <span>· {formatCount(cost.unpriced_model_steps)} unpriced</span>
        <span>· {formatCount(cost.unevaluated_model_steps)} unevaluated</span>
      </div>
    </div>
  )
}

function WorkflowUsageSummary({
  data,
  state,
  index,
}: {
  data: UsageRollup
  state: WorkflowTopologyState
  index: WorkflowRenderIndex
}) {
  const loadedSessionIds = new Set(index.sessionById.keys())
  const sessionGroups = data.session_breakdown?.groups ?? []
  const mappedGroups = sessionGroups.filter((group) => loadedSessionIds.has(group.session_id))
  const outsideLoadedScope = sessionGroups.length - mappedGroups.length
  const costsBySession = new Map(
    (data.session_cost_breakdown?.groups ?? []).map((group) => [group.session_id, group.cost]),
  )
  const usageRemainder = data.session_breakdown?.remainder ?? null
  const costRemainder = data.session_cost_breakdown?.remainder ?? null
  const displayedCurrencies =
    data.cost?.currencies.slice(0, WORKFLOW_RENDERED_COST_DETAIL_LIMIT) ?? []
  const displayedUnpricedReasons =
    data.cost?.unpriced_reasons.slice(0, WORKFLOW_RENDERED_COST_DETAIL_LIMIT) ?? []
  return (
    <DataCard
      title="Causal-budget usage"
      description="Selected event-time window over the focus session's shared causal budget."
      actions={<WorkflowAccuracyBadge label="shared usage" accuracy={data.accuracy} />}
    >
      <div className="border-b border-border p-4 text-sm text-muted-foreground">
        <div className="break-all font-mono text-xs">Budget {state.focus.causal_budget_id}</div>
        <div className="mt-2">
          Events from {formatDateTime(data.start_at)} inclusive to {formatDateTime(data.end_at)}{" "}
          exclusive, observed {formatDateTime(data.as_of)}. These are window totals, not lifetime
          totals.
        </div>
      </div>
      {data.accuracy.kind !== "exact" && (
        <div
          className="border-b border-border bg-chart-1/5 px-4 py-3 text-sm text-chart-1"
          role="status"
        >
          The store reports {data.accuracy.kind} aggregate coverage. Totals are not presented as
          complete.
        </div>
      )}
      <div className="grid gap-0 border-b border-border sm:grid-cols-2 xl:grid-cols-5">
        {[
          ["Matching sessions", formatCount(data.matching_session_count)],
          ["Active sessions", formatCount(data.active_session_count)],
          ["Model steps", formatCount(data.totals.model_steps)],
          ["Tokens", formatCount(data.totals.usage.total_tokens)],
          ["Tool calls", formatCount(data.totals.tool_calls)],
        ].map(([label, value]) => (
          <div
            key={label}
            className="border-b border-border p-4 last:border-b-0 sm:border-r xl:border-b-0"
          >
            <div className="text-xs uppercase text-muted-foreground">{label}</div>
            <div className="mt-1 text-xl font-semibold">{value}</div>
          </div>
        ))}
      </div>
      <div className="border-b border-border p-4">
        <div className="flex flex-wrap items-center gap-2 text-sm font-medium">
          <CircleDollarSign className="h-4 w-4 text-muted-foreground" /> Estimated cost
          {data.cost && <WorkflowAccuracyBadge label="cost" accuracy={data.cost.accuracy} />}
        </div>
        {data.cost === null ? (
          <p className="mt-2 text-sm text-muted-foreground">
            No dashboard price book is configured. Usage remains available; cost is unavailable
            rather than zero.
          </p>
        ) : (
          <div className="mt-2 space-y-2 text-sm">
            <p className="text-xs text-muted-foreground">
              Price book {data.cost.price_book_version}, generated{" "}
              {data.cost.price_book_generated_at}.
            </p>
            <div className="flex flex-wrap gap-3">
              {data.cost.accuracy.kind === "truncated" ? (
                <span className="text-muted-foreground">
                  Cost evaluation exceeded its bound, so no partial evaluated totals are published.
                </span>
              ) : displayedCurrencies.length > 0 ? (
                displayedCurrencies.map((currency) => (
                  <Badge key={currency.currency} variant="outline">
                    {formatDecimal(currency.total_cost)} {currency.currency}
                  </Badge>
                ))
              ) : (
                <span className="text-muted-foreground">No model steps were priced.</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {formatCount(data.cost.priced_model_steps)} priced ·{" "}
              {formatCount(data.cost.unpriced_model_steps)} unpriced ·{" "}
              {formatCount(data.cost.unevaluated_model_steps)} unevaluated. Currencies are never
              combined.
            </p>
            {data.cost.currencies.length > displayedCurrencies.length && (
              <p className="text-xs text-muted-foreground">
                {formatCount(data.cost.currencies.length - displayedCurrencies.length)} additional
                currency totals are omitted from rendering.
              </p>
            )}
            {displayedUnpricedReasons.length > 0 && (
              <div className="text-xs text-muted-foreground">
                <div className="font-medium text-foreground">Unpriced reasons</div>
                <ul className="mt-1 space-y-1">
                  {displayedUnpricedReasons.map((reason) => (
                    <li key={reason.reason} className="flex flex-wrap justify-between gap-2">
                      <span>{reason.reason}</span>
                      <span>{formatCount(reason.model_steps)} steps</span>
                    </li>
                  ))}
                </ul>
                {data.cost.unpriced_reasons.length > displayedUnpricedReasons.length && (
                  <p className="mt-1">
                    {formatCount(
                      data.cost.unpriced_reasons.length - displayedUnpricedReasons.length,
                    )}{" "}
                    additional reasons are omitted from rendering.
                  </p>
                )}
              </div>
            )}
            {(data.cost.accuracy.kind !== "exact" ||
              data.cost.unpriced_model_steps !== "0" ||
              data.cost.unevaluated_model_steps !== "0") && (
              <p className="text-sm text-chart-1" role="status">
                {workflowAccuracyExplanation(
                  data.cost.accuracy,
                  "Displayed currency totals cover only the model steps that were priced.",
                )}
              </p>
            )}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-3 text-sm">
        <span className="font-medium">Bounded per-session detail</span>
        {data.session_breakdown && (
          <WorkflowAccuracyBadge label="usage detail" accuracy={data.session_breakdown.accuracy} />
        )}
        {data.session_cost_breakdown && (
          <WorkflowAccuracyBadge
            label="cost detail"
            accuracy={data.session_cost_breakdown.accuracy}
          />
        )}
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Loaded session</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Model steps</TableHead>
            <TableHead className="text-right">Tokens</TableHead>
            <TableHead className="text-right">Tool calls</TableHead>
            <TableHead className="text-right">Cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {mappedGroups.length > 0 ? (
            mappedGroups.map((group) => (
              <TableRow key={group.session_id}>
                <TableCell>
                  <Link
                    to="/sessions/$sessionId"
                    params={{ sessionId: group.session_id }}
                    search={{}}
                    className="font-mono text-xs text-primary hover:underline"
                  >
                    {group.session_id}
                  </Link>
                </TableCell>
                <TableCell>
                  <Badge variant={statusVariant(group.status)}>{group.status}</Badge>
                </TableCell>
                <TableCell className="text-right">
                  {formatCount(group.totals.model_steps)}
                </TableCell>
                <TableCell className="text-right">
                  {formatCount(group.totals.usage.total_tokens)}
                </TableCell>
                <TableCell className="text-right">{formatCount(group.totals.tool_calls)}</TableCell>
                <TableCell className="text-right">
                  <SessionCostCell cost={costsBySession.get(group.session_id)} />
                </TableCell>
              </TableRow>
            ))
          ) : usageRemainder === null && costRemainder === null ? (
            <TableRow>
              <TableCell colSpan={6}>
                <StateMessage>
                  No returned usage groups map to the currently loaded topology.
                </StateMessage>
              </TableCell>
            </TableRow>
          ) : null}
          {(usageRemainder !== null || costRemainder !== null) && (
            <TableRow>
              <TableCell>
                <div className="font-medium">Omitted-session remainder</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {usageRemainder !== null && (
                    <span>
                      {formatCount(usageRemainder.group_count)} usage groups ·{" "}
                      {formatCount(usageRemainder.active_session_count)} active
                    </span>
                  )}
                  {usageRemainder !== null && costRemainder !== null && <span> · </span>}
                  {costRemainder !== null && (
                    <span>{formatCount(costRemainder.group_count)} cost groups</span>
                  )}
                </div>
              </TableCell>
              <TableCell>
                <Badge variant="outline">aggregate remainder</Badge>
              </TableCell>
              <TableCell className="text-right">
                {usageRemainder === null ? "—" : formatCount(usageRemainder.totals.model_steps)}
              </TableCell>
              <TableCell className="text-right">
                {usageRemainder === null
                  ? "—"
                  : formatCount(usageRemainder.totals.usage.total_tokens)}
              </TableCell>
              <TableCell className="text-right">
                {usageRemainder === null ? "—" : formatCount(usageRemainder.totals.tool_calls)}
              </TableCell>
              <TableCell className="text-right">
                <SessionCostCell cost={costRemainder?.cost} />
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {(usageRemainder !== null || costRemainder !== null || outsideLoadedScope > 0) && (
        <div className="space-y-1 border-t border-border px-4 py-3 text-xs text-muted-foreground">
          {usageRemainder !== null && (
            <p>
              The usage remainder is included in the shared totals and displayed as a separate
              aggregate row; it is not redistributed to loaded sessions.
            </p>
          )}
          {costRemainder !== null && (
            <p>
              The cost remainder is displayed independently because cost coverage can differ from
              usage-detail coverage.
            </p>
          )}
          {outsideLoadedScope > 0 && (
            <p>
              {formatCount(outsideLoadedScope)} returned session groups are outside the loaded
              topology and are not mapped to rows here.
            </p>
          )}
        </div>
      )}
    </DataCard>
  )
}

export function WorkflowPage() {
  const { sessionId } = useParams({ from: "/sessions/$sessionId/workflow" })
  return <WorkflowPageForSession key={sessionId} sessionId={sessionId} />
}

function WorkflowPageForSession({ sessionId }: { sessionId: string }) {
  const search = useSearch({ from: "/sessions/$sessionId/workflow" })
  const navigate = useNavigate({ from: "/sessions/$sessionId/workflow" })
  const usageCapability = useDashboardCapability({ kind: "surface", surface: "usage" })
  const usageEnabled = usageCapability.enabled
  const usageUnavailableText = dashboardCapabilityUnavailableText(usageCapability)
  const topologyCoordinator = useRef(new LatestWorkflowRequestCoordinator())
  const mountedRef = useRef(true)
  const pendingTicketRef = useRef<WorkflowRequestTicket | null>(null)
  const topologyRef = useRef<WorkflowTopologyState | undefined>(undefined)
  const topologyPendingRef = useRef(false)
  const usageRefreshPendingRef = useRef(false)
  const terminalUsageReconciler = useRef(new WorkflowTerminalUsageReconciler())
  const usageAnchor = useRef(new Date())
  const [topology, setTopology] = useState<WorkflowTopologyState>()
  const [topologyReadError, setTopologyReadError] = useState<unknown>(null)
  const [pendingRead, setPendingRead] = useState<TopologyPendingRead | null>(null)
  const [topologyRefreshFailures, setTopologyRefreshFailures] = useState(0)
  const [interactionError, setInteractionError] = useState<string | null>(null)
  const topologyShapeKey = workflowTopologyShapeKey(search)
  const topologySearch = useMemo(
    () => validateWorkflowSearch(JSON.parse(topologyShapeKey) as Record<string, unknown>),
    [topologyShapeKey],
  )

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const finishTopologyRead = useCallback((ticket: WorkflowRequestTicket) => {
    topologyCoordinator.current.finish(ticket)
    if (pendingTicketRef.current !== ticket) return
    pendingTicketRef.current = null
    topologyPendingRef.current = false
    setPendingRead(null)
  }, [])

  const readTopology = useCallback(
    async (
      continuation: WorkflowContinuation | undefined,
      kind: TopologyPendingRead,
    ): Promise<boolean | null> => {
      const previous = topologyRef.current
      if (continuation !== undefined && previous === undefined) return null
      const ticket = topologyCoordinator.current.begin()
      pendingTicketRef.current = ticket
      topologyPendingRef.current = true
      setPendingRead(kind)
      try {
        const request = buildWorkflowTopologyRequest(sessionId, topologySearch, continuation)
        const response = await fetchSessionTopology(sessionId, request, ticket.signal)
        let next: WorkflowTopologyState | undefined
        const committed = topologyCoordinator.current.commit(ticket, () => {
          next = mergeWorkflowTopologyResponse(previous, sessionId, request, response)
          topologyRef.current = next
          setTopology(next)
          setTopologyReadError(null)
        })
        if (!committed) return null
        finishTopologyRead(ticket)
        return true
      } catch (error) {
        if (isAbortError(error) && ticket.signal.aborted) return null
        const committed = topologyCoordinator.current.commit(ticket, () => {
          setTopologyReadError(error)
        })
        if (!committed) return null
        finishTopologyRead(ticket)
        return false
      }
    },
    [finishTopologyRead, sessionId, topologySearch],
  )

  useEffect(() => {
    if (search.invalid) return
    setTopologyRefreshFailures(0)
    void readTopology(undefined, topologyRef.current === undefined ? "initial" : "refresh")
    return () => {
      topologyCoordinator.current.cancel()
      pendingTicketRef.current = null
      topologyPendingRef.current = false
    }
  }, [readTopology, search.invalid])

  const usageSearchKey = JSON.stringify({
    range: search.range,
    start_at: search.start_at,
    end_at: search.end_at,
  })
  const usageRefreshAuthority = JSON.stringify([
    sessionId,
    topology?.focus.causal_budget_id ?? null,
    usageSearchKey,
  ])
  const [usageRefreshBackoff, setUsageRefreshBackoff] = useState(() => ({
    authority: usageRefreshAuthority,
    failures: 0,
  }))
  const usageRefreshFailures =
    usageRefreshBackoff.authority === usageRefreshAuthority ? usageRefreshBackoff.failures : 0
  const usageRequestState = useMemo(() => {
    if (!usageEnabled || topology === undefined || search.invalid) {
      return { request: null, error: null }
    }
    try {
      return {
        request: buildWorkflowUsageRequest(topology.focus.causal_budget_id, search, {
          now: usageAnchor.current,
          pricing: dashboardConfig.priceBook,
        }),
        error: null,
      }
    } catch (error) {
      return {
        request: null,
        error: error instanceof Error ? error.message : "The usage request is invalid.",
      }
    }
  }, [search, topology, usageEnabled])
  const usageRequest = usageRequestState.request
  const usageAvailable = usageRequest !== null
  const usage = useQuery({
    queryKey: [
      "workflow-usage",
      sessionId,
      topology?.focus.causal_budget_id ?? null,
      usageSearchKey,
    ],
    queryFn: ({ signal }) => {
      const current = topologyRef.current
      if (current === undefined) throw new Error("Workflow topology is unavailable.")
      return fetchUsageRollup(
        buildWorkflowUsageRequest(current.focus.causal_budget_id, search, {
          now: usageAnchor.current,
          pricing: dashboardConfig.priceBook,
        }),
        signal,
      )
    },
    enabled: usageRequest !== null,
    gcTime: 0,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    retry: retryAggregateRequest,
  })
  const usageData = usage.data
  const usageIsFetching = usage.isFetching
  const refetchUsage = usage.refetch

  const refreshTopology = useCallback(async (): Promise<boolean | null> => {
    if (topologyPendingRef.current) return null
    const succeeded = await readTopology(undefined, "refresh")
    if (succeeded !== null) {
      setTopologyRefreshFailures((current) => (succeeded ? 0 : Math.min(current + 1, 30)))
    }
    return succeeded
  }, [readTopology])

  const refreshUsage = useCallback(async (): Promise<boolean | null> => {
    if (!usageAvailable || usageIsFetching || usageRefreshPendingRef.current) return null
    usageRefreshPendingRef.current = true
    usageAnchor.current = new Date()
    const recordResult = (succeeded: boolean) => {
      setUsageRefreshBackoff((current) => {
        const failures = current.authority === usageRefreshAuthority ? current.failures : 0
        const nextFailures = succeeded ? 0 : Math.min(failures + 1, 30)
        if (current.authority === usageRefreshAuthority && current.failures === nextFailures) {
          return current
        }
        return { authority: usageRefreshAuthority, failures: nextFailures }
      })
    }
    try {
      const result = await refetchUsage()
      const succeeded = !result.isError
      recordResult(succeeded)
      return succeeded
    } catch {
      recordResult(false)
      return false
    } finally {
      usageRefreshPendingRef.current = false
    }
  }, [refetchUsage, usageAvailable, usageIsFetching, usageRefreshAuthority])

  const topologyActive = topology ? workflowTopologyContainsActiveNodes(topology) : false
  const [terminalUsageRefreshSignal, setTerminalUsageRefreshSignal] = useState(0)
  useEffect(() => {
    if (topology === undefined) return
    if (
      terminalUsageReconciler.current.observe(usageRefreshAuthority, topologyActive, usageAvailable)
    ) {
      setTerminalUsageRefreshSignal((current) => current + 1)
    }
  }, [topology, topologyActive, usageAvailable, usageRefreshAuthority])

  useEffect(() => {
    const requestPending = usageIsFetching || usageRefreshPendingRef.current
    if (
      terminalUsageRefreshSignal === 0 ||
      !usageAvailable ||
      !terminalUsageReconciler.current.claim(usageRefreshAuthority, requestPending)
    ) {
      return
    }
    void refreshUsage().finally(() => {
      const queued = terminalUsageReconciler.current.finish(usageRefreshAuthority)
      if (queued && mountedRef.current) {
        setTerminalUsageRefreshSignal((current) => current + 1)
      }
    })
  }, [
    refreshUsage,
    terminalUsageRefreshSignal,
    usageAvailable,
    usageIsFetching,
    usageRefreshAuthority,
  ])

  useEffect(() => {
    if (!topologyActive) return
    let timer: number | undefined
    let disposed = false
    const schedule = () => {
      if (disposed) return
      if (timer !== undefined) window.clearTimeout(timer)
      timer = undefined
      if (
        !workflowShouldAutoRefresh({
          documentVisible: document.visibilityState === "visible",
          hasActiveNodes: topologyActive,
          requestPending: pendingRead !== null,
        })
      ) {
        return
      }
      timer = window.setTimeout(async () => {
        timer = undefined
        await refreshTopology()
        schedule()
      }, workflowRefreshDelay(topologyRefreshFailures))
    }
    const visibilityChanged = () => schedule()
    document.addEventListener("visibilitychange", visibilityChanged)
    schedule()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
      document.removeEventListener("visibilitychange", visibilityChanged)
    }
  }, [pendingRead, refreshTopology, topologyActive, topologyRefreshFailures])

  useEffect(() => {
    if (!topologyActive || !usageAvailable) return
    let timer: number | undefined
    let disposed = false
    const schedule = () => {
      if (disposed) return
      if (timer !== undefined) window.clearTimeout(timer)
      timer = undefined
      if (
        !workflowShouldAutoRefresh({
          documentVisible: document.visibilityState === "visible",
          hasActiveNodes: topologyActive,
          requestPending: usageIsFetching,
        })
      ) {
        return
      }
      timer = window.setTimeout(async () => {
        timer = undefined
        await refreshUsage()
        schedule()
      }, workflowRefreshDelay(usageRefreshFailures))
    }
    const visibilityChanged = () => schedule()
    document.addEventListener("visibilitychange", visibilityChanged)
    schedule()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
      document.removeEventListener("visibilitychange", visibilityChanged)
    }
  }, [refreshUsage, topologyActive, usageAvailable, usageIsFetching, usageRefreshFailures])

  const renderIndex = useMemo(
    () => (topology ? buildWorkflowRenderIndex(topology, topologySearch) : null),
    [topology, topologySearch],
  )
  const visibility = useMemo(
    () => (topology && renderIndex ? workflowNodeVisibility(topology, search, renderIndex) : null),
    [renderIndex, search, topology],
  )
  const resolvedUsageWindow = usageData
    ? { startAt: usageData.start_at, endAt: usageData.end_at }
    : usageRequest
      ? { startAt: usageRequest.start_at, endAt: usageRequest.end_at }
      : null

  function applySearch(next: WorkflowSearch) {
    try {
      setInteractionError(null)
      void navigate({ search: workflowSearchForUrl(next), resetScroll: false })
    } catch (error) {
      setInteractionError(error instanceof Error ? error.message : "The Workflow URL is invalid.")
    }
  }

  if (search.invalid) {
    return (
      <Page>
        <PageHeader title="Workflow" />
        <StateMessage tone="danger" className="rounded-lg border border-destructive/30 py-12">
          <div role="alert" className="space-y-3">
            <div className="font-medium">The Workflow URL is invalid or exceeds safe bounds.</div>
            <Button type="button" variant="outline" onClick={() => void navigate({ search: {} })}>
              Reset Workflow view
            </Button>
          </div>
        </StateMessage>
      </Page>
    )
  }

  const presentedError =
    topologyReadError === null ? null : workflowTopologyError(topologyReadError)
  return (
    <Page data-testid="workflow-page">
      <PageHeader
        title="Workflow"
        description="Bounded operational topology and causal-budget usage for one focus session."
        actions={
          <div className="flex flex-wrap gap-2">
            <Link
              to="/sessions/$sessionId"
              params={{ sessionId }}
              search={{}}
              className={buttonVariants({ variant: "outline" })}
            >
              <ArrowLeft className="h-4 w-4" /> Session details
            </Link>
            <Button
              type="button"
              variant="outline"
              disabled={pendingRead !== null && (!usageAvailable || usage.isFetching)}
              onClick={() => void Promise.all([refreshTopology(), refreshUsage()])}
            >
              <RefreshCw
                className={cn("h-4 w-4", (pendingRead || usage.isFetching) && "animate-spin")}
              />
              Refresh snapshots
            </Button>
          </div>
        }
      />

      <WorkflowControls
        key={workflowControlsKey(search)}
        search={search}
        resolvedWindow={resolvedUsageWindow}
        usageEnabled={usageEnabled}
        onApply={applySearch}
        onClearFilters={() => applySearch(workflowSearchWithoutFilters(search))}
      />

      {interactionError && (
        <StateMessage
          tone="danger"
          className="rounded-md border border-destructive/30 p-4"
          role="alert"
        >
          {interactionError}
        </StateMessage>
      )}

      {topology === undefined || renderIndex === null ? (
        presentedError ? (
          <StateMessage tone="danger" className="rounded-lg border border-destructive/30 py-12">
            <div role="alert" className="space-y-3">
              <div className="font-medium">{presentedError.title}</div>
              <div>{presentedError.detail}</div>
              <Button
                type="button"
                variant="outline"
                onClick={() => void readTopology(undefined, "initial")}
              >
                Retry topology
              </Button>
            </div>
          </StateMessage>
        ) : (
          <StateMessage
            className="rounded-lg border border-border py-12"
            role="status"
            aria-live="polite"
          >
            Loading the bounded Workflow topology...
          </StateMessage>
        )
      ) : (
        <>
          {presentedError && (
            <div
              className="flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
              role="alert"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                <div className="font-medium">{presentedError.title}</div>
                <div className="mt-1">
                  {presentedError.detail} The last confirmed topology remains visible.
                </div>
              </div>
            </div>
          )}
          <div className="flex items-start gap-3 rounded-md border border-border bg-muted/30 px-4 py-3 text-sm">
            <Clock className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div>
              <div className="font-medium">
                Sessions observed {formatDateTime(topology.observedAt)}
              </div>
              <div className="mt-1 text-muted-foreground">
                {topology.taskStatus === "available" && topology.taskObservedAt
                  ? `Tasks observed ${formatDateTime(topology.taskObservedAt)}. `
                  : "No task-store observation is available. "}
                Session and task stores are sampled independently (`cross_store_atomic: false`).{" "}
                {topologyActive
                  ? WORKFLOW_REFRESH_LABEL
                  : "Every loaded node is terminal, so routine refresh is stopped."}
              </div>
            </div>
          </div>
          {workflowTopologyContainsMixedSnapshots(topology) && (
            <div
              className="flex items-start gap-3 rounded-md border border-chart-1/30 bg-chart-1/5 px-4 py-3 text-sm text-chart-1"
              role="status"
            >
              <GitBranch className="mt-0.5 h-4 w-4 shrink-0" />
              Loaded pages contain multiple observation times. Retained rows remain visible until
              their current cursor chain confirms or removes them.
            </div>
          )}
          {topology.taskStatus !== "available" && (
            <div
              className="rounded-md border border-border bg-muted/30 px-4 py-3 text-sm"
              role="status"
            >
              Task linkage is{" "}
              {topology.taskStatus === "not_configured" ? "not configured" : "unsupported"}. Session
              ancestry and child branches remain available.
            </div>
          )}

          <DataCard
            title="Execution topology"
            description={`${renderIndex.sessionById.size} loaded sessions. Filters do not search unloaded branches.`}
            actions={pendingRead ? <Badge variant="outline">refreshing</Badge> : undefined}
            contentClassName="p-4"
          >
            {visibility && visibility.visibleNodes.size > 0 && (
              <AncestorPath
                state={topology}
                index={renderIndex}
                search={search}
                visibleNodes={visibility.visibleNodes}
                directMatches={visibility.directMatches}
                pendingRead={pendingRead}
                onSessionExpanded={(id, expanded) =>
                  applySearch(workflowSearchWithExpansion(search, "session", id, expanded))
                }
                onTaskExpanded={(id, expanded) =>
                  applySearch(workflowSearchWithExpansion(search, "task", id, expanded))
                }
                onFocusCollapsed={(collapsed) =>
                  applySearch(workflowSearchWithFocusCollapsed(search, collapsed))
                }
                onLoadMore={(continuation) => void readTopology(continuation, continuation.kind)}
              />
            )}
            {visibility && visibility.visibleNodes.size > 0 ? (
              <ul className="space-y-3" aria-label="Loaded Workflow topology">
                <SessionTreeNode
                  node={topology.focus}
                  state={topology}
                  index={renderIndex}
                  search={search}
                  visibleNodes={visibility.visibleNodes}
                  directMatches={visibility.directMatches}
                  pendingRead={pendingRead}
                  onSessionExpanded={(id, expanded) =>
                    applySearch(workflowSearchWithExpansion(search, "session", id, expanded))
                  }
                  onTaskExpanded={(id, expanded) =>
                    applySearch(workflowSearchWithExpansion(search, "task", id, expanded))
                  }
                  onFocusCollapsed={(collapsed) =>
                    applySearch(workflowSearchWithFocusCollapsed(search, collapsed))
                  }
                  onLoadMore={(continuation) => void readTopology(continuation, continuation.kind)}
                  path={new Set()}
                  placement="focus"
                />
              </ul>
            ) : (
              <StateMessage>No loaded nodes match the current filters.</StateMessage>
            )}
            {visibility && visibility.visibleNodes.size > 0 && (
              <UnattachedRestoredExpansions
                state={topology}
                index={renderIndex}
                search={search}
                visibleNodes={visibility.visibleNodes}
                directMatches={visibility.directMatches}
                pendingRead={pendingRead}
                onSessionExpanded={(id, expanded) =>
                  applySearch(workflowSearchWithExpansion(search, "session", id, expanded))
                }
                onTaskExpanded={(id, expanded) =>
                  applySearch(workflowSearchWithExpansion(search, "task", id, expanded))
                }
                onFocusCollapsed={(collapsed) =>
                  applySearch(workflowSearchWithFocusCollapsed(search, collapsed))
                }
                onLoadMore={(continuation) => void readTopology(continuation, continuation.kind)}
              />
            )}
          </DataCard>

          {!usageEnabled ? (
            <StateMessage className="rounded-md border border-border bg-muted/30 py-8">
              <div role="status">
                <div className="font-medium">Causal-budget usage is unavailable</div>
                <div className="mt-1">{usageUnavailableText}</div>
              </div>
            </StateMessage>
          ) : usageRequestState.error ? (
            <StateMessage tone="danger" className="rounded-md border border-destructive/30 py-8">
              <div role="alert">
                <div className="font-medium">The causal-budget usage query is invalid.</div>
                <div className="mt-1">{usageRequestState.error}</div>
              </div>
            </StateMessage>
          ) : usageData ? (
            <>
              {usage.isError && (
                <StateMessage
                  tone="danger"
                  className="rounded-md border border-destructive/30 p-4"
                  role="alert"
                >
                  Usage refresh failed. The last confirmed bounded rollup remains visible.
                </StateMessage>
              )}
              <WorkflowUsageSummary data={usageData} state={topology} index={renderIndex} />
            </>
          ) : usage.isError ? (
            <StateMessage
              tone="danger"
              className="rounded-md border border-destructive/30 py-8"
              role="alert"
            >
              {usage.error instanceof Error
                ? usage.error.message
                : "Causal-budget usage could not be loaded."}
            </StateMessage>
          ) : (
            <StateMessage className="rounded-md border border-border py-8" role="status">
              Loading the bounded causal-budget rollup...
            </StateMessage>
          )}
        </>
      )}
    </Page>
  )
}
