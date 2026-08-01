import type { ApiSessionTopologyNode, ApiTaskTopologyNode } from "./generated/server-api"
import { validateWorkflowSearch, type WorkflowSearch } from "./workflow-search.ts"
import {
  type WorkflowTopologyState,
  workflowSessionNodes,
  workflowTaskNodes,
} from "./workflow-topology.ts"

export type WorkflowNodeKind = "session" | "task"

export type WorkflowNodeVisibility = {
  directMatches: ReadonlySet<string>
  visibleNodes: ReadonlySet<string>
}

export type WorkflowCausalBudgetRelationship = "focus" | "shared" | "different"

export type WorkflowTaskCausalBudgetRelationship =
  | "shared"
  | "different"
  | "unlinked"
  | "session_not_loaded"

export type WorkflowNodePlacement<T> = {
  node: T
  restored: boolean
}

export type WorkflowRenderIndex = {
  sessionById: ReadonlyMap<string, ApiSessionTopologyNode>
  taskById: ReadonlyMap<string, ApiTaskTopologyNode>
  focusPathSessionIds: ReadonlySet<string>
  sessionChildrenByParent: ReadonlyMap<
    string,
    readonly WorkflowNodePlacement<ApiSessionTopologyNode>[]
  >
  tasksBySession: ReadonlyMap<string, readonly WorkflowNodePlacement<ApiTaskTopologyNode>[]>
  taskChildrenByParent: ReadonlyMap<string, readonly WorkflowNodePlacement<ApiTaskTopologyNode>[]>
  unattachedSessionRoots: readonly ApiSessionTopologyNode[]
  unattachedTaskRoots: readonly ApiTaskTopologyNode[]
  parentByNode: ReadonlyMap<string, string>
}

export type WorkflowTopologyError = {
  kind: "missing" | "inconsistent" | "oversized" | "unsupported" | "failed"
  title: string
  detail: string
}

export const WORKFLOW_REFRESH_BASE_MS = 5_000
export const WORKFLOW_REFRESH_MAX_MS = 60_000

/**
 * Queue one final usage read after an observed active topology becomes terminal.
 *
 * The route owns request execution; this coordinator only makes the transition,
 * authority, and single-flight rules deterministic and independently testable.
 */
export class WorkflowTerminalUsageReconciler {
  private authority: string | null = null
  private wasActive = false
  private pending = false
  private running = false

  observe(authority: string, active: boolean, usageAvailable: boolean): boolean {
    if (authority !== this.authority) {
      this.authority = authority
      this.wasActive = active
      this.pending = false
      this.running = false
      return false
    }

    if (active) {
      this.wasActive = true
      this.pending = false
      return false
    }

    // A session can resume and terminate again while the previous terminal
    // catch-up is still running. Preserve one queued successor in that case.
    const queued = this.wasActive && usageAvailable && !this.pending
    this.wasActive = false
    if (queued) this.pending = true
    return queued
  }

  claim(authority: string, requestPending: boolean): boolean {
    if (authority !== this.authority || requestPending || !this.pending || this.running) {
      return false
    }
    this.pending = false
    this.running = true
    return true
  }

  finish(authority: string): boolean {
    if (authority !== this.authority) return false
    this.running = false
    return this.pending
  }
}

export function workflowNodeKey(kind: WorkflowNodeKind, id: string): string {
  return `${kind}\u0000${id}`
}

export function workflowCausalBudgetRelationship(
  node: Pick<ApiSessionTopologyNode, "id" | "causal_budget_id">,
  focus: Pick<ApiSessionTopologyNode, "id" | "causal_budget_id">,
): WorkflowCausalBudgetRelationship {
  if (node.id === focus.id) return "focus"
  return node.causal_budget_id === focus.causal_budget_id ? "shared" : "different"
}

export function workflowTaskCausalBudgetRelationship(
  node: Pick<ApiTaskTopologyNode, "session_id">,
  state: WorkflowTopologyState,
  index: Pick<WorkflowRenderIndex, "sessionById">,
): WorkflowTaskCausalBudgetRelationship {
  if (node.session_id === null) return "unlinked"
  const session = index.sessionById.get(node.session_id)
  if (session === undefined) return "session_not_loaded"
  return session.causal_budget_id === state.focus.causal_budget_id ? "shared" : "different"
}

function appendPlacement<T extends { id: string }>(
  placements: Map<string, WorkflowNodePlacement<T>[]>,
  key: string,
  node: T,
  restored: boolean,
): void {
  const values = placements.get(key)
  if (values === undefined) {
    placements.set(key, [{ node, restored }])
  } else {
    values.push({ node, restored })
  }
}

function traversePlacements<T extends { id: string }>(
  rootIds: Iterable<string>,
  childrenByParent: ReadonlyMap<string, readonly WorkflowNodePlacement<T>[]>,
  rendered: Set<string>,
): void {
  const pending = [...rootIds]
  while (pending.length > 0) {
    const nodeId = pending.pop()!
    if (rendered.has(nodeId)) continue
    rendered.add(nodeId)
    const children = childrenByParent.get(nodeId) ?? []
    for (let index = children.length - 1; index >= 0; index -= 1) {
      pending.push(children[index]!.node.id)
    }
  }
}

/**
 * Build the complete bounded render graph once per confirmed topology state.
 *
 * Tasks deliberately have both task-session and task-parent edges. The tree
 * gives an expanded parent-task branch primary placement authority and uses
 * the session link only when no loaded parent placement is available. This
 * keeps every loaded node operational without duplicating rows or subtrees.
 */
export function buildWorkflowRenderIndex(
  state: WorkflowTopologyState,
  search: WorkflowSearch,
): WorkflowRenderIndex {
  const sessionById = new Map(workflowSessionNodes(state).map((node) => [node.id, node] as const))
  const taskById = new Map(workflowTaskNodes(state).map((node) => [node.id, node] as const))
  const focusPathSessionIds = new Set([...state.ancestors.map((node) => node.id), state.focus.id])
  const sessionParentById = new Map<string, string>()
  const taskParentById = new Map<string, string>()
  const taskSessionById = new Map<string, string>()
  for (const edge of state.edges) {
    if (edge.kind === "session_parent") sessionParentById.set(edge.source_id, edge.target_id)
    else if (edge.kind === "task_parent") taskParentById.set(edge.source_id, edge.target_id)
    else taskSessionById.set(edge.source_id, edge.target_id)
  }

  const sessionChildrenByParent = new Map<string, WorkflowNodePlacement<ApiSessionTopologyNode>[]>()
  const sessionPlacementParent = new Map<string, string>()
  for (const [parentId, branch] of state.sessionBranches) {
    for (const node of branch.nodes) {
      if (sessionPlacementParent.has(node.id)) continue
      sessionPlacementParent.set(node.id, parentId)
      appendPlacement(sessionChildrenByParent, parentId, node, false)
    }
  }
  for (const node of state.expandedSessionParents.values()) {
    if (node.id === state.focus.id || sessionPlacementParent.has(node.id)) continue
    const parentId = sessionParentById.get(node.id)
    if (parentId === undefined) continue
    sessionPlacementParent.set(node.id, parentId)
    appendPlacement(sessionChildrenByParent, parentId, node, true)
  }

  const rawTasksBySession = new Map<string, WorkflowNodePlacement<ApiTaskTopologyNode>[]>()
  const taskPlacementSession = new Map<string, string>()
  for (const [sessionId, branch] of state.linkedTaskBranches) {
    for (const node of branch.nodes) {
      if (taskPlacementSession.has(node.id)) continue
      taskPlacementSession.set(node.id, sessionId)
      appendPlacement(rawTasksBySession, sessionId, node, false)
    }
  }
  const rawTaskChildrenByParent = new Map<string, WorkflowNodePlacement<ApiTaskTopologyNode>[]>()
  const taskPlacementParent = new Map<string, string>()
  for (const [parentId, branch] of state.taskChildBranches) {
    for (const node of branch.nodes) {
      if (taskPlacementParent.has(node.id)) continue
      taskPlacementParent.set(node.id, parentId)
      appendPlacement(rawTaskChildrenByParent, parentId, node, false)
    }
  }
  for (const node of state.expandedTaskParents.values()) {
    if (!taskPlacementSession.has(node.id)) {
      const sessionId = taskSessionById.get(node.id)
      if (sessionId !== undefined) {
        taskPlacementSession.set(node.id, sessionId)
        appendPlacement(rawTasksBySession, sessionId, node, true)
      }
    }
    if (!taskPlacementParent.has(node.id)) {
      const parentId = taskParentById.get(node.id)
      if (parentId !== undefined) {
        taskPlacementParent.set(node.id, parentId)
        appendPlacement(rawTaskChildrenByParent, parentId, node, true)
      }
    }
  }

  const expandedTaskIds = new Set(search.expanded_task_id ?? [])
  const canonicalTaskParent = new Map<string, string>()
  for (const [taskId, parentId] of taskPlacementParent) {
    if (expandedTaskIds.has(parentId)) canonicalTaskParent.set(taskId, parentId)
  }
  const tasksBySession = new Map<string, WorkflowNodePlacement<ApiTaskTopologyNode>[]>()
  for (const [sessionId, placements] of rawTasksBySession) {
    for (const placement of placements) {
      if (canonicalTaskParent.has(placement.node.id)) continue
      appendPlacement(tasksBySession, sessionId, placement.node, placement.restored)
    }
  }
  const taskChildrenByParent = new Map<string, WorkflowNodePlacement<ApiTaskTopologyNode>[]>()
  for (const [parentId, placements] of rawTaskChildrenByParent) {
    for (const placement of placements) {
      if (canonicalTaskParent.get(placement.node.id) !== parentId) continue
      appendPlacement(taskChildrenByParent, parentId, placement.node, placement.restored)
    }
  }

  // The ancestor path and focus are rendered by dedicated UI surfaces. Walk
  // only their off-path children here so an explicitly expanded ancestor can
  // expose sibling branches without duplicating the path or focus nodes.
  const renderedSessionIds = new Set(focusPathSessionIds)
  for (const pathSessionId of focusPathSessionIds) {
    const sideChildren = (sessionChildrenByParent.get(pathSessionId) ?? []).filter(
      (placement) => !focusPathSessionIds.has(placement.node.id),
    )
    traversePlacements(
      sideChildren.map((placement) => placement.node.id),
      sessionChildrenByParent,
      renderedSessionIds,
    )
  }
  const unattachedSessionCandidates = [...state.expandedSessionParents.values()].filter(
    (node) => node.id !== state.focus.id && !renderedSessionIds.has(node.id),
  )
  const unattachedSessionIds = new Set(unattachedSessionCandidates.map((node) => node.id))
  const unattachedSessionRoots = unattachedSessionCandidates.filter((node) => {
    const parentId = sessionPlacementParent.get(node.id)
    return parentId === undefined || !unattachedSessionIds.has(parentId)
  })
  for (const root of unattachedSessionRoots) {
    traversePlacements([root.id], sessionChildrenByParent, renderedSessionIds)
  }
  // Server topology rejects cycles, but retain a deterministic defensive root
  // for any malformed component instead of silently dropping requested data.
  for (const node of unattachedSessionCandidates) {
    if (renderedSessionIds.has(node.id)) continue
    unattachedSessionRoots.push(node)
    traversePlacements([node.id], sessionChildrenByParent, renderedSessionIds)
  }

  const renderedTaskIds = new Set<string>()
  for (const sessionId of renderedSessionIds) {
    traversePlacements(
      (tasksBySession.get(sessionId) ?? []).map((placement) => placement.node.id),
      taskChildrenByParent,
      renderedTaskIds,
    )
  }
  const unattachedTaskCandidates = [...state.expandedTaskParents.values()].filter(
    (node) => !renderedTaskIds.has(node.id),
  )
  const unattachedTaskIds = new Set(unattachedTaskCandidates.map((node) => node.id))
  const unattachedTaskRoots = unattachedTaskCandidates.filter((node) => {
    const parentId = canonicalTaskParent.get(node.id)
    return parentId === undefined || !unattachedTaskIds.has(parentId)
  })
  for (const root of unattachedTaskRoots) {
    traversePlacements([root.id], taskChildrenByParent, renderedTaskIds)
  }
  for (const node of unattachedTaskCandidates) {
    if (renderedTaskIds.has(node.id)) continue
    unattachedTaskRoots.push(node)
    traversePlacements([node.id], taskChildrenByParent, renderedTaskIds)
  }

  const parentByNode = new Map<string, string>()
  for (const [nodeId, parentId] of sessionParentById) {
    if (!sessionById.has(parentId)) continue
    parentByNode.set(workflowNodeKey("session", nodeId), workflowNodeKey("session", parentId))
  }
  for (const node of taskById.values()) {
    const parentId = canonicalTaskParent.get(node.id)
    if (parentId !== undefined && taskById.has(parentId)) {
      parentByNode.set(workflowNodeKey("task", node.id), workflowNodeKey("task", parentId))
      continue
    }
    const sessionId = taskPlacementSession.get(node.id)
    if (sessionId !== undefined && sessionById.has(sessionId)) {
      parentByNode.set(workflowNodeKey("task", node.id), workflowNodeKey("session", sessionId))
    }
  }

  return {
    sessionById,
    taskById,
    focusPathSessionIds,
    sessionChildrenByParent,
    tasksBySession,
    taskChildrenByParent,
    unattachedSessionRoots,
    unattachedTaskRoots,
    parentByNode,
  }
}

export function workflowSessionChildrenForParent(
  index: WorkflowRenderIndex,
  parentSessionId: string,
): readonly WorkflowNodePlacement<ApiSessionTopologyNode>[] {
  return index.sessionChildrenByParent.get(parentSessionId) ?? []
}

export function workflowSessionSideChildrenForPathParent(
  index: WorkflowRenderIndex,
  parentSessionId: string,
): readonly WorkflowNodePlacement<ApiSessionTopologyNode>[] {
  return (index.sessionChildrenByParent.get(parentSessionId) ?? []).filter(
    (placement) => !index.focusPathSessionIds.has(placement.node.id),
  )
}

export function workflowTasksForSession(
  index: WorkflowRenderIndex,
  sessionId: string,
): readonly WorkflowNodePlacement<ApiTaskTopologyNode>[] {
  return index.tasksBySession.get(sessionId) ?? []
}

export function workflowTaskChildrenForParent(
  index: WorkflowRenderIndex,
  parentTaskId: string,
): readonly WorkflowNodePlacement<ApiTaskTopologyNode>[] {
  return index.taskChildrenByParent.get(parentTaskId) ?? []
}

export function workflowUnattachedExpandedSessions(
  index: WorkflowRenderIndex,
): readonly ApiSessionTopologyNode[] {
  return index.unattachedSessionRoots
}

export function workflowUnattachedExpandedTasks(
  index: WorkflowRenderIndex,
): readonly ApiTaskTopologyNode[] {
  return index.unattachedTaskRoots
}

export function workflowFilterCount(search: WorkflowSearch): number {
  return (
    (search.status?.length ?? 0) +
    (search.node_type?.length ?? 0) +
    (search.agent_name === undefined ? 0 : 1) +
    (search.environment_name === undefined ? 0 : 1)
  )
}

export function workflowNodeVisibility(
  state: WorkflowTopologyState,
  search: WorkflowSearch,
  index: WorkflowRenderIndex = buildWorkflowRenderIndex(state, search),
): WorkflowNodeVisibility {
  const sessions = [...index.sessionById.values()]
  const tasks = [...index.taskById.values()]
  const loadedKeys = new Set([
    ...sessions.map((node) => workflowNodeKey("session", node.id)),
    ...tasks.map((node) => workflowNodeKey("task", node.id)),
  ])
  if (workflowFilterCount(search) === 0) {
    return { directMatches: loadedKeys, visibleNodes: loadedKeys }
  }

  const statuses = search.status === undefined ? null : new Set<string>(search.status)
  const nodeTypes = search.node_type === undefined ? null : new Set(search.node_type)
  const directMatches = new Set<string>()
  for (const node of sessions) {
    if (nodeTypes !== null && !nodeTypes.has("session")) continue
    if (statuses !== null && !statuses.has(node.status)) continue
    if (search.agent_name !== undefined && node.agent_name !== search.agent_name) continue
    if (
      search.environment_name !== undefined &&
      node.environment_name !== search.environment_name
    ) {
      continue
    }
    directMatches.add(workflowNodeKey("session", node.id))
  }
  for (const node of tasks) {
    if (nodeTypes !== null && !nodeTypes.has("task")) continue
    if (statuses !== null && !statuses.has(node.status)) continue
    if (search.agent_name !== undefined && node.assigned_agent_name !== search.agent_name) {
      continue
    }
    // Task topology deliberately has no environment projection. An environment
    // filter therefore cannot directly match a task; a task may still appear as
    // structural context for a matching descendant.
    if (search.environment_name !== undefined) continue
    directMatches.add(workflowNodeKey("task", node.id))
  }

  const visibleNodes = new Set(directMatches)
  const pending = [...directMatches]
  while (pending.length > 0) {
    const child = pending.pop()!
    const parent = index.parentByNode.get(child)
    if (parent === undefined || visibleNodes.has(parent)) continue
    visibleNodes.add(parent)
    pending.push(parent)
  }
  // Ancestors are returned as one bounded path from the root to the focus
  // session's parent. When a filter directly matches any node on that path,
  // retain the complete path so the UI never presents separated nodes as
  // though they were adjacent.
  if (state.ancestors.some((node) => directMatches.has(workflowNodeKey("session", node.id)))) {
    for (const node of state.ancestors) {
      visibleNodes.add(workflowNodeKey("session", node.id))
    }
  }
  // The page is scoped to the focus session. Keep that anchor visible whenever
  // any loaded node matches, including the unusual case where only an ancestor
  // matches and the ordinary parent-edge walk points away from the focus.
  if (directMatches.size > 0) {
    visibleNodes.add(workflowNodeKey("session", state.focus.id))
  }
  return { directMatches, visibleNodes }
}

export function workflowSearchWithoutFilters(search: WorkflowSearch): WorkflowSearch {
  return validateWorkflowSearch({
    expanded_session_id: search.expanded_session_id,
    expanded_task_id: search.expanded_task_id,
    focus_collapsed: search.focus_collapsed,
    range: search.range,
    start_at: search.start_at,
    end_at: search.end_at,
  })
}

export function workflowSearchForNewFocus(search: WorkflowSearch): WorkflowSearch {
  return validateWorkflowSearch({
    status: search.status,
    agent_name: search.agent_name,
    environment_name: search.environment_name,
    node_type: search.node_type,
    range: search.range,
    start_at: search.start_at,
    end_at: search.end_at,
  })
}

export function workflowControlsKey(search: WorkflowSearch): string {
  return JSON.stringify(workflowSearchForNewFocus(search))
}

export function workflowTopologyShapeKey(search: WorkflowSearch): string {
  return JSON.stringify({
    expanded_session_id: search.expanded_session_id,
    expanded_task_id: search.expanded_task_id,
  })
}

export function workflowSearchWithExpansion(
  search: WorkflowSearch,
  kind: WorkflowNodeKind,
  id: string,
  expanded: boolean,
): WorkflowSearch {
  const field = kind === "session" ? "expanded_session_id" : "expanded_task_id"
  const values = new Set(search[field] ?? [])
  if (expanded) values.add(id)
  else values.delete(id)
  return validateWorkflowSearch({
    ...search,
    [field]: [...values],
  })
}

export function workflowSearchWithFocusCollapsed(
  search: WorkflowSearch,
  collapsed: boolean,
): WorkflowSearch {
  return validateWorkflowSearch({
    ...search,
    focus_collapsed: collapsed ? true : undefined,
  })
}

export function workflowTopologyError(error: unknown): WorkflowTopologyError {
  const status =
    error !== null &&
    typeof error === "object" &&
    "status" in error &&
    typeof error.status === "number" &&
    Number.isInteger(error.status)
      ? error.status
      : undefined
  if (status !== undefined) {
    if (status === 404) {
      return {
        kind: "missing",
        title: "Workflow session not found",
        detail: "The focus session or one requested expanded parent no longer exists.",
      }
    }
    if (status === 409) {
      return {
        kind: "inconsistent",
        title: "Workflow topology is inconsistent",
        detail: "Cayu rejected contradictory lineage or stale continuation authority.",
      }
    }
    if (status === 413) {
      return {
        kind: "oversized",
        title: "Workflow topology exceeds its bounds",
        detail: "Collapse branches or load fewer expansions before retrying.",
      }
    }
    if (status === 501) {
      return {
        kind: "unsupported",
        title: "Workflow topology is unsupported",
        detail: "The configured session store cannot serve bounded topology reads.",
      }
    }
  }
  return {
    kind: "failed",
    title: "Workflow topology could not be loaded",
    detail: error instanceof Error ? error.message : "The topology request failed.",
  }
}

export function workflowRefreshDelay(failureCount: number): number {
  if (!Number.isSafeInteger(failureCount) || failureCount < 0) {
    throw new RangeError("Workflow refresh failures must be a non-negative safe integer.")
  }
  return Math.min(WORKFLOW_REFRESH_MAX_MS, WORKFLOW_REFRESH_BASE_MS * 2 ** failureCount)
}

export function workflowShouldAutoRefresh({
  documentVisible,
  hasActiveNodes,
  requestPending,
}: {
  documentVisible: boolean
  hasActiveNodes: boolean
  requestPending: boolean
}): boolean {
  return documentVisible && hasActiveNodes && !requestPending
}
