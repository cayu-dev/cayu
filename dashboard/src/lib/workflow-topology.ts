import type {
  ApiExecutionTopologyEdge,
  ApiSessionTopologyBranch,
  ApiSessionTopologyNode,
  ApiTaskTopologyChildBranch,
  ApiTaskTopologyNode,
  ApiTaskTopologySessionBranch,
  PriceBook,
  SessionTopologyRequest,
  SessionTopologyResponse,
  UsageRollupRequest,
} from "./generated/server-api"
import { buildUsageRollupRequest } from "./usage-rollup.ts"
import { validateUsageRollupSearch } from "./usage-rollup-search.ts"
import {
  WORKFLOW_SESSION_EXPANSION_LIMIT,
  WORKFLOW_TASK_EXPANSION_LIMIT,
  type WorkflowSearch,
} from "./workflow-search.ts"

export const WORKFLOW_ANCESTOR_DEPTH_LIMIT = 32
export const WORKFLOW_BRANCH_PAGE_SIZE = 25
export const WORKFLOW_SESSION_GROUP_LIMIT = 100
export const WORKFLOW_MAX_SESSION_NODES = 500
export const WORKFLOW_MAX_TASK_NODES = 500
export const WORKFLOW_MAX_EDGES = 1500
export const WORKFLOW_MAX_RESULT_BYTES = 4 * 1024 * 1024
export const WORKFLOW_MAX_CLIENT_STATE_BYTES = 4 * 1024 * 1024
export const WORKFLOW_CURSOR_MAX_BYTES = 4096
export const WORKFLOW_MAX_REQUEST_BYTES = 256 * 1024
const WORKFLOW_IDENTIFIER_MAX_BYTES = 1024

export type WorkflowRequestTicket = Readonly<{
  generation: number
  signal: AbortSignal
}>

/**
 * Own one coordinator per independently authoritative Workflow read.
 *
 * Starting newer work aborts the previous request, while commit() also checks
 * object identity so a response that already escaped cancellation cannot
 * mutate current state.
 */
export class LatestWorkflowRequestCoordinator {
  private active: WorkflowRequestTicket | null = null
  private activeController: AbortController | null = null
  private generation = 0

  begin(): WorkflowRequestTicket {
    this.cancel()
    const controller = new AbortController()
    const ticket = Object.freeze({ generation: this.generation + 1, signal: controller.signal })
    this.generation = ticket.generation
    this.active = ticket
    this.activeController = controller
    return ticket
  }

  commit(ticket: WorkflowRequestTicket, apply: () => void): boolean {
    if (this.active !== ticket || ticket.signal.aborted) return false
    apply()
    return true
  }

  finish(ticket: WorkflowRequestTicket): void {
    if (this.active !== ticket) return
    this.active = null
    this.activeController = null
  }

  cancel(): void {
    this.activeController?.abort()
    this.active = null
    this.activeController = null
  }
}

export type WorkflowContinuation =
  | { kind: "session_children"; scopeId: string; cursor: string }
  | { kind: "task_session"; scopeId: string; cursor: string }
  | { kind: "task_children"; scopeId: string; cursor: string }

export type WorkflowBranchPage<T> = {
  nodes: readonly T[]
  nextCursor: string | null
  hasMore: boolean
  pageCount: number
  observedAt: string
  oldestObservedAt: string
  // A first-page refresh starts a new cursor chain while older loaded rows may
  // remain visible. These fields distinguish that retained tail from pages
  // already confirmed by the current chain.
  currentPageChainStartedAt: string
  retainedNodeIds: readonly string[]
  mixedSnapshot: boolean
}

export type WorkflowTopologyState = {
  focus: ApiSessionTopologyNode
  ancestors: readonly ApiSessionTopologyNode[]
  expandedSessionParents: ReadonlyMap<string, ApiSessionTopologyNode>
  sessionBranches: ReadonlyMap<string, WorkflowBranchPage<ApiSessionTopologyNode>>
  taskStatus: "available" | "not_configured" | "unsupported"
  taskObservedAt: string | null
  linkedTaskBranches: ReadonlyMap<string, WorkflowBranchPage<ApiTaskTopologyNode>>
  expandedTaskParents: ReadonlyMap<string, ApiTaskTopologyNode>
  taskChildBranches: ReadonlyMap<string, WorkflowBranchPage<ApiTaskTopologyNode>>
  edges: readonly ApiExecutionTopologyEdge[]
  observedAt: string
  crossStoreAtomic: false
}

type ServerBranch<T> = {
  key: string
  nodes: T[]
  nextCursor: string | null
  hasMore: boolean
}

function requireValidSearch(search: WorkflowSearch): void {
  if (search.invalid) {
    throw new Error("The Workflow URL contains invalid or over-limit state.")
  }
}

function hasUnsafeAuthorityText(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index)
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const next = value.charCodeAt(index + 1)
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true
      index += 1
      continue
    }
    if (
      (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) ||
      codeUnit <= 0x1f ||
      (codeUnit >= 0x7f && codeUnit <= 0x9f)
    ) {
      return true
    }
  }
  return false
}

function boundedAuthority(value: string, label: string, maxBytes: number): string {
  if (value.length > maxBytes) {
    throw new Error(`${label} exceeds its ${maxBytes}-byte limit.`)
  }
  if (value.trim() === "" || value.trim() !== value || hasUnsafeAuthorityText(value)) {
    throw new Error(`${label} must be a clean non-empty identifier.`)
  }
  if (new TextEncoder().encode(value).byteLength > maxBytes) {
    throw new Error(`${label} exceeds its ${maxBytes}-byte limit.`)
  }
  return value
}

function expandedSessionIds(focusSessionId: string, search: WorkflowSearch): string[] {
  const values = new Set<string>([
    boundedAuthority(focusSessionId, "Focus session ID", WORKFLOW_IDENTIFIER_MAX_BYTES),
  ])
  for (const value of search.expanded_session_id ?? []) {
    values.add(boundedAuthority(value, "Expanded session ID", WORKFLOW_IDENTIFIER_MAX_BYTES))
  }
  if (values.size > WORKFLOW_SESSION_EXPANSION_LIMIT + 1) {
    throw new Error(
      `A Workflow request cannot expand more than ${WORKFLOW_SESSION_EXPANSION_LIMIT} sessions in addition to its focus.`,
    )
  }
  return [...values]
}

function expandedTaskIds(search: WorkflowSearch): string[] {
  const values = (search.expanded_task_id ?? []).map((value) =>
    boundedAuthority(value, "Expanded task ID", WORKFLOW_IDENTIFIER_MAX_BYTES),
  )
  if (values.length > WORKFLOW_TASK_EXPANSION_LIMIT) {
    throw new Error(
      `A Workflow request cannot expand more than ${WORKFLOW_TASK_EXPANSION_LIMIT} tasks.`,
    )
  }
  return values
}

export function buildWorkflowTopologyRequest(
  focusSessionId: string,
  search: WorkflowSearch,
  continuation?: WorkflowContinuation,
): SessionTopologyRequest {
  requireValidSearch(search)
  const sessionIds = expandedSessionIds(focusSessionId, search)
  const taskIds = expandedTaskIds(search)
  const request: SessionTopologyRequest = {
    ancestor_depth_limit: WORKFLOW_ANCESTOR_DEPTH_LIMIT,
    child_limit: WORKFLOW_BRANCH_PAGE_SIZE,
    expanded_parent_ids: sessionIds,
    child_cursors: {},
    linked_task_session_ids: sessionIds,
    task_session_cursors: {},
    expanded_task_parent_ids: taskIds,
    task_child_cursors: {},
    task_session_limit: WORKFLOW_BRANCH_PAGE_SIZE,
    task_child_limit: WORKFLOW_BRANCH_PAGE_SIZE,
    max_result_bytes: WORKFLOW_MAX_RESULT_BYTES,
  }
  if (continuation !== undefined) {
    boundedAuthority(continuation.scopeId, "Continuation scope", WORKFLOW_IDENTIFIER_MAX_BYTES)
    boundedAuthority(continuation.cursor, "Continuation cursor", WORKFLOW_CURSOR_MAX_BYTES)
    if (continuation.kind === "session_children") {
      if (!sessionIds.includes(continuation.scopeId)) {
        throw new Error("A session continuation must belong to an expanded session branch.")
      }
      request.child_cursors = { [continuation.scopeId]: continuation.cursor }
    } else if (continuation.kind === "task_session") {
      if (!sessionIds.includes(continuation.scopeId)) {
        throw new Error("A task-session continuation must belong to a linked session branch.")
      }
      request.task_session_cursors = { [continuation.scopeId]: continuation.cursor }
    } else {
      if (!taskIds.includes(continuation.scopeId)) {
        throw new Error("A task continuation must belong to an expanded task branch.")
      }
      request.task_child_cursors = { [continuation.scopeId]: continuation.cursor }
    }
  }
  requireWorkflowRequestBound(request)
  return request
}

function requireWorkflowRequestBound(request: SessionTopologyRequest): void {
  if (new TextEncoder().encode(JSON.stringify(request)).byteLength > WORKFLOW_MAX_REQUEST_BYTES) {
    throw new Error(
      `The Workflow topology request exceeds the server's ${WORKFLOW_MAX_REQUEST_BYTES}-byte limit.`,
    )
  }
}

function retainedBranchCursors<T>(
  scopeIds: readonly string[],
  branches: ReadonlyMap<string, WorkflowBranchPage<T>>,
): Record<string, string> {
  const cursors: Record<string, string> = {}
  for (const scopeId of scopeIds) {
    const branch = branches.get(scopeId)
    if (
      branch === undefined ||
      branch.retainedNodeIds.length === 0 ||
      !branch.hasMore ||
      branch.nextCursor === null
    ) {
      continue
    }
    cursors[scopeId] = boundedAuthority(
      branch.nextCursor,
      "Refresh continuation cursor",
      WORKFLOW_CURSOR_MAX_BYTES,
    )
  }
  return cursors
}

/**
 * Continue every refreshed branch whose older loaded pages remain visible.
 *
 * A first-page refresh intentionally preserves already loaded later pages.
 * Subsequent refresh cycles must walk the new cursor chain so those retained
 * rows are eventually reconciled without issuing one request per branch.
 */
export function buildWorkflowTopologyRefreshRequest(
  focusSessionId: string,
  search: WorkflowSearch,
  state: WorkflowTopologyState,
): SessionTopologyRequest {
  if (state.focus.id !== focusSessionId) {
    throw new Error("The Workflow refresh state does not match the focus session.")
  }
  const request = buildWorkflowTopologyRequest(focusSessionId, search)
  request.child_cursors = retainedBranchCursors(
    request.expanded_parent_ids ?? [],
    state.sessionBranches,
  )
  request.task_session_cursors = retainedBranchCursors(
    request.linked_task_session_ids ?? [],
    state.linkedTaskBranches,
  )
  request.task_child_cursors = retainedBranchCursors(
    request.expanded_task_parent_ids ?? [],
    state.taskChildBranches,
  )
  requireWorkflowRequestBound(request)
  return request
}

export function buildWorkflowUsageRequest(
  causalBudgetId: string,
  search: WorkflowSearch,
  {
    now = new Date(),
    pricing = null,
    sessionGroupLimit = WORKFLOW_SESSION_GROUP_LIMIT,
  }: {
    now?: Date
    pricing?: PriceBook | null
    sessionGroupLimit?: number
  } = {},
): UsageRollupRequest {
  requireValidSearch(search)
  if (!Number.isInteger(sessionGroupLimit) || sessionGroupLimit < 1 || sessionGroupLimit > 100) {
    throw new Error("Workflow session-group limits must be integers from 1 through 100.")
  }
  const request = buildUsageRollupRequest(
    validateUsageRollupSearch({
      range: search.range,
      start_at: search.start_at,
      end_at: search.end_at,
    }),
    { now, pricing },
  )
  return {
    ...request,
    session_filter: {
      causal_budget_id: boundedAuthority(
        causalBudgetId,
        "Causal budget ID",
        WORKFLOW_IDENTIFIER_MAX_BYTES,
      ),
    },
    session_group_limit: sessionGroupLimit,
  }
}

function sessionStructure(node: ApiSessionTopologyNode): string {
  // Session.model is intentionally absent: ResumeRequest may update it while
  // the durable session and its topology identity remain unchanged.
  return JSON.stringify([
    node.id,
    node.parent_session_id,
    node.causal_budget_id,
    node.agent_name,
    node.provider_name,
    node.runtime_name,
    node.runtime_version,
    JSON.stringify(node.runtime_build_provenance),
    node.environment_name,
    node.created_at,
  ])
}

function taskStructure(node: ApiTaskTopologyNode): string {
  // A task may acquire or replace session_id when it starts. Parentage and
  // creation identity remain immutable across that lifecycle transition.
  return JSON.stringify([node.id, node.parent_task_id, node.created_at])
}

function mergeCanonicalNodes<T extends { id: string }>(
  values: Iterable<T>,
  incoming: Iterable<T>,
  structure: (value: T) => string,
  label: string,
): Map<string, T> {
  const result = new Map<string, T>()
  const merge = (value: T) => {
    const current = result.get(value.id)
    if (current !== undefined && structure(current) !== structure(value)) {
      throw new Error(`The Workflow response changed durable ${label} identity for ${value.id}.`)
    }
    result.set(value.id, value)
  }
  for (const value of values) merge(value)
  for (const value of incoming) merge(value)
  return result
}

function requireCursorShape(branch: ServerBranch<unknown>, label: string): void {
  if (branch.hasMore !== (branch.nextCursor !== null)) {
    throw new Error(`${label} continuation state is inconsistent.`)
  }
}

function mergeNodePage<T extends { id: string }>(
  previous: WorkflowBranchPage<T> | undefined,
  incoming: ServerBranch<T>,
  requestedCursor: string | undefined,
  observedAt: string,
  structure: (value: T) => string,
  label: string,
): WorkflowBranchPage<T> {
  const incomingNodes = incoming.nodes
  const incomingIds = new Set<string>()
  for (const node of incomingNodes) {
    if (incomingIds.has(node.id)) throw new Error(`${label} returned duplicate node identities.`)
    incomingIds.add(node.id)
    const loaded = previous?.nodes.find((candidate) => candidate.id === node.id)
    if (loaded !== undefined && structure(loaded) !== structure(node)) {
      throw new Error(`${label} changed durable node identity for ${node.id}.`)
    }
  }
  if (requestedCursor !== undefined) {
    if (
      previous === undefined ||
      !previous.hasMore ||
      previous.nextCursor === null ||
      previous.nextCursor !== requestedCursor
    ) {
      throw new Error("The Workflow continuation does not match the loaded branch boundary.")
    }
    const retainedIds = new Set(previous.retainedNodeIds)
    const confirmedNodes = previous.nodes.filter((node) => !retainedIds.has(node.id))
    const indexes = new Map(confirmedNodes.map((node, index) => [node.id, index]))
    for (const node of incomingNodes) {
      const existingIndex = indexes.get(node.id)
      if (existingIndex === undefined) {
        indexes.set(node.id, confirmedNodes.length)
        confirmedNodes.push(node)
      } else {
        confirmedNodes[existingIndex] = node
      }
      retainedIds.delete(node.id)
    }
    const retainedNodes = incoming.hasMore
      ? previous.nodes.filter((node) => retainedIds.has(node.id))
      : []
    return {
      nodes: [...confirmedNodes, ...retainedNodes],
      nextCursor: incoming.nextCursor,
      hasMore: incoming.hasMore,
      pageCount: previous.pageCount + 1,
      observedAt,
      oldestObservedAt:
        retainedNodes.length > 0 ? previous.oldestObservedAt : previous.currentPageChainStartedAt,
      currentPageChainStartedAt: previous.currentPageChainStartedAt,
      retainedNodeIds: retainedNodes.map((node) => node.id),
      mixedSnapshot: previous.nodes.length > 0 || previous.mixedSnapshot,
    }
  }
  if (previous === undefined || !incoming.hasMore) {
    return {
      nodes: [...incomingNodes],
      nextCursor: incoming.nextCursor,
      hasMore: incoming.hasMore,
      pageCount: 1,
      observedAt,
      oldestObservedAt: observedAt,
      currentPageChainStartedAt: observedAt,
      retainedNodeIds: [],
      mixedSnapshot: false,
    }
  }
  const prefixIds = new Set(incomingNodes.map((node) => node.id))
  const retainedTail = previous.nodes.filter((node) => !prefixIds.has(node.id))
  if (retainedTail.length === 0) {
    return {
      nodes: [...incomingNodes],
      nextCursor: incoming.nextCursor,
      hasMore: incoming.hasMore,
      pageCount: 1,
      observedAt,
      oldestObservedAt: observedAt,
      currentPageChainStartedAt: observedAt,
      retainedNodeIds: [],
      mixedSnapshot: false,
    }
  }
  return {
    nodes: [...incomingNodes, ...retainedTail],
    nextCursor: incoming.nextCursor,
    hasMore: incoming.hasMore,
    pageCount: 1,
    observedAt,
    oldestObservedAt: previous.oldestObservedAt,
    currentPageChainStartedAt: observedAt,
    retainedNodeIds: retainedTail.map((node) => node.id),
    mixedSnapshot: true,
  }
}

function mergeBranches<T extends { id: string }>(
  previous: ReadonlyMap<string, WorkflowBranchPage<T>> | undefined,
  responseBranches: readonly ServerBranch<T>[],
  expectedKeys: readonly string[],
  cursors: Record<string, string> | undefined,
  pageLimit: number,
  observedAt: string,
  label: string,
  structure: (value: T) => string,
): Map<string, WorkflowBranchPage<T>> {
  const expected = new Set(expectedKeys)
  const incoming = new Map<string, ServerBranch<T>>()
  for (const branch of responseBranches) {
    if (!expected.has(branch.key) || incoming.has(branch.key)) {
      throw new Error(`${label} returned an unexpected or duplicate branch.`)
    }
    if (branch.nodes.length > pageLimit) {
      throw new Error(`${label} exceeded the requested page limit.`)
    }
    requireCursorShape(branch, label)
    incoming.set(branch.key, branch)
  }
  if (incoming.size !== expected.size) {
    throw new Error(`${label} omitted a requested branch.`)
  }

  const result = new Map<string, WorkflowBranchPage<T>>()
  for (const key of expectedKeys) {
    const branch = incoming.get(key)!
    const merged = mergeNodePage(
      previous?.get(key),
      branch,
      cursors?.[key],
      observedAt,
      structure,
      label,
    )
    result.set(key, merged)
  }
  return result
}

function sessionBranches(
  response: SessionTopologyResponse,
): ServerBranch<ApiSessionTopologyNode>[] {
  return response.branches.map((branch: ApiSessionTopologyBranch) => ({
    key: branch.parent_session_id,
    nodes: branch.children,
    nextCursor: branch.next_cursor,
    hasMore: branch.has_more,
  }))
}

function taskSessionBranches(
  response: SessionTopologyResponse,
): ServerBranch<ApiTaskTopologyNode>[] {
  return response.task_projection.session_branches.map((branch: ApiTaskTopologySessionBranch) => ({
    key: branch.session_id,
    nodes: branch.tasks,
    nextCursor: branch.next_cursor,
    hasMore: branch.has_more,
  }))
}

function taskChildBranches(response: SessionTopologyResponse): ServerBranch<ApiTaskTopologyNode>[] {
  return response.task_projection.child_branches.map((branch: ApiTaskTopologyChildBranch) => ({
    key: branch.parent_task_id,
    nodes: branch.children,
    nextCursor: branch.next_cursor,
    hasMore: branch.has_more,
  }))
}

function setEquals(values: Iterable<string>, expected: readonly string[]): boolean {
  const actual = new Set(values)
  return actual.size === expected.length && expected.every((value) => actual.has(value))
}

function allSessionNodes(state: WorkflowTopologyState): ApiSessionTopologyNode[] {
  return [
    state.focus,
    ...state.ancestors,
    ...state.expandedSessionParents.values(),
    ...[...state.sessionBranches.values()].flatMap((branch) => branch.nodes),
  ]
}

function allTaskNodes(state: WorkflowTopologyState): ApiTaskTopologyNode[] {
  return [
    ...state.expandedTaskParents.values(),
    ...[...state.linkedTaskBranches.values()].flatMap((branch) => branch.nodes),
    ...[...state.taskChildBranches.values()].flatMap((branch) => branch.nodes),
  ]
}

function canonicalizeStateNodes(
  state: WorkflowTopologyState,
  authoritativeSessionNodes: readonly ApiSessionTopologyNode[],
  authoritativeTaskNodes: readonly ApiTaskTopologyNode[],
): WorkflowTopologyState {
  const sessionValues = allSessionNodes(state)
  const sessions = mergeCanonicalNodes(
    sessionValues,
    authoritativeSessionNodes,
    sessionStructure,
    "session",
  )
  const taskValues = allTaskNodes(state)
  const tasks = mergeCanonicalNodes(taskValues, authoritativeTaskNodes, taskStructure, "task")
  if (sessions.size > WORKFLOW_MAX_SESSION_NODES) {
    throw new Error(`The loaded Workflow exceeds ${WORKFLOW_MAX_SESSION_NODES} session nodes.`)
  }
  if (tasks.size > WORKFLOW_MAX_TASK_NODES) {
    throw new Error(`The loaded Workflow exceeds ${WORKFLOW_MAX_TASK_NODES} task nodes.`)
  }
  const canonicalSessionBranches = new Map(
    [...state.sessionBranches].map(([key, branch]) => [
      key,
      { ...branch, nodes: branch.nodes.map((node) => sessions.get(node.id)!) },
    ]),
  )
  const canonicalTaskBranch = (
    branches: ReadonlyMap<string, WorkflowBranchPage<ApiTaskTopologyNode>>,
  ) =>
    new Map(
      [...branches].map(([key, branch]) => [
        key,
        { ...branch, nodes: branch.nodes.map((node) => tasks.get(node.id)!) },
      ]),
    )
  const canonicalLinkedTaskBranches = new Map(
    [...state.linkedTaskBranches].map(([sessionId, branch]) => {
      const nodes = branch.nodes
        .map((node) => tasks.get(node.id)!)
        .filter((node) => node.session_id === sessionId)
      const nodeIds = new Set(nodes.map((node) => node.id))
      const retainedNodeIds = branch.retainedNodeIds.filter((nodeId) => nodeIds.has(nodeId))
      const hasRetainedNodes = retainedNodeIds.length > 0
      return [
        sessionId,
        {
          ...branch,
          nodes,
          oldestObservedAt: hasRetainedNodes
            ? branch.oldestObservedAt
            : branch.currentPageChainStartedAt,
          retainedNodeIds,
          mixedSnapshot: branch.pageCount > 1 || hasRetainedNodes,
        },
      ]
    }),
  )
  return {
    ...state,
    focus: sessions.get(state.focus.id)!,
    ancestors: state.ancestors.map((node) => sessions.get(node.id)!),
    expandedSessionParents: new Map(
      [...state.expandedSessionParents].map(([key, node]) => [key, sessions.get(node.id)!]),
    ),
    sessionBranches: canonicalSessionBranches,
    linkedTaskBranches: canonicalLinkedTaskBranches,
    expandedTaskParents: new Map(
      [...state.expandedTaskParents].map(([key, node]) => [key, tasks.get(node.id)!]),
    ),
    taskChildBranches: canonicalTaskBranch(state.taskChildBranches),
  }
}

function requireBoundedClientState(state: WorkflowTopologyState): void {
  const serializable = {
    focus: state.focus,
    ancestors: state.ancestors,
    expanded_session_parents: [...state.expandedSessionParents],
    session_branches: [...state.sessionBranches],
    task_status: state.taskStatus,
    task_observed_at: state.taskObservedAt,
    linked_task_branches: [...state.linkedTaskBranches],
    expanded_task_parents: [...state.expandedTaskParents],
    task_child_branches: [...state.taskChildBranches],
    edges: state.edges,
    observed_at: state.observedAt,
    cross_store_atomic: state.crossStoreAtomic,
  }
  if (
    new TextEncoder().encode(JSON.stringify(serializable)).byteLength >
    WORKFLOW_MAX_CLIENT_STATE_BYTES
  ) {
    throw new Error(
      `The loaded Workflow exceeds the ${WORKFLOW_MAX_CLIENT_STATE_BYTES}-byte client limit.`,
    )
  }
}

function mergeEdges(
  previous: readonly ApiExecutionTopologyEdge[] | undefined,
  incoming: readonly ApiExecutionTopologyEdge[],
  sessionIds: ReadonlySet<string>,
  taskIds: ReadonlySet<string>,
  authoritativeSessionIds: ReadonlySet<string>,
  authoritativeTaskIds: ReadonlySet<string>,
): ApiExecutionTopologyEdge[] {
  const byIdentity = new Map<string, ApiExecutionTopologyEdge>()
  const authority = new Map<string, string>()
  const retainedPrevious = (previous ?? []).filter((edge) => {
    const sourceIsAuthoritative =
      edge.kind === "session_parent"
        ? authoritativeSessionIds.has(edge.source_id)
        : authoritativeTaskIds.has(edge.source_id)
    return !sourceIsAuthoritative
  })
  for (const edge of [...retainedPrevious, ...incoming]) {
    const sourceLoaded =
      edge.kind === "session_parent" ? sessionIds.has(edge.source_id) : taskIds.has(edge.source_id)
    if (!sourceLoaded) continue
    const authorityKey = `${edge.kind}\u0000${edge.source_id}`
    const currentTarget = authority.get(authorityKey)
    if (currentTarget !== undefined && currentTarget !== edge.target_id) {
      throw new Error("The Workflow response changed durable edge authority.")
    }
    authority.set(authorityKey, edge.target_id)
    const key = `${authorityKey}\u0000${edge.target_id}`
    const targetLoaded =
      edge.kind === "task_parent" ? taskIds.has(edge.target_id) : sessionIds.has(edge.target_id)
    byIdentity.set(key, {
      ...edge,
      target_loaded: targetLoaded,
    })
  }
  if (byIdentity.size > WORKFLOW_MAX_EDGES) {
    throw new Error(`The loaded Workflow exceeds ${WORKFLOW_MAX_EDGES} edges.`)
  }
  return [...byIdentity.values()]
}

export function mergeWorkflowTopologyResponse(
  previous: WorkflowTopologyState | undefined,
  focusSessionId: string,
  request: SessionTopologyRequest,
  response: SessionTopologyResponse,
): WorkflowTopologyState {
  if (response.focus.id !== focusSessionId || (previous && previous.focus.id !== focusSessionId)) {
    throw new Error("The Workflow response does not match the requested focus session.")
  }
  if (response.cross_store_atomic !== false) {
    throw new Error("The Workflow response has an unsupported snapshot contract.")
  }
  if (previous && previous.focus.causal_budget_id !== response.focus.causal_budget_id) {
    throw new Error("The Workflow response changed the focus causal-budget identity.")
  }
  const responseSessionNodes = [
    response.focus,
    ...response.ancestors,
    ...response.expanded_parents,
    ...response.branches.flatMap((branch) => branch.children),
  ]
  const responseTaskNodes =
    response.task_projection.status === "available"
      ? [
          ...response.task_projection.expanded_parents,
          ...response.task_projection.session_branches.flatMap((branch) => branch.tasks),
          ...response.task_projection.child_branches.flatMap((branch) => branch.children),
        ]
      : []
  if (previous) {
    mergeCanonicalNodes(
      allSessionNodes(previous),
      responseSessionNodes,
      sessionStructure,
      "session",
    )
    if (response.task_projection.status === "available") {
      mergeCanonicalNodes(allTaskNodes(previous), responseTaskNodes, taskStructure, "task")
    }
  }
  const ancestorIds = response.ancestors.map((node) => node.id)
  if (
    previous &&
    previous.ancestors.map((node) => node.id).join("\u0000") !== ancestorIds.join("\u0000")
  ) {
    throw new Error("The Workflow response changed the durable ancestor path.")
  }

  const expectedSessionParents = request.expanded_parent_ids?.length
    ? request.expanded_parent_ids
    : [focusSessionId]
  if (
    !setEquals(
      response.expanded_parents.map((node) => node.id),
      expectedSessionParents,
    )
  ) {
    throw new Error("The Workflow response does not match the requested session expansions.")
  }
  const mergedSessionBranches = mergeBranches(
    previous?.sessionBranches,
    sessionBranches(response),
    expectedSessionParents,
    request.child_cursors,
    request.child_limit ?? WORKFLOW_BRANCH_PAGE_SIZE,
    response.observed_at,
    "Session topology",
    sessionStructure,
  )

  const expectedTaskSessions = request.linked_task_session_ids?.length
    ? request.linked_task_session_ids
    : [focusSessionId]
  const expectedTaskParents = request.expanded_task_parent_ids ?? []
  const taskAvailable = response.task_projection.status === "available"
  const taskObservedAt = response.task_projection.observed_at
  if (
    taskAvailable &&
    !setEquals(
      response.task_projection.expanded_parents.map((node) => node.id),
      expectedTaskParents,
    )
  ) {
    throw new Error("The Workflow response does not match the requested task expansions.")
  }
  let linkedTaskBranches: Map<string, WorkflowBranchPage<ApiTaskTopologyNode>>
  let taskChildren: Map<string, WorkflowBranchPage<ApiTaskTopologyNode>>
  if (taskAvailable) {
    if (taskObservedAt === null) {
      throw new Error("An available Workflow task projection requires an observation timestamp.")
    }
    linkedTaskBranches = mergeBranches(
      previous?.taskStatus === "available" ? previous.linkedTaskBranches : undefined,
      taskSessionBranches(response),
      expectedTaskSessions,
      request.task_session_cursors,
      request.task_session_limit ?? WORKFLOW_BRANCH_PAGE_SIZE,
      taskObservedAt,
      "Task session topology",
      taskStructure,
    )
    taskChildren = mergeBranches(
      previous?.taskStatus === "available" ? previous.taskChildBranches : undefined,
      taskChildBranches(response),
      expectedTaskParents,
      request.task_child_cursors,
      request.task_child_limit ?? WORKFLOW_BRANCH_PAGE_SIZE,
      taskObservedAt,
      "Task child topology",
      taskStructure,
    )
  } else {
    linkedTaskBranches = new Map()
    taskChildren = new Map()
  }

  const state = canonicalizeStateNodes(
    {
      focus: response.focus,
      ancestors: response.ancestors,
      expandedSessionParents: new Map(response.expanded_parents.map((node) => [node.id, node])),
      sessionBranches: mergedSessionBranches,
      taskStatus: response.task_projection.status,
      taskObservedAt,
      linkedTaskBranches,
      expandedTaskParents: taskAvailable
        ? new Map(response.task_projection.expanded_parents.map((node) => [node.id, node]))
        : new Map(),
      taskChildBranches: taskChildren,
      edges: [],
      observedAt: response.observed_at,
      crossStoreAtomic: false,
    },
    responseSessionNodes,
    responseTaskNodes,
  )
  const sessionIds = new Set(allSessionNodes(state).map((node) => node.id))
  const taskIds = new Set(allTaskNodes(state).map((node) => node.id))
  const authoritativeSessionIds = new Set(responseSessionNodes.map((node) => node.id))
  const authoritativeTaskIds = new Set(responseTaskNodes.map((node) => node.id))
  const merged = {
    ...state,
    edges: mergeEdges(
      previous?.edges,
      response.edges,
      sessionIds,
      taskIds,
      authoritativeSessionIds,
      authoritativeTaskIds,
    ),
  }
  requireBoundedClientState(merged)
  return merged
}

export function workflowSessionNodes(state: WorkflowTopologyState): ApiSessionTopologyNode[] {
  return [...new Map(allSessionNodes(state).map((node) => [node.id, node])).values()]
}

export function workflowTaskNodes(state: WorkflowTopologyState): ApiTaskTopologyNode[] {
  return [...new Map(allTaskNodes(state).map((node) => [node.id, node])).values()]
}

export function workflowTopologyContainsMixedSnapshots(state: WorkflowTopologyState): boolean {
  return [state.sessionBranches, state.linkedTaskBranches, state.taskChildBranches].some(
    (branches) => [...branches.values()].some((branch) => branch.mixedSnapshot),
  )
}

const TERMINAL_SESSION_STATUSES = new Set(["completed", "failed", "interrupted"])
const TERMINAL_TASK_STATUSES = new Set(["completed", "failed", "cancelled"])

export function workflowTopologyContainsActiveNodes(state: WorkflowTopologyState): boolean {
  return (
    workflowSessionNodes(state).some((node) => !TERMINAL_SESSION_STATUSES.has(node.status)) ||
    workflowTaskNodes(state).some((node) => !TERMINAL_TASK_STATUSES.has(node.status))
  )
}
