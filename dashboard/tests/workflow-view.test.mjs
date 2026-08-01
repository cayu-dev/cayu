import assert from "node:assert/strict"
import test from "node:test"

import { validateWorkflowSearch } from "../src/lib/workflow-search.ts"
import {
  buildWorkflowRenderIndex,
  WORKFLOW_REFRESH_BASE_MS,
  WORKFLOW_REFRESH_MAX_MS,
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
} from "../src/lib/workflow-view.ts"

class HttpError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

function sessionNode(id, parentSessionId = null, overrides = {}) {
  return {
    id,
    agent_name: "assistant",
    provider_name: "provider",
    model: "model",
    parent_session_id: parentSessionId,
    causal_budget_id: "budget-1",
    runtime_name: "cayu",
    runtime_version: "1",
    environment_name: "production",
    status: "completed",
    created_at: "2026-07-01T00:00:00Z",
    updated_at: "2026-07-01T00:01:00Z",
    last_activity_at: "2026-07-01T00:01:00Z",
    ...overrides,
  }
}

function taskNode(id, sessionId, parentTaskId = null, overrides = {}) {
  return {
    id,
    type: "review",
    title: `Task ${id}`,
    status: "completed",
    status_reason: null,
    session_id: sessionId,
    parent_task_id: parentTaskId,
    assigned_agent_name: "assistant",
    created_at: "2026-07-01T00:00:00Z",
    updated_at: "2026-07-01T00:01:00Z",
    truncated_fields: [],
    ...overrides,
  }
}

function branch(nodes) {
  return {
    nodes,
    nextCursor: null,
    hasMore: false,
    pageCount: 1,
    observedAt: "2026-07-01T00:02:00Z",
    oldestObservedAt: "2026-07-01T00:02:00Z",
    currentPageChainStartedAt: "2026-07-01T00:02:00Z",
    retainedNodeIds: [],
    mixedSnapshot: false,
  }
}

function topologyState() {
  const root = sessionNode("root")
  const focus = sessionNode("focus", "root", { status: "running" })
  const child = sessionNode("child", "focus", {
    agent_name: "researcher",
    environment_name: "sandbox",
    status: "failed",
  })
  const parentTask = taskNode("task-parent", "child", null, {
    assigned_agent_name: "researcher",
    status: "failed",
  })
  const childTask = taskNode("task-child", "child", "task-parent", {
    assigned_agent_name: "reviewer",
    status: "running",
  })
  return {
    focus,
    ancestors: [root],
    expandedSessionParents: new Map([[focus.id, focus]]),
    sessionBranches: new Map([[focus.id, branch([child])]]),
    taskStatus: "available",
    taskObservedAt: "2026-07-01T00:02:00Z",
    linkedTaskBranches: new Map([[child.id, branch([parentTask])]]),
    expandedTaskParents: new Map([[parentTask.id, parentTask]]),
    taskChildBranches: new Map([[parentTask.id, branch([childTask])]]),
    edges: [
      {
        kind: "session_parent",
        source_id: "focus",
        target_id: "root",
        target_loaded: true,
      },
      {
        kind: "session_parent",
        source_id: "child",
        target_id: "focus",
        target_loaded: true,
      },
      {
        kind: "task_session",
        source_id: "task-parent",
        target_id: "child",
        target_loaded: true,
      },
      {
        kind: "task_parent",
        source_id: "task-child",
        target_id: "task-parent",
        target_loaded: true,
      },
    ],
    observedAt: "2026-07-01T00:02:00Z",
    crossStoreAtomic: false,
  }
}

test("loaded-node filters retain structural ancestors without broadening direct matches", () => {
  const visibility = workflowNodeVisibility(
    topologyState(),
    validateWorkflowSearch({
      status: "running",
      node_type: "task",
      expanded_session_id: "child",
      expanded_task_id: "task-parent",
    }),
  )

  assert.deepEqual([...visibility.directMatches], [workflowNodeKey("task", "task-child")])
  assert.deepEqual(
    [...visibility.visibleNodes].sort(),
    [
      workflowNodeKey("session", "child"),
      workflowNodeKey("session", "focus"),
      workflowNodeKey("session", "root"),
      workflowNodeKey("task", "task-child"),
      workflowNodeKey("task", "task-parent"),
    ].sort(),
  )
})

test("session nodes identify their causal-budget relationship to the focus", () => {
  const focus = sessionNode("focus")

  assert.equal(workflowCausalBudgetRelationship(focus, focus), "focus")
  assert.equal(workflowCausalBudgetRelationship(sessionNode("shared"), focus), "shared")
  assert.equal(
    workflowCausalBudgetRelationship(
      sessionNode("different", null, { causal_budget_id: "budget-2" }),
      focus,
    ),
    "different",
  )

  const state = topologyState()
  state.sessionBranches.set(
    "focus",
    branch([
      sessionNode("shared-session", "focus"),
      sessionNode("different-session", "focus", { causal_budget_id: "budget-2" }),
    ]),
  )
  const index = buildWorkflowRenderIndex(state, validateWorkflowSearch({}))
  assert.equal(
    workflowTaskCausalBudgetRelationship(taskNode("shared-task", "shared-session"), state, index),
    "shared",
  )
  assert.equal(
    workflowTaskCausalBudgetRelationship(
      taskNode("different-task", "different-session"),
      state,
      index,
    ),
    "different",
  )
  assert.equal(
    workflowTaskCausalBudgetRelationship(taskNode("unlinked-task", null), state, index),
    "unlinked",
  )
  assert.equal(
    workflowTaskCausalBudgetRelationship(taskNode("unknown-task", "not-loaded"), state, index),
    "session_not_loaded",
  )
})

test("restored expansions attach through typed edges or remain explicitly unattached", () => {
  const state = topologyState()
  const pagedSession = sessionNode("paged-session", "focus")
  const restoredSession = sessionNode("restored-session", "focus")
  const unattachedSession = sessionNode("unattached-session", "missing-parent")
  state.sessionBranches.set("focus", branch([pagedSession]))
  state.expandedSessionParents = new Map([
    [state.focus.id, state.focus],
    [restoredSession.id, restoredSession],
    [unattachedSession.id, unattachedSession],
  ])
  state.edges.push(
    {
      kind: "session_parent",
      source_id: restoredSession.id,
      target_id: state.focus.id,
      target_loaded: true,
    },
    {
      kind: "session_parent",
      source_id: unattachedSession.id,
      target_id: "missing-parent",
      target_loaded: false,
    },
  )
  const sessionIndex = buildWorkflowRenderIndex(
    state,
    validateWorkflowSearch({
      expanded_session_id: [restoredSession.id, unattachedSession.id],
      expanded_task_id: "task-parent",
    }),
  )

  assert.deepEqual(
    workflowSessionChildrenForParent(sessionIndex, "focus").map((placement) => ({
      id: placement.node.id,
      restored: placement.restored,
    })),
    [
      { id: "paged-session", restored: false },
      { id: "restored-session", restored: true },
    ],
  )
  assert.deepEqual(
    workflowUnattachedExpandedSessions(sessionIndex).map((node) => node.id),
    ["unattached-session"],
  )

  const pagedTask = taskNode("paged-task", "focus")
  const restoredLinkedTask = taskNode("restored-linked-task", "focus")
  const restoredChildTask = taskNode("restored-child-task", "focus", "paged-task")
  const taskUnderUnattachedSession = taskNode("task-under-unattached-session", "unattached-session")
  const unattachedTask = taskNode("unattached-task", "missing-session")
  state.linkedTaskBranches.set("focus", branch([pagedTask]))
  state.linkedTaskBranches.set("unattached-session", branch([]))
  state.expandedTaskParents = new Map([
    [pagedTask.id, pagedTask],
    [restoredLinkedTask.id, restoredLinkedTask],
    [restoredChildTask.id, restoredChildTask],
    [taskUnderUnattachedSession.id, taskUnderUnattachedSession],
    [unattachedTask.id, unattachedTask],
  ])
  state.taskChildBranches.set(pagedTask.id, branch([]))
  state.edges.push(
    {
      kind: "task_session",
      source_id: restoredLinkedTask.id,
      target_id: "focus",
      target_loaded: true,
    },
    {
      kind: "task_parent",
      source_id: restoredChildTask.id,
      target_id: "paged-task",
      target_loaded: true,
    },
    {
      kind: "task_session",
      source_id: taskUnderUnattachedSession.id,
      target_id: "unattached-session",
      target_loaded: true,
    },
    {
      kind: "task_session",
      source_id: unattachedTask.id,
      target_id: "missing-session",
      target_loaded: false,
    },
  )
  const taskIndex = buildWorkflowRenderIndex(
    state,
    validateWorkflowSearch({
      expanded_session_id: [restoredSession.id, unattachedSession.id],
      expanded_task_id: [...state.expandedTaskParents.keys()],
    }),
  )

  assert.deepEqual(
    workflowTasksForSession(taskIndex, "focus").map((placement) => ({
      id: placement.node.id,
      restored: placement.restored,
    })),
    [
      { id: "paged-task", restored: false },
      { id: "restored-linked-task", restored: true },
    ],
  )
  assert.deepEqual(
    workflowTaskChildrenForParent(taskIndex, "paged-task").map((placement) => ({
      id: placement.node.id,
      restored: placement.restored,
    })),
    [{ id: "restored-child-task", restored: true }],
  )
  assert.deepEqual(
    workflowUnattachedExpandedTasks(taskIndex).map((node) => node.id),
    ["unattached-task"],
  )
})

test("unattached expansion components render only their structural roots", () => {
  const state = topologyState()
  const outsideParent = sessionNode("outside-parent", "missing-session")
  const outsideChild = sessionNode("outside-child", outsideParent.id)
  state.sessionBranches = new Map([
    [state.focus.id, branch([])],
    [outsideParent.id, branch([outsideChild])],
    [outsideChild.id, branch([])],
  ])
  state.expandedSessionParents = new Map([
    [state.focus.id, state.focus],
    [outsideParent.id, outsideParent],
    [outsideChild.id, outsideChild],
  ])
  state.linkedTaskBranches = new Map([
    [state.focus.id, branch([])],
    [outsideParent.id, branch([])],
    [outsideChild.id, branch([])],
  ])
  state.expandedTaskParents = new Map()
  state.taskChildBranches = new Map()
  state.edges = [
    {
      kind: "session_parent",
      source_id: outsideParent.id,
      target_id: "missing-session",
      target_loaded: false,
    },
    {
      kind: "session_parent",
      source_id: outsideChild.id,
      target_id: outsideParent.id,
      target_loaded: true,
    },
  ]
  const sessionIndex = buildWorkflowRenderIndex(
    state,
    validateWorkflowSearch({
      expanded_session_id: [outsideParent.id, outsideChild.id],
    }),
  )

  assert.deepEqual(
    workflowUnattachedExpandedSessions(sessionIndex).map((node) => node.id),
    [outsideParent.id],
  )
  assert.deepEqual(
    workflowSessionChildrenForParent(sessionIndex, outsideParent.id).map(
      (placement) => placement.node.id,
    ),
    [outsideChild.id],
  )

  const outsideTaskParent = taskNode("outside-task-parent", "missing-session")
  const outsideTaskChild = taskNode("outside-task-child", "missing-session", outsideTaskParent.id)
  state.expandedTaskParents = new Map([
    [outsideTaskParent.id, outsideTaskParent],
    [outsideTaskChild.id, outsideTaskChild],
  ])
  state.taskChildBranches = new Map([
    [outsideTaskParent.id, branch([outsideTaskChild])],
    [outsideTaskChild.id, branch([])],
  ])
  state.edges.push(
    {
      kind: "task_session",
      source_id: outsideTaskParent.id,
      target_id: "missing-session",
      target_loaded: false,
    },
    {
      kind: "task_parent",
      source_id: outsideTaskChild.id,
      target_id: outsideTaskParent.id,
      target_loaded: true,
    },
    {
      kind: "task_session",
      source_id: outsideTaskChild.id,
      target_id: "missing-session",
      target_loaded: false,
    },
  )
  const taskIndex = buildWorkflowRenderIndex(
    state,
    validateWorkflowSearch({
      expanded_session_id: [outsideParent.id, outsideChild.id],
      expanded_task_id: [outsideTaskParent.id, outsideTaskChild.id],
    }),
  )

  assert.deepEqual(
    workflowUnattachedExpandedTasks(taskIndex).map((node) => node.id),
    [outsideTaskParent.id],
  )
  assert.deepEqual(
    workflowTaskChildrenForParent(taskIndex, outsideTaskParent.id).map(
      (placement) => placement.node.id,
    ),
    [outsideTaskChild.id],
  )
})

test("expanded ancestors own off-path branches without duplicating the focus path", () => {
  const state = topologyState()
  const root = sessionNode("root")
  const middle = sessionNode("middle", root.id)
  const focus = sessionNode("focus", middle.id, { status: "running" })
  const rootSibling = sessionNode("root-sibling", root.id)
  const middleSibling = sessionNode("middle-sibling", middle.id)
  const focusChild = sessionNode("focus-child", focus.id)
  state.focus = focus
  state.ancestors = [root, middle]
  state.expandedSessionParents = new Map([
    [root.id, root],
    [middle.id, middle],
    [focus.id, focus],
  ])
  state.sessionBranches = new Map([
    [root.id, branch([middle, rootSibling])],
    [middle.id, branch([focus, middleSibling])],
    [focus.id, branch([focusChild])],
  ])
  state.edges = [
    {
      kind: "session_parent",
      source_id: middle.id,
      target_id: root.id,
      target_loaded: true,
    },
    {
      kind: "session_parent",
      source_id: focus.id,
      target_id: middle.id,
      target_loaded: true,
    },
    ...[rootSibling, middleSibling, focusChild].map((node) => ({
      kind: "session_parent",
      source_id: node.id,
      target_id: node.parent_session_id,
      target_loaded: true,
    })),
  ]
  const index = buildWorkflowRenderIndex(
    state,
    validateWorkflowSearch({ expanded_session_id: [root.id, middle.id] }),
  )

  assert.deepEqual([...index.focusPathSessionIds], [root.id, middle.id, focus.id])
  assert.deepEqual(
    workflowSessionSideChildrenForPathParent(index, root.id).map(({ node }) => node.id),
    [rootSibling.id],
  )
  assert.deepEqual(
    workflowSessionSideChildrenForPathParent(index, middle.id).map(({ node }) => node.id),
    [middleSibling.id],
  )
  assert.deepEqual(workflowUnattachedExpandedSessions(index), [])
})

test("task-parent placement wins over the task's secondary session link", () => {
  const state = topologyState()
  const parent = taskNode("parent", state.focus.id)
  const child = taskNode("child", state.focus.id, parent.id)
  state.linkedTaskBranches = new Map([[state.focus.id, branch([parent, child])]])
  state.expandedTaskParents = new Map([[parent.id, parent]])
  state.taskChildBranches = new Map([[parent.id, branch([child])]])
  state.edges = [
    {
      kind: "task_session",
      source_id: parent.id,
      target_id: state.focus.id,
      target_loaded: true,
    },
    {
      kind: "task_session",
      source_id: child.id,
      target_id: state.focus.id,
      target_loaded: true,
    },
    {
      kind: "task_parent",
      source_id: child.id,
      target_id: parent.id,
      target_loaded: true,
    },
  ]
  const index = buildWorkflowRenderIndex(
    state,
    validateWorkflowSearch({ expanded_task_id: parent.id }),
  )

  assert.deepEqual(
    workflowTasksForSession(index, state.focus.id).map((placement) => placement.node.id),
    [parent.id],
  )
  assert.deepEqual(
    workflowTaskChildrenForParent(index, parent.id).map((placement) => placement.node.id),
    [child.id],
  )
  assert.equal(index.parentByNode.get(workflowNodeKey("task", child.id)), "task\u0000parent")
})

test("render-index lookups never rescan the bounded edge collection", () => {
  const focus = sessionNode("session-0")
  const sessions = Array.from({ length: 499 }, (_, index) =>
    sessionNode(`session-${index + 1}`, focus.id),
  )
  const tasks = Array.from({ length: 500 }, (_, index) =>
    taskNode(`task-${index}`, `session-${index}`),
  )
  const edgeValues = [
    ...sessions.map((node) => ({
      kind: "session_parent",
      source_id: node.id,
      target_id: focus.id,
      target_loaded: true,
    })),
    ...tasks.flatMap((node, index) => [
      {
        kind: "task_session",
        source_id: node.id,
        target_id: node.session_id,
        target_loaded: true,
      },
      ...(index === 0
        ? []
        : [
            {
              kind: "task_parent",
              source_id: node.id,
              target_id: `task-${index - 1}`,
              target_loaded: true,
            },
          ]),
    ]),
  ]
  let edgeIterations = 0
  const state = {
    focus,
    ancestors: [],
    expandedSessionParents: new Map([[focus.id, focus]]),
    sessionBranches: new Map([[focus.id, branch(sessions)]]),
    taskStatus: "available",
    taskObservedAt: "2026-07-01T00:02:00Z",
    linkedTaskBranches: new Map([[focus.id, branch(tasks)]]),
    expandedTaskParents: new Map(),
    taskChildBranches: new Map(),
    edges: {
      [Symbol.iterator]() {
        edgeIterations += 1
        return edgeValues[Symbol.iterator]()
      },
    },
    observedAt: "2026-07-01T00:02:00Z",
    crossStoreAtomic: false,
  }
  const index = buildWorkflowRenderIndex(state, validateWorkflowSearch({}))
  assert.equal(edgeIterations, 1)

  for (const session of [focus, ...sessions]) {
    workflowSessionChildrenForParent(index, session.id)
  }
  for (const task of tasks) {
    workflowTasksForSession(index, task.session_id)
    workflowTaskChildrenForParent(index, task.id)
    workflowTaskCausalBudgetRelationship(task, state, index)
  }
  workflowUnattachedExpandedSessions(index)
  workflowUnattachedExpandedTasks(index)
  assert.equal(edgeIterations, 1)
})

test("environment filters match sessions only while agent filters include task assignees", () => {
  const state = topologyState()
  const environment = workflowNodeVisibility(
    state,
    validateWorkflowSearch({ environment_name: "sandbox" }),
  )
  assert.deepEqual([...environment.directMatches], [workflowNodeKey("session", "child")])
  assert.equal(environment.visibleNodes.has(workflowNodeKey("task", "task-parent")), false)

  const assignee = workflowNodeVisibility(
    state,
    validateWorkflowSearch({
      agent_name: "reviewer",
      expanded_session_id: "child",
      expanded_task_id: "task-parent",
    }),
  )
  assert.deepEqual([...assignee.directMatches], [workflowNodeKey("task", "task-child")])
})

test("an ancestor-only match retains the complete path to the focus session", () => {
  const state = topologyState()
  const root = sessionNode("root", null, { agent_name: "root-agent" })
  const middle = sessionNode("middle", "root", { agent_name: "middle-agent" })
  const focus = sessionNode("focus", "middle", { status: "running" })
  state.focus = focus
  state.ancestors = [root, middle]
  state.expandedSessionParents = new Map([[focus.id, focus]])
  state.edges = [
    {
      kind: "session_parent",
      source_id: "middle",
      target_id: "root",
      target_loaded: true,
    },
    {
      kind: "session_parent",
      source_id: "focus",
      target_id: "middle",
      target_loaded: true,
    },
    ...state.edges.filter((edge) => edge.kind !== "session_parent" || edge.source_id !== "focus"),
  ]

  const visibility = workflowNodeVisibility(
    state,
    validateWorkflowSearch({ agent_name: "root-agent" }),
  )

  assert.deepEqual([...visibility.directMatches], [workflowNodeKey("session", "root")])
  assert.equal(visibility.visibleNodes.has(workflowNodeKey("session", "middle")), true)
  assert.equal(visibility.visibleNodes.has(workflowNodeKey("session", "focus")), true)
})

test("clearing filters preserves branch expansion, focus collapse, and usage range", () => {
  const search = validateWorkflowSearch({
    status: ["running", "failed"],
    node_type: "session",
    agent_name: "assistant",
    environment_name: "production",
    expanded_session_id: ["child-b", "child-a"],
    expanded_task_id: "task-a",
    focus_collapsed: true,
    range: "custom",
    start_at: "2026-07-01T00:00:00.000001Z",
    end_at: "2026-07-02T00:00:00.000001Z",
  })
  assert.equal(workflowFilterCount(search), 5)

  assert.deepEqual(workflowSearchWithoutFilters(search), {
    status: undefined,
    agent_name: undefined,
    environment_name: undefined,
    node_type: undefined,
    expanded_session_id: ["child-a", "child-b"],
    expanded_task_id: ["task-a"],
    focus_collapsed: true,
    range: "custom",
    start_at: "2026-07-01T00:00:00.000001Z",
    end_at: "2026-07-02T00:00:00.000001Z",
    invalid: undefined,
  })
})

test("expansion and focus helpers produce canonical shareable state", () => {
  const initial = validateWorkflowSearch({
    status: "failed",
    expanded_session_id: "child-b",
    expanded_task_id: "task-a",
    focus_collapsed: true,
    range: "7d",
  })
  const expanded = workflowSearchWithExpansion(initial, "session", "child-a", true)
  assert.deepEqual(expanded.expanded_session_id, ["child-a", "child-b"])
  assert.deepEqual(
    workflowSearchWithExpansion(expanded, "session", "child-b", false).expanded_session_id,
    ["child-a"],
  )
  assert.equal(workflowSearchWithFocusCollapsed(initial, true).focus_collapsed, true)
  assert.equal(workflowSearchWithFocusCollapsed(initial, false).focus_collapsed, undefined)
  assert.deepEqual(workflowSearchForNewFocus(initial), {
    status: ["failed"],
    agent_name: undefined,
    environment_name: undefined,
    node_type: undefined,
    expanded_session_id: undefined,
    expanded_task_id: undefined,
    focus_collapsed: undefined,
    range: "7d",
    start_at: undefined,
    end_at: undefined,
    invalid: undefined,
  })
  assert.equal(
    workflowControlsKey(initial),
    workflowControlsKey(
      validateWorkflowSearch({ status: "failed", expanded_session_id: "other", range: "7d" }),
    ),
  )
  assert.equal(
    workflowTopologyShapeKey(initial),
    workflowTopologyShapeKey(
      validateWorkflowSearch({
        agent_name: "different-filter",
        expanded_session_id: "child-b",
        expanded_task_id: "task-a",
        range: "365d",
      }),
    ),
  )
})

test("topology errors have stable operator-facing classifications", () => {
  assert.equal(workflowTopologyError(new HttpError("missing", 404)).kind, "missing")
  assert.equal(workflowTopologyError(new HttpError("conflict", 409)).kind, "inconsistent")
  assert.equal(workflowTopologyError(new HttpError("large", 413)).kind, "oversized")
  assert.equal(workflowTopologyError(new HttpError("unsupported", 501)).kind, "unsupported")
  assert.deepEqual(workflowTopologyError(new Error("network unavailable")), {
    kind: "failed",
    title: "Workflow topology could not be loaded",
    detail: "network unavailable",
  })
})

test("active Workflow refresh is visibility-aware, single-flight, and exponentially bounded", () => {
  assert.equal(workflowRefreshDelay(0), WORKFLOW_REFRESH_BASE_MS)
  assert.equal(workflowRefreshDelay(1), WORKFLOW_REFRESH_BASE_MS * 2)
  assert.equal(workflowRefreshDelay(30), WORKFLOW_REFRESH_MAX_MS)
  assert.throws(() => workflowRefreshDelay(-1), /non-negative safe integer/)

  assert.equal(
    workflowShouldAutoRefresh({
      documentVisible: true,
      hasActiveNodes: true,
      requestPending: false,
    }),
    true,
  )
  assert.equal(
    workflowShouldAutoRefresh({
      documentVisible: false,
      hasActiveNodes: true,
      requestPending: false,
    }),
    false,
  )
  assert.equal(
    workflowShouldAutoRefresh({
      documentVisible: true,
      hasActiveNodes: false,
      requestPending: false,
    }),
    false,
  )
  assert.equal(
    workflowShouldAutoRefresh({
      documentVisible: true,
      hasActiveNodes: true,
      requestPending: true,
    }),
    false,
  )
})

test("terminal usage reconciliation queues once behind any active usage read", () => {
  const reconciler = new WorkflowTerminalUsageReconciler()

  assert.equal(reconciler.observe("session-a", false, true), false)
  assert.equal(reconciler.claim("session-a", false), false)
  assert.equal(reconciler.observe("session-a", true, true), false)
  assert.equal(reconciler.observe("session-a", false, true), true)
  assert.equal(reconciler.claim("session-a", true), false)
  assert.equal(reconciler.claim("session-a", false), true)
  assert.equal(reconciler.claim("session-a", false), false)
  assert.equal(reconciler.finish("session-a"), false)
  assert.equal(reconciler.claim("session-a", false), false)

  assert.equal(reconciler.observe("session-a", true, true), false)
  assert.equal(reconciler.observe("session-a", false, true), true)
  assert.equal(reconciler.claim("session-a", false), true)
  assert.equal(reconciler.observe("session-a", true, true), false)
  assert.equal(reconciler.observe("session-a", false, true), true)
  assert.equal(reconciler.claim("session-a", false), false)
  assert.equal(reconciler.finish("session-a"), true)
  assert.equal(reconciler.claim("session-a", false), true)
  assert.equal(reconciler.finish("session-a"), false)
  assert.equal(reconciler.observe("session-b", false, true), false)
  assert.equal(reconciler.claim("session-b", false), false)
})
