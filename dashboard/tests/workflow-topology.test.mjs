import assert from "node:assert/strict"
import test from "node:test"

import { parseDashboardSearch, stringifyDashboardSearch } from "../src/lib/search-params.ts"
import {
  validateWorkflowSearch,
  WORKFLOW_URL_MAX_BYTES,
  workflowSearchForUrl,
} from "../src/lib/workflow-search.ts"
import {
  buildWorkflowTopologyRefreshRequest,
  buildWorkflowTopologyRequest,
  buildWorkflowUsageRequest,
  LatestWorkflowRequestCoordinator,
  mergeWorkflowTopologyResponse,
  WORKFLOW_MAX_REQUEST_BYTES,
  workflowSessionNodes,
  workflowTaskNodes,
  workflowTopologyContainsActiveNodes,
  workflowTopologyContainsMixedSnapshots,
} from "../src/lib/workflow-topology.ts"

function fixedByteIdentifier(prefix, fill, byteLength = 1024) {
  assert.equal(new TextEncoder().encode(fill).byteLength, 1)
  return `${prefix}${fill.repeat(byteLength - prefix.length)}`
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
    environment_name: "local",
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

function topologyResponse({
  focus = sessionNode("focus", null, { status: "running" }),
  ancestors = [],
  expandedParents = [focus],
  branches = [
    {
      parent_session_id: focus.id,
      children: [sessionNode("child-1", focus.id), sessionNode("child-2", focus.id)],
      next_cursor: "session-page-2",
      has_more: true,
    },
  ],
  taskStatus = "available",
  taskSessionBranches = [
    {
      session_id: focus.id,
      tasks: [taskNode("task-1", focus.id, null, { status: "running" })],
      next_cursor: "task-page-2",
      has_more: true,
    },
  ],
  expandedTaskParents = [],
  taskChildBranches = [],
  edges,
  observedAt = "2026-07-01T00:02:00Z",
} = {}) {
  const available = taskStatus === "available"
  return {
    scope: "session_focus",
    observed_at: observedAt,
    cross_store_atomic: false,
    focus,
    ancestors,
    expanded_parents: expandedParents,
    branches,
    unique_node_count: new Set([
      focus.id,
      ...ancestors.map((node) => node.id),
      ...expandedParents.map((node) => node.id),
      ...branches.flatMap((branch) => branch.children.map((node) => node.id)),
    ]).size,
    task_projection: {
      status: taskStatus,
      observed_at: available ? observedAt : null,
      session_branches: available ? taskSessionBranches : [],
      expanded_parents: available ? expandedTaskParents : [],
      child_branches: available ? taskChildBranches : [],
      unique_node_count: available
        ? new Set([
            ...taskSessionBranches.flatMap((branch) => branch.tasks.map((node) => node.id)),
            ...expandedTaskParents.map((node) => node.id),
            ...taskChildBranches.flatMap((branch) => branch.children.map((node) => node.id)),
          ]).size
        : 0,
    },
    edges: edges ?? [
      {
        kind: "session_parent",
        source_id: "child-1",
        target_id: focus.id,
        target_loaded: true,
      },
      {
        kind: "task_session",
        source_id: "task-1",
        target_id: focus.id,
        target_loaded: true,
      },
    ],
  }
}

test("Workflow URL state is bounded, canonical, shareable, and cursor-free", () => {
  const search = validateWorkflowSearch(
    parseDashboardSearch(
      "?status=running&status=failed&node_type=task&node_type=session&agent_name=true&environment_name=null&expanded_session_id=child%2F2&expanded_session_id=child%2F1&expanded_task_id=task%2F1&range=7d",
    ),
  )
  assert.deepEqual(search, {
    status: ["running", "failed"],
    agent_name: "true",
    environment_name: "null",
    node_type: ["session", "task"],
    expanded_session_id: ["child/1", "child/2"],
    expanded_task_id: ["task/1"],
    focus_collapsed: undefined,
    range: "7d",
    start_at: undefined,
    end_at: undefined,
    invalid: undefined,
  })
  const serialized = stringifyDashboardSearch(workflowSearchForUrl(search))
  const params = new URLSearchParams(serialized)
  assert.deepEqual(params.getAll("expanded_session_id"), ["child/1", "child/2"])
  assert.equal(params.has("cursor"), false)
  assert.deepEqual(validateWorkflowSearch(parseDashboardSearch(serialized)), search)
})

test("invalid and over-limit Workflow URL values fail closed", () => {
  const invalidStatus = validateWorkflowSearch({ status: "not-a-status" })
  assert.equal(invalidStatus.invalid, true)
  assert.equal(invalidStatus.status, undefined)
  assert.throws(() => workflowSearchForUrl(invalidStatus), /cannot be serialized/)
  assert.throws(() => buildWorkflowTopologyRequest("focus", invalidStatus), /invalid or over-limit/)

  const serializedInvalid = stringifyDashboardSearch(invalidStatus)
  assert.equal(serializedInvalid, "?invalid=true")
  const invalidRoundTrip = validateWorkflowSearch(parseDashboardSearch(serializedInvalid))
  assert.equal(invalidRoundTrip.invalid, true)
  assert.throws(
    () => buildWorkflowTopologyRequest("focus", invalidRoundTrip),
    /invalid or over-limit/,
  )

  const allowedExpansions = Array.from({ length: 49 }, (_, index) => `session-${index}`)
  const allowed = validateWorkflowSearch({ expanded_session_id: allowedExpansions })
  assert.equal(allowed.invalid, undefined)
  assert.equal(buildWorkflowTopologyRequest("focus", allowed).expanded_parent_ids.length, 50)

  const expansions = Array.from({ length: 50 }, (_, index) => `session-${index}`)
  const overLimit = validateWorkflowSearch({ expanded_session_id: expansions })
  assert.equal(overLimit.invalid, true)
  assert.equal(overLimit.expanded_session_id.length, 49)
  assert.throws(() => buildWorkflowTopologyRequest("focus", overLimit), /invalid or over-limit/)

  const malformedWindow = validateWorkflowSearch({
    range: "custom",
    start_at: "not-rfc3339",
    end_at: "also-invalid",
  })
  assert.equal(malformedWindow.invalid, true)
  assert.equal(malformedWindow.start_at, undefined)
  assert.equal(malformedWindow.end_at, undefined)
  assert.throws(() => workflowSearchForUrl(malformedWindow), /cannot be serialized/)

  const oversizedFilter = validateWorkflowSearch({ agent_name: "a".repeat(1025) })
  assert.equal(oversizedFilter.invalid, true)
  assert.equal(oversizedFilter.agent_name, undefined)

  const ordinaryExpansions = validateWorkflowSearch({
    expanded_session_id: Array.from(
      { length: 49 },
      (_, index) => `0198f743-${String(index).padStart(4, "0")}-7000-8000-000000000000`,
    ),
  })
  assert.equal(ordinaryExpansions.invalid, undefined)
  assert.ok(
    new TextEncoder().encode(stringifyDashboardSearch(workflowSearchForUrl(ordinaryExpansions)))
      .byteLength <= WORKFLOW_URL_MAX_BYTES,
  )

  const oversizedAggregate = validateWorkflowSearch({
    expanded_task_id: Array.from({ length: 2 }, (_, index) =>
      fixedByteIdentifier(`task-${index}-`, '"'),
    ),
  })
  assert.deepEqual(oversizedAggregate, { invalid: true })
  assert.throws(() => workflowSearchForUrl(oversizedAggregate), /cannot be serialized/)
})

test("topology requests batch expansions and scope a single continuation", () => {
  const search = validateWorkflowSearch({
    expanded_session_id: ["child-1"],
    expanded_task_id: ["task-1"],
  })
  const initial = buildWorkflowTopologyRequest("focus", search)
  assert.deepEqual(initial, {
    ancestor_depth_limit: 32,
    child_limit: 25,
    expanded_parent_ids: ["focus", "child-1"],
    child_cursors: {},
    linked_task_session_ids: ["focus", "child-1"],
    task_session_cursors: {},
    expanded_task_parent_ids: ["task-1"],
    task_child_cursors: {},
    task_session_limit: 25,
    task_child_limit: 25,
    max_result_bytes: 4 * 1024 * 1024,
  })
  assert.deepEqual(
    buildWorkflowTopologyRequest("focus", search, {
      kind: "session_children",
      scopeId: "child-1",
      cursor: "opaque-session-cursor",
    }).child_cursors,
    { "child-1": "opaque-session-cursor" },
  )
  assert.deepEqual(
    buildWorkflowTopologyRequest("focus", search, {
      kind: "task_session",
      scopeId: "focus",
      cursor: "opaque-task-session-cursor",
    }).task_session_cursors,
    { focus: "opaque-task-session-cursor" },
  )
  assert.throws(
    () =>
      buildWorkflowTopologyRequest("focus", search, {
        kind: "task_children",
        scopeId: "not-expanded",
        cursor: "cursor",
      }),
    /expanded task branch/,
  )
  assert.throws(
    () =>
      buildWorkflowTopologyRequest("focus", search, {
        kind: "session_children",
        scopeId: "focus",
        cursor: "c".repeat(4097),
      }),
    /4096-byte limit/,
  )

  const quoteHeavySearch = {
    expanded_session_id: Array.from({ length: 49 }, (_, index) =>
      fixedByteIdentifier(`session-${String(index).padStart(2, "0")}-`, '"'),
    ),
    expanded_task_id: Array.from({ length: 50 }, (_, index) =>
      fixedByteIdentifier(`task-${String(index).padStart(2, "0")}-`, '"'),
    ),
  }
  assert.throws(
    () => buildWorkflowTopologyRequest("focus", quoteHeavySearch),
    new RegExp(`${WORKFLOW_MAX_REQUEST_BYTES}-byte limit`),
  )
})

test("latest Workflow request coordination aborts and rejects stale responses", async () => {
  const coordinator = new LatestWorkflowRequestCoordinator()
  let resolveFirst
  let visibleState = "initial"

  const firstTicket = coordinator.begin()
  const lateFirstResponse = new Promise((resolve) => {
    resolveFirst = resolve
  }).then((value) => coordinator.commit(firstTicket, () => (visibleState = value)))

  const secondTicket = coordinator.begin()
  assert.equal(firstTicket.signal.aborted, true)
  assert.equal(
    coordinator.commit(secondTicket, () => {
      visibleState = "new response"
    }),
    true,
  )

  resolveFirst("stale response")
  assert.equal(await lateFirstResponse, false)
  assert.equal(visibleState, "new response")

  coordinator.finish(secondTicket)
  assert.equal(
    coordinator.commit(secondTicket, () => {
      visibleState = "committed twice"
    }),
    false,
  )

  const cancelledTicket = coordinator.begin()
  coordinator.cancel()
  assert.equal(cancelledTicket.signal.aborted, true)
  assert.equal(
    coordinator.commit(cancelledTicket, () => {
      visibleState = "cancelled response"
    }),
    false,
  )
  assert.equal(visibleState, "new response")
})

test("Workflow usage requests preserve exact windows and use only the focus budget scope", () => {
  const startAt = "2026-07-01T00:00:00.000001Z"
  const endAt = "2026-07-02T08:30:00.123456+06:00"
  const request = buildWorkflowUsageRequest(
    "budget-1",
    validateWorkflowSearch({ range: "custom", start_at: startAt, end_at: endAt }),
  )
  assert.equal(request.start_at, startAt)
  assert.equal(request.end_at, endAt)
  assert.deepEqual(request.session_filter, { causal_budget_id: "budget-1" })
  assert.equal(request.session_group_limit, 100)
  assert.equal(request.pricing, null)
})

test("topology pages merge by branch without duplicating nodes or losing prior pages", () => {
  const search = validateWorkflowSearch({})
  const initialRequest = buildWorkflowTopologyRequest("focus", search)
  const initial = mergeWorkflowTopologyResponse(
    undefined,
    "focus",
    initialRequest,
    topologyResponse(),
  )
  assert.deepEqual(
    initial.sessionBranches.get("focus").nodes.map((node) => node.id),
    ["child-1", "child-2"],
  )
  assert.equal(initial.sessionBranches.get("focus").nextCursor, "session-page-2")
  assert.equal(initial.sessionBranches.get("focus").observedAt, "2026-07-01T00:02:00Z")
  assert.equal(initial.sessionBranches.get("focus").oldestObservedAt, "2026-07-01T00:02:00Z")
  assert.equal(
    initial.sessionBranches.get("focus").currentPageChainStartedAt,
    "2026-07-01T00:02:00Z",
  )
  assert.deepEqual(initial.sessionBranches.get("focus").retainedNodeIds, [])
  assert.equal(initial.sessionBranches.get("focus").mixedSnapshot, false)
  assert.equal(workflowTopologyContainsMixedSnapshots(initial), false)
  assert.equal(initial.taskStatus, "available")
  assert.deepEqual(
    workflowTaskNodes(initial).map((node) => node.id),
    ["task-1"],
  )

  const nextRequest = buildWorkflowTopologyRequest("focus", search, {
    kind: "session_children",
    scopeId: "focus",
    cursor: "session-page-2",
  })
  const next = mergeWorkflowTopologyResponse(
    initial,
    "focus",
    nextRequest,
    topologyResponse({
      branches: [
        {
          parent_session_id: "focus",
          children: [sessionNode("child-2", "focus"), sessionNode("child-3", "focus")],
          next_cursor: null,
          has_more: false,
        },
      ],
      observedAt: "2026-07-01T00:03:00Z",
    }),
  )
  assert.deepEqual(
    next.sessionBranches.get("focus").nodes.map((node) => node.id),
    ["child-1", "child-2", "child-3"],
  )
  assert.equal(next.sessionBranches.get("focus").pageCount, 2)
  assert.equal(next.sessionBranches.get("focus").hasMore, false)
  assert.equal(next.sessionBranches.get("focus").observedAt, "2026-07-01T00:03:00Z")
  assert.equal(next.sessionBranches.get("focus").oldestObservedAt, "2026-07-01T00:02:00Z")
  assert.equal(next.sessionBranches.get("focus").mixedSnapshot, true)
  assert.equal(workflowTopologyContainsMixedSnapshots(next), true)

  const refreshed = mergeWorkflowTopologyResponse(
    next,
    "focus",
    initialRequest,
    topologyResponse({
      branches: [
        {
          parent_session_id: "focus",
          children: [
            sessionNode("child-0", "focus"),
            sessionNode("child-1", "focus", { status: "failed" }),
          ],
          next_cursor: "new-first-page",
          has_more: true,
        },
      ],
      observedAt: "2026-07-01T00:04:00Z",
    }),
  )
  assert.deepEqual(
    refreshed.sessionBranches.get("focus").nodes.map((node) => node.id),
    ["child-0", "child-1", "child-2", "child-3"],
  )
  assert.equal(refreshed.sessionBranches.get("focus").nodes[1].status, "failed")
  assert.equal(refreshed.sessionBranches.get("focus").hasMore, true)
  assert.equal(refreshed.sessionBranches.get("focus").nextCursor, "new-first-page")
  assert.equal(refreshed.sessionBranches.get("focus").observedAt, "2026-07-01T00:04:00Z")
  assert.equal(refreshed.sessionBranches.get("focus").oldestObservedAt, "2026-07-01T00:02:00Z")
  assert.equal(refreshed.sessionBranches.get("focus").pageCount, 1)
  assert.equal(
    refreshed.sessionBranches.get("focus").currentPageChainStartedAt,
    "2026-07-01T00:04:00Z",
  )
  assert.deepEqual(refreshed.sessionBranches.get("focus").retainedNodeIds, ["child-2", "child-3"])
  assert.equal(refreshed.sessionBranches.get("focus").mixedSnapshot, true)

  const refreshedContinuationRequest = buildWorkflowTopologyRefreshRequest(
    "focus",
    search,
    refreshed,
  )
  assert.deepEqual(refreshedContinuationRequest.child_cursors, {
    focus: "new-first-page",
  })
  const reconciledTail = mergeWorkflowTopologyResponse(
    refreshed,
    "focus",
    refreshedContinuationRequest,
    topologyResponse({
      branches: [
        {
          parent_session_id: "focus",
          children: [sessionNode("child-3", "focus")],
          next_cursor: null,
          has_more: false,
        },
      ],
      observedAt: "2026-07-01T00:04:30Z",
    }),
  )
  assert.deepEqual(
    reconciledTail.sessionBranches.get("focus").nodes.map((node) => node.id),
    ["child-0", "child-1", "child-3"],
  )
  assert.equal(reconciledTail.sessionBranches.get("focus").pageCount, 2)
  assert.equal(reconciledTail.sessionBranches.get("focus").hasMore, false)
  assert.deepEqual(reconciledTail.sessionBranches.get("focus").retainedNodeIds, [])
  assert.equal(reconciledTail.sessionBranches.get("focus").oldestObservedAt, "2026-07-01T00:04:00Z")

  const completeRefresh = mergeWorkflowTopologyResponse(
    reconciledTail,
    "focus",
    initialRequest,
    topologyResponse({
      branches: [
        {
          parent_session_id: "focus",
          children: [sessionNode("child-0", "focus")],
          next_cursor: null,
          has_more: false,
        },
      ],
      observedAt: "2026-07-01T00:05:00Z",
    }),
  )
  assert.deepEqual(
    completeRefresh.sessionBranches.get("focus").nodes.map((node) => node.id),
    ["child-0"],
  )
  assert.equal(completeRefresh.sessionBranches.get("focus").pageCount, 1)
  assert.equal(completeRefresh.sessionBranches.get("focus").hasMore, false)
  assert.equal(completeRefresh.sessionBranches.get("focus").observedAt, "2026-07-01T00:05:00Z")
  assert.equal(
    completeRefresh.sessionBranches.get("focus").oldestObservedAt,
    "2026-07-01T00:05:00Z",
  )
  assert.equal(completeRefresh.sessionBranches.get("focus").mixedSnapshot, false)
  assert.equal(workflowTopologyContainsMixedSnapshots(completeRefresh), false)
})

test("routine Workflow refresh reconciles active rows retained beyond the first page", () => {
  const search = validateWorkflowSearch({})
  const firstPageRequest = buildWorkflowTopologyRequest("focus", search)
  const focus = sessionNode("focus")
  const firstPage = Array.from({ length: 25 }, (_, index) =>
    sessionNode(`child-${String(index).padStart(2, "0")}`, "focus"),
  )
  const initial = mergeWorkflowTopologyResponse(
    undefined,
    "focus",
    firstPageRequest,
    topologyResponse({
      focus,
      expandedParents: [focus],
      branches: [
        {
          parent_session_id: "focus",
          children: firstPage,
          next_cursor: "original-page-2",
          has_more: true,
        },
      ],
      taskStatus: "not_configured",
      edges: [],
    }),
  )
  const pageTwoRequest = buildWorkflowTopologyRequest("focus", search, {
    kind: "session_children",
    scopeId: "focus",
    cursor: "original-page-2",
  })
  const loaded = mergeWorkflowTopologyResponse(
    initial,
    "focus",
    pageTwoRequest,
    topologyResponse({
      focus,
      expandedParents: [focus],
      branches: [
        {
          parent_session_id: "focus",
          children: [sessionNode("active-page-2", "focus", { status: "running" })],
          next_cursor: null,
          has_more: false,
        },
      ],
      taskStatus: "not_configured",
      edges: [],
      observedAt: "2026-07-01T00:03:00Z",
    }),
  )
  assert.equal(workflowTopologyContainsActiveNodes(loaded), true)

  const refreshedFirstPage = mergeWorkflowTopologyResponse(
    loaded,
    "focus",
    firstPageRequest,
    topologyResponse({
      focus,
      expandedParents: [focus],
      branches: [
        {
          parent_session_id: "focus",
          children: firstPage,
          next_cursor: "refresh-page-2",
          has_more: true,
        },
      ],
      taskStatus: "not_configured",
      edges: [],
      observedAt: "2026-07-01T00:04:00Z",
    }),
  )
  assert.deepEqual(refreshedFirstPage.sessionBranches.get("focus").retainedNodeIds, [
    "active-page-2",
  ])

  const retainedPageRequest = buildWorkflowTopologyRefreshRequest(
    "focus",
    search,
    refreshedFirstPage,
  )
  assert.deepEqual(retainedPageRequest.child_cursors, { focus: "refresh-page-2" })
  const reconciled = mergeWorkflowTopologyResponse(
    refreshedFirstPage,
    "focus",
    retainedPageRequest,
    topologyResponse({
      focus,
      expandedParents: [focus],
      branches: [
        {
          parent_session_id: "focus",
          children: [sessionNode("active-page-2", "focus")],
          next_cursor: null,
          has_more: false,
        },
      ],
      taskStatus: "not_configured",
      edges: [],
      observedAt: "2026-07-01T00:05:00Z",
    }),
  )
  assert.deepEqual(reconciled.sessionBranches.get("focus").retainedNodeIds, [])
  assert.equal(workflowTopologyContainsActiveNodes(reconciled), false)
})

test("routine Workflow refresh batches retained cursors across branch kinds", () => {
  const search = validateWorkflowSearch({
    expanded_session_id: ["session-parent"],
    expanded_task_id: ["task-parent"],
  })
  const retainedPage = (cursor) => ({
    nodes: [],
    nextCursor: cursor,
    hasMore: true,
    pageCount: 1,
    observedAt: "2026-07-01T00:03:00Z",
    oldestObservedAt: "2026-07-01T00:02:00Z",
    currentPageChainStartedAt: "2026-07-01T00:03:00Z",
    retainedNodeIds: ["retained"],
    mixedSnapshot: true,
  })
  const request = buildWorkflowTopologyRefreshRequest("focus", search, {
    focus: sessionNode("focus"),
    sessionBranches: new Map([["focus", retainedPage("session-cursor")]]),
    linkedTaskBranches: new Map([["session-parent", retainedPage("task-session-cursor")]]),
    taskChildBranches: new Map([["task-parent", retainedPage("task-child-cursor")]]),
  })
  assert.deepEqual(request.child_cursors, { focus: "session-cursor" })
  assert.deepEqual(request.task_session_cursors, {
    "session-parent": "task-session-cursor",
  })
  assert.deepEqual(request.task_child_cursors, { "task-parent": "task-child-cursor" })
})

test("current session-model and task-attachment projections replace retained values", () => {
  const search = validateWorkflowSearch({ expanded_task_id: ["task-parent"] })
  const request = buildWorkflowTopologyRequest("focus", search)
  const taskParent = taskNode("task-parent", null)
  const initialTask = taskNode("task-child", "focus", "task-parent", { status: "pending" })
  const initial = mergeWorkflowTopologyResponse(
    undefined,
    "focus",
    request,
    topologyResponse({
      taskSessionBranches: [
        {
          session_id: "focus",
          tasks: [initialTask],
          next_cursor: "old-focus-task-page",
          has_more: true,
        },
      ],
      expandedTaskParents: [taskParent],
      taskChildBranches: [
        {
          parent_task_id: "task-parent",
          children: [initialTask],
          next_cursor: null,
          has_more: false,
        },
      ],
      edges: [
        {
          kind: "task_parent",
          source_id: "task-child",
          target_id: "task-parent",
          target_loaded: true,
        },
        {
          kind: "task_session",
          source_id: "task-child",
          target_id: "focus",
          target_loaded: true,
        },
      ],
    }),
  )

  const refreshedFocus = sessionNode("focus", null, { model: "model-v2", status: "running" })
  const refreshed = mergeWorkflowTopologyResponse(
    initial,
    "focus",
    request,
    topologyResponse({
      focus: refreshedFocus,
      expandedParents: [refreshedFocus],
      taskSessionBranches: [
        {
          session_id: "focus",
          tasks: [taskNode("task-other", "focus", null, { status: "running" })],
          next_cursor: "new-focus-task-page",
          has_more: true,
        },
      ],
      expandedTaskParents: [taskParent],
      taskChildBranches: [
        {
          parent_task_id: "task-parent",
          children: [
            taskNode("task-child", "worker-session", "task-parent", { status: "running" }),
          ],
          next_cursor: null,
          has_more: false,
        },
      ],
      edges: [
        {
          kind: "task_parent",
          source_id: "task-child",
          target_id: "task-parent",
          target_loaded: true,
        },
        {
          kind: "task_session",
          source_id: "task-child",
          target_id: "worker-session",
          target_loaded: false,
        },
        {
          kind: "task_session",
          source_id: "task-other",
          target_id: "focus",
          target_loaded: true,
        },
      ],
      observedAt: "2026-07-01T00:03:00Z",
    }),
  )
  assert.equal(refreshed.focus.model, "model-v2")
  assert.equal(refreshed.taskChildBranches.get("task-parent").nodes[0].session_id, "worker-session")
  assert.deepEqual(
    refreshed.linkedTaskBranches.get("focus").nodes.map((node) => node.id),
    ["task-other"],
  )
  assert.deepEqual(refreshed.linkedTaskBranches.get("focus").retainedNodeIds, [])
  assert.equal(refreshed.linkedTaskBranches.get("focus").mixedSnapshot, false)
  assert.deepEqual(
    refreshed.edges
      .filter((edge) => edge.kind === "task_session" && edge.source_id === "task-child")
      .map((edge) => edge.target_id),
    ["worker-session"],
  )
})

test("topology projection rejects mismatched focus, cursors, identities, and client overgrowth", () => {
  const search = validateWorkflowSearch({})
  const request = buildWorkflowTopologyRequest("focus", search)
  assert.throws(
    () =>
      mergeWorkflowTopologyResponse(
        undefined,
        "focus",
        request,
        topologyResponse({ focus: sessionNode("other") }),
      ),
    /focus session/,
  )

  const initial = mergeWorkflowTopologyResponse(undefined, "focus", request, topologyResponse())
  const wrongCursorRequest = buildWorkflowTopologyRequest("focus", search, {
    kind: "session_children",
    scopeId: "focus",
    cursor: "wrong-cursor",
  })
  assert.throws(
    () =>
      mergeWorkflowTopologyResponse(
        initial,
        "focus",
        wrongCursorRequest,
        topologyResponse({
          branches: [
            {
              parent_session_id: "focus",
              children: [],
              next_cursor: null,
              has_more: false,
            },
          ],
        }),
      ),
    /continuation does not match/,
  )

  const correctCursorRequest = buildWorkflowTopologyRequest("focus", search, {
    kind: "session_children",
    scopeId: "focus",
    cursor: "session-page-2",
  })
  assert.throws(
    () =>
      mergeWorkflowTopologyResponse(
        initial,
        "focus",
        correctCursorRequest,
        topologyResponse({
          branches: [
            {
              parent_session_id: "focus",
              children: [sessionNode("child-2", "changed-parent")],
              next_cursor: null,
              has_more: false,
            },
          ],
        }),
      ),
    /changed durable session identity|changed durable node identity/,
  )

  assert.throws(
    () =>
      mergeWorkflowTopologyResponse(
        undefined,
        "focus",
        request,
        topologyResponse({
          branches: [
            {
              parent_session_id: "focus",
              children: [sessionNode("focus", "different-parent")],
              next_cursor: null,
              has_more: false,
            },
          ],
        }),
      ),
    /durable session identity/,
  )

  const expandedIds = Array.from({ length: 20 }, (_, index) => `parent-${index}`)
  const largeSearch = validateWorkflowSearch({ expanded_session_id: expandedIds })
  const largeRequest = buildWorkflowTopologyRequest("focus", largeSearch)
  const parents = largeRequest.expanded_parent_ids.map((id) =>
    id === "focus" ? sessionNode(id) : sessionNode(id, "focus"),
  )
  const branches = largeRequest.expanded_parent_ids.map((parentId, parentIndex) => ({
    parent_session_id: parentId,
    children: Array.from({ length: 25 }, (_, childIndex) =>
      sessionNode(`child-${parentIndex}-${childIndex}`, parentId),
    ),
    next_cursor: null,
    has_more: false,
  }))
  assert.throws(
    () =>
      mergeWorkflowTopologyResponse(
        undefined,
        "focus",
        largeRequest,
        topologyResponse({
          expandedParents: parents,
          branches,
          taskSessionBranches: largeRequest.linked_task_session_ids.map((sessionId) => ({
            session_id: sessionId,
            tasks: [],
            next_cursor: null,
            has_more: false,
          })),
        }),
      ),
    /500 session nodes/,
  )

  const oversizedChildren = Array.from({ length: 25 }, (_, index) =>
    sessionNode(`large-${index}`, "focus", { agent_name: "a".repeat(180_000) }),
  )
  assert.throws(
    () =>
      mergeWorkflowTopologyResponse(
        undefined,
        "focus",
        request,
        topologyResponse({
          branches: [
            {
              parent_session_id: "focus",
              children: oversizedChildren,
              next_cursor: null,
              has_more: false,
            },
          ],
        }),
      ),
    /byte client limit/,
  )
})

test("task availability and active-state refresh policy remain explicit", () => {
  const request = buildWorkflowTopologyRequest("focus", validateWorkflowSearch({}))
  const unavailable = mergeWorkflowTopologyResponse(
    undefined,
    "focus",
    request,
    topologyResponse({ taskStatus: "not_configured" }),
  )
  assert.equal(unavailable.taskStatus, "not_configured")
  assert.deepEqual(workflowTaskNodes(unavailable), [])
  assert.equal(workflowTopologyContainsActiveNodes(unavailable), true)

  const terminal = mergeWorkflowTopologyResponse(
    undefined,
    "focus",
    request,
    topologyResponse({
      focus: sessionNode("focus"),
      taskStatus: "not_configured",
    }),
  )
  assert.equal(workflowTopologyContainsActiveNodes(terminal), false)
  assert.deepEqual(
    workflowSessionNodes(terminal).map((node) => node.id),
    ["focus", "child-1", "child-2"],
  )
})

test("generated-contract topology helper forwards encoded identity, body, and cancellation", async () => {
  const originalWindow = globalThis.window
  const originalFetch = globalThis.fetch
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
  const calls = []
  globalThis.fetch = async (input, init) => {
    calls.push({ input: String(input), init })
    return new Response(JSON.stringify(topologyResponse()), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  }
  const controller = new AbortController()
  const body = buildWorkflowTopologyRequest("focus/1", validateWorkflowSearch({}))
  try {
    const { fetchSessionTopology } = await import("../src/lib/api.ts")
    await fetchSessionTopology("focus/1", body, controller.signal)
  } finally {
    globalThis.fetch = originalFetch
    if (originalWindow === undefined) delete globalThis.window
    else globalThis.window = originalWindow
  }

  assert.equal(calls.length, 1)
  assert.equal(calls[0].input, "/api/sessions/focus%2F1/topology")
  assert.equal(calls[0].init.method, "POST")
  assert.equal(calls[0].init.signal, controller.signal)
  assert.deepEqual(JSON.parse(calls[0].init.body), body)
  assert.equal(new Headers(calls[0].init.headers).get("content-type"), "application/json")
  assert.equal(calls[0].init.credentials, "same-origin")
})
