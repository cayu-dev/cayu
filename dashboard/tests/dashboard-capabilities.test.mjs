import assert from "node:assert/strict"
import test from "node:test"

import {
  DASHBOARD_ROUTE_REQUIREMENTS,
  dashboardCapabilityEnabled,
  dashboardCapabilityUnavailableText,
  resolveDashboardCapability,
} from "../src/lib/dashboard-capabilities.ts"

function operation(enabled, unavailableReason = null) {
  return {
    enabled,
    unavailable_reason: enabled ? null : unavailableReason,
  }
}

function capabilities() {
  return {
    cayu_version: "0.1.0",
    configured_store_roles: ["session"],
    actor: null,
    surfaces: {
      dashboard: {
        configured: true,
        read: operation(true),
        mutate: operation(false, "unsupported"),
      },
      tasks: {
        configured: false,
        read: operation(false, "not_configured"),
        mutate: operation(false, "not_configured"),
      },
      reviewed_knowledge: {
        configured: true,
        read: operation(true),
        mutate: operation(true),
      },
      artifacts: {
        configured: true,
        read: operation(true),
        mutate: operation(false, "unsupported"),
      },
      usage: {
        configured: true,
        read: operation(true),
        mutate: operation(false, "unsupported"),
      },
      pricing: {
        configured: false,
        read: operation(false, "not_configured"),
        mutate: operation(false, "unsupported"),
      },
      evaluation_promotion: {
        configured: true,
        read: operation(true),
        mutate: operation(false, "unsupported"),
      },
      evals: {
        configured: false,
        read: operation(false, "not_configured"),
        mutate: operation(false, "not_configured"),
      },
    },
    mutations: {
      session_execution: operation(true),
      session_interruption: operation(false, "unsupported"),
      provider_operation_resolution: operation(true),
      pending_action_resolution: operation(true),
      session_annotations: operation(true),
      task_lifecycle: operation(false, "not_configured"),
      knowledge_review: operation(true),
    },
  }
}

test("capability requirements resolve surface reads, surface mutations, and mutation families", () => {
  const snapshot = capabilities()

  assert.equal(
    resolveDashboardCapability(snapshot, { kind: "surface", surface: "artifacts" }).enabled,
    true,
  )
  assert.deepEqual(
    resolveDashboardCapability(snapshot, {
      kind: "surface",
      surface: "artifacts",
      operation: "mutate",
    }),
    operation(false, "unsupported"),
  )
  assert.equal(
    resolveDashboardCapability(snapshot, {
      kind: "surface",
      surface: "evaluation_promotion",
    }).enabled,
    true,
  )
  assert.equal(
    resolveDashboardCapability(snapshot, {
      kind: "surface",
      surface: "evaluation_promotion",
      operation: "mutate",
    }).enabled,
    false,
  )
  assert.deepEqual(
    resolveDashboardCapability(snapshot, {
      kind: "mutation",
      mutation: "session_interruption",
    }),
    operation(false, "unsupported"),
  )
})

test("navigation availability follows readable operations rather than configuration alone", () => {
  const snapshot = capabilities()

  assert.equal(
    dashboardCapabilityEnabled(snapshot, { kind: "surface", surface: "reviewed_knowledge" }),
    true,
  )
  assert.equal(dashboardCapabilityEnabled(snapshot, { kind: "surface", surface: "tasks" }), false)
  assert.equal(dashboardCapabilityEnabled(snapshot, { kind: "surface", surface: "pricing" }), false)
  assert.equal(dashboardCapabilityEnabled(snapshot, { kind: "surface", surface: "usage" }), true)
  assert.equal(
    dashboardCapabilityEnabled(snapshot, {
      kind: "mutation",
      mutation: "session_execution",
    }),
    true,
  )
})

test("a previous v4 response without a newer optional surface fails closed", () => {
  const snapshot = capabilities()

  assert.deepEqual(
    resolveDashboardCapability(snapshot, {
      kind: "surface",
      surface: "workflow",
    }),
    operation(false, "unsupported"),
  )
  assert.equal(
    dashboardCapabilityEnabled(snapshot, {
      kind: "surface",
      surface: "workflow",
    }),
    false,
  )
})

test("direct optional routes share the same server-authoritative requirements as navigation", () => {
  const snapshot = capabilities()

  assert.equal("/evals" in DASHBOARD_ROUTE_REQUIREMENTS, false)

  assert.equal(dashboardCapabilityEnabled(snapshot, DASHBOARD_ROUTE_REQUIREMENTS["/"]), true)
  assert.equal(dashboardCapabilityEnabled(snapshot, DASHBOARD_ROUTE_REQUIREMENTS["/tasks"]), false)
  assert.equal(dashboardCapabilityEnabled(snapshot, DASHBOARD_ROUTE_REQUIREMENTS["/usage"]), true)
  assert.equal(
    dashboardCapabilityEnabled(snapshot, DASHBOARD_ROUTE_REQUIREMENTS["/knowledge"]),
    true,
  )
  assert.equal(
    dashboardCapabilityEnabled(snapshot, DASHBOARD_ROUTE_REQUIREMENTS["/artifacts"]),
    true,
  )
  assert.equal(
    dashboardCapabilityEnabled(
      snapshot,
      DASHBOARD_ROUTE_REQUIREMENTS["/sessions/$sessionId/workflow"],
    ),
    false,
  )
  snapshot.surfaces.workflow = {
    configured: true,
    read: operation(true),
    mutate: operation(false, "unsupported"),
  }
  assert.equal(
    dashboardCapabilityEnabled(
      snapshot,
      DASHBOARD_ROUTE_REQUIREMENTS["/sessions/$sessionId/workflow"],
    ),
    true,
  )
  assert.equal(dashboardCapabilityEnabled(snapshot, DASHBOARD_ROUTE_REQUIREMENTS["/run"]), true)
})

test("unavailable explanations distinguish missing configuration from unsupported operations", () => {
  assert.equal(dashboardCapabilityUnavailableText(operation(true)), null)
  assert.equal(
    dashboardCapabilityUnavailableText(operation(false, "not_configured")),
    "This feature is not configured for this Cayu deployment.",
  )
  assert.equal(
    dashboardCapabilityUnavailableText(operation(false, "unsupported")),
    "This operation is not supported by this Cayu deployment.",
  )
})
