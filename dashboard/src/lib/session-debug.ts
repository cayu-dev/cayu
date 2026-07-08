import type { SessionEvent } from "./api"
import type { ApiEventRecord } from "./generated/server-api"

export type DebugEvent = Pick<
  ApiEventRecord | SessionEvent,
  "type" | "payload" | "tool_name" | "timestamp"
> & {
  id?: string
}

export type SessionDebugSummary = {
  kind: "failure" | "interruption" | "tool_issue"
  label: string
  detail: string
  eventType: string | null
  toolName: string | null
}

export function isFailureEventType(type: string): boolean {
  return (
    type === "session.failed" ||
    type === "tool.call.failed" ||
    type === "tool.call.blocked" ||
    type.endsWith(".failed") ||
    type.includes(".error")
  )
}

export function latestFailureEvent<T extends DebugEvent>(events: readonly T[]): T | null {
  for (const event of [...events].reverse()) {
    if (isFailureEventType(event.type)) return event
  }
  return null
}

export function summarizeFailureEvent(event: DebugEvent): SessionDebugSummary {
  const payload = objectPayload(event.payload)
  const errorType = optionalString(payload?.error_type)
  const error = optionalString(payload?.error)
  const message = optionalString(payload?.message)
  const reason = optionalString(payload?.reason)
  const detail = error || message || reason || "Inspect the event payload for details."

  if (event.type === "tool.call.failed") {
    const toolName = event.tool_name || optionalString(payload?.tool_name)
    return {
      kind: "tool_issue",
      label: toolName ? `Tool failed: ${toolName}` : "Tool call failed",
      detail,
      eventType: event.type,
      toolName,
    }
  }

  if (event.type === "tool.call.blocked") {
    const toolName = event.tool_name || optionalString(payload?.tool_name)
    return {
      kind: "tool_issue",
      label: toolName ? `Tool blocked: ${toolName}` : "Tool call blocked",
      detail,
      eventType: event.type,
      toolName,
    }
  }

  return {
    kind: "failure",
    label: errorType ? `Session failed: ${errorType}` : "Session failed",
    detail,
    eventType: event.type,
    toolName: event.tool_name || optionalString(payload?.tool_name),
  }
}

export function summarizeInterruptionEvent(event: DebugEvent): SessionDebugSummary | null {
  const payload = objectPayload(event.payload)
  const interruptionType = optionalString(payload?.interruption_type)
  if (interruptionType === "tool_approval_required") {
    const approval = objectPayload(payload?.approval)
    const toolName = optionalString(approval?.tool_name) || event.tool_name
    return {
      kind: "interruption",
      label: "Awaiting approval",
      detail: toolName ? `Tool approval required for ${toolName}.` : "Tool approval required.",
      eventType: event.type,
      toolName,
    }
  }
  if (interruptionType === "user_input_required") {
    const userInput = objectPayload(payload?.user_input)
    return {
      kind: "interruption",
      label: "Awaiting user input",
      detail: optionalString(userInput?.question) || "The session is waiting for an answer.",
      eventType: event.type,
      toolName: event.tool_name || optionalString(userInput?.tool_name),
    }
  }
  if (event.type === "session.awaiting_user_input") {
    return {
      kind: "interruption",
      label: "Awaiting user input",
      detail: optionalString(payload?.question) || "The session is waiting for an answer.",
      eventType: event.type,
      toolName: event.tool_name,
    }
  }
  return null
}

export function summarizeSessionDebugState({
  status,
  latestEvent,
  terminalEvent,
  countsByType = {},
}: {
  status: string
  latestEvent: DebugEvent | null | undefined
  terminalEvent?: DebugEvent | null
  countsByType?: Record<string, number>
}): SessionDebugSummary | null {
  const interruption = latestEvent ? summarizeInterruptionEvent(latestEvent) : null
  if (status === "interrupted" && interruption) return interruption

  const failureEvent =
    (terminalEvent && isFailureEventType(terminalEvent.type) ? terminalEvent : null) ||
    (latestEvent && isFailureEventType(latestEvent.type) ? latestEvent : null)
  if (failureEvent) return summarizeFailureEvent(failureEvent)

  if (status === "failed") {
    return {
      kind: "failure",
      label: "Failed session",
      detail: "Session status is failed. Inspect events for details.",
      eventType: null,
      toolName: null,
    }
  }

  if (status === "interrupted") {
    return {
      kind: "interruption",
      label: "Interrupted session",
      detail: "Session status is interrupted. Inspect events for pending action details.",
      eventType: null,
      toolName: null,
    }
  }

  const failedToolCalls = countsByType["tool.call.failed"] || 0
  const blockedToolCalls = countsByType["tool.call.blocked"] || 0
  const toolDebugCalls = failedToolCalls + blockedToolCalls
  if (toolDebugCalls > 0) {
    const label =
      failedToolCalls > 0
        ? failedToolCalls === 1
          ? "Tool failure recorded"
          : "Tool failures recorded"
        : blockedToolCalls === 1
          ? "Tool block recorded"
          : "Tool blocks recorded"
    const detail =
      failedToolCalls > 0
        ? failedToolCalls === 1
          ? "One tool call failed during this session."
          : `${failedToolCalls} tool calls failed during this session.`
        : blockedToolCalls === 1
          ? "One tool call was blocked during this session."
          : `${blockedToolCalls} tool calls were blocked during this session.`
    return {
      kind: "tool_issue",
      label,
      detail,
      eventType: failedToolCalls > 0 ? "tool.call.failed" : "tool.call.blocked",
      toolName: null,
    }
  }

  return null
}

export function objectPayload(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

export function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null
}
