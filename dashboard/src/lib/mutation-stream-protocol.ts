import type { SseErrorEnvelope, SseEventEnvelope } from "./generated/server-api"

export const MUTATION_EVENT_ID_MAX_CHARS = 512

function objectRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function parseJsonObject(raw: string, description: string): Record<string, unknown> {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch (error) {
    throw new MutationStreamProtocolError(`${description} is not valid JSON.`, error)
  }
  const record = objectRecord(value)
  if (record === null) {
    throw new MutationStreamProtocolError(`${description} must be a JSON object.`)
  }
  return record
}

function nullableString(value: unknown): boolean {
  return value === null || typeof value === "string"
}

function optionalNullableString(value: unknown): boolean {
  return value === undefined || nullableString(value)
}

export function isMutationSseMarkerComponent(value: unknown): value is string {
  if (typeof value !== "string" || value.length === 0 || value !== value.trim()) return false
  for (const character of value) {
    const codePoint = character.codePointAt(0)
    if (codePoint !== undefined && (codePoint < 0x20 || codePoint === 0x7f)) return false
  }
  return true
}

export function isMutationSseEventId(value: unknown): value is string {
  if (!isMutationSseMarkerComponent(value)) return false
  let characters = 0
  for (const _character of value) {
    characters += 1
    if (characters > MUTATION_EVENT_ID_MAX_CHARS) return false
  }
  return true
}

export class MutationStreamProtocolError extends Error {
  readonly cause: unknown

  constructor(message: string, cause?: unknown) {
    super(message)
    this.name = "MutationStreamProtocolError"
    this.cause = cause
  }
}

export function assertMutationSseEvent(value: unknown): asserts value is SseEventEnvelope {
  const event = objectRecord(value)
  if (
    event === null ||
    !isMutationSseEventId(event.id) ||
    typeof event.type !== "string" ||
    event.type.length === 0 ||
    !isMutationSseMarkerComponent(event.session_id) ||
    typeof event.timestamp !== "string" ||
    event.timestamp.length === 0 ||
    !nullableString(event.agent_name) ||
    !nullableString(event.tool_name) ||
    !optionalNullableString(event.environment_name) ||
    !optionalNullableString(event.workflow_name) ||
    objectRecord(event.payload) === null
  ) {
    throw new MutationStreamProtocolError("Malformed SSE event from Cayu server.")
  }
}

export function parseMutationSseEvent(raw: string): SseEventEnvelope {
  const event = parseJsonObject(raw, "SSE event")
  assertMutationSseEvent(event)
  return event as SseEventEnvelope
}

export function parseMutationSseError(raw: string): SseErrorEnvelope {
  const error = parseJsonObject(raw, "SSE error")
  const validCode =
    error.code === "runtime_failed" ||
    error.code === "observer_lagged" ||
    error.code === "event_frame_too_large" ||
    error.code === "replay_idle_timeout"
  if (
    error.type !== "stream.error" ||
    (error.kind !== "runtime" && error.kind !== "observer") ||
    !validCode ||
    (error.kind === "runtime") !== (error.code === "runtime_failed") ||
    typeof error.error !== "string" ||
    error.error.length === 0 ||
    typeof error.error_type !== "string" ||
    error.error_type.length === 0 ||
    typeof error.retryable !== "boolean" ||
    (error.session_id !== null && !isMutationSseMarkerComponent(error.session_id))
  ) {
    throw new MutationStreamProtocolError("Malformed SSE error from Cayu server.")
  }
  return error as SseErrorEnvelope
}

export function expectedMutationSseId(event: SseEventEnvelope): string {
  return `${event.session_id}:${event.id}`
}
