import type { QueryClient } from "@tanstack/react-query"
import type { SessionDetail } from "./api.ts"

const MAX_LABEL_KEY_LENGTH = 128
const MAX_LABEL_VALUE_LENGTH = 512
const RESERVED_LABEL_PREFIX = "cayu:"
const RUNTIME_METADATA_KEYS = new Set(["subagent"])
const RUNTIME_METADATA_PREFIX = "cayu:"

export type SessionEditValidation<T> = { ok: true; value: T } | { ok: false; message: string }
export type SessionMetadataOwnership = {
  userMetadata: Record<string, unknown>
  runtimeMetadata: Record<string, unknown>
}

function jsonObject(value: unknown): Record<string, unknown> | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function invalidDurableText(value: string): string | null {
  if (value.includes("\0")) return "must not contain NUL characters"
  for (const character of value) {
    const codePoint = character.codePointAt(0)
    if (codePoint !== undefined && codePoint >= 0xd800 && codePoint <= 0xdfff) {
      return "must not contain Unicode surrogate code points"
    }
  }
  return null
}

function unicodeLength(value: string): number {
  return Array.from(value).length
}

function isRuntimeMetadataKey(key: string): boolean {
  return RUNTIME_METADATA_KEYS.has(key) || key.startsWith(RUNTIME_METADATA_PREFIX)
}

export function splitSessionMetadata(metadata: Record<string, unknown>): SessionMetadataOwnership {
  const userEntries: Array<[string, unknown]> = []
  const runtimeEntries: Array<[string, unknown]> = []
  for (const entry of Object.entries(metadata)) {
    const destination = isRuntimeMetadataKey(entry[0]) ? runtimeEntries : userEntries
    destination.push(entry)
  }
  return {
    userMetadata: Object.fromEntries(userEntries),
    runtimeMetadata: Object.fromEntries(runtimeEntries),
  }
}

function invalidDurableJson(value: unknown, path = "$"): string | null {
  if (typeof value === "string") {
    const invalid = invalidDurableText(value)
    return invalid === null ? null : `${path} ${invalid}.`
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return `${path} must contain a finite JSON number.`
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      return `${path} contains an integer outside JavaScript's safe range. Represent exact large identifiers as strings.`
    }
    return null
  }
  if (value === null || typeof value === "boolean") return null
  if (Array.isArray(value)) {
    for (const [index, item] of value.entries()) {
      const invalid = invalidDurableJson(item, `${path}[${index}]`)
      if (invalid !== null) return invalid
    }
    return null
  }
  const object = jsonObject(value)
  if (object !== null) {
    for (const [key, item] of Object.entries(object)) {
      const invalidKey = invalidDurableText(key)
      if (invalidKey !== null) return `${path} key ${invalidKey}.`
      const invalid = invalidDurableJson(item, `${path}.${key}`)
      if (invalid !== null) return invalid
    }
    return null
  }
  return `${path} must contain a JSON-compatible value.`
}

function parseObjectDraft(
  draft: string,
  fieldName: "Labels" | "Metadata",
): SessionEditValidation<Record<string, unknown>> {
  let parsed: unknown
  try {
    parsed = JSON.parse(draft)
  } catch (error) {
    const detail = error instanceof SyntaxError ? error.message : "The value could not be parsed."
    return { ok: false, message: `${fieldName} must be valid JSON. ${detail}` }
  }
  const object = jsonObject(parsed)
  if (object === null) {
    return { ok: false, message: `${fieldName} must be a JSON object, such as {}.` }
  }
  const invalid = invalidDurableJson(object)
  if (invalid !== null) return { ok: false, message: `${fieldName} ${invalid}` }
  return { ok: true, value: object }
}

export function parseSessionLabelsDraft(
  draft: string,
): SessionEditValidation<Record<string, string>> {
  const parsed = parseObjectDraft(draft, "Labels")
  if (!parsed.ok) return parsed

  const labels: Array<[string, string]> = []
  for (const [key, value] of Object.entries(parsed.value)) {
    if (key.trim().length === 0) {
      return { ok: false, message: "Label keys cannot be blank." }
    }
    if (key !== key.trim()) {
      return {
        ok: false,
        message: `Label key ${JSON.stringify(key)} cannot start or end with whitespace.`,
      }
    }
    if (unicodeLength(key) > MAX_LABEL_KEY_LENGTH) {
      return {
        ok: false,
        message: `Label key ${JSON.stringify(key)} exceeds ${MAX_LABEL_KEY_LENGTH} characters.`,
      }
    }
    if (key.startsWith(RESERVED_LABEL_PREFIX)) {
      return {
        ok: false,
        message: `Label keys starting with ${JSON.stringify(RESERVED_LABEL_PREFIX)} are reserved for Cayu.`,
      }
    }
    if (typeof value !== "string") {
      return { ok: false, message: `Label ${JSON.stringify(key)} must have a string value.` }
    }
    if (value.trim().length === 0) {
      return { ok: false, message: `Label ${JSON.stringify(key)} cannot be blank.` }
    }
    if (value !== value.trim()) {
      return {
        ok: false,
        message: `Label ${JSON.stringify(key)} cannot start or end with whitespace.`,
      }
    }
    if (unicodeLength(value) > MAX_LABEL_VALUE_LENGTH) {
      return {
        ok: false,
        message: `Label ${JSON.stringify(key)} exceeds ${MAX_LABEL_VALUE_LENGTH} characters.`,
      }
    }
    labels.push([key, value])
  }
  return { ok: true, value: Object.fromEntries(labels) }
}

export function parseSessionMetadataDraft(
  draft: string,
): SessionEditValidation<Record<string, unknown>> {
  const parsed = parseObjectDraft(draft, "Metadata")
  if (!parsed.ok) return parsed
  for (const key of Object.keys(parsed.value)) {
    if (isRuntimeMetadataKey(key)) {
      return {
        ok: false,
        message: `Metadata key ${JSON.stringify(key)} is managed by Cayu and cannot be edited.`,
      }
    }
  }
  return parsed
}

export function formatSessionEditDraft(value: Record<string, unknown>): string {
  return JSON.stringify(value, null, 2)
}

function errorStatus(error: unknown): number | null {
  if (error === null || typeof error !== "object" || !("status" in error)) return null
  const status = error.status
  return typeof status === "number" ? status : null
}

export function sessionEditErrorMessage(error: unknown, fieldName: "labels" | "metadata"): string {
  const status = errorStatus(error)
  if (status === 401) return "Authentication is required to edit this session."
  if (status === 403) return "You do not have permission to edit this session."
  if (status === 404) return "This session no longer exists. Reload the page before editing."
  if (status === 422 && error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  if (status !== null && status >= 500) {
    return `The server could not save the session ${fieldName}. Your draft has been preserved.`
  }
  if (error instanceof TypeError) {
    return `The Cayu server could not be reached. Your ${fieldName} draft has been preserved.`
  }
  if (error instanceof Error && error.message.trim().length > 0) return error.message
  return `The session ${fieldName} could not be saved. Your draft has been preserved.`
}

export function applyConfirmedSessionEdit(queryClient: QueryClient, session: SessionDetail): void {
  queryClient.setQueryData(["session", session.id], session)
  void queryClient.invalidateQueries({ queryKey: ["session-state", session.id] })
  void queryClient.invalidateQueries({ queryKey: ["session-summary", session.id] })
  void queryClient.invalidateQueries({ queryKey: ["sessions-summary"] })
}
