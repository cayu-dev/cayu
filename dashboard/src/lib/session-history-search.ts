export const TRANSCRIPT_ROLE_FILTERS = ["user", "assistant", "system", "tool"] as const

export type TranscriptRoleFilter = (typeof TRANSCRIPT_ROLE_FILTERS)[number]

export type SessionHistorySearch = {
  event_id?: string
  event_type?: string
  tool_name?: string
  agent_name?: string
  environment_name?: string
  workflow_name?: string
  transcript_role?: TranscriptRoleFilter
  include_thinking?: false
}

function optionalSearchValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function transcriptRoleFilter(value: unknown): TranscriptRoleFilter | undefined {
  return typeof value === "string" &&
    TRANSCRIPT_ROLE_FILTERS.includes(value as TranscriptRoleFilter)
    ? (value as TranscriptRoleFilter)
    : undefined
}

export function validateSessionHistorySearch(
  search: Record<string, unknown>,
): SessionHistorySearch {
  const role = transcriptRoleFilter(search.transcript_role)
  const includeThinking = !(
    search.include_thinking === false ||
    search.include_thinking === "false" ||
    search.include_thinking === "0"
  )
  return {
    event_id: optionalSearchValue(search.event_id),
    event_type: optionalSearchValue(search.event_type),
    tool_name: optionalSearchValue(search.tool_name),
    agent_name: optionalSearchValue(search.agent_name),
    environment_name: optionalSearchValue(search.environment_name),
    workflow_name: optionalSearchValue(search.workflow_name),
    transcript_role: role,
    include_thinking: includeThinking ? undefined : false,
  }
}
