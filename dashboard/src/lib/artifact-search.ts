export type ArtifactSearch = {
  q?: string
  session_id?: string
  agent_name?: string
  environment_name?: string
  scope?: "session" | "environment"
  artifact?: string
  /** List scope; also identifies previews in existing direct bookmarks. */
  artifact_store_id?: string
  /** Exact preview identity, independent of list filters and pagination. */
  preview_store_id?: string
  offset?: string
}

export function validateArtifactSearch(search: Record<string, unknown>): ArtifactSearch {
  const text = (value: unknown) =>
    typeof value === "string" && value.trim() ? value.trim() : undefined
  const offset =
    typeof search.offset === "string" || typeof search.offset === "number"
      ? Number(search.offset)
      : Number.NaN
  return {
    q: text(search.q),
    session_id: text(search.session_id),
    agent_name: text(search.agent_name),
    environment_name: text(search.environment_name),
    artifact: text(search.artifact),
    artifact_store_id: text(search.artifact_store_id),
    preview_store_id: text(search.preview_store_id),
    scope: search.scope === "session" || search.scope === "environment" ? search.scope : undefined,
    offset: Number.isSafeInteger(offset) && offset > 0 ? String(offset) : undefined,
  }
}
