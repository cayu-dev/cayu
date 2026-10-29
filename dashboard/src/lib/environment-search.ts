/** Browser dot-segment normalization makes these two names unsafe as path params. */
export type EnvironmentSearch = { q?: string; name?: "." | ".." }

export function validateEnvironmentSearch(search: Record<string, unknown>): EnvironmentSearch {
  return {
    q: typeof search.q === "string" ? search.q : undefined,
    name: search.name === "." || search.name === ".." ? search.name : undefined,
  }
}
