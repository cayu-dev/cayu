/** Prototype-only adapter for unchanged pages that read their filters from the URL. */
import { dashboardConfig } from "../lib/config"
import { currentQueryParam as productionQueryParam } from "../lib/links"

export type { LinkQueryValue } from "../lib/links"
export { dashboardPath, replaceDashboardLocation } from "../lib/links"

export function currentQueryParam(name: string): string {
  // Vite's dev server also serves the ordinary dashboard; keep its query behavior.
  if (!dashboardConfig.basePath.endsWith("#")) return productionQueryParam(name)
  const hash = window.location.hash
  const query = hash.includes("?") ? hash.slice(hash.indexOf("?") + 1) : ""
  return new URLSearchParams(query).get(name)?.trim() ?? ""
}
