import { dashboardConfig } from "./config"

export type LinkQueryValue = string | number | boolean | null | undefined

export function dashboardPath(path: string, query: Record<string, LinkQueryValue> = {}): string {
  const basePath =
    dashboardConfig.basePath === "/" ? "" : dashboardConfig.basePath.replace(/\/+$/g, "")
  const normalizedPath = path.startsWith("/") ? path : `/${path}`
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue
    const serialized = String(value).trim()
    if (serialized === "") continue
    params.set(key, serialized)
  }
  const search = params.toString()
  return `${basePath}${normalizedPath}${search ? `?${search}` : ""}`
}

export function currentQueryParam(name: string): string {
  return new URLSearchParams(window.location.search).get(name)?.trim() ?? ""
}

export function replaceDashboardLocation(
  path: string,
  query: Record<string, LinkQueryValue> = {},
): void {
  window.history.replaceState(null, "", dashboardPath(path, query))
}
