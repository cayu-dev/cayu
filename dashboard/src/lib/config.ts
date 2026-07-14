import type { ModelCatalog, PricingCatalog } from "./generated/server-api"

export type DashboardConfig = {
  basePath: string
  apiBaseUrl: string
  pricingCatalog: PricingCatalog | ModelCatalog | null
}

declare global {
  interface Window {
    __CAYU_DASHBOARD_CONFIG__?: Partial<DashboardConfig>
  }
}

export const dashboardConfig: DashboardConfig = {
  basePath: normalizePublicPath(window.__CAYU_DASHBOARD_CONFIG__?.basePath ?? "/"),
  apiBaseUrl: normalizePublicUrl(window.__CAYU_DASHBOARD_CONFIG__?.apiBaseUrl ?? "/api"),
  pricingCatalog: window.__CAYU_DASHBOARD_CONFIG__?.pricingCatalog ?? null,
}

export function apiUrl(path: string): string {
  return joinPath(dashboardConfig.apiBaseUrl, path)
}

export function dashboardAsset(path: string): string {
  return joinPath(dashboardConfig.basePath, path)
}

function normalizePublicPath(path: string): string {
  const value = path.trim()
  if (!value || value === "/") return "/"
  return `/${value.replace(/^\/+|\/+$/g, "")}`
}

function normalizePublicUrl(url: string): string {
  const value = url.trim()
  if (!value || value === "/") return "/"
  if (/^[a-z][a-z\d+\-.]*:\/\//i.test(value)) return value.replace(/\/+$/g, "")
  return normalizePublicPath(value)
}

function joinPath(base: string, path: string): string {
  const suffix = path.replace(/^\/+/, "")
  if (!suffix) return base
  return base === "/" ? `/${suffix}` : `${base}/${suffix}`
}
