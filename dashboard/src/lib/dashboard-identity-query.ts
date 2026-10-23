import { ApiClientError, fetchDashboardIdentity } from "./api.ts"
import type { AppInfoResponse } from "./generated/server-api"

export type DashboardIdentitySnapshot =
  | { identity: AppInfoResponse; error: null }
  | { identity: null; error: string }

/** Permanent rejection replaces cached authority; transport failures keep the last snapshot. */
export async function fetchDashboardIdentitySnapshot(): Promise<DashboardIdentitySnapshot> {
  try {
    return { identity: await fetchDashboardIdentity(), error: null }
  } catch (error) {
    if (
      error instanceof TypeError ||
      (error instanceof ApiClientError &&
        (error.status === 408 ||
          error.status === 429 ||
          (error.status >= 500 && error.status <= 599)))
    )
      throw error
    return {
      identity: null,
      error: error instanceof Error ? error.message : "Could not validate dashboard identity.",
    }
  }
}
