/**
 * Settling an approval after the application was updated.
 *
 * A pending approval keeps the execution profile its run started with. When the
 * application changed underneath it, the server refuses the settle with 409
 * `profile_changed`. The same decision can be sent again with `profile_adoption`:
 * the server checks the paused call against the current tool and policy, and the
 * application's execution-profile policy decides whether to adopt the update.
 */

import type { InboxProfileAdoption } from "../../lib/generated/server-api"

export const PROFILE_ADOPTION_REASON = "Settled in the dashboard after the application was updated."

/** Whether a settle failed only because the application changed while the item waited. */
export function isProfileChangedConflict(error: unknown): boolean {
  if (!error || typeof error !== "object") return false
  const { status, detail } = error as { status?: unknown; detail?: unknown }
  return (
    status === 409 &&
    !!detail &&
    typeof detail === "object" &&
    (detail as { code?: unknown }).code === "profile_changed"
  )
}

/** The same decision again, asking to adopt the updated application. */
export function withProfileAdoption<T extends object>(
  body: T,
  idempotencyKey: string,
): T & { profile_adoption: InboxProfileAdoption } {
  return {
    ...body,
    profile_adoption: { idempotency_key: idempotencyKey, reason: PROFILE_ADOPTION_REASON },
  }
}
