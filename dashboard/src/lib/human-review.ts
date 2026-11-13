/**
 * Protected human review over the dashboard's `/api`: the `ReviewApi` its
 * `ProtectedReviewPart` reads through. Responses are protected material: never cached.
 */
import type { ReviewApi } from "../parts"
import { throwResponseError } from "./api"
import { apiUrl } from "./config"
import type { HumanReviewLongFieldView, HumanReviewView } from "./generated/server-api"

export const dashboardReviewApi: ReviewApi = {
  async inspect(sessionId, purpose, signal) {
    const response = await fetch(
      apiUrl(
        `/sessions/${encodeURIComponent(sessionId)}/human-review?purpose=${encodeURIComponent(purpose)}`,
      ),
      { credentials: "include", cache: "no-store", signal },
    )
    if (!response.ok) await throwResponseError(response)
    return (await response.json()) as HumanReviewView
  },
  async longField(sessionId, reference, key, signal) {
    const response = await fetch(
      apiUrl(`/sessions/${encodeURIComponent(sessionId)}/human-review/long-field`),
      {
        method: "POST",
        credentials: "include",
        cache: "no-store",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ review_reference: reference, key }),
        signal,
      },
    )
    if (!response.ok) await throwResponseError(response)
    return (await response.json()) as HumanReviewLongFieldView
  },
}
