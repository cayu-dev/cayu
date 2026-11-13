/**
 * Protected review's reads from client.js: `reviewApiFromCayuClient(await connect())` is
 * the `api` a `ProtectedReviewPart` takes.
 */
import type { ReviewApi } from "./protected-review.tsx"
import type { HumanReviewLongFieldView, HumanReviewReference, HumanReviewView } from "./types.ts"

/** The client.js methods protected review calls; a `CayuClient` from `connect()` has them. */
export interface CayuReviewClient {
  getHumanReview(
    sessionId: string,
    options: { purpose: string; signal?: AbortSignal },
  ): Promise<unknown>
  readHumanReviewLongField(
    sessionId: string,
    reference: HumanReviewReference,
    key: string,
    options?: { signal?: AbortSignal },
  ): Promise<unknown>
}

export function reviewApiFromCayuClient(client: CayuReviewClient): ReviewApi {
  return {
    inspect: async (sessionId, purpose, signal) =>
      (await client.getHumanReview(sessionId, { purpose, signal })) as HumanReviewView,
    longField: async (sessionId, reference, key, signal) =>
      (await client.readHumanReviewLongField(sessionId, reference, key, {
        signal,
      })) as HumanReviewLongFieldView,
  }
}
