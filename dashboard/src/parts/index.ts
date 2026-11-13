/**
 * The Developer dashboard's pages that an Agent's own app can show too, as parts: each a
 * component plus exactly what its host injects (./host). The dashboard's own routes use
 * these same parts. This entry has what every part shares; each part has its own entry
 * (./inbox).
 *
 * Parts import their components through `@kit/ui`, `@kit/lib` and `@kit/icons`. The
 * dashboard maps `@kit` to src/kit (its own stock shadcn components); an app maps it to its
 * own kit, so a part renders with the app's components. A host's stylesheet must also let
 * Tailwind see the parts' files (`@source`), or their classes aren't generated.
 */
export {
  DASHBOARD_SOURCE_CAYU_VERSION as CAYU_VERSION,
  SUPPORTED_SERVER_CONTRACT_VERSION as SERVER_CONTRACT_VERSION,
} from "../lib/release-metadata.ts"
export { relativeTime } from "./format"
export type {
  InboxApi,
  InboxCountsQuery,
  InboxHost,
  InboxSearch,
  PartChange,
  PartLink,
  PartsHost,
  PartTarget,
  ProtectedReview,
  ProtectedReviewDecision,
} from "./host"
export { PartsHostProvider, usePartsHost } from "./host"
export { partKeys } from "./keys"
export { ProtectedReviewPart, type ReviewApi } from "./protected-review"
export { type CayuReviewClient, reviewApiFromCayuClient } from "./review-api"
export { EmptyState, ErrorState } from "./states"
export type * from "./types"
