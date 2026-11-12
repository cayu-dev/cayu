import { z } from "zod";
import type { Requester } from "@kit/data";

/**
 * GENERATED. `GET /app/overview`. The generated route (integrations/http/generated/overview.py)
 * fills `inbox` from Cayu's inbox as the signed-in person, and `metrics` from the
 * Agent's `overview_metrics()` hook, the one Python function the Agent writes for it.
 * `inbox.unavailable` says why there are no counts: Cayu doesn't open its inbox to this
 * person, or couldn't count it.
 */
export const overview = z.object({
  as_of: z.string(),
  metrics: z.array(z.object({ label: z.string(), value: z.string(), description: z.string().nullable() })),
  inbox: z.object({
    total: z.number(),
    by_kind: z.array(z.object({ kind: z.string(), label: z.string(), count: z.number(), to: z.string().nullable() })),
    empty: z.string(),
    unavailable: z.string().nullable(),
  }),
});
export type Overview = z.infer<typeof overview>;

export function overviewApi(request: Requester) {
  return { get: (signal?: AbortSignal) => request("overview", { signal, schema: overview }) };
}
export type OverviewApi = ReturnType<typeof overviewApi>;
