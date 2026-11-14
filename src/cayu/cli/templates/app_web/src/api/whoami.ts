import { z } from "zod";
import type { Requester } from "@kit/data";

/** GENERATED. `GET /app/whoami` (integrations/http/generated/whoami.py): who is signed in, and what their role allows. */
export const whoami = z.object({
  person: z.object({ subject: z.string(), name: z.string().nullable(), email: z.string().nullable() }),
  role: z.enum(["admin", "developer", "member"]),
  // Whether they may decide what waits in Cayu's inbox, and change the Agent's state or settings.
  may_decide: z.boolean(),
  may_change: z.boolean(),
});
export type Whoami = z.infer<typeof whoami>;

export function whoamiApi(request: Requester) {
  return { me: (signal?: AbortSignal) => request("whoami", { signal, schema: whoami }) };
}
