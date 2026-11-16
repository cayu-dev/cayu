import { z } from "zod";
import type { Requester } from "@kit/data";

/**
 * GENERATED. `/app/knowledge` (integrations/http/generated/knowledge.py): what the Agent knows,
 * as plain statements, what it asked to remember, and correcting or forgetting a statement.
 * The server decides who reads which statements and who may change each one (`may_change`).
 */
const source = z.object({
  kind: z.enum(["conversation", "document", "person", "agent", "app"]),
  label: z.string(),
  // The conversation it was learned in, when the person may open it from the app.
  conversation: z.string().nullable(),
});
export type KnowledgeSource = z.infer<typeof source>;

export const statement = z.object({
  id: z.string(),
  // What a correction or forget names, so it changes only what the person saw.
  version: z.number(),
  text: z.string(),
  source,
  learned_at: z.string(),
  corrected: z.object({ by: z.string(), at: z.string() }).nullable(),
  // Who it belongs to, such as an organization; null when the app has no such groups.
  group: z.string().nullable(),
  may_change: z.boolean(),
  // False when the text shown has a secret taken out, which a correction would lose.
  may_correct: z.boolean(),
});
export type Statement = z.infer<typeof statement>;

export const knowledge = z.object({ statements: z.array(statement), query: z.string().nullable(), more: z.boolean() });
export type Knowledge = z.infer<typeof knowledge>;

export const suggestion = z.object({
  id: z.string(),
  text: z.string(),
  source,
  suggested_at: z.string(),
  group: z.string().nullable(),
  // Where to keep or drop it: Cayu's Inbox, filtered to it. Null when the person can't open it.
  review: z.string().nullable(),
});
export type Suggestion = z.infer<typeof suggestion>;

export const suggestions = z.object({ suggestions: z.array(suggestion), more: z.boolean(), reviewed_by: z.string() });
export type Suggestions = z.infer<typeof suggestions>;

const changed = z.object({ id: z.string(), version: z.number(), status: z.enum(["corrected", "forgotten"]) });

export function knowledgeApi(request: Requester) {
  return {
    list: (q: string, limit: number, signal?: AbortSignal) => request("knowledge", { signal, query: { q, limit }, schema: knowledge }),
    suggestions: (signal?: AbortSignal) => request("knowledge/suggestions", { signal, schema: suggestions }),
    correct: (item: Statement, text: string) =>
      request("knowledge/correct", { method: "POST", body: { id: item.id, version: item.version, text }, schema: changed }),
    forget: (item: Statement, reason: string) =>
      request("knowledge/forget", { method: "POST", body: { id: item.id, version: item.version, reason: reason || null }, schema: changed }),
  };
}
export type KnowledgeApi = ReturnType<typeof knowledgeApi>;
