import { z } from "zod";
import type { Requester } from "@kit/data";
import { session } from "../app/services";

/**
 * THE AGENT'S OWN. b2b's routes under /app (integrations/http/__init__.py) and Cayu's scoped
 * routes under /app/resources. The server decides who may see what; the app only shows what
 * it answers.
 */

export const SHARING = ["personal", "department", "organization"] as const;
export type Sharing = (typeof SHARING)[number];
export const ROLES = ["member", "admin"] as const;
export type Role = (typeof ROLES)[number];

export const membership = z.object({
  subject: z.string(),
  organization: z.string(),
  department: z.string().nullable(),
  role: z.enum(ROLES),
});
export type Membership = z.infer<typeof membership>;

export const me = z.object({ subject: z.string(), agent_name: z.string(), memberships: z.array(membership) });
export type Me = z.infer<typeof me>;

export const conversation = z.object({
  id: z.string(),
  status: z.string(),
  labels: z.record(z.string(), z.string()),
  metadata: z.record(z.string(), z.unknown()),
  last_activity_at: z.string().nullable(),
});
export type Conversation = z.infer<typeof conversation>;

const conversationPage = z.object({ sessions: z.array(conversation), next_cursor: z.string().nullable() });
export type ConversationPage = z.infer<typeof conversationPage>;

export const message = z.object({
  role: z.string(),
  content: z.array(z.object({ type: z.string(), text: z.string().optional() })),
});
export type Message = z.infer<typeof message>;

const transcriptPage = z.object({ records: z.array(message), next_offset: z.number().nullable() });

// Transcript records per request, and the largest page the server allows. The server refuses
// a page over that size rather than shortening it, so a refused page is asked for in halves.
const TRANSCRIPT_PAGE = 20;
const TRANSCRIPT_PAGE_BYTES = 4_194_304;

/** A failure the app shows as it is: the server's own words, and its status. */
export class RequestFailed extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * A reply that failed after it started: the message it answers is already in the
 * conversation, `sessionId`.
 */
export class ReplyFailed extends RequestFailed {
  readonly sessionId: string;
  constructor(status: number, message: string, sessionId: string) {
    super(status, message);
    this.sessionId = sessionId;
  }
}

// A member ID is the identity provider's subject and may contain '/'. A browser resolves '.'
// and '..' segments before sending a URL, so such an ID would reach another member: refuse it.
function memberPath(organization: string, subject: string): string {
  const segments = subject.split("/");
  if (segments.includes(".") || segments.includes("..")) {
    throw new RequestFailed(422, "A member ID has no '.' or '..' path segments.");
  }
  return `organizations/${encodeURIComponent(organization)}/members/${segments.map(encodeURIComponent).join("/")}`;
}

const isServerFailure = (error: unknown) => error instanceof Error && "kind" in error && (error.kind === "server" || error.kind === "unavailable");

export function b2bApi(request: Requester) {
  return {
    me: (signal?: AbortSignal) => request("me", { signal, schema: me }),
    demoPeople: (signal?: AbortSignal) => request("dev/people", { signal, schema: z.array(z.string()) }),
    demoSignIn: (subject: string) => request("dev/sign-in", { method: "POST", body: { subject } }),
    labels: (organization: string, sharing: Sharing) =>
      request("conversation-labels", { method: "POST", body: { organization, sharing }, schema: z.record(z.string(), z.string()) }),
    conversations: (organization: string | null, cursor: string | null, limit = 50, signal?: AbortSignal) =>
      request("resources/sessions/query", {
        method: "POST",
        body: { labels: organization ? { organization } : {}, limit, cursor },
        signal,
        schema: conversationPage,
        // A query sent as POST: it changes nothing.
        read: true,
      }),
    conversation: (id: string, signal?: AbortSignal) => request(`resources/sessions/${encodeURIComponent(id)}`, { signal, schema: conversation }),
    transcript: async (id: string, signal?: AbortSignal) => {
      // Read in pages, each well under the server's byte limit, until the transcript ends.
      const records: Message[] = [];
      let offset: number | null = 0;
      let limit = TRANSCRIPT_PAGE;
      while (offset !== null) {
        let page: z.infer<typeof transcriptPage>;
        try {
          page = await request(`resources/sessions/${encodeURIComponent(id)}/records/transcript`, {
            query: { offset, limit, max_bytes: TRANSCRIPT_PAGE_BYTES },
            signal,
            schema: transcriptPage,
          });
        } catch (caught) {
          if (!isServerFailure(caught) || limit === 1) throw caught;
          limit = Math.ceil(limit / 2);
          continue;
        }
        records.push(...page.records);
        offset = page.next_offset;
        limit = TRANSCRIPT_PAGE;
      }
      return records;
    },
    members: (organization: string, signal?: AbortSignal) =>
      request(`organizations/${encodeURIComponent(organization)}/members`, { signal, schema: z.array(membership) }),
    putMember: (organization: string, subject: string, department: string | null, role: Role) =>
      request(memberPath(organization, subject), { method: "PUT", body: { department, role } }),
    removeMember: (organization: string, subject: string) => request(memberPath(organization, subject), { method: "DELETE" }),
  };
}
export type B2bApi = ReturnType<typeof b2bApi>;

export function titleOf(item: Conversation): string {
  const title = item.metadata["cayu:title"];
  return typeof title === "string" && title ? title : "Untitled conversation";
}

/** The text a person reads: never the model's reasoning or tool calls. */
export function textOf(item: Message): string {
  return item.content
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("")
    .trim();
}

async function errorDetail(response: Response): Promise<string> {
  const fallback = `The request failed (${response.status}).`;
  try {
    const detail = (await response.json()).detail;
    if (typeof detail === "string" && detail) return detail;
    // A validation error lists what was wrong; the first item says enough.
    if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
    return fallback;
  } catch {
    return fallback;
  }
}

/**
 * Starts or continues a conversation and calls `onText` with the reply as it streams. Returns
 * the conversation's ID. A failure once the conversation is known, including a lost
 * connection, is a `ReplyFailed` naming it.
 *
 * The run lives in this response: stopping reading it stops the agent's reply.
 */
export async function stream(path: "resources/runs" | "resources/resume", body: unknown, onText: (text: string) => void): Promise<string> {
  const response = await fetch(`/app/${path}`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.status === 401) session.expire();
  if (!response.ok || response.body === null) {
    throw new RequestFailed(response.status, await errorDetail(response));
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let sessionId = "";
  let reply = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        if (frame.startsWith("event: access_revoked")) {
          throw new ReplyFailed(403, "Your access to this conversation ended.", sessionId);
        }
        const data = frame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("");
        if (!data) continue;
        const event = JSON.parse(data);
        sessionId = event.session_id ?? sessionId;
        if (event.type === "model.text.delta") {
          reply += event.payload?.delta ?? "";
          onText(reply);
        } else if (event.type === "session.failed") {
          throw new ReplyFailed(500, event.payload?.error ?? "The agent couldn't finish this reply.", sessionId);
        }
      }
    }
  } catch (caught) {
    if (caught instanceof ReplyFailed || !sessionId) throw caught;
    throw new ReplyFailed(0, "The connection was lost; open the conversation to see how far the reply got.", sessionId);
  } finally {
    // Stop reading a stream the app no longer follows.
    await reader.cancel().catch(() => undefined);
  }
  return sessionId;
}

/** What an error says to a person. */
export const messageOf = (caught: unknown) => (caught instanceof Error ? caught.message : String(caught));

/** The status of a failed request, from either kind of error. */
export const statusOf = (caught: unknown): number | undefined =>
  caught instanceof Error && "status" in caught && typeof caught.status === "number" ? caught.status : undefined;
