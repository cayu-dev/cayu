import { z } from "zod";
import { createRequester, type Requester } from "@kit/data";
import { session } from "../app/services";

/**
 * THE AGENT'S OWN. Conversations through Cayu's own API at /api, the one the Developer
 * dashboard uses. This app's people are the Agent's operators, who sign in to it as they do
 * to the dashboard, so Cayu checks each request and its role rules apply: every role reads,
 * and starting or continuing a conversation needs a role that may change things (`admin`,
 * `developer`). Nothing here is reimplemented on the Agent's side.
 */
export const cayuRequest: Requester = createRequester({ baseUrl: new URL("/api/", window.location.origin).href });

export const conversation = z.object({
  id: z.string(),
  status: z.string(),
  agent_name: z.string(),
  parent_session_id: z.string().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
});
export type Conversation = z.infer<typeof conversation>;

const conversationPage = z.object({ sessions: z.array(conversation), next_cursor: z.string().nullable() });
const displays = z.object({ items: z.array(z.object({ session_id: z.string(), title: z.string().nullable() })) });
const detail = conversation.extend({ metadata: z.record(z.string(), z.unknown()) });

export const message = z.object({
  role: z.string(),
  content: z.array(z.looseObject({ type: z.string(), text: z.string().optional() })),
});
export type Message = z.infer<typeof message>;

const transcriptPage = z.object({ messages: z.array(message), next_offset: z.number(), has_more: z.boolean() });

// Transcript messages per request.
const TRANSCRIPT_PAGE = 100;

/** A failure the app shows as it is: the server's own words, and its status. */
export class RequestFailed extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** A reply that failed after it started: the message it answers is already in the conversation, `sessionId`. */
export class ReplyFailed extends RequestFailed {
  readonly sessionId: string;
  constructor(status: number, message: string, sessionId: string) {
    super(status, message);
    this.sessionId = sessionId;
  }
}

export const conversations = {
  /** The agent's top-level conversations, newest activity first, with their titles. */
  list: async (agent: string, cursor: string | null, limit = 50, signal?: AbortSignal) => {
    const page = await cayuRequest("sessions", {
      query: { agent_name: agent, limit, cursor },
      signal,
      schema: conversationPage,
    });
    const rows = page.sessions.filter((item) => item.parent_session_id === null);
    const ids = rows.map((item) => `ids=${encodeURIComponent(item.id)}`).join("&");
    const titles = rows.length ? await cayuRequest(`sessions/display?${ids}`, { signal, schema: displays }) : { items: [] };
    const named = new Map(titles.items.map((item) => [item.session_id, item.title]));
    return { rows: rows.map((item) => ({ ...item, title: named.get(item.id) ?? null })), next_cursor: page.next_cursor };
  },
  conversation: async (id: string, signal?: AbortSignal) => {
    const found = await cayuRequest(`sessions/${encodeURIComponent(id)}`, { signal, schema: detail });
    const messages: Message[] = [];
    for (let offset = 0; ;) {
      const page = await cayuRequest(`sessions/${encodeURIComponent(id)}/transcript`, {
        query: { offset, limit: TRANSCRIPT_PAGE, include_thinking: false },
        signal,
        schema: transcriptPage,
      });
      messages.push(...page.messages);
      if (!page.has_more) break;
      offset = page.next_offset;
    }
    return { conversation: found, messages };
  },
};

export function titleOf(item: { title?: string | null; metadata?: Record<string, unknown> }): string {
  const title = item.title ?? item.metadata?.["cayu:title"];
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
 * Starts a conversation (`POST /api/run`) or continues one (`POST /api/resume`) and calls
 * `onText` with the reply as it streams. Returns the conversation's ID. A failure once the
 * conversation is known, including a lost connection, is a `ReplyFailed` naming it.
 *
 * The run lives in this response: stopping reading it stops the agent's reply.
 */
export async function stream(path: "run" | "resume", body: unknown, onText: (text: string) => void): Promise<string> {
  const response = await fetch(`/api/${path}`, {
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
      // Whole, so a line ending split across two reads still becomes one.
      buffer = (buffer + value).replaceAll("\r\n", "\n");
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
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
        } else if (event.type === "session.failed" || event.type === "stream.error") {
          throw new ReplyFailed(500, event.payload?.error ?? event.error ?? "The agent couldn't finish this reply.", sessionId);
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
