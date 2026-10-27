// The application's API under /app. The server decides who may see what; this interface only
// shows what it answers.

export type Sharing = "personal" | "department" | "organization";
export type Role = "member" | "admin";

export interface Membership {
  subject: string;
  organization: string;
  department: string | null;
  role: Role;
}

export interface Me {
  subject: string;
  agent_name: string;
  memberships: Membership[];
}

export interface Conversation {
  id: string;
  status: string;
  labels: Record<string, string>;
  metadata: Record<string, unknown>;
  last_activity_at: string | null;
}

export interface Message {
  role: string;
  content: { type: string; text?: string }[];
}

// Transcript records per request, and the largest page the server allows. The server refuses
// a page over that size rather than shortening it, so a refused page is asked for in halves.
const TRANSCRIPT_PAGE = 20;
const TRANSCRIPT_PAGE_BYTES = 4_194_304;

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// A reply that failed after it started: the message it answers is already in the
// conversation, `sessionId`.
export class ReplyFailed extends ApiError {
  readonly sessionId: string;

  constructor(status: number, message: string, sessionId: string) {
    super(status, message);
    this.sessionId = sessionId;
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/app/${path}`, {
    method,
    credentials: "same-origin",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await errorDetail(response));
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
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

// A member ID is the identity provider's subject and may contain '/'. A browser resolves '.'
// and '..' segments before sending a URL, so such an ID would reach another member: refuse it.
function memberPath(organization: string, subject: string): string {
  const segments = subject.split("/");
  if (segments.includes(".") || segments.includes("..")) {
    throw new ApiError(422, "A member ID has no '.' or '..' path segments.");
  }
  const member = segments.map(encodeURIComponent).join("/");
  return `organizations/${encodeURIComponent(organization)}/members/${member}`;
}

export interface ConversationPage {
  sessions: Conversation[];
  next_cursor: string | null;
}

export const api = {
  me: () => request<Me>("GET", "me"),
  demoPeople: () => request<string[]>("GET", "dev/people"),
  demoSignIn: (subject: string) => request<void>("POST", "dev/sign-in", { subject }),
  demoSignOut: () => request<void>("DELETE", "dev/sign-in"),
  labels: (organization: string, sharing: Sharing) =>
    request<Record<string, string>>("POST", "conversation-labels", { organization, sharing }),
  conversations: (organization: string, cursor: string | null = null) =>
    request<ConversationPage>("POST", "resources/sessions/query", {
      labels: { organization },
      limit: 50,
      cursor,
    }),
  conversation: (id: string) =>
    request<Conversation>("GET", `resources/sessions/${encodeURIComponent(id)}`),
  transcript: async (id: string) => {
    // Read in pages, each well under the server's byte limit, until the transcript ends.
    const records: Message[] = [];
    let offset: number | null = 0;
    let limit = TRANSCRIPT_PAGE;
    while (offset !== null) {
      let page: { records: Message[]; next_offset: number | null };
      try {
        page = await request(
          "GET",
          `resources/sessions/${encodeURIComponent(id)}/records/transcript?offset=${offset}&limit=${limit}&max_bytes=${TRANSCRIPT_PAGE_BYTES}`,
        );
      } catch (caught) {
        if (!(caught instanceof ApiError) || caught.status < 500 || limit === 1) throw caught;
        limit = Math.ceil(limit / 2);
        continue;
      }
      records.push(...page.records);
      offset = page.next_offset;
      limit = TRANSCRIPT_PAGE;
    }
    return records;
  },
  members: (organization: string) =>
    request<Membership[]>("GET", `organizations/${encodeURIComponent(organization)}/members`),
  putMember: async (organization: string, subject: string, department: string | null, role: Role) =>
    request<void>("PUT", memberPath(organization, subject), { department, role }),
  removeMember: async (organization: string, subject: string) =>
    request<void>("DELETE", memberPath(organization, subject)),
};

export function titleOf(conversation: Conversation): string {
  const title = conversation.metadata["cayu:title"];
  return typeof title === "string" && title ? title : "Untitled conversation";
}

// The text a person reads: never the model's reasoning or tool calls.
export function textOf(message: Message): string {
  return message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("")
    .trim();
}

// Starts or continues a conversation and calls `onText` with the reply as it streams. Returns
// the conversation's ID. A failure once the conversation is known, including a lost
// connection, is a `ReplyFailed` naming it.
//
// The run lives in this response: stopping reading it stops the agent's reply.
export async function stream(
  path: "resources/runs" | "resources/resume",
  body: unknown,
  onText: (text: string) => void,
): Promise<string> {
  const response = await fetch(`/app/${path}`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok || response.body === null) {
    throw new ApiError(response.status, await errorDetail(response));
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
          throw new ReplyFailed(
            500,
            event.payload?.error ?? "The agent couldn't finish this reply.",
            sessionId,
          );
        }
      }
    }
  } catch (caught) {
    if (caught instanceof ReplyFailed || !sessionId) throw caught;
    throw new ReplyFailed(
      0,
      "The connection was lost; open the conversation to see how far the reply got.",
      sessionId,
    );
  } finally {
    // Stop reading a stream the interface no longer follows.
    await reader.cancel().catch(() => undefined);
  }
  return sessionId;
}
