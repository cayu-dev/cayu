export type Session = {
  id: string
  status: string
  agent_name: string
  created_at: string
  updated_at: string
}

export type SessionEvent = {
  id: string
  type: string
  agent_name: string | null
  tool_name: string | null
  payload: Record<string, unknown>
  timestamp: string
}

export type TranscriptMessage = {
  role: string
  content: Array<{ type: string; text?: string; tool_name?: string; [key: string]: unknown }>
}

export type SessionDetail = {
  session: Session
  events: SessionEvent[]
  transcript: TranscriptMessage[]
}

export type Task = {
  id: string
  type: string
  title: string | null
  status: string
  session_id: string | null
  created_at: string
  completed_at: string | null
}

export type SSEEvent = {
  id: string
  type: string
  session_id: string
  agent_name: string | null
  tool_name: string | null
  payload: Record<string, unknown>
  timestamp: string
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(await responseErrorMessage(res))
  }
  return res.json() as Promise<T>
}

async function responseErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    const body = (await response.json().catch(() => null)) as { detail?: unknown } | null
    if (typeof body?.detail === "string") {
      return body.detail
    }
  }
  return `Request failed with HTTP ${response.status}`
}

export async function fetchSessions(): Promise<Session[]> {
  // GET /api/sessions returns a paginated envelope; the dashboard shows the first page.
  const page = await fetchJson<{ sessions: Session[] }>("/api/sessions")
  return page.sessions
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return fetchJson<SessionDetail>(`/api/sessions/${id}`)
}

export async function fetchTasks(): Promise<Task[]> {
  return fetchJson<Task[]>("/api/tasks")
}

export async function streamRun(
  prompt: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  const { fetchEventSource } = await import("@microsoft/fetch-event-source")
  try {
    await fetchEventSource("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
      async onopen(response) {
        if (!response.ok) {
          throw new Error(`Run failed with HTTP ${response.status}`)
        }
      },
      onmessage(msg) {
        if (msg.data) onEvent(JSON.parse(msg.data))
      },
      onerror(error) {
        throw error
      },
    })
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  } finally {
    onDone()
  }
}

export async function streamResume(
  sessionId: string,
  prompt: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  const { fetchEventSource } = await import("@microsoft/fetch-event-source")
  try {
    await fetchEventSource("/api/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, prompt }),
      async onopen(response) {
        if (!response.ok) {
          throw new Error(`Resume failed with HTTP ${response.status}`)
        }
      },
      onmessage(msg) {
        if (msg.data) onEvent(JSON.parse(msg.data))
      },
      onerror(error) {
        throw error
      },
    })
  } catch (error) {
    onError(error instanceof Error ? error.message : String(error))
  } finally {
    onDone()
  }
}
