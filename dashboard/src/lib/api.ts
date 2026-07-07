import { apiUrl } from "./config"

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

export type KnowledgeEntry = {
  entry_id: string
  namespace: string
  kind: string
  visibility: string
  status: string
  title: string | null
  labels: Record<string, string>
  aspects: string[]
  impact_targets: string[]
  source_type: string | null
  source_uri: string | null
  source_id: string | null
  created_by_type: string
  created_by: string
  created_at: string
  updated_at: string
  importance: number | null
  importance_source: string | null
  confidence: number | null
  chunk_count?: number
  text_preview: string | null
}

export type KnowledgeEntryDetail = KnowledgeEntry & {
  text: string
  metadata: Record<string, unknown>
  expires_at: string | null
  chunks: KnowledgeChunk[]
  chunk_limit: number
  chunk_max_bytes: number
}

export type KnowledgeChunk = {
  chunk_id: string
  entry_id: string
  chunk_index: number
  text: string
  content_hash: string | null
  source_uri: string | null
  metadata: Record<string, unknown>
}

export type KnowledgePendingPage = {
  entries: KnowledgeEntry[]
  truncated: boolean
  limit: number
  max_bytes: number
  total_entries_known: number | null
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

async function postJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: "POST" })
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
  const page = await fetchJson<{ sessions: Session[] }>(apiUrl("/sessions"))
  return page.sessions
}

export async function fetchSession(id: string): Promise<SessionDetail> {
  return fetchJson<SessionDetail>(apiUrl(`/sessions/${id}`))
}

export async function fetchTasks(): Promise<Task[]> {
  return fetchJson<Task[]>(apiUrl("/tasks"))
}

export async function fetchPendingKnowledge(): Promise<KnowledgePendingPage> {
  return fetchJson<KnowledgePendingPage>(apiUrl("/knowledge/pending"))
}

export async function fetchPendingKnowledgeEntry(entryId: string): Promise<KnowledgeEntryDetail> {
  return fetchJson<KnowledgeEntryDetail>(
    apiUrl(`/knowledge/pending/${encodeURIComponent(entryId)}`),
  )
}

export async function approveKnowledge(entryId: string): Promise<KnowledgeEntry> {
  return postJson<KnowledgeEntry>(apiUrl(`/knowledge/${encodeURIComponent(entryId)}/approve`))
}

export async function rejectKnowledge(entryId: string): Promise<KnowledgeEntry> {
  return postJson<KnowledgeEntry>(apiUrl(`/knowledge/${encodeURIComponent(entryId)}/reject`))
}

export async function streamRun(
  prompt: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
) {
  const { fetchEventSource } = await import("@microsoft/fetch-event-source")
  try {
    await fetchEventSource(apiUrl("/run"), {
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
    await fetchEventSource(apiUrl("/resume"), {
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
