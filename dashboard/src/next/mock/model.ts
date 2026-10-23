/**
 * The fake API's data: one app's records, in the runtime's own response shapes.
 * Seeds come from builder.ts (made-up apps) or from snapshots of real deployments passed
 * in through CAYU_PROTOTYPE_SEEDS, which stay outside the repository.
 */
import type { SessionUsageSummary } from "@/lib/generated/server-api"
import type {
  AgentsPage,
  AppInfo,
  InboxItem,
  KnowledgeEntry,
  Session,
  SessionEvent,
  SessionOverview,
  TaskItem,
  TaskType,
  TranscriptMessage,
} from "../api/types"

export type SeedSession = {
  session: Session
  metadata: Record<string, unknown>
  usage: SessionUsageSummary
  transcript: TranscriptMessage[]
  events: SessionEvent[]
  overview: SessionOverview
}

export type AppSeed = {
  id: string
  label: string
  description: string
  /** The seed's "now", so relative times read the same whenever it is opened. */
  now: string
  app: AppInfo
  specialists: AgentsPage
  sessions: SeedSession[]
  inbox: InboxItem[]
  tasks: TaskItem[]
  /** Inputs for seed tasks that have no session to read one from. */
  taskInputs?: Record<string, Record<string, unknown>>
  totalSessions?: number
  knowledge?: KnowledgeEntry[]
  taskTypes?: TaskType[]
}
