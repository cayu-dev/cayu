/** A freshly scaffolded app: one specialist and no sessions yet, for the empty states. */
import { specialist, tool } from "../builder"
import type { AppSeed } from "../model"

const NOW = Date.parse("2026-10-21T23:30:00Z")
const ctx = { now: NOW, runtime: "0.13.0", provider: "anthropic", model: "claude-sonnet-5-5" }
const helper = specialist(ctx, "assistant", "The specialist `cayu new` generates", [
  tool("read_file", "Read a file in the workspace."),
  tool("write_file", "Write a file in the workspace.", undefined, "idempotent"),
])

export const newApp: AppSeed = {
  id: "new-app",
  label: "New agent",
  description: "Just scaffolded: one specialist, no sessions",
  now: new Date(NOW).toISOString(),
  app: { name: "my-agent", runtime_version: ctx.runtime },
  specialists: { agents: [helper], total_count: 1 },
  sessions: [],
  inbox: [],
  tasks: [],
}
