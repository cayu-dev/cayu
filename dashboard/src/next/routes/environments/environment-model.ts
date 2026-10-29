/**
 * Plain words for an environment summary. The server reports class names (`DockerRunner`,
 * `SQLiteKnowledgeStore`) and branch capability flags; the pages say what they mean.
 */
import type { EnvironmentSummary } from "@/lib/api"

export const environmentRoute = (name: string) =>
  name === "." || name === ".."
    ? ({ to: "/environments", search: { name } } as const)
    : ({ to: "/environments/$name", params: { name } } as const)

/** Split a class name into words, dropping a known suffix: `LambdaMicroVMRunner` → "Lambda MicroVM". */
function words(className: string, suffix: string) {
  const base = className.endsWith(suffix) ? className.slice(0, -suffix.length) : className
  return base.replace(/([a-z0-9])([A-Z])/g, "$1 $2").trim() || className
}

const RUNNERS: Record<string, string> = {
  LocalRunner: "This machine",
  DockerRunner: "Docker container",
  LambdaMicroVMRunner: "AWS Lambda microVM",
  MicrosandboxRunner: "Microsandbox VM",
  E2BRunner: "E2B sandbox",
  BrokeredSandboxRunner: "Brokered sandbox",
  DeferredRunner: "Chosen when a session starts",
}

/** Where the environment's commands run, as a short phrase. */
export function runsOn(env: EnvironmentSummary): string {
  if (!env.runner_type) return "No runner"
  return (
    (Object.hasOwn(RUNNERS, env.runner_type) ? RUNNERS[env.runner_type] : undefined) ??
    `${words(env.runner_type, "Runner")} runner`
  )
}

/**
 * One line under "Commands run in". Without a runner, Cayu's command, Git and text-search
 * tools return "No runner configured"; file tools and the agent's own tools still work.
 */
export function runsOnDetail(env: EnvironmentSummary): string {
  if (!env.runner_type)
    return env.workspace_id
      ? "Shell, Git and text-search tools aren't available. File tools and the agent's own tools still work."
      : "Shell, Git and text-search tools aren't available. The agent's own tools still work."
  return env.is_factory
    ? "A fresh environment is set up for each session and removed when it ends."
    : "Every session shares the same one."
}

const KNOWLEDGE: Record<string, string> = {
  SQLiteKnowledgeStore: "SQLite",
  PostgresKnowledgeStore: "Postgres",
  PostgresEmbeddingKnowledgeStore: "Postgres with semantic search",
  InMemoryKnowledgeStore: "In memory (cleared on restart)",
  InMemoryEmbeddingKnowledgeStore: "In memory with semantic search (cleared on restart)",
}

export function knowledgeText(env: EnvironmentSummary): string {
  if (!env.knowledge_store_type) return "None"
  return (
    (Object.hasOwn(KNOWLEDGE, env.knowledge_store_type)
      ? KNOWLEDGE[env.knowledge_store_type]
      : undefined) ?? words(env.knowledge_store_type, "KnowledgeStore")
  )
}

const BINDINGS: Record<string, string> = {
  GitRepositoryBinding: "Git repository",
  NativeBinding: "Folder",
  DeterministicWorkspaceBinding: "Fixed folder",
  BrokeredSandboxCodingBinding: "Sandboxed repository",
}

/** What the workspace is, or null when the environment has none. */
export function workspaceText(env: EnvironmentSummary): string | null {
  if (!env.workspace_id) return null
  if (!env.binding_type) return "Workspace"
  return (
    (Object.hasOwn(BINDINGS, env.binding_type) ? BINDINGS[env.binding_type] : undefined) ??
    words(env.binding_type, "Binding")
  )
}

/**
 * Workspace branches, in a few words for the list. A branch is a private copy of the
 * workspace that the agent's code opens on request (`Workspace.create_branch`); sessions
 * don't get one automatically, and it isn't a Git branch.
 */
export function branchShort(env: EnvironmentSummary): string {
  if (!env.workspace_id) return "No workspace"
  return env.workspace_branch_capabilities.isolation ? "Supports branches" : "No branches"
}

/** What the workspace's branch support means, as sentences for the detail page. */
export function branchDetails(env: EnvironmentSummary): string[] {
  const caps = env.workspace_branch_capabilities
  if (!env.workspace_id) return ["This environment has no workspace."]
  if (!caps.isolation) return ["Not supported. Code works on the workspace files directly."]
  const lines = [
    "The agent's code can open a branch: a private copy of the workspace whose changes are kept or thrown away together. Sessions don't get one automatically.",
  ]
  if (caps.net_changes) lines.push("A branch lists the files it changed before they're applied.")
  if (caps.publication === "cooperative_atomic")
    lines.push(
      "Changes are applied all at once, as long as nothing else edits those files meanwhile. Shell commands and direct file access aren't covered.",
    )
  if (caps.retention === "durable") lines.push("Open branches survive a server restart.")
  else if (caps.retention === "process_local")
    lines.push("Open branches are lost if the server restarts.")
  return lines
}

const VAULTS: Record<string, string> = {
  LocalEnvVault: "From the server's environment variables",
  SecretsManagerVault: "From AWS Secrets Manager",
  StaticVault: "Fixed values set in code",
  ChainVault: "From several sources, in order",
  RoutedVault: "From different sources per secret",
}

export function secretsText(env: EnvironmentSummary): string {
  if (!env.vault_type) return "None"
  return (
    (Object.hasOwn(VAULTS, env.vault_type) ? VAULTS[env.vault_type] : undefined) ??
    words(env.vault_type, "Vault")
  )
}

const PROXIES: Record<string, string> = {
  AllowlistProxy: "Limited to an allowlist of hosts",
  PassthroughProxy: "Unrestricted",
}

export function networkText(env: EnvironmentSummary): string {
  if (!env.proxy_type) return "Not managed by Cayu"
  return (
    (Object.hasOwn(PROXIES, env.proxy_type) ? PROXIES[env.proxy_type] : undefined) ??
    words(env.proxy_type, "Proxy")
  )
}

export function mcpText(count: number): string {
  if (count === 0) return "None"
  return count === 1 ? "1 MCP server" : `${count} MCP servers`
}

/** Whether a session of this environment can save files. */
export const savesFiles = (env: EnvironmentSummary) => !!env.artifact_store_id

/** Attached handles are bounded process-local observations, not a global open-branch count. */
export function branchObservationText(
  summary: EnvironmentSummary["workspace_branch_lifecycle"],
): string {
  if (summary.attached_count === 0)
    return summary.truncated
      ? "Branch observations are unavailable or incomplete."
      : "No branch handles are attached here."
  const states = new Map<string, number>()
  for (const status of summary.statuses) states.set(status, (states.get(status) ?? 0) + 1)
  const detail = [...states]
    .map(([status, count]) => `${count} ${status.replaceAll("_", " ")}`)
    .join(", ")
  return `${summary.attached_count} attached branch handle${summary.attached_count === 1 ? "" : "s"}${summary.truncated ? " (incomplete)" : ""}: ${detail}. This is not a count of all open branches.`
}

/** Storage is a declaration, not an inference from its opaque ID or a health check. */
export function artifactStorageText(env: EnvironmentSummary): string {
  if (!savesFiles(env)) return "No file storage"
  if (env.artifact_store_kind === "local_disk") return "Local disk"
  if (env.artifact_store_kind === "s3") return "S3"
  return "Storage kind unavailable"
}

export function mcpNamesText(env: EnvironmentSummary): string {
  if (env.mcp_server_count === 0) return "None"
  const names = env.mcp_server_names
  return Array.isArray(names) &&
    names.length === env.mcp_server_count &&
    names.every((name) => typeof name === "string" && name.length > 0)
    ? names.join(", ")
    : `${mcpText(env.mcp_server_count)} (names unavailable)`
}

/** Only a complete, supported native observation proves a count (including zero). */
export function environmentActivity(env: EnvironmentSummary, enabled: boolean) {
  const count = env.session_count
  const last = env.last_activity_at
  const asOf = env.activity_as_of
  if (
    !enabled ||
    typeof count !== "string" ||
    !/^(0|[1-9]\d*)$/.test(count) ||
    typeof asOf !== "string" ||
    !Number.isFinite(Date.parse(asOf))
  )
    return null
  if (
    count === "0" ? last !== null : typeof last !== "string" || !Number.isFinite(Date.parse(last))
  )
    return null
  return { count, last: last ?? null, asOf }
}
