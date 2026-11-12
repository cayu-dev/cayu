/**
 * Environments: where the agent's sessions run. The list gives one row per environment in
 * plain words (where commands run, the workspace, files, knowledge, MCP servers, and when a
 * session last ran there); an environment's page adds its recent sessions and files.
 * An environment created for each session by a factory is named by its factory, as Trace
 * names it, and its page reads what the latest session got. Other class names, IDs and raw
 * capability records stay in Debug.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useParams, useSearch } from "@tanstack/react-router"
import { Box, ChevronRight, RefreshCw } from "lucide-react"
import type { ReactNode } from "react"
import { Button } from "@/components/ui/button"
import type { EnvironmentSummary } from "@/lib/api"
import { useArtifacts } from "@/lib/artifact-query"
import { formatBytes, formatCount } from "@/lib/format"
import { cn } from "@/lib/utils"
import { overviewForDisplay } from "../api/overview"
import type { SessionOverview } from "../api/types"
import { useEnvironments } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import { relativeTime } from "../app/session-model"
import { kindOf, sessionTitle } from "./artifacts/artifact-model"
import { DebugDetails } from "./artifacts/debug"
import {
  EmptyState,
  ErrorState,
  NavLink,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  StateDot,
  sessionRoute,
  TopBar,
  useRowClick,
} from "./common"
import {
  artifactStorageText,
  branchDetails,
  branchObservationText,
  branchShort,
  brief,
  environmentActivity,
  environmentRoute,
  knowledgeDetail,
  knowledgeText,
  mcpNamesText,
  mcpShort,
  networkText,
  noFileStoreText,
  runsOn,
  runsOnDetail,
  runsOnShort,
  SESSION_SETUP_EVENT,
  savesFiles,
  secretsText,
  sessionSetupFacts,
  setupText,
  workspaceLabel,
  workspaceText,
} from "./environments/environment-model"

export function EnvironmentsPage() {
  const search = useSearch({ from: "/environments" })
  return search.name ? (
    <EnvironmentView key={search.name} name={search.name} />
  ) : (
    <EnvironmentList />
  )
}

function EnvironmentList() {
  const environments = useEnvironments()
  const search = useSearch({ from: "/environments" })
  const { environmentDetailsEnabled = false } = useServices()
  const now = useNow()
  const rowClick = useRowClick()
  const needle = search.q?.trim().toLocaleLowerCase() ?? ""
  const rows = (environments.data?.environments ?? []).filter(
    (env) => !needle || env.name.toLocaleLowerCase().includes(needle),
  )

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <Refresh
            fetching={environments.isFetching}
            onRefresh={() => void environments.refetch()}
          />
        }
      >
        Environments
      </TopBar>
      <PageBody width="max-w-6xl">
        <p className="text-sm text-muted-foreground">
          Where your agent's sessions run, and what they can reach there.
        </p>
        {search.q && (
          <p className="mt-3 text-sm text-muted-foreground">
            Matching “{search.q}” ·{" "}
            <NavLink to="/environments" search={{}} className="underline">
              Show all environments
            </NavLink>
          </p>
        )}
        {environments.isPending && (
          <p role="status" className="mt-4 text-sm text-muted-foreground">
            Loading environments…
          </p>
        )}
        {environments.error && (
          <div className="mt-4">
            <ErrorState error={environments.error} />
            {rows.length > 0 && <StaleInventory />}
          </div>
        )}
        {environments.isSuccess && rows.length === 0 && (
          <div className="mt-4">
            <EmptyState title={needle ? "No matching environments" : "No environments"}>
              The agent registers its environments in code. Sessions run in the default one until it
              adds more.
            </EmptyState>
          </div>
        )}
        {rows.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="w-full text-sm md:min-w-[820px]">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Environment</th>
                  <th className="hidden px-3 py-2 font-medium md:table-cell">Commands run in</th>
                  <th className="hidden px-3 py-2 font-medium md:table-cell">Workspace</th>
                  <th className="hidden px-3 py-2 font-medium md:table-cell">Files</th>
                  <th className="hidden px-3 py-2 font-medium md:table-cell">Knowledge</th>
                  <th className="px-3 py-2 text-right font-medium">Last activity</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((env) => {
                  const activity = environmentActivity(env, environmentDetailsEnabled)
                  const workspace = workspaceText(env)
                  return (
                    <tr
                      key={env.name}
                      onClick={rowClick(environmentRoute(env.name))}
                      className={ROW_CLASS}
                    >
                      <td className="max-w-[16rem] px-3 py-2.5 max-md:max-w-[15rem]">
                        <div className="flex min-w-0 items-center gap-2">
                          <Box
                            className="size-4 shrink-0 text-muted-foreground"
                            aria-hidden="true"
                          />
                          <NavLink
                            {...environmentRoute(env.name)}
                            className={cn("truncate font-medium", ROW_LINK_CLASS)}
                          >
                            {env.name}
                          </NavLink>
                        </div>
                        <div className="mt-0.5 truncate pl-6 text-xs text-muted-foreground">
                          <span className="md:hidden">
                            {runsOn(env)} · {brief(workspaceLabel(env, "No workspace"))} ·{" "}
                          </span>
                          {mcpShort(env)}
                        </div>
                      </td>
                      <td className="hidden px-3 py-2.5 md:table-cell">
                        <div>{runsOn(env)}</div>
                        <div className="mt-0.5 text-xs text-muted-foreground">
                          {runsOnShort(env)}
                        </div>
                      </td>
                      <td className="hidden px-3 py-2.5 md:table-cell">
                        <div>{brief(workspaceLabel(env))}</div>
                        {workspace && (
                          <div className="mt-0.5 text-xs text-muted-foreground">
                            {branchShort(env)}
                          </div>
                        )}
                      </td>
                      <td className="hidden px-3 py-2.5 text-muted-foreground md:table-cell">
                        {brief(artifactStorageText(env))}
                      </td>
                      <td className="hidden px-3 py-2.5 text-muted-foreground md:table-cell">
                        {brief(knowledgeText(env))}
                      </td>
                      <td className="px-3 py-2.5 text-right whitespace-nowrap tabular-nums text-muted-foreground">
                        <div title={activity?.last ?? undefined}>
                          {activity
                            ? activity.last
                              ? relativeTime(Date.parse(activity.last), now())
                              : "None yet"
                            : "Unavailable"}
                        </div>
                        <div
                          className="mt-0.5 text-xs"
                          title={activity ? `Observed ${activity.asOf}` : undefined}
                        >
                          {activity
                            ? sessionCountText(activity.count)
                            : "Session count unavailable"}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        {environments.isSuccess && rows.length > 0 && (
          <p className="mt-3 text-xs text-muted-foreground">
            {rows.length === 1 ? "1 environment" : `${rows.length} environments`}
          </p>
        )}
        {rows.length > 0 && <p className="mt-2 text-xs text-muted-foreground">{COUNT_SCOPE}</p>}
      </PageBody>
    </div>
  )
}

const COUNT_SCOPE = "Counts include all retained sessions, including eval trials and forks."
const sessionCountText = (count: string) =>
  `${formatCount(count)} ${count === "1" ? "session" : "sessions"}`
const StaleInventory = () => (
  <p role="status" className="mt-2 text-sm text-muted-foreground">
    Couldn't refresh environments; showing the last inventory loaded.
  </p>
)

function Refresh({ fetching, onRefresh }: { fetching: boolean; onRefresh: () => void }) {
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label="Refresh"
      title="Refresh"
      disabled={fetching}
      onClick={onRefresh}
    >
      <RefreshCw className={cn(fetching && "animate-spin")} />
    </Button>
  )
}

function Section({
  title,
  action,
  children,
}: {
  title: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="mt-8">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  )
}

function Facts({ rows }: { rows: Array<[label: string, value: ReactNode]> }) {
  return (
    <dl className="rounded-xl border border-border px-3 py-1">
      {rows.map(([label, value]) => (
        <div
          key={label}
          className="grid grid-cols-[minmax(0,8.5rem)_minmax(0,1fr)] gap-3 border-b border-border py-2 text-sm last:border-0"
        >
          <dt className="text-muted-foreground">{label}</dt>
          <dd className="min-w-0 [overflow-wrap:anywhere]">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

const List = ({ children }: { children: ReactNode }) => (
  <div className="overflow-hidden rounded-xl border border-border">{children}</div>
)

const linkClass =
  "text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"

function minutes(seconds: number) {
  if (seconds < 120) return `${seconds} seconds`
  return `${Math.round(seconds / 60)} minutes`
}

export function EnvironmentPage() {
  const { name } = useParams({ strict: false }) as { name: string }
  return <EnvironmentView name={name} />
}

function EnvironmentView({ name }: { name: string }) {
  const environments = useEnvironments()
  const queryClient = useQueryClient()
  const env = environments.data?.environments.find((e) => e.name === name)
  // Refresh rereads the page's sessions and the latest session's setup, not only the inventory.
  const refresh = () => {
    void environments.refetch()
    for (const key of ["environment-sessions", "environment-latest-setup"])
      void queryClient.invalidateQueries({ queryKey: [key, name] })
  }

  const header = (
    <TopBar actions={<Refresh fetching={environments.isFetching} onRefresh={refresh} />}>
      <span className="flex min-w-0 items-center gap-1">
        <NavLink
          to="/environments"
          search={{}}
          className="shrink-0 text-muted-foreground hover:text-foreground"
        >
          Environments
        </NavLink>
        <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
        <span className="truncate">{name}</span>
      </span>
    </TopBar>
  )
  if (environments.error && !env)
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {header}
        <PageBody width="max-w-3xl">
          <ErrorState error={environments.error} />
        </PageBody>
      </div>
    )
  if (environments.isPending)
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {header}
        <PageBody width="max-w-3xl">
          <p role="status" className="text-sm text-muted-foreground">
            Loading environment…
          </p>
        </PageBody>
      </div>
    )
  if (!env)
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {header}
        <PageBody width="max-w-3xl">
          <EmptyState title="Environment not found">
            This agent has no environment named “{name}”.{" "}
            <NavLink to="/environments" search={{}} className="underline">
              See all environments
            </NavLink>
          </EmptyState>
        </PageBody>
      </div>
    )
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      {header}
      <EnvironmentDetail env={env} stale={!!environments.error} />
    </div>
  )
}

function EnvironmentDetail({ env, stale }: { env: EnvironmentSummary; stale: boolean }) {
  const {
    overviewQueriesEnabled = true,
    artifactsEnabled = true,
    environmentDetailsEnabled = false,
  } = useServices()
  const activity = environmentActivity(env, environmentDetailsEnabled)
  const api = useApi()
  const now = useNow()
  const recent = useQuery({
    queryKey: ["environment-sessions", env.name, overviewQueriesEnabled],
    queryFn: async ({ signal }) => {
      const page = await api.sessions(
        {
          environment_name: env.name,
          limit: 8,
          order_by: "updated_at_desc",
          include: overviewQueriesEnabled ? ["overview"] : [],
        },
        signal,
      )
      const overviews = overviewQueriesEnabled ? (page.overviews ?? []).map(overviewForDisplay) : []
      const byId = new Map(overviews.map((o) => [o.session_id, o]))
      return {
        rows: page.sessions.map((session) => ({ session, overview: byId.get(session.id) })),
      }
    },
  })
  const files = useArtifacts(
    { environment_name: env.name, limit: 5 },
    artifactsEnabled && savesFiles(env),
  )
  const branches = branchDetails(env)
  const fileCount = files.data?.total_count
  const knownCount =
    fileCount == null
      ? "Count unavailable"
      : fileCount === 0
        ? "None yet"
        : fileCount === 1
          ? "1 file"
          : `${fileCount} files`
  const savedFiles = !artifactsEnabled
    ? "Unavailable"
    : files.error
      ? files.data
        ? `${knownCount} (last read; refresh failed)`
        : "Unavailable"
      : files.isPending
        ? "…"
        : files.data
          ? knownCount
          : "Unavailable"

  return (
    <PageBody width="max-w-3xl">
      <div className="flex items-center gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted">
          <Box className="size-4" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold">{env.name}</h1>
          <p className="text-sm text-muted-foreground">
            {setupText(env) ?? `${runsOn(env)} · ${workspaceLabel(env, "No workspace")}`}
          </p>
        </div>
      </div>

      {stale && <StaleInventory />}
      <Section title="Activity">
        <Facts
          rows={[
            ["Sessions", activity ? sessionCountText(activity.count) : "Unavailable"],
            [
              "Last activity",
              activity ? (
                activity.last ? (
                  <time key="last-activity" dateTime={activity.last} title={activity.last}>
                    {relativeTime(Date.parse(activity.last), now())}
                  </time>
                ) : (
                  "None yet"
                )
              ) : (
                "Unavailable"
              ),
            ],
            [
              "Observed",
              activity ? (
                <time key="observed" dateTime={activity.asOf} title={activity.asOf}>
                  {relativeTime(Date.parse(activity.asOf), now())}
                </time>
              ) : (
                "Unavailable"
              ),
            ],
          ]}
        />
        <p className="mt-2 text-xs text-muted-foreground">{COUNT_SCOPE}</p>
      </Section>

      {env.is_factory && !recent.error && (
        <LatestSessionSetup
          env={env}
          latest={recent.isSuccess ? (recent.data.rows[0] ?? null) : undefined}
        />
      )}

      <Section title="Where it runs">
        <Facts
          rows={[
            ["Commands run in", <RunsOn key="where" env={env} />],
            ["Secrets", secretsText(env)],
            ["Network", networkText(env)],
            ...(env.lifecycle_policy?.lifecycle_timeout_seconds
              ? ([
                  [
                    "Setup limit",
                    `Setup gives up after ${minutes(env.lifecycle_policy.lifecycle_timeout_seconds)}`,
                  ],
                ] as Array<[string, ReactNode]>)
              : []),
          ]}
        />
      </Section>

      <Section title="Workspace">
        <Facts
          rows={[
            ["Workspace", workspaceLabel(env)],
            [
              "Branches",
              <div key="branches" className="space-y-1">
                {branches.map((line) => (
                  <p key={line}>{line}</p>
                ))}
              </div>,
            ],
            ...(env.workspace_branch_capabilities.isolation
              ? ([
                  ["Observed branches", branchObservationText(env.workspace_branch_lifecycle)],
                ] as Array<[string, ReactNode]>)
              : []),
            ...(env.workspace_instructions
              ? ([["Instructions", env.workspace_instructions]] as Array<[string, ReactNode]>)
              : []),
          ]}
        />
      </Section>

      <Section
        title="Files"
        action={
          savesFiles(env) && artifactsEnabled ? (
            <NavLink to="/artifacts" search={{ environment_name: env.name }} className={linkClass}>
              See all files
            </NavLink>
          ) : undefined
        }
      >
        <Facts
          rows={
            savesFiles(env)
              ? [
                  ["Storage", artifactStorageText(env)],
                  [
                    "Where they go",
                    "Sessions here can save files, and they're kept after the session ends.",
                  ],
                  ["Saved so far", savedFiles],
                ]
              : [["Where they go", noFileStoreText(env)]]
          }
        />
      </Section>

      <Section
        title="Knowledge"
        action={
          env.knowledge_store_type ? (
            <NavLink to="/knowledge" className={linkClass}>
              Open knowledge
            </NavLink>
          ) : undefined
        }
      >
        <Facts rows={[["Stored in", knowledgeDetail(env)]]} />
      </Section>

      <Section title="MCP servers">
        <Facts rows={[["Declared servers", mcpNamesText(env)]]} />
      </Section>

      <Section
        title="Recent sessions"
        action={
          <NavLink
            to="/sessions"
            search={{ environment_name: env.name, include_eval: true, include_forks: true }}
            className={linkClass}
          >
            All sessions here
          </NavLink>
        }
      >
        {recent.isPending && (
          <p role="status" className="text-sm text-muted-foreground">
            Loading sessions…
          </p>
        )}
        {recent.error && <ErrorState error={recent.error} />}
        {recent.isSuccess && recent.data.rows.length === 0 && (
          <EmptyState title="No sessions yet">
            Sessions that run in this environment show up here.
          </EmptyState>
        )}
        {recent.isSuccess && recent.data.rows.length > 0 && (
          <List>
            {recent.data.rows.map(({ session, overview }) => (
              <NavLink
                key={session.id}
                {...sessionRoute(session.id)}
                className="flex min-w-0 items-center gap-2 border-b border-border px-3 py-2 text-sm last:border-0 hover:bg-muted/60"
              >
                <StateDot state={overview?.state} />
                <span className="min-w-0 flex-1 truncate">{sessionTitle(overview)}</span>
                <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">
                  {session.agent_name}
                </span>
                <span className="w-16 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                  {relativeTime(Date.parse(session.updated_at), now())}
                </span>
              </NavLink>
            ))}
          </List>
        )}
      </Section>

      {savesFiles(env) && (
        <Section title="Recent files">
          {!artifactsEnabled ? (
            <p className="text-sm text-muted-foreground">Files are unavailable for this agent.</p>
          ) : (
            files.isPending && (
              <p role="status" className="text-sm text-muted-foreground">
                Loading files…
              </p>
            )
          )}
          {artifactsEnabled && files.error && <ErrorState error={files.error} />}
          {artifactsEnabled &&
            files.isSuccess &&
            files.data &&
            files.data.artifacts.length === 0 && (
              <EmptyState title="No files yet">
                Files that sessions here save show up here.
              </EmptyState>
            )}
          {artifactsEnabled && files.isSuccess && files.data && files.data.artifacts.length > 0 && (
            <List>
              {files.data.artifacts.map((artifact) => {
                const kind = kindOf(artifact)
                const Icon = kind.icon
                return (
                  <NavLink
                    key={`${artifact.artifact_store_id}:${artifact.id}`}
                    to="/artifacts"
                    search={{
                      environment_name: env.name,
                      artifact: artifact.id,
                      preview_store_id: artifact.artifact_store_id ?? undefined,
                    }}
                    className="flex min-w-0 items-center gap-2 border-b border-border px-3 py-2 text-sm last:border-0 hover:bg-muted/60"
                  >
                    <Icon className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                    <span className="min-w-0 flex-1 truncate">{artifact.filename}</span>
                    <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">
                      {kind.label} · {formatBytes(artifact.size_bytes)}
                    </span>
                    <span className="w-16 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                      {relativeTime(Date.parse(artifact.created_at), now())}
                    </span>
                  </NavLink>
                )
              })}
            </List>
          )}
        </Section>
      )}

      <DebugDetails
        className="mt-8"
        rows={[
          ["Name", env.name],
          ["Workspace ID", env.workspace_id],
          ["Artifact store ID", env.artifact_store_id],
          ["Runner", env.runner_type],
          ["Binding", env.binding_type],
          ["Knowledge store", env.knowledge_store_type],
          ["Vault", env.vault_type],
          ["Proxy", env.proxy_type],
          ["MCP servers", String(env.mcp_server_count)],
          ["Last activity", activity?.last],
          ["Activity observed at", activity?.asOf],
          ["Factory-backed", env.is_factory ? "yes" : "no"],
          ["Factory", env.factory_type],
          [
            "Branches",
            `${env.workspace_branch_lifecycle.attached_count} attached${
              env.workspace_branch_lifecycle.statuses.length
                ? ` · ${env.workspace_branch_lifecycle.statuses.join(", ")}`
                : ""
            }${env.workspace_branch_lifecycle.truncated ? " · truncated" : ""}`,
          ],
        ]}
        json={[
          ["Branch capabilities", env.workspace_branch_capabilities],
          ["Branch lifecycle", env.workspace_branch_lifecycle],
          ["Lifecycle policy", env.lifecycle_policy],
          ["Execution snapshots", env.execution_snapshot_capability ?? null],
          ["Workspace checkpoint policy", env.workspace_checkpoint_policy ?? null],
          ["Raw environment", env],
        ]}
      />
      <div className="h-6" />
    </PageBody>
  )
}

function RunsOn({ env }: { env: EnvironmentSummary }) {
  return (
    <div>
      <p>{runsOn(env)}</p>
      <p className="mt-0.5 text-xs text-muted-foreground">{runsOnDetail(env)}</p>
    </div>
  )
}

/**
 * A factory environment's parts exist only once a session starts, so the page reads what the
 * newest session's environment had from its `environment.binding.completed` event.
 */
function LatestSessionSetup({
  env,
  latest,
}: {
  env: EnvironmentSummary
  /** The newest session here; null when there is none, undefined while loading. */
  latest:
    | { session: { id: string; updated_at: string }; overview?: SessionOverview }
    | null
    | undefined
}) {
  const api = useApi()
  const now = useNow()
  const setup = useQuery<ReturnType<typeof sessionSetupFacts>>({
    queryKey: ["environment-latest-setup", env.name, latest?.session.id],
    enabled: !!latest,
    // A session still setting up has no binding event yet; look again until it has one.
    refetchInterval: (query) =>
      query.state.data == null && latest?.overview?.state === "running" ? 5000 : false,
    queryFn: async ({ signal }) => {
      if (!latest) return null
      const page = await api.events(
        latest.session.id,
        { event_type: SESSION_SETUP_EVENT, order_by: "sequence_desc", limit: 1 },
        signal,
      )
      return sessionSetupFacts(page.events[0]?.payload as Record<string, unknown> | undefined)
    },
  })
  const facts = setup.data
  return (
    <Section title="Latest session">
      {latest === undefined ? (
        <p role="status" className="text-sm text-muted-foreground">
          Loading sessions…
        </p>
      ) : latest === null ? (
        <p className="text-sm text-muted-foreground">
          No session has run here yet. Each one gets the environment{" "}
          {env.factory_type ?? "the factory"} creates for it.
        </p>
      ) : setup.isPending ? (
        <p role="status" className="text-sm text-muted-foreground">
          Loading what it got…
        </p>
      ) : setup.error ? (
        <ErrorState error={setup.error} />
      ) : (
        <Facts
          rows={[
            [
              "Session",
              <span key="session">
                <NavLink
                  {...sessionRoute(latest.session.id)}
                  className="underline-offset-4 hover:underline"
                >
                  {sessionTitle(latest.overview)}
                </NavLink>
                <span className="text-muted-foreground">
                  {" "}
                  · {relativeTime(Date.parse(latest.session.updated_at), now())}
                </span>
              </span>,
            ],
            ["Commands", facts?.commands ?? "Not recorded"],
            ["Workspace", facts?.workspace ?? "Not recorded"],
          ]}
        />
      )}
    </Section>
  )
}
