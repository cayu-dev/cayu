import { useQuery } from "@tanstack/react-query"
import { Boxes, Database, FolderGit2, KeyRound, Search, ServerCog } from "lucide-react"
import { type ReactNode, useEffect, useMemo, useState } from "react"
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Input } from "../components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import { type EnvironmentSummary, fetchEnvironments } from "../lib/api"
import { formatCount } from "../lib/format"
import { cn } from "../lib/utils"

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(handle)
  }, [delayMs, value])
  return debounced
}

function environmentMatches(environment: EnvironmentSummary, query: string) {
  const needle = query.trim().toLowerCase()
  if (!needle) return true
  return [
    environment.name,
    environment.workspace_id,
    environment.artifact_store_id,
    environment.runner_type,
    environment.binding_type,
    environment.vault_type,
    environment.proxy_type,
    environment.knowledge_store_type,
    environment.workspace_instructions,
  ].some((value) => value?.toLowerCase().includes(needle))
}

function capabilityCount(environment: EnvironmentSummary) {
  return [
    environment.workspace_id,
    environment.artifact_store_id,
    environment.runner_type,
    environment.binding_type,
    environment.vault_type,
    environment.proxy_type,
    environment.knowledge_store_type,
    environment.workspace_instructions,
  ].filter(Boolean).length
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="grid grid-cols-[8rem_minmax(0,1fr)] gap-3 text-sm">
      <div className="text-muted-foreground">{label}</div>
      <div className="min-w-0 break-words">{value}</div>
    </div>
  )
}

function CapabilityTile({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof ServerCog
  label: string
  value: string | null
}) {
  return (
    <div className="rounded-md border border-border p-3">
      <div className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
        <Icon className="h-4 w-4 shrink-0 text-primary" />
        <span>{label}</span>
      </div>
      <div className="mt-2 truncate font-mono text-sm font-medium">{value ?? "-"}</div>
    </div>
  )
}

function EnvironmentDetail({ environment }: { environment: EnvironmentSummary | null }) {
  if (environment === null) {
    return (
      <StateMessage className="py-16">
        Select an environment to inspect workspace, runner, artifact, and secret wiring.
      </StateMessage>
    )
  }

  return (
    <div className="space-y-5 p-4 sm:p-5">
      <div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Badge variant={environment.is_factory ? "outline" : "secondary"}>
            {environment.is_factory ? "factory" : "static"}
          </Badge>
          {environment.artifact_store_id && <Badge variant="outline">artifacts</Badge>}
          {environment.runner_type && <Badge variant="outline">runner</Badge>}
          {environment.vault_type && <Badge variant="outline">vault</Badge>}
          {environment.mcp_server_count > 0 && (
            <Badge variant="outline">{formatCount(environment.mcp_server_count)} MCP</Badge>
          )}
        </div>
        <h2 className="mt-3 truncate text-xl font-semibold">{environment.name}</h2>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        <CapabilityTile icon={FolderGit2} label="Workspace" value={environment.workspace_id} />
        <CapabilityTile
          icon={Database}
          label="Artifact Store"
          value={environment.artifact_store_id}
        />
        <CapabilityTile icon={ServerCog} label="Runner" value={environment.runner_type} />
        <CapabilityTile icon={Boxes} label="Binding" value={environment.binding_type} />
        <CapabilityTile icon={KeyRound} label="Vault" value={environment.vault_type} />
        <CapabilityTile
          icon={Database}
          label="Knowledge Store"
          value={environment.knowledge_store_type}
        />
      </div>

      <div className="space-y-3 rounded-md border border-border p-4">
        <DetailRow label="proxy" value={environment.proxy_type ?? "-"} />
        <DetailRow label="instructions" value={environment.workspace_instructions ?? "-"} />
        <DetailRow label="mcp servers" value={formatCount(environment.mcp_server_count)} />
        <DetailRow
          label="metadata"
          value={<PayloadViewer value={environment.metadata} maxHeight="max-h-44" />}
        />
        <DetailRow
          label="bound workspace"
          value={
            environment.bound_workspace ? (
              <PayloadViewer value={environment.bound_workspace} maxHeight="max-h-44" />
            ) : (
              "-"
            )
          }
        />
      </div>
    </div>
  )
}

export function EnvironmentsPage() {
  const [search, setSearch] = useState("")
  const [selectedEnvironmentName, setSelectedEnvironmentName] = useState<string | null>(null)
  const debouncedSearch = useDebouncedValue(search, 300)

  const environments = useQuery({
    queryKey: ["environments"],
    queryFn: fetchEnvironments,
    staleTime: 15_000,
  })

  const filteredEnvironments = useMemo(() => {
    const items = environments.data?.environments ?? []
    return items.filter((environment) => environmentMatches(environment, debouncedSearch))
  }, [debouncedSearch, environments.data?.environments])

  useEffect(() => {
    if (
      selectedEnvironmentName &&
      filteredEnvironments.some((environment) => environment.name === selectedEnvironmentName)
    ) {
      return
    }
    setSelectedEnvironmentName(filteredEnvironments[0]?.name ?? null)
  }, [filteredEnvironments, selectedEnvironmentName])

  const selectedEnvironment =
    filteredEnvironments.find((environment) => environment.name === selectedEnvironmentName) ??
    filteredEnvironments[0] ??
    null
  const error = environments.error instanceof Error ? environments.error.message : null

  return (
    <Page>
      <PageHeader
        title="Environments"
        description="Inspect registered execution environments, workspace bindings, artifact stores, and runtime services."
      />

      <div className="grid min-h-[calc(100vh-10rem)] gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(28rem,1.2fr)]">
        <DataCard
          title="Registered Environments"
          description={`${formatCount(environments.data?.total_count)} configured environments`}
        >
          <div className="border-b border-border p-4">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search environments, runners, workspaces, stores..."
                className="pl-8"
              />
            </div>
          </div>
          {environments.isLoading ? (
            <StateMessage>Loading environments...</StateMessage>
          ) : error ? (
            <StateMessage tone="danger">{error}</StateMessage>
          ) : filteredEnvironments.length === 0 ? (
            <StateMessage>No environments match the current search.</StateMessage>
          ) : (
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Environment</TableHead>
                    <TableHead>Runner</TableHead>
                    <TableHead className="text-right">Services</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredEnvironments.map((environment) => (
                    <TableRow
                      key={environment.name}
                      className={cn(
                        "cursor-pointer",
                        selectedEnvironment?.name === environment.name && "bg-muted/60",
                      )}
                      onClick={() => setSelectedEnvironmentName(environment.name)}
                    >
                      <TableCell>
                        <div className="min-w-0">
                          <div className="truncate font-medium">{environment.name}</div>
                          <div className="truncate text-xs text-muted-foreground">
                            {environment.is_factory ? "factory-backed" : "static"}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="min-w-32">
                        <span className="font-mono text-xs">{environment.runner_type ?? "-"}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        {formatCount(capabilityCount(environment))}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </DataCard>

        <DataCard title="Environment Detail" description="Read-only runtime wiring.">
          <EnvironmentDetail environment={selectedEnvironment} />
        </DataCard>
      </div>
    </Page>
  )
}
