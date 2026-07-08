import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Search } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button, buttonVariants } from "../components/ui/button"
import { Input } from "../components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import { fetchSessionsPage, type SessionListQuery } from "../lib/api"
import { formatDateTime } from "../lib/format"

type SessionDateOrder = "updated_at_desc" | "updated_at_asc" | "created_at_desc" | "created_at_asc"
type SessionStatusFilter = "all" | Exclude<SessionListQuery["status"], null | undefined>

const SESSION_STATUSES = [
  "pending",
  "running",
  "interrupting",
  "completed",
  "failed",
  "interrupted",
]

const selectClassName =
  "h-8 min-w-32 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "completed"
      ? "default"
      : status === "running"
        ? "secondary"
        : status === "failed"
          ? "destructive"
          : "outline"
  return <Badge variant={variant}>{status}</Badge>
}

function optionalFilter(value: string) {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value)

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedValue(value), delayMs)
    return () => window.clearTimeout(timer)
  }, [delayMs, value])

  return debouncedValue
}

export function SessionsPage() {
  const [search, setSearch] = useState("")
  const [status, setStatus] = useState<SessionStatusFilter>("all")
  const [orderBy, setOrderBy] = useState<SessionDateOrder>("updated_at_desc")
  const debouncedSearch = useDebouncedValue(search, 300)
  const query = useMemo<SessionListQuery>(
    () => ({
      limit: 100,
      order_by: orderBy,
      q: optionalFilter(debouncedSearch),
      status: status === "all" ? undefined : status,
    }),
    [debouncedSearch, orderBy, status],
  )

  const {
    data: sessionsPage,
    error,
    isError,
    isLoading,
  } = useQuery({
    queryKey: ["sessions", query],
    queryFn: () => fetchSessionsPage(query),
  })
  const errorMessage = error instanceof Error ? error.message : "Failed to load sessions."
  const list = sessionsPage?.sessions || []
  const totalCount = sessionsPage?.total_count ?? list.length
  const hasFilters = search.trim() !== "" || status !== "all"

  function clearFilters() {
    setSearch("")
    setStatus("all")
  }

  return (
    <Page>
      <PageHeader
        title="Sessions"
        actions={
          <Link to="/run" className={buttonVariants()}>
            New Run
          </Link>
        }
      />

      <DataCard
        title="Sessions"
        description={`${list.length} shown${totalCount !== list.length ? ` of ${totalCount}` : ""} matching sessions`}
      >
        <div className="flex min-w-0 flex-wrap items-center gap-2 border-b border-border p-4">
          <div className="relative min-w-56 flex-[2_1_18rem]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search sessions..."
              className="pl-8"
            />
          </div>
          <select
            value={status}
            onChange={(event) => setStatus(event.target.value as SessionStatusFilter)}
            className={selectClassName}
            aria-label="Filter by status"
          >
            <option value="all">All statuses</option>
            {SESSION_STATUSES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
          <select
            value={orderBy}
            onChange={(event) => setOrderBy(event.target.value as SessionDateOrder)}
            className={selectClassName}
            aria-label="Sort sessions"
          >
            <option value="updated_at_desc">Updated newest</option>
            <option value="updated_at_asc">Updated oldest</option>
            <option value="created_at_desc">Created newest</option>
            <option value="created_at_asc">Created oldest</option>
          </select>
          {hasFilters && (
            <Button variant="outline" size="sm" onClick={clearFilters}>
              Clear
            </Button>
          )}
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Session ID</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={6}>
                  <StateMessage>Loading...</StateMessage>
                </TableCell>
              </TableRow>
            ) : isError ? (
              <TableRow>
                <TableCell colSpan={6}>
                  <StateMessage tone="danger">{errorMessage}</StateMessage>
                </TableCell>
              </TableRow>
            ) : !list.length ? (
              <TableRow>
                <TableCell colSpan={6}>
                  <StateMessage>
                    {hasFilters ? "No sessions match the current filters." : "No sessions yet"}
                  </StateMessage>
                </TableCell>
              </TableRow>
            ) : (
              list.map((s) => (
                <TableRow key={s.id} className="cursor-pointer">
                  <TableCell>
                    <Link
                      to="/sessions/$sessionId"
                      params={{ sessionId: s.id }}
                      className="font-mono text-sm text-primary hover:underline"
                    >
                      {s.id}
                    </Link>
                  </TableCell>
                  <TableCell className="text-sm">{s.agent_name}</TableCell>
                  <TableCell>
                    <StatusBadge status={s.status} />
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">{s.provider_name}</TableCell>
                  <TableCell className="max-w-56 truncate text-sm text-muted-foreground">
                    {s.model}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDateTime(s.updated_at || s.created_at)}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </DataCard>
    </Page>
  )
}
