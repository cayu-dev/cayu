import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { buttonVariants } from "../components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import { fetchSessions } from "../lib/api"
import { formatDateTime } from "../lib/format"

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

export function SessionsPage() {
  const {
    data: sessions,
    error,
    isError,
    isLoading,
  } = useQuery({
    queryKey: ["sessions"],
    queryFn: fetchSessions,
  })
  const errorMessage = error instanceof Error ? error.message : "Failed to load sessions."

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

      <DataCard>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Session ID</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={4}>
                  <StateMessage>Loading...</StateMessage>
                </TableCell>
              </TableRow>
            ) : isError ? (
              <TableRow>
                <TableCell colSpan={4}>
                  <StateMessage tone="danger">{errorMessage}</StateMessage>
                </TableCell>
              </TableRow>
            ) : !sessions?.length ? (
              <TableRow>
                <TableCell colSpan={4}>
                  <StateMessage>No sessions yet</StateMessage>
                </TableCell>
              </TableRow>
            ) : (
              sessions.map((s) => (
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
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDateTime(s.created_at)}
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
