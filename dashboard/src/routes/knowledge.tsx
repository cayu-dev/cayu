import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Check, X } from "lucide-react"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { Card, CardContent } from "../components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import {
  approveKnowledge,
  fetchPendingKnowledge,
  type KnowledgeEntry,
  rejectKnowledge,
} from "../lib/api"

function Labels({ labels }: { labels: Record<string, string> }) {
  const entries = Object.entries(labels)
  if (entries.length === 0) {
    return <span className="text-muted-foreground">-</span>
  }
  return (
    <div className="flex flex-wrap gap-1">
      {entries.map(([key, value]) => (
        <Badge key={key} variant="outline">
          {key}={value}
        </Badge>
      ))}
    </div>
  )
}

function Aspects({ aspects }: { aspects: string[] }) {
  if (aspects.length === 0) {
    return <span className="text-muted-foreground">-</span>
  }
  return (
    <div className="flex flex-wrap gap-1">
      {aspects.map((aspect) => (
        <Badge key={aspect} variant="secondary">
          {aspect}
        </Badge>
      ))}
    </div>
  )
}

function EntryActions({
  entry,
  disabled,
  onApprove,
  onReject,
}: {
  entry: KnowledgeEntry
  disabled: boolean
  onApprove: (entryId: string) => void
  onReject: (entryId: string) => void
}) {
  return (
    <div className="flex items-center justify-end gap-2">
      <Button
        variant="outline"
        size="sm"
        disabled={disabled}
        onClick={() => onApprove(entry.entry_id)}
      >
        <Check className="h-3.5 w-3.5" />
        Approve
      </Button>
      <Button
        variant="destructive"
        size="sm"
        disabled={disabled}
        onClick={() => onReject(entry.entry_id)}
      >
        <X className="h-3.5 w-3.5" />
        Reject
      </Button>
    </div>
  )
}

export function KnowledgePage() {
  const queryClient = useQueryClient()
  const pending = useQuery({
    queryKey: ["knowledge", "pending"],
    queryFn: fetchPendingKnowledge,
  })

  const approve = useMutation({
    mutationFn: approveKnowledge,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["knowledge", "pending"] }),
  })
  const reject = useMutation({
    mutationFn: rejectKnowledge,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["knowledge", "pending"] }),
  })

  const error =
    pending.error instanceof Error
      ? pending.error.message
      : approve.error instanceof Error
        ? approve.error.message
        : reject.error instanceof Error
          ? reject.error.message
          : null
  const entries = pending.data?.entries ?? []
  const mutating = approve.isPending || reject.isPending

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Knowledge Review</h1>
          <p className="text-sm text-muted-foreground">
            {pending.data?.total_entries_known ?? 0} pending entries
          </p>
        </div>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Entry</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead>Scope</TableHead>
                <TableHead>Aspects</TableHead>
                <TableHead>Created</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {pending.isLoading ? (
                <TableRow>
                  <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                    Loading...
                  </TableCell>
                </TableRow>
              ) : error ? (
                <TableRow>
                  <TableCell colSpan={6} className="py-8 text-center text-destructive">
                    {error}
                  </TableCell>
                </TableRow>
              ) : entries.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                    No pending knowledge entries
                  </TableCell>
                </TableRow>
              ) : (
                entries.map((entry) => (
                  <TableRow key={entry.entry_id}>
                    <TableCell className="max-w-xl whitespace-normal">
                      <div className="font-medium">{entry.title || entry.entry_id}</div>
                      <div className="mt-1 text-sm text-muted-foreground">
                        {entry.text_preview || "No preview available"}
                      </div>
                      <div className="mt-2 font-mono text-xs text-muted-foreground">
                        {entry.entry_id}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{entry.kind}</Badge>
                    </TableCell>
                    <TableCell className="max-w-xs whitespace-normal">
                      <div className="mb-2 font-mono text-xs text-muted-foreground">
                        {entry.namespace}
                      </div>
                      <Labels labels={entry.labels} />
                    </TableCell>
                    <TableCell className="max-w-xs whitespace-normal">
                      <Aspects aspects={entry.aspects} />
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(entry.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <EntryActions
                        entry={entry}
                        disabled={mutating}
                        onApprove={approve.mutate}
                        onReject={reject.mutate}
                      />
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
