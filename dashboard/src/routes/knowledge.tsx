import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Check, X } from "lucide-react"
import { type ReactNode, useEffect, useState } from "react"
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
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
  fetchPendingKnowledgeEntry,
  type KnowledgeEntryDetail,
  rejectKnowledge,
} from "../lib/api"
import { formatDateTime } from "../lib/format"

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
  entry: { entry_id: string }
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

function MetadataBlock({ metadata }: { metadata: Record<string, unknown> }) {
  if (Object.keys(metadata).length === 0) {
    return <span className="text-muted-foreground">-</span>
  }
  return <PayloadViewer value={metadata} maxHeight="max-h-56" />
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="grid grid-cols-[7rem_1fr] gap-3 text-sm">
      <div className="text-muted-foreground">{label}</div>
      <div className="min-w-0">{value}</div>
    </div>
  )
}

function KnowledgeDetail({
  entry,
  isLoading,
  error,
  disabled,
  onApprove,
  onReject,
}: {
  entry: KnowledgeEntryDetail | null
  isLoading: boolean
  error: string | null
  disabled: boolean
  onApprove: (entryId: string) => void
  onReject: (entryId: string) => void
}) {
  if (isLoading) {
    return <StateMessage className="py-10">Loading...</StateMessage>
  }
  if (error !== null) {
    return (
      <StateMessage tone="danger" className="py-10">
        {error}
      </StateMessage>
    )
  }
  if (entry === null) {
    return <StateMessage className="py-10">Select a pending entry</StateMessage>
  }
  return (
    <div className="space-y-5 p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="break-words text-lg font-semibold">{entry.title || entry.entry_id}</h2>
          <div className="mt-1 break-all font-mono text-xs text-muted-foreground">
            {entry.entry_id}
          </div>
        </div>
        <EntryActions entry={entry} disabled={disabled} onApprove={onApprove} onReject={onReject} />
      </div>

      <div className="space-y-3">
        <DetailRow label="Kind" value={<Badge variant="outline">{entry.kind}</Badge>} />
        <DetailRow label="Namespace" value={<span className="font-mono">{entry.namespace}</span>} />
        <DetailRow label="Labels" value={<Labels labels={entry.labels} />} />
        <DetailRow label="Aspects" value={<Aspects aspects={entry.aspects} />} />
        <DetailRow label="Source" value={entry.source_type || "-"} />
        <DetailRow label="Source ID" value={entry.source_id || "-"} />
        <DetailRow label="Created" value={formatDateTime(entry.created_at)} />
      </div>

      <div>
        <div className="mb-2 text-sm font-medium">Text</div>
        <div className="max-h-[24rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-background p-4 text-sm">
          {entry.text}
        </div>
      </div>

      <div>
        <div className="mb-2 text-sm font-medium">Chunks</div>
        <div className="space-y-3">
          {entry.chunks.length === 0 ? (
            <div className="text-sm text-muted-foreground">No chunks available</div>
          ) : (
            entry.chunks.map((chunk) => (
              <div
                key={chunk.chunk_id}
                className="rounded-md border border-border bg-background p-3"
              >
                <div className="mb-2 flex items-center justify-between gap-3">
                  <Badge variant="outline">#{chunk.chunk_index}</Badge>
                  <span className="truncate font-mono text-xs text-muted-foreground">
                    {chunk.chunk_id}
                  </span>
                </div>
                <div className="max-h-44 overflow-auto whitespace-pre-wrap break-words text-sm">
                  {chunk.text}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div>
        <div className="mb-2 text-sm font-medium">Metadata</div>
        <MetadataBlock metadata={entry.metadata} />
      </div>
    </div>
  )
}

export function KnowledgePage() {
  const queryClient = useQueryClient()
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)
  const pending = useQuery({
    queryKey: ["knowledge", "pending"],
    queryFn: fetchPendingKnowledge,
  })
  const detail = useQuery({
    queryKey: ["knowledge", "pending", selectedEntryId],
    queryFn: () => fetchPendingKnowledgeEntry(selectedEntryId!),
    enabled: selectedEntryId !== null,
  })

  const approve = useMutation({
    mutationFn: approveKnowledge,
    onSuccess: (_entry, entryId) => {
      if (selectedEntryId === entryId) {
        setSelectedEntryId(null)
      }
      void queryClient.invalidateQueries({ queryKey: ["knowledge", "pending"] })
    },
  })
  const reject = useMutation({
    mutationFn: rejectKnowledge,
    onSuccess: (_entry, entryId) => {
      if (selectedEntryId === entryId) {
        setSelectedEntryId(null)
      }
      void queryClient.invalidateQueries({ queryKey: ["knowledge", "pending"] })
    },
  })

  const entries = pending.data?.entries ?? []
  useEffect(() => {
    if (selectedEntryId === null || pending.isLoading) {
      return
    }
    if (!entries.some((entry) => entry.entry_id === selectedEntryId)) {
      setSelectedEntryId(null)
    }
  }, [entries, pending.isLoading, selectedEntryId])

  const error =
    pending.error instanceof Error
      ? pending.error.message
      : approve.error instanceof Error
        ? approve.error.message
        : reject.error instanceof Error
          ? reject.error.message
          : null
  const detailError = detail.error instanceof Error ? detail.error.message : null
  const mutating = approve.isPending || reject.isPending

  return (
    <Page>
      <PageHeader
        title="Knowledge Review"
        description={`${pending.data?.total_entries_known ?? 0} pending entries`}
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_28rem]">
        <DataCard>
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
                  <TableCell colSpan={6}>
                    <StateMessage>Loading...</StateMessage>
                  </TableCell>
                </TableRow>
              ) : error ? (
                <TableRow>
                  <TableCell colSpan={6}>
                    <StateMessage tone="danger">{error}</StateMessage>
                  </TableCell>
                </TableRow>
              ) : entries.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6}>
                    <StateMessage>No pending knowledge entries</StateMessage>
                  </TableCell>
                </TableRow>
              ) : (
                entries.map((entry) => (
                  <TableRow
                    key={entry.entry_id}
                    className={
                      selectedEntryId === entry.entry_id ? "bg-muted/50" : "cursor-pointer"
                    }
                    onClick={() => setSelectedEntryId(entry.entry_id)}
                  >
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
                      {formatDateTime(entry.created_at)}
                    </TableCell>
                    <TableCell onClick={(event) => event.stopPropagation()}>
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
        </DataCard>

        <DataCard>
          <KnowledgeDetail
            entry={detail.data ?? null}
            isLoading={detail.isLoading}
            error={detailError}
            disabled={mutating}
            onApprove={approve.mutate}
            onReject={reject.mutate}
          />
        </DataCard>
      </div>
    </Page>
  )
}
