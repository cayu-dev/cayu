import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  BookOpen,
  Check,
  FileText,
  Layers,
  type LucideIcon,
  RefreshCw,
  ShieldCheck,
  Tag,
  X,
} from "lucide-react"
import { type ReactNode, useEffect, useMemo, useState } from "react"
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
  approveKnowledge,
  fetchPendingKnowledge,
  fetchPendingKnowledgeEntry,
  type KnowledgeEntry,
  type KnowledgeEntryDetail,
  type KnowledgePendingPage,
  rejectKnowledge,
} from "../lib/api"
import { formatCount, formatDateTime } from "../lib/format"
import { cn } from "../lib/utils"

type PendingEntry = KnowledgePendingPage["entries"][number]

const PENDING_LIST_QUERY = {
  limit: 50,
  max_bytes: 40_000,
} as const

function Labels({ labels, max = 8 }: { labels: Record<string, string>; max?: number }) {
  const entries = Object.entries(labels)
  if (entries.length === 0) {
    return <span className="text-muted-foreground">-</span>
  }
  return (
    <div className="flex flex-wrap gap-1">
      {entries.slice(0, max).map(([key, value]) => (
        <Badge key={key} variant="outline" className="font-mono text-[10px]">
          {key}={value}
        </Badge>
      ))}
      {entries.length > max && (
        <Badge variant="outline" className="font-mono text-[10px]">
          +{entries.length - max}
        </Badge>
      )}
    </div>
  )
}

function Aspects({ aspects, max = 8 }: { aspects: string[]; max?: number }) {
  if (aspects.length === 0) {
    return <span className="text-muted-foreground">-</span>
  }
  return (
    <div className="flex flex-wrap gap-1">
      {aspects.slice(0, max).map((aspect) => (
        <Badge key={aspect} variant="secondary" className="font-mono text-[10px]">
          {aspect}
        </Badge>
      ))}
      {aspects.length > max && (
        <Badge variant="secondary" className="font-mono text-[10px]">
          +{aspects.length - max}
        </Badge>
      )}
    </div>
  )
}

function ReviewActions({
  entryId,
  disabled,
  isApproving,
  isRejecting,
  onApprove,
  onReject,
}: {
  entryId: string
  disabled: boolean
  isApproving: boolean
  isRejecting: boolean
  onApprove: (entryId: string) => void
  onReject: (entryId: string) => void
}) {
  return (
    <div className="flex items-center justify-end gap-2">
      <Button variant="outline" size="sm" disabled={disabled} onClick={() => onApprove(entryId)}>
        <Check className="h-3.5 w-3.5" />
        {isApproving ? "Approving..." : "Approve"}
      </Button>
      <Button variant="destructive" size="sm" disabled={disabled} onClick={() => onReject(entryId)}>
        <X className="h-3.5 w-3.5" />
        {isRejecting ? "Rejecting..." : "Reject"}
      </Button>
    </div>
  )
}

function MetadataBlock({ metadata }: { metadata: Record<string, unknown> }) {
  if (Object.keys(metadata).length === 0) {
    return <span className="text-muted-foreground">-</span>
  }
  return <PayloadViewer value={metadata} maxHeight="max-h-72" />
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-3 text-sm">
      <div className="text-muted-foreground">{label}</div>
      <div className="min-w-0 break-words">{value}</div>
    </div>
  )
}

function SummaryStat({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: LucideIcon
  label: string
  value: ReactNode
  detail: ReactNode
}) {
  return (
    <DataCard contentClassName="p-4">
      <div className="flex min-w-0 gap-3">
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
        <div className="min-w-0">
          <div className="text-sm font-medium text-muted-foreground">{label}</div>
          <div className="mt-1 truncate text-2xl font-semibold">{value}</div>
          <div className="mt-1 truncate text-xs text-muted-foreground">{detail}</div>
        </div>
      </div>
    </DataCard>
  )
}

function sourceLabel(entry: Pick<PendingEntry, "source_type" | "source_id" | "source_uri">) {
  if (entry.source_type && entry.source_id) return `${entry.source_type}:${entry.source_id}`
  if (entry.source_type) return entry.source_type
  if (entry.source_id) return entry.source_id
  if (entry.source_uri) return entry.source_uri
  return "-"
}

function entryTitle(entry: Pick<PendingEntry, "entry_id" | "title">) {
  return entry.title || entry.entry_id
}

function entryPreview(entry: Pick<PendingEntry, "text_preview">) {
  return entry.text_preview || "No preview available."
}

function PendingEntryButton({
  entry,
  selected,
  disabled,
  onSelect,
}: {
  entry: PendingEntry
  selected: boolean
  disabled: boolean
  onSelect: (entryId: string) => void
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onSelect(entry.entry_id)}
      className={cn(
        "w-full rounded-md border border-border p-3 text-left transition-colors",
        selected ? "border-primary/40 bg-primary/5" : "hover:bg-muted/60",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate font-medium">{entryTitle(entry)}</div>
          <div className="mt-1 line-clamp-2 text-sm text-muted-foreground">
            {entryPreview(entry)}
          </div>
        </div>
        <Badge variant="outline" className="shrink-0">
          {entry.kind}
        </Badge>
      </div>
      <div className="mt-3 flex min-w-0 flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span className="font-mono">{entry.namespace}</span>
        <span>{formatDateTime(entry.created_at)}</span>
        <span>{formatCount(entry.chunk_count)} chunks</span>
      </div>
      <div className="mt-3 space-y-2">
        <Labels labels={entry.labels} max={4} />
        <Aspects aspects={entry.aspects} max={4} />
      </div>
    </button>
  )
}

function KnowledgeDetail({
  entry,
  isLoading,
  error,
  disabled,
  isLoadingSelection,
  mutatingEntryId,
  action,
  onApprove,
  onReject,
}: {
  entry: KnowledgeEntryDetail | null
  isLoading: boolean
  error: string | null
  disabled: boolean
  isLoadingSelection: boolean
  mutatingEntryId: string | null
  action: "approve" | "reject" | null
  onApprove: (entryId: string) => void
  onReject: (entryId: string) => void
}) {
  if (isLoading) {
    return <StateMessage className="py-16">Loading entry...</StateMessage>
  }
  if (error !== null) {
    return (
      <StateMessage tone="danger" className="py-16">
        {error}
      </StateMessage>
    )
  }
  if (entry === null) {
    return <StateMessage className="py-16">Select a pending entry to review.</StateMessage>
  }

  const isCurrentAction = mutatingEntryId === entry.entry_id
  return (
    <div className="min-h-0 space-y-5 p-4 sm:p-5">
      <div className="flex min-w-0 flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">pending</Badge>
            <Badge variant="outline">{entry.kind}</Badge>
            <Badge variant="outline">{entry.visibility}</Badge>
          </div>
          <h2 className="mt-3 break-words text-xl font-semibold">{entryTitle(entry)}</h2>
          <div className="mt-1 break-all font-mono text-xs text-muted-foreground">
            {entry.entry_id}
          </div>
        </div>
        <ReviewActions
          entryId={entry.entry_id}
          disabled={disabled || isLoadingSelection}
          isApproving={isCurrentAction && action === "approve"}
          isRejecting={isCurrentAction && action === "reject"}
          onApprove={onApprove}
          onReject={onReject}
        />
      </div>

      {isLoadingSelection && (
        <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
          Loading selected entry...
        </div>
      )}

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="space-y-3 rounded-md border border-border bg-muted/30 p-3">
          <DetailRow
            label="Namespace"
            value={<span className="font-mono">{entry.namespace}</span>}
          />
          <DetailRow label="Labels" value={<Labels labels={entry.labels} />} />
          <DetailRow label="Aspects" value={<Aspects aspects={entry.aspects} />} />
          <DetailRow
            label="Targets"
            value={
              entry.impact_targets.length === 0 ? (
                <span className="text-muted-foreground">-</span>
              ) : (
                <Aspects aspects={entry.impact_targets} />
              )
            }
          />
        </div>
        <div className="space-y-3 rounded-md border border-border bg-muted/30 p-3">
          <DetailRow label="Source" value={sourceLabel(entry)} />
          <DetailRow label="URI" value={entry.source_uri || "-"} />
          <DetailRow
            label="Created by"
            value={
              entry.created_by
                ? `${entry.created_by_type}:${entry.created_by}`
                : entry.created_by_type
            }
          />
          <DetailRow label="Created" value={formatDateTime(entry.created_at)} />
        </div>
      </div>

      <section>
        <div className="mb-2 flex items-center justify-between gap-3">
          <h3 className="text-sm font-medium">Captured Text</h3>
          <span className="text-xs text-muted-foreground">
            {entry.text.length.toLocaleString()} chars
          </span>
        </div>
        <div className="max-h-[28rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-background p-4 text-sm leading-relaxed">
          {entry.text}
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between gap-3">
          <h3 className="text-sm font-medium">Chunks</h3>
          <span className="text-xs text-muted-foreground">
            {entry.chunks.length} loaded / limit {entry.chunk_limit}
          </span>
        </div>
        <div className="space-y-3">
          {entry.chunks.length === 0 ? (
            <div className="rounded-md border border-border p-4 text-sm text-muted-foreground">
              No chunks available.
            </div>
          ) : (
            entry.chunks.map((chunk) => (
              <div
                key={chunk.chunk_id}
                className="rounded-md border border-border bg-background p-3"
              >
                <div className="mb-2 flex min-w-0 items-center justify-between gap-3">
                  <Badge variant="outline">chunk {chunk.chunk_index}</Badge>
                  <span className="truncate font-mono text-xs text-muted-foreground">
                    {chunk.chunk_id}
                  </span>
                </div>
                <div className="max-h-48 overflow-auto whitespace-pre-wrap break-words text-sm leading-relaxed">
                  {chunk.text}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-medium">Metadata</h3>
        <MetadataBlock metadata={entry.metadata} />
      </section>
    </div>
  )
}

export function KnowledgePage() {
  const queryClient = useQueryClient()
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)

  const pending = useQuery({
    queryKey: ["knowledge", "pending", PENDING_LIST_QUERY],
    queryFn: () => fetchPendingKnowledge(PENDING_LIST_QUERY),
  })
  const detail = useQuery({
    queryKey: ["knowledge", "pending", selectedEntryId],
    queryFn: () => fetchPendingKnowledgeEntry(selectedEntryId!),
    enabled: selectedEntryId !== null,
    placeholderData: keepPreviousData,
  })

  const entries = pending.data?.entries ?? []
  const summary = useMemo(() => {
    const kinds = new Set(entries.map((entry) => entry.kind))
    const sources = new Set(entries.map(sourceLabel).filter((value) => value !== "-"))
    const chunks = entries.reduce((total, entry) => total + entry.chunk_count, 0)
    return { kinds: kinds.size, sources: sources.size, chunks }
  }, [entries])

  useEffect(() => {
    if (pending.isLoading) return
    if (entries.length === 0) {
      setSelectedEntryId(null)
      return
    }
    const firstEntry = entries[0]
    if (
      firstEntry !== undefined &&
      (selectedEntryId === null || !entries.some((entry) => entry.entry_id === selectedEntryId))
    ) {
      setSelectedEntryId(firstEntry.entry_id)
    }
  }, [entries, pending.isLoading, selectedEntryId])

  function onReviewSuccess(entryId: string) {
    if (selectedEntryId === entryId) {
      setSelectedEntryId(null)
    }
    queryClient.removeQueries({ queryKey: ["knowledge", "pending", entryId] })
    void queryClient.invalidateQueries({ queryKey: ["knowledge", "pending"] })
  }

  const approve = useMutation<KnowledgeEntry, Error, string>({
    mutationFn: approveKnowledge,
    onSuccess: (_entry, entryId) => onReviewSuccess(entryId),
  })
  const reject = useMutation<KnowledgeEntry, Error, string>({
    mutationFn: rejectKnowledge,
    onSuccess: (_entry, entryId) => onReviewSuccess(entryId),
  })

  const listError = pending.error instanceof Error ? pending.error.message : null
  const actionError =
    approve.error instanceof Error
      ? approve.error.message
      : reject.error instanceof Error
        ? reject.error.message
        : null
  const detailError = detail.error instanceof Error ? detail.error.message : null
  const selectedDetail = selectedEntryId === null ? null : (detail.data ?? null)
  const isLoadingSelection =
    selectedEntryId !== null &&
    (detail.isPlaceholderData ||
      (detail.data !== undefined && detail.data.entry_id !== selectedEntryId))
  const mutatingEntryId = approve.variables ?? reject.variables ?? null
  const action = approve.isPending ? "approve" : reject.isPending ? "reject" : null
  const mutating = approve.isPending || reject.isPending

  return (
    <Page>
      <PageHeader
        title="Knowledge Review"
        description="Review model-captured pending knowledge before it becomes active recall."
        actions={
          <Button variant="outline" size="sm" onClick={() => void pending.refetch()}>
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
        }
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryStat
          icon={BookOpen}
          label="Pending Entries"
          value={formatCount(pending.data?.total_entries_known ?? entries.length)}
          detail={pending.data?.truncated ? "list truncated" : "ready for review"}
        />
        <SummaryStat
          icon={Tag}
          label="Kinds"
          value={formatCount(summary.kinds)}
          detail="distinct pending kinds"
        />
        <SummaryStat
          icon={Layers}
          label="Chunks"
          value={formatCount(summary.chunks)}
          detail={
            pending.data ? `preview budget ${formatCount(pending.data.max_bytes)} bytes` : "loading"
          }
        />
        <SummaryStat
          icon={ShieldCheck}
          label="Sources"
          value={formatCount(summary.sources)}
          detail="captured source groups"
        />
      </div>

      {actionError && (
        <DataCard contentClassName="p-4">
          <StateMessage tone="danger" className="py-2 text-left">
            {actionError}
          </StateMessage>
        </DataCard>
      )}

      <div className="grid min-h-[34rem] gap-4 xl:grid-cols-[minmax(18rem,26rem)_minmax(0,1fr)]">
        <DataCard
          title="Pending Queue"
          description={`${entries.length} loaded from the review scope`}
          contentClassName="p-3"
        >
          {pending.isLoading ? (
            <StateMessage className="py-12">Loading pending knowledge...</StateMessage>
          ) : listError ? (
            <StateMessage tone="danger" className="py-12">
              {listError}
            </StateMessage>
          ) : entries.length === 0 ? (
            <StateMessage className="py-12">No pending knowledge entries.</StateMessage>
          ) : (
            <div className="max-h-[58rem] space-y-2 overflow-auto pr-1">
              {entries.map((entry) => (
                <PendingEntryButton
                  key={entry.entry_id}
                  entry={entry}
                  selected={entry.entry_id === selectedEntryId}
                  disabled={mutating}
                  onSelect={setSelectedEntryId}
                />
              ))}
            </div>
          )}
        </DataCard>

        <DataCard
          title="Review Detail"
          description={
            selectedEntryId === null ? "Select an entry to inspect evidence." : selectedEntryId
          }
          contentClassName="min-h-0"
          actions={
            selectedDetail ? (
              <div className="hidden items-center gap-2 lg:flex">
                <FileText className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">
                  {formatCount(selectedDetail.chunks.length)} chunks loaded
                </span>
              </div>
            ) : null
          }
        >
          <KnowledgeDetail
            entry={selectedDetail}
            isLoading={detail.isLoading}
            error={detailError}
            disabled={mutating}
            isLoadingSelection={isLoadingSelection}
            mutatingEntryId={mutatingEntryId}
            action={action}
            onApprove={approve.mutate}
            onReject={reject.mutate}
          />
        </DataCard>
      </div>
    </Page>
  )
}
