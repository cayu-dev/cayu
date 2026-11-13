/**
 * The Inbox's filters: text search, the advanced filters, and one chip per kind with its
 * exact count. The host keeps the filters (`search`, `onSearch`), usually in the URL.
 */

import { cn } from "@kit/lib"
import type { InboxSearch } from "../host"
import { selectClassName } from "../styles"
import type { InboxItem } from "../types"
import { useInboxCounts } from "./queries"

const KINDS: [InboxItem["kind"] | undefined, string][] = [
  [undefined, "All"],
  ["approval", "Approvals"],
  ["question", "Questions"],
  ["recovery", "Recovery"],
  ["knowledge", "Knowledge"],
]

export function InboxFilters({
  search,
  onSearch,
  advancedFilters = true,
  pageSize,
  onPageSize,
}: {
  search: InboxSearch
  /** Show other filters; the host navigates, or keeps them in state. */
  onSearch: (next: InboxSearch) => void
  /** The session, specialist and page-size filters. An app for one Agent can leave them out. */
  advancedFilters?: boolean
  /** The page size the advanced filters offer; without `onPageSize` there's no choice. */
  pageSize?: number
  onPageSize?: (size: number) => void
}) {
  const all = useInboxCounts({
    q: search.q,
    session_id: search.session_id,
    agent_name: search.agent_name,
  })
  const countFor = (kind: InboxItem["kind"] | undefined) =>
    all.data?.complete === true ? (kind ? all.data.counts?.[kind] : all.data.total_count) : null

  return (
    <>
      <form
        key={JSON.stringify(search)}
        className="mt-4 flex flex-wrap gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          const values = new FormData(event.currentTarget)
          const text = (name: string) => String(values.get(name) ?? "").trim() || undefined
          onSearch({
            ...search,
            q: text("q"),
            ...(advancedFilters
              ? { session_id: text("session_id"), agent_name: text("agent_name") }
              : {}),
          })
        }}
      >
        <label className="text-xs">
          Search Inbox
          <input name="q" defaultValue={search.q} className={cn("ml-2", selectClassName)} />
        </label>
        <button
          type="submit"
          className="h-8 rounded-lg border border-border px-3 text-sm hover:bg-muted focus-visible:outline-ring"
        >
          Filter
        </button>
        {advancedFilters && (
          <details className="w-full text-sm" open={!!search.session_id || !!search.agent_name}>
            <summary className="cursor-pointer text-muted-foreground">Advanced filters</summary>
            <div className="mt-3 flex flex-wrap items-end gap-3">
              <label className="text-xs">
                Session
                <input
                  name="session_id"
                  defaultValue={search.session_id}
                  className={cn("ml-2", selectClassName)}
                />
              </label>
              <label className="text-xs">
                Specialist
                <input
                  name="agent_name"
                  defaultValue={search.agent_name}
                  className={cn("ml-2", selectClassName)}
                />
              </label>
              {onPageSize && (
                <label className="mt-3 block text-xs">
                  Inbox page size
                  <select
                    aria-label="Inbox page size"
                    value={pageSize}
                    onChange={(event) => onPageSize(Number(event.target.value))}
                    className={cn("ml-2", selectClassName)}
                  >
                    {[25, 50, 100].map((size) => (
                      <option key={size} value={size}>
                        {size}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          </details>
        )}
      </form>
      <fieldset className="mt-4 flex min-w-0 flex-wrap gap-1.5">
        <legend className="sr-only">Filter by kind</legend>
        {KINDS.map(([kind, label]) => (
          <button
            key={label}
            type="button"
            aria-pressed={search.kind === kind}
            onClick={() => onSearch({ ...search, kind })}
            className={cn(
              "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px]",
              search.kind === kind
                ? "border-foreground bg-foreground text-background"
                : "border-border hover:bg-muted",
            )}
          >
            {label}
            {(countFor(kind) ?? 0) > 0 && (
              <span
                className={cn(
                  "rounded-full px-1.5 text-xs font-medium tabular-nums",
                  search.kind === kind ? "bg-background/20" : "bg-warning/15 text-warning",
                )}
              >
                {countFor(kind)}
              </span>
            )}
          </button>
        ))}
      </fieldset>
    </>
  )
}
