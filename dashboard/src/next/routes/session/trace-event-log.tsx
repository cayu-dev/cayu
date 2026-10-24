import { useQuery, useQueryClient } from "@tanstack/react-query"
import { type ReactNode, useState } from "react"
import {
  eventHistoryFilters,
  eventHistoryQuery,
  initialPageNavigation,
  navigateToNewerPage,
  navigateToOlderPage,
} from "@/lib/session-history"
import type { SessionHistorySearch } from "@/lib/session-history-search"
import type { SessionEvent } from "../../api/types"
import { useApi } from "../../app/services"
import { NOISY_EVENT_TYPES } from "../../app/trace-model"
import { ErrorState } from "../common"

const fields = [
  ["event_id", "Event ID"],
  ["event_type", "Event type"],
  ["tool_name", "Tool"],
  ["agent_name", "Specialist"],
  ["environment_name", "Environment"],
  ["workflow_name", "Workflow"],
] as const

type TraceEventLogProps = {
  sessionId: string
  search: SessionHistorySearch
  onSearch: (search: SessionHistorySearch) => void
  selected?: string
  onSelect: (id: string | undefined) => void
  renderDetail: (event: SessionEvent, observedAt: number) => ReactNode
}

/** Filter navigation resets its page, while the session owns the pause choice. */
export function TraceEventLog(props: TraceEventLogProps) {
  const [paused, setPaused] = useState(false)
  return (
    <TraceEventLogPage
      key={JSON.stringify(props.search)}
      {...props}
      paused={paused}
      setPaused={setPaused}
    />
  )
}

function TraceEventLogPage({
  sessionId,
  search,
  onSearch,
  selected,
  onSelect,
  renderDetail,
  paused,
  setPaused,
}: TraceEventLogProps & { paused: boolean; setPaused: (paused: boolean) => void }) {
  const api = useApi()
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState(search)
  const [hideNoise, setHideNoise] = useState(true)
  const [navigation, setNavigation] = useState(() => initialPageNavigation<number | null>(null))
  const suppressNoise = hideNoise && !search.event_type && !search.event_id
  const queryKey = ["trace-events", sessionId, search, navigation.current, hideNoise]
  const query = useQuery({
    queryKey,
    queryFn: ({ signal }) =>
      api.events(
        sessionId,
        {
          ...eventHistoryQuery(eventHistoryFilters(search)),
          exclude_event_type: suppressNoise ? "model.text.delta" : undefined,
          before_sequence: navigation.current ?? undefined,
          order_by: "sequence_desc",
          limit: 100,
        },
        signal,
      ),
    meta: { sessionProgress: !paused && navigation.current === null },
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const selectedQuery = useQuery({
    queryKey: ["trace-event", sessionId, selected],
    staleTime: 0,
    enabled: !!selected,
    queryFn: ({ signal }) => api.events(sessionId, { event_id: selected, limit: 1 }, signal),
  })
  const events = query.data?.events ?? []
  const visible = suppressNoise
    ? events.filter((event) => !NOISY_EVENT_TYPES.has(event.type))
    : events
  const event = selectedQuery.data?.events[0]
  return (
    <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
      <div className="min-w-0 flex-1 overflow-auto p-4">
        <form
          className="mb-3 flex flex-wrap items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            setNavigation(initialPageNavigation(null))
            onSearch(draft)
          }}
        >
          {fields.map(([key, label]) => (
            <label key={key} className="text-xs">
              {label}
              <input
                className="mt-1 block w-36 rounded border bg-background px-2 py-1"
                value={draft[key] ?? ""}
                onChange={(event) => setDraft({ ...draft, [key]: event.target.value || undefined })}
              />
            </label>
          ))}
          <button type="submit" className="rounded border px-3 py-1 text-sm">
            Apply filters
          </button>
          <button
            type="button"
            className="rounded border px-3 py-1 text-sm"
            onClick={() => {
              setDraft({})
              setNavigation(initialPageNavigation(null))
              onSearch({})
            }}
          >
            Clear filters
          </button>
        </form>
        <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
          <label>
            <input
              type="checkbox"
              checked={hideNoise}
              onChange={(event) => setHideNoise(event.target.checked)}
            />{" "}
            Hide noisy events
          </label>
          <button
            type="button"
            className="rounded border px-2 py-1"
            onClick={() => {
              // Pause freezes an observed page. Let its initial explicit read finish;
              // otherwise cancelling it leaves a paused view with no snapshot.
              if (!paused && query.data) void queryClient.cancelQueries({ queryKey, exact: true })
              else if (paused) void query.refetch()
              setPaused(!paused)
            }}
          >
            {paused ? "Resume live tail" : "Pause live tail"}
          </button>
          <span>
            {navigation.current === null ? "Latest page" : "Older history"} · {visible.length}{" "}
            visible of {events.length} loaded events
          </span>
        </div>
        {query.error && <ErrorState error={query.error} />}
        {query.isPending && <p>Loading events…</p>}
        <ol className="space-y-1">
          {visible.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                aria-pressed={selected === item.id}
                className="flex w-full gap-3 rounded px-2 py-2 text-left text-xs hover:bg-muted focus-visible:outline"
                onClick={() => onSelect(item.id)}
              >
                <span className="font-mono">{item.sequence}</span>
                <span>{new Date(item.timestamp).toLocaleTimeString()}</span>
                <span className="break-all">{item.type}</span>
              </button>
            </li>
          ))}
        </ol>
        {!query.isPending && visible.length === 0 && (
          <p className="py-4 text-sm text-muted-foreground">No visible events on this page.</p>
        )}
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            className="rounded border px-3 py-1 text-sm"
            disabled={navigation.newer.length === 0}
            onClick={() => setNavigation(navigateToNewerPage(navigation))}
          >
            Newer events
          </button>
          <button
            type="button"
            className="rounded border px-3 py-1 text-sm"
            disabled={!query.data?.has_more || query.isFetching}
            onClick={() => {
              const next = query.data?.next_sequence
              if (next != null) setNavigation(navigateToOlderPage(navigation, next))
            }}
          >
            Older events
          </button>
          {navigation.current !== null && (
            <button
              type="button"
              className="rounded border px-3 py-1 text-sm"
              onClick={() => setNavigation(initialPageNavigation(null))}
            >
              Latest events
            </button>
          )}
        </div>
      </div>
      {selected && (
        <aside
          aria-label="Event detail"
          className="max-h-[45vh] overflow-auto border-t lg:max-h-none lg:w-[420px] lg:border-l lg:border-t-0"
        >
          {selectedQuery.error ? (
            <ErrorState error={selectedQuery.error} />
          ) : event ? (
            renderDetail(event, selectedQuery.dataUpdatedAt)
          ) : (
            <p className="p-4">
              {selectedQuery.isPending ? "Loading event…" : "Event is not retained."}
            </p>
          )}
        </aside>
      )}
    </div>
  )
}
