import * as React from "react";
import { functionalUpdate, type ColumnFiltersState, type PaginationState, type SortingState, type Updater } from "@tanstack/react-table";
import { COLLECTION_PAGE_SIZE, type DataTableState } from "./data-table";

/** Sets the search params from the current ones (React Router's `setSearchParams` fits). */
export type SetSearchParams = (next: (current: URLSearchParams) => URLSearchParams, options?: { replace?: boolean }) => void;
/** `useSearchParams()` from React Router, or an equivalent pair. */
export type SearchParamsState = readonly [URLSearchParams, SetSearchParams];

/** How long typing pauses before the search reaches the URL. */
const SEARCH_DEBOUNCE_MS = 250;

/**
 * Applies the search-param changes made in one tick as one navigation. A sort
 * change followed by "back to page 1" would otherwise each start from the old
 * URL, and the second would undo the first. It replaces the history entry only
 * when every change asked to.
 */
function useBatchedSearchUpdate(setSearch: SetSearchParams) {
  const batch = React.useRef<{ patch: Record<string, string | null>; replace: boolean } | null>(null);
  return React.useCallback(
    (patch: Record<string, string | null>, replace = false) => {
      if (!batch.current) {
        const current = { patch: {} as Record<string, string | null>, replace: true };
        batch.current = current;
        queueMicrotask(() => {
          batch.current = null;
          setSearch(
            (params) => {
              const next = new URLSearchParams(params);
              for (const [key, value] of Object.entries(current.patch)) {
                if (value === null || value === "") next.delete(key);
                else next.set(key, value);
              }
              return next;
            },
            { replace: current.replace },
          );
        });
      }
      Object.assign(batch.current.patch, patch);
      batch.current.replace &&= replace;
    },
    [setSearch],
  );
}

/**
 * A client table's search, sort, page and column filters, kept in the URL: `q`,
 * `sort` (`column` or `-column`), `page` (left out on page 1) and the params named
 * in `filters`. Spread the result into `DataTable`; `query` and `setQuery` feed a
 * `DataTableToolbar`. The search types into a draft that reaches the URL once
 * typing pauses, so fast typing isn't lost to navigation.
 */
export function useUrlTableState({
  searchParams,
  param,
  filters = {},
  initialSort,
  pageSize = COLLECTION_PAGE_SIZE,
}: {
  searchParams: SearchParamsState;
  /** Prefixes the params when a page has more than one table. */
  param?: string;
  /** Column id → search param, for column filters the URL holds. */
  filters?: Record<string, string>;
  initialSort?: SortingState;
  pageSize?: number;
}) {
  const [search, setSearch] = searchParams;
  const update = useBatchedSearchUpdate(setSearch);
  const name = (key: string) => (param ? `${param}-${key}` : key);
  const qParam = name("q");
  const sortParam = name("sort");
  const pageParam = name("page");

  const urlQuery = search.get(qParam) ?? "";
  const [draft, setDraft] = React.useState(urlQuery);
  // Searches sent to the URL that it hasn't shown yet. When the URL catches up with one
  // of ours, the draft (perhaps typed further meanwhile) stays; a change from elsewhere
  // (Back, a link, Clear filters) replaces it.
  const [sent, setSent] = React.useState<string[]>([]);
  const [seen, setSeen] = React.useState(urlQuery);
  if (urlQuery !== seen) {
    setSeen(urlQuery);
    const ours = sent.indexOf(urlQuery);
    if (ours >= 0) setSent(sent.slice(ours + 1));
    else {
      setSent([]);
      setDraft(urlQuery);
    }
  }
  React.useEffect(() => {
    if (draft === urlQuery) return;
    const timer = setTimeout(
      () => {
        setSent((current) => [...current, draft]);
        update({ [qParam]: draft, [pageParam]: null }, true);
      },
      draft === "" ? 0 : SEARCH_DEBOUNCE_MS,
    );
    return () => clearTimeout(timer);
  }, [draft, urlQuery, update, qParam, pageParam]);

  const sortValue = search.get(sortParam);
  const sorting: SortingState = sortValue ? [{ id: sortValue.replace(/^-/, ""), desc: sortValue.startsWith("-") }] : (initialSort ?? []);
  const requested = Number(search.get(pageParam));
  const page = Number.isInteger(requested) && requested > 1 ? requested : 1;
  const pagination: PaginationState = { pageIndex: page - 1, pageSize };
  const columnFilters: ColumnFiltersState = Object.entries(filters).flatMap(([id, key]) => {
    const value = search.get(key);
    return value ? [{ id, value }] : [];
  });

  return {
    state: { globalFilter: draft, sorting, pagination, columnFilters } satisfies DataTableState,
    query: draft,
    setQuery: setDraft,
    onGlobalFilterChange: (updater: Updater<unknown>) => {
      const value = functionalUpdate(updater, draft as unknown);
      setDraft(typeof value === "string" ? value : "");
    },
    onSortingChange: (updater: Updater<SortingState>) => {
      const [next] = functionalUpdate(updater, sorting);
      update({ [sortParam]: next ? `${next.desc ? "-" : ""}${next.id}` : null, [pageParam]: null });
    },
    onPaginationChange: (updater: Updater<PaginationState>, options: { replace?: boolean } = {}) => {
      const next = functionalUpdate(updater, pagination).pageIndex + 1;
      if (next !== page) update({ [pageParam]: next > 1 ? String(next) : null }, options.replace);
    },
    onColumnFiltersChange: (updater: Updater<ColumnFiltersState>) => {
      const next = functionalUpdate(updater, columnFilters);
      const patch: Record<string, string | null> = { [pageParam]: null };
      for (const [id, key] of Object.entries(filters)) {
        const value = next.find((filter) => filter.id === id)?.value;
        patch[key] = typeof value === "string" ? value : null;
      }
      update(patch);
    },
  };
}
