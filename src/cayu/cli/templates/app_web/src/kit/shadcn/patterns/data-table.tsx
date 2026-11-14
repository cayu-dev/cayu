import * as React from "react";
import {
  columnFilteringFeature,
  createColumnHelper,
  createFilteredRowModel,
  createPaginatedRowModel,
  createSortedRowModel,
  globalFilteringFeature,
  rowPaginationFeature,
  rowSortingFeature,
  tableFeatures,
  useTable,
  type ColumnDef,
  type ColumnFiltersState,
  type OnChangeFn,
  type PaginationState,
  type RowData,
  type SortingState,
  type Updater,
} from "@tanstack/react-table";
import { ArrowDown01Icon, ArrowLeft01Icon, ArrowRight01Icon, ArrowUp01Icon, ArrowUpDownIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "../components/ui/button";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { Skeleton } from "../components/ui/skeleton";
import { Field, FieldLabel, Input } from "../components/field";
import { AsyncContent, type ContentState } from "./async-content";
import type { ActionResult } from "./action";

/** Rows per page in a client table. */
export const COLLECTION_PAGE_SIZE = 12;

/** The TanStack Table features every kit table uses. */
export const dataTableFeatures = tableFeatures({
  columnFilteringFeature,
  globalFilteringFeature,
  rowSortingFeature,
  rowPaginationFeature,
  filteredRowModel: createFilteredRowModel(),
  sortedRowModel: createSortedRowModel(),
  paginatedRowModel: createPaginatedRowModel(),
});
export type DataTableFeatures = typeof dataTableFeatures;

/** One column of the client `DataTable`. */
export type DataColumn<Row extends RowData> = {
  key: string;
  label: string;
  cell: (row: Row) => React.ReactNode;
  /** Makes the column sortable: numbers numerically, text in natural order ("PR 2" before "PR 10"). */
  sortValue?: (row: Row) => string | number;
  /** Makes the column filterable from outside: a column filter `{ id: key, value }` keeps rows whose value matches. */
  filterValue?: (row: Row) => string;
};

/** The table state an owner (the URL) can hold, in TanStack Table's shapes. */
export type DataTableState = {
  globalFilter: string;
  sorting: SortingState;
  pagination: PaginationState;
  columnFilters: ColumnFiltersState;
};

/** Controlled state and change handlers; `useUrlTableState` returns exactly these. */
export type DataTableStateProps = {
  state?: Partial<DataTableState>;
  onGlobalFilterChange?: OnChangeFn<unknown>;
  onSortingChange?: OnChangeFn<SortingState>;
  onPaginationChange?: (updater: Updater<PaginationState>, options?: { replace?: boolean }) => void;
  onColumnFiltersChange?: OnChangeFn<ColumnFiltersState>;
};

export type DataTableProps<Row extends RowData> = DataTableStateProps & {
  rows: Row[];
  columns: DataColumn<Row>[];
  rowId: (row: Row) => string;
  /** The text a search matches, per row. Without it the table has no search. */
  searchText?: (row: Row) => string;
  /** Names the table (its caption, and the heading unless `showHeading` is false). */
  label: string;
  showHeading?: boolean;
  /** The count under the rows: "3 pull requests". */
  itemLabel?: { singular: string; plural: string };
  showCount?: boolean;
  /** Search and filters of your own (a `DataTableToolbar`), in place of the built-in search. */
  toolbar?: React.ReactNode;
  /** Show the built-in search. Defaults to whether there is `searchText`. */
  filterable?: boolean;
  /** Beside the heading, or at the end of the toolbar row. */
  actions?: React.ReactNode;
  /** Overrides the derived state (loading, error, …); `queryTableProps` fills these. */
  contentState?: ContentState;
  contentMessage?: string;
  busy?: boolean;
  onRetry?: () => Promise<ActionResult>;
  emptyTitle?: string;
  emptyDescription?: React.ReactNode;
  emptyAction?: React.ReactNode;
  pageSize?: number;
};

function compareNatural(a: unknown, b: unknown) {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a ?? "").localeCompare(String(b ?? ""), "en-US", { numeric: true });
}

/** Whether a search or any column filter narrows the rows. */
function isFiltered(query: string, filters: ColumnFiltersState) {
  return query !== "" || filters.some(({ value }) => value !== undefined && value !== null && value !== "" && !(Array.isArray(value) && value.length === 0));
}

/** A search field with its label, and any filters of your own beside it. */
export function DataTableToolbar({
  searchLabel,
  query,
  onQueryChange,
  searchRef,
  children,
}: {
  searchLabel: string;
  query: string;
  onQueryChange: (query: string) => void;
  searchRef?: React.Ref<HTMLInputElement>;
  children?: React.ReactNode;
}) {
  return (
    <div data-slot="data-table-toolbar" className="flex flex-wrap items-end gap-3">
      <Field className="w-full sm:w-72">
        <FieldLabel>{searchLabel}</FieldLabel>
        <Input ref={searchRef} type="search" value={query} onChange={(event) => onQueryChange(event.target.value)} />
      </Field>
      {children}
    </div>
  );
}

function SortIcon({ direction }: { direction: false | "asc" | "desc" }) {
  const icon = direction === "asc" ? ArrowUp01Icon : direction === "desc" ? ArrowDown01Icon : ArrowUpDownIcon;
  return <HugeiconsIcon icon={icon} strokeWidth={2} data-icon="inline-end" className={direction ? undefined : "opacity-50"} />;
}

/**
 * A client table of finite rows on the stock shadcn Table and TanStack Table:
 * a search over `searchText`, sortable columns (`sortValue`), pages of twelve, a
 * count, and loading, empty, no-results and error states. Spread
 * `useUrlTableState(…)` to keep its search, sort and page in the URL, and
 * `queryTableProps(query, …)` for a query's states.
 */
export function DataTable<Row extends RowData>({
  rows,
  columns,
  rowId,
  searchText,
  state,
  onGlobalFilterChange,
  onSortingChange,
  onPaginationChange,
  onColumnFiltersChange,
  label,
  showHeading = true,
  itemLabel = { singular: "result", plural: "results" },
  showCount = true,
  toolbar,
  filterable,
  actions,
  contentState,
  contentMessage,
  busy = false,
  onRetry,
  emptyTitle,
  emptyDescription,
  emptyAction,
  pageSize = COLLECTION_PAGE_SIZE,
}: DataTableProps<Row>) {
  // Read through a ref: TanStack calls these during render, and they must not invalidate the table.
  const callbacks = React.useRef({ rowId, searchText });
  callbacks.current = { rowId, searchText };

  const byKey = new Map(columns.map((column) => [column.key, column]));
  const columnShape = columns.map((column) => `${column.key}:${column.sortValue ? 1 : 0}${column.filterValue ? 1 : 0}`).join("|");
  const latestColumns = React.useRef(columns);
  latestColumns.current = columns;
  // Rebuilt only when the set of columns changes; cells render from the latest `columns` below.
  const columnDefs = React.useMemo(() => {
    const helper = createColumnHelper<DataTableFeatures, Row>();
    return latestColumns.current.map((column) =>
      helper.accessor((row) => latestColumns.current.find((latest) => latest.key === column.key)?.sortValue?.(row) ?? "", {
        id: column.key,
        header: column.label,
        enableSorting: !!column.sortValue,
        sortFn: (a, b, id) => compareNatural(a.getValue(id), b.getValue(id)),
        enableColumnFilter: !!column.filterValue,
        filterFn: (row, _id, value) => {
          const actual = latestColumns.current.find((latest) => latest.key === column.key)?.filterValue?.(row.original);
          return Array.isArray(value) ? value.includes(actual) : actual === value;
        },
      }),
    ) as ColumnDef<DataTableFeatures, Row>[];
  }, [columnShape]);

  const firstColumn = columns[0]?.key;
  // Only the parts the owner holds: an undefined key would replace the table's own state.
  const controlled = Object.fromEntries(Object.entries(state ?? {}).filter(([, value]) => value !== undefined)) as Partial<DataTableState>;
  const table = useTable({
    features: dataTableFeatures,
    data: rows,
    columns: columnDefs,
    getRowId: (row) => callbacks.current.rowId(row),
    state: controlled,
    getColumnCanGlobalFilter: (column) => column.id === firstColumn,
    globalFilterFn: (row, _columnId, value) => {
      const text = callbacks.current.searchText?.(row.original);
      return text === undefined || text.toLocaleLowerCase().includes(String(value).toLocaleLowerCase());
    },
    enableSortingRemoval: false,
    sortDescFirst: false,
    autoResetPageIndex: false,
    initialState: { pagination: { pageIndex: 0, pageSize } },
    ...(onGlobalFilterChange && { onGlobalFilterChange }),
    ...(onSortingChange && { onSortingChange }),
    ...(onColumnFiltersChange && { onColumnFiltersChange }),
    ...(onPaginationChange && { onPaginationChange: (updater: Updater<PaginationState>) => onPaginationChange(updater) }),
  });

  const query = String(table.atoms.globalFilter.get() ?? "");
  const filtered = isFiltered(query, table.atoms.columnFilters.get());
  const { pageIndex } = table.atoms.pagination.get();
  const pageRows = table.getRowModel().rows;
  const pageCount = Math.max(1, table.getPageCount());
  const total = table.getRowCount();

  // A page past the end (a stale link, fewer rows now) goes to the last page.
  React.useEffect(() => {
    if (pageIndex > pageCount - 1) table.setPageIndex(pageCount - 1);
  }, [pageIndex, pageCount, table]);

  const resolved: ContentState = contentState ?? (pageRows.length ? "ready" : filtered ? "no-results" : "empty");
  const search = filterable ?? !!searchText;
  const toolbarRow =
    toolbar ??
    (search ? (
      <DataTableToolbar
        searchLabel={`Filter ${label.toLocaleLowerCase()}`}
        query={query}
        onQueryChange={(next) => {
          table.setGlobalFilter(next);
          table.setPageIndex(0);
        }}
      />
    ) : null);
  const first = pageIndex * pageSize + 1;
  const last = Math.min(first + pageRows.length - 1, total);
  const count = `${total} ${total === 1 ? itemLabel.singular : itemLabel.plural}`;
  const showFooter = resolved === "ready" || resolved === "refreshing" || resolved === "refresh-error";

  const skeleton = (
    <div className="overflow-hidden rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column.key}>{column.label}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {Array.from({ length: Math.min(pageSize, 5) }, (_, index) => (
            <TableRow key={index}>
              {columns.map((column) => (
                <TableCell key={column.key}>
                  <Skeleton className="h-4 w-3/4" />
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );

  return (
    <section data-slot="data-table" className="flex w-full min-w-0 flex-col gap-4">
      {showHeading || (actions && !toolbarRow) ? (
        <div className="flex flex-wrap items-center justify-between gap-3">
          {showHeading ? <h2 className="text-base font-medium">{label}</h2> : null}
          {toolbarRow ? null : actions}
        </div>
      ) : null}
      {toolbarRow ? (
        <div data-slot="data-table-controls" className="flex flex-wrap items-end gap-3">
          <div className="min-w-0 grow">{toolbarRow}</div>
          {actions && !showHeading ? <div className="flex items-end gap-2">{actions}</div> : null}
        </div>
      ) : null}
      <AsyncContent
        state={resolved}
        message={contentMessage}
        onRetry={onRetry}
        skeleton={skeleton}
        emptyTitle={emptyTitle}
        emptyDescription={emptyDescription}
        emptyAction={emptyAction}
        onClearFilters={() => {
          table.setGlobalFilter("");
          table.setColumnFilters([]);
          table.setPageIndex(0);
        }}
      >
        <div className="overflow-hidden rounded-lg border">
          {/* On phones the table keeps its columns readable and scrolls sideways, rather than squeezing them to a word a line. */}
          <Table className="max-md:w-max max-md:min-w-full">
            <TableCaption className="sr-only">{label}</TableCaption>
            <TableHeader className="bg-muted/50">
              {table.getHeaderGroups().map((group) => (
                <TableRow key={group.id} className="hover:bg-transparent">
                  {group.headers.map((header) => {
                    const direction = header.column.getIsSorted();
                    const column = byKey.get(header.column.id);
                    return (
                      <TableHead
                        key={header.id}
                        className="px-3 text-muted-foreground"
                        aria-sort={direction === "asc" ? "ascending" : direction === "desc" ? "descending" : undefined}
                      >
                        {header.column.getCanSort() ? (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-2.5 text-muted-foreground"
                            onClick={() => {
                              header.column.toggleSorting(undefined, false);
                              table.setPageIndex(0);
                            }}
                          >
                            {column?.label}
                            <SortIcon direction={direction} />
                          </Button>
                        ) : (
                          column?.label
                        )}
                      </TableHead>
                    );
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {pageRows.map((row) => (
                <TableRow key={row.id}>
                  {row.getAllCells().map((cell) => (
                    <TableCell key={cell.id} className="px-3 py-2.5 whitespace-normal">
                      <div className="max-md:max-w-72">{byKey.get(cell.column.id)?.cell(row.original)}</div>
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </AsyncContent>
      {showFooter && (showCount || pageCount > 1) ? (
        <div data-slot="data-table-footer" className="flex flex-wrap items-center justify-between gap-3">
          <p role="status" className="text-sm text-muted-foreground tabular-nums">
            {busy ? "Loading…" : pageCount > 1 ? `${first}–${last} of ${count}` : count}
          </p>
          {pageCount > 1 ? (
            <nav aria-label={`${label} pages`} className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={!table.getCanPreviousPage()} onClick={() => table.previousPage()}>
                <HugeiconsIcon icon={ArrowLeft01Icon} strokeWidth={2} data-icon="inline-start" />
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled={!table.getCanNextPage()} onClick={() => table.nextPage()}>
                Next
                <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} data-icon="inline-end" />
              </Button>
            </nav>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
