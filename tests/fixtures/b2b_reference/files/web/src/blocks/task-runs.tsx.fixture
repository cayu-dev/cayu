import type * as React from "react";
import { Link } from "react-router";
import type { UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";
import { DataTable } from "@kit/patterns";
import { defineStatuses } from "@kit/ui";
import { formatDateTime, formatDuration, formatRelative } from "@kit/lib";
import { queryTableProps } from "@kit/data";

/**
 * BUILDING BLOCK. A list of an Agent's Cayu tasks, one row per run of its work,
 * with a slot for what the Agent says about each run. The generated
 * route groups tasks by the key the Agent names (a pass, a digest, a scan), so a run that
 * spans several tasks after a lost worker is still one row.
 */
export const taskRun = z.object({
  id: z.string(),
  title: z.string(),
  task_type: z.string(),
  status: z.string(),
  started_at: z.string(),
  ended_at: z.string().nullable(),
  attempts: z.number(),
  tasks: z.number(),
});
export type TaskRun = z.infer<typeof taskRun>;

export const taskRunStatus = defineStatuses<string>({
  queued: { label: "Queued", tone: "neutral" },
  scheduled: { label: "Scheduled", tone: "waiting" },
  running: { label: "Running", tone: "progress" },
  completed: { label: "Completed", tone: "success" },
  partly_failed: { label: "Partly failed", tone: "warning" },
  failed: { label: "Failed", tone: "danger" },
  // Not every task of the run could be read, so it isn't shown as ended.
  partial: { label: "Not all read", tone: "neutral" },
});

export function TaskRuns<T extends TaskRun>({
  label,
  query,
  itemLabel,
  summary,
  extraColumns = [],
  onOpen,
  emptyDescription,
}: {
  label: string;
  query: UseQueryResult<T[]>;
  itemLabel: { singular: string; plural: string };
  /** The Agent's slot: what this run did, in its own words. */
  summary?: (row: T) => React.ReactNode;
  extraColumns?: { key: string; label: string; cell: (row: T) => React.ReactNode }[];
  onOpen?: (row: T) => string | null;
  emptyDescription: string;
}) {
  return (
    <div data-origin="block">
      <DataTable
        {...queryTableProps(query, `Couldn't load the ${itemLabel.plural}.`)}
        label={label}
        showHeading={false}
        rows={query.data ?? []}
        rowId={(row) => row.id}
        searchText={(row) => `${row.title} ${row.id}`}
        itemLabel={itemLabel}
        emptyTitle={`No ${itemLabel.plural} yet`}
        emptyDescription={emptyDescription}
        columns={[
          {
            key: "run",
            label: itemLabel.singular[0].toUpperCase() + itemLabel.singular.slice(1),
            sortValue: (row) => row.started_at,
            cell: (row) => {
              const to = onOpen?.(row);
              return (
                <span className="flex min-w-48 flex-col gap-0.5">
                  {to ? (
                    <Link to={to} className="font-medium underline-offset-4 hover:underline">
                      {row.title}
                    </Link>
                  ) : (
                    <span className="font-medium">{row.title}</span>
                  )}
                  <code className="text-footnote text-text-secondary">{row.id}</code>
                </span>
              );
            },
          },
          {
            key: "started",
            label: "Started",
            sortValue: (row) => row.started_at,
            cell: (row) => (
              <time dateTime={row.started_at} title={formatDateTime(row.started_at)}>
                {formatRelative(row.started_at)}
              </time>
            ),
          },
          {
            key: "duration",
            label: "Took",
            cell: (row) =>
              row.ended_at ? <span className="tabular-nums">{formatDuration((Date.parse(row.ended_at) - Date.parse(row.started_at)) / 1000)}</span> : "—",
          },
          { key: "status", label: "Status", sortValue: (row) => row.status, cell: (row) => <taskRunStatus.Badge status={row.status} /> },
          ...extraColumns,
          ...(summary
            ? [{ key: "summary", label: "What it did", cell: (row: T) => <span className="block max-w-96 text-body-sm" data-origin="own">{summary(row)}</span> }]
            : []),
        ]}
      />
    </div>
  );
}
