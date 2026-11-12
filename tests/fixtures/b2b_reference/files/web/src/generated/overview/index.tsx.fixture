import type * as React from "react";
import { Link } from "react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { AppPageDescription, AppPageHeader, AppPageSection, AppPageTitle, Metric, MetricGroup } from "@kit/layouts/app";
import { AsyncContent, DataTable } from "@kit/patterns";
import { Alert, AlertDescription, AlertTitle, Skeleton } from "@kit/ui";
import { formatDateTime } from "@kit/lib";
import { queryContentProps } from "@kit/data";
import { DashboardSquare01Icon } from "@kit/icons";
import type { AppArea } from "../../app/area";
import { useAreaApi } from "../../app/services";
import { overviewApi, type Overview, type OverviewApi } from "../../api/overview";

export const overviewQuery = (api: OverviewApi) => queryOptions({ queryKey: ["overview"], queryFn: ({ signal }) => api.get(signal), refetchInterval: 30_000 });

/**
 * GENERATED. The Overview area. Every Agent's app has it: the Agent's metrics, what waits
 * for a person in Cayu's inbox, then the sections the Agent adds (its own components).
 */
export function overviewArea({ description, sections = [] }: { description: string; sections?: React.ComponentType[] }): AppArea {
  return {
    id: "overview",
    label: "Overview",
    icon: DashboardSquare01Icon,
    path: "/",
    routes: () => [{ index: true, element: <OverviewPage description={description} sections={sections} /> }],
  };
}

function OverviewPage({ description, sections }: { description: string; sections: React.ComponentType[] }) {
  const api = useAreaApi(overviewApi);
  const overview = useQuery(overviewQuery(api));
  return (
    <div className="flex flex-col gap-page-section" data-origin="generated">
      <AppPageHeader>
        <div className="flex flex-col gap-1">
          <AppPageTitle>Overview</AppPageTitle>
          <AppPageDescription>
            {description}
            {overview.data ? ` As of ${formatDateTime(overview.data.as_of)}.` : null}
          </AppPageDescription>
        </div>
      </AppPageHeader>
      <AsyncContent {...queryContentProps(overview, "Couldn't load the overview.")} skeleton={<Skeleton className="h-64 w-full" />}>
        {overview.data ? <OverviewContent data={overview.data} /> : null}
      </AsyncContent>
      {sections.map((Section, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: the Agent's sections are a fixed list
        <div key={index} data-origin="own">
          <Section />
        </div>
      ))}
    </div>
  );
}

function OverviewContent({ data }: { data: Overview }) {
  return (
    <div className="flex flex-col gap-page-section">
      <div data-origin="own">
        <MetricGroup>
          {data.metrics.map((metric) => (
            <Metric key={metric.label} label={metric.label} value={metric.value} description={metric.description ?? undefined} />
          ))}
        </MetricGroup>
      </div>
      <AppPageSection title="Waiting for a person" description="Items in Cayu's inbox for this Agent, by kind, and where to decide them.">
        {data.inbox.unavailable ? (
          <Alert variant="info">
            <AlertTitle>No inbox counts for you</AlertTitle>
            <AlertDescription>{data.inbox.unavailable}</AlertDescription>
          </Alert>
        ) : (
        <DataTable
          label="Waiting for a person"
          showHeading={false}
          rows={data.inbox.by_kind}
          rowId={(row) => row.kind}
          itemLabel={{ singular: "kind", plural: "kinds" }}
          emptyTitle="Nothing is waiting"
          emptyDescription={data.inbox.empty}
          columns={[
            { key: "label", label: "Kind", cell: (row) => row.label },
            { key: "count", label: "Waiting", cell: (row) => <span className="tabular-nums">{row.count}</span> },
            { key: "to", label: "", cell: (row) => (row.to ? <DecideLink to={row.to} /> : "—") },
          ]}
        />
        )}
      </AppPageSection>
    </div>
  );
}

/** Where to decide an inbox kind: one of the app's pages, or Cayu's Developer dashboard (`/cayu/...`), outside the app. */
function DecideLink({ to }: { to: string }) {
  const className = "underline underline-offset-4";
  return to.startsWith("/cayu/") ? (
    <a href={to} className={className}>
      Decide
    </a>
  ) : (
    <Link to={to} className={className}>
      Decide
    </Link>
  );
}
