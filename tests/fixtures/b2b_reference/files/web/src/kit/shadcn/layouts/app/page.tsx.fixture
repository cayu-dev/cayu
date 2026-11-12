import * as React from "react";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "../../lib/utils";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";

/** The routed page's frame inside the shell: centred, with the page gutter. */
export function AppPage({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="app-page" className={cn("mx-auto w-full max-w-6xl min-w-0 p-4 md:p-6 lg:p-8", className)} {...props} />;
}

/**
 * The page's title row. Children are the title (and anything beside it). With
 * `description` or `actions`, the title and description stack on the left and the
 * actions sit on the right, wrapping below on narrow screens.
 */
export function AppPageHeader({
  className,
  description,
  actions,
  children,
  ...props
}: React.ComponentProps<"div"> & { description?: React.ReactNode; actions?: React.ReactNode }) {
  const slotted = description != null || actions != null;
  return (
    <div data-slot="app-page-header" className={cn("flex flex-wrap items-start justify-between gap-4", className)} {...props}>
      {slotted ? (
        <>
          <div className="flex min-w-0 flex-1 basis-64 flex-col gap-1">
            {children}
            {description != null ? <AppPageDescription>{description}</AppPageDescription> : null}
          </div>
          {actions != null ? <AppPageActions>{actions}</AppPageActions> : null}
        </>
      ) : (
        children
      )}
    </div>
  );
}

export function AppPageTitle({ className, ...props }: React.ComponentProps<"h1">) {
  return <h1 data-slot="app-page-title" className={cn("min-w-0 text-2xl font-semibold tracking-tight outline-none", className)} {...props} />;
}

/** The text under a page title. A div, so it can hold links and code. */
export function AppPageDescription({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="app-page-description" className={cn("max-w-3xl text-sm text-muted-foreground", className)} {...props} />;
}

export function AppPageActions({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="app-page-actions" className={cn("flex max-w-full flex-wrap items-start gap-2", className)} {...props} />;
}

/** A titled part of a page, with an optional description and actions on the right. */
export function AppPageSection({
  title,
  description,
  actions,
  children,
  className,
  ...props
}: Omit<React.ComponentProps<"section">, "title"> & { title: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode }) {
  const titleId = React.useId();
  return (
    <section data-slot="app-page-section" aria-labelledby={titleId} className={cn("flex min-w-0 flex-col gap-4", className)} {...props}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 id={titleId} className="text-lg font-semibold tracking-tight">
            {title}
          </h2>
          {description ? <p className="max-w-3xl text-sm text-muted-foreground">{description}</p> : null}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

/** A record's page: its sections, a page-section gap apart. */
export function AppDetailPage({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="app-detail-page" className={cn("flex min-w-0 flex-col gap-page-section", className)} {...props} />;
}

/** "Back to …": a ghost button, usually rendering the router's Link (`render={<Link to="/runs" />}`). */
export function AppBackLink({ children, className, ...props }: Omit<React.ComponentProps<typeof Button>, "variant" | "size">) {
  return (
    <Button data-slot="app-back-link" variant="ghost" size="sm" className={cn("-ml-2.5 self-start text-muted-foreground", className)} {...props}>
      <HugeiconsIcon icon={ArrowLeft01Icon} strokeWidth={2} data-icon="inline-start" />
      {children}
    </Button>
  );
}

/** A record's header: the way back, its title with a status badge, a description and actions. */
export function AppDetailHeader({
  back,
  title,
  titleRef,
  badge,
  description,
  actions,
  className,
  ...props
}: Omit<React.ComponentProps<"div">, "title" | "children"> & {
  back: React.ReactNode;
  title: React.ReactNode;
  titleRef?: React.Ref<HTMLHeadingElement>;
  badge?: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div data-slot="app-detail-header" className={cn("flex min-w-0 flex-col gap-3", className)} {...props}>
      {back}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-1 basis-64 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-3">
            <AppPageTitle ref={titleRef} tabIndex={-1} className="wrap-anywhere">
              {title}
            </AppPageTitle>
            {badge}
          </div>
          {description ? <AppPageDescription>{description}</AppPageDescription> : null}
        </div>
        {actions ? <AppPageActions>{actions}</AppPageActions> : null}
      </div>
    </div>
  );
}

/** A record's fields as a definition list, in a card; `plain` drops the card for facts inside one. */
export function AppMetadata({ className, variant = "default", ...props }: React.ComponentProps<"dl"> & { variant?: "default" | "plain" }) {
  return (
    <dl
      data-slot="app-metadata"
      data-variant={variant}
      className={cn(
        "grid min-w-0",
        variant === "default"
          ? "grid-cols-1 gap-x-8 gap-y-5 rounded-xl bg-card p-4 text-card-foreground ring-1 ring-foreground/10 sm:grid-cols-2 xl:grid-cols-4"
          : "grid-cols-[repeat(auto-fill,minmax(min(100%,12rem),1fr))] gap-4",
        className,
      )}
      {...props}
    />
  );
}

export function AppMetadataItem({ label, children, className, ...props }: React.ComponentProps<"div"> & { label: React.ReactNode }) {
  return (
    <div data-slot="app-metadata-item" className={cn("flex min-w-0 flex-col gap-1", className)} {...props}>
      <dt className="text-xs font-medium text-muted-foreground">{label}</dt>
      <dd className="text-sm wrap-anywhere">{children}</dd>
    </div>
  );
}

/** Metrics as stock Cards: four a row, fewer when there are fewer, six as two rows of three. */
export function MetricGroup({ className, ...props }: React.ComponentProps<"dl">) {
  return (
    <dl
      data-slot="metric-group"
      className={cn(
        "grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4",
        "sm:has-[>:only-child]:grid-cols-1",
        "xl:has-[>:nth-child(2):last-child]:grid-cols-2 xl:has-[>:nth-child(3):last-child]:grid-cols-3 xl:has-[>:nth-child(6):last-child]:grid-cols-3",
        className,
      )}
      {...props}
    />
  );
}

export function Metric({
  label,
  value,
  description,
  children,
  className,
  ...props
}: Omit<React.ComponentProps<"div">, "children"> & { label: React.ReactNode; value: React.ReactNode; description?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <Card data-slot="metric" className={cn("gap-1.5 px-4", className)} {...props}>
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd data-slot="metric-value" className="text-2xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
      {description || children ? (
        <dd className="flex flex-col gap-1 text-xs text-muted-foreground">
          {description ? <p>{description}</p> : null}
          {children}
        </dd>
      ) : null}
    </Card>
  );
}
