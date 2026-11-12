import type * as React from "react";
import { Cancel01Icon, ExclamationMarkIcon, MinusSignIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "../lib/utils";
import { Spinner } from "./ui/spinner";

/** Where a step stands. `attention` stopped for a person: it hasn't failed and won't move on by itself. */
export type TimelineState = "pending" | "active" | "attention" | "succeeded" | "failed" | "cancelled" | "skipped";

const STATE_TEXT: Record<TimelineState, string> = {
  pending: "Not started",
  active: "In progress",
  attention: "Needs attention",
  succeeded: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
  skipped: "Skipped",
};

const MARKER: Record<TimelineState, string> = {
  pending: "border border-border bg-background",
  active: "bg-info/15 text-info",
  attention: "bg-warning/15 text-warning",
  succeeded: "bg-success/15 text-success",
  failed: "bg-destructive/15 text-destructive",
  cancelled: "bg-muted text-muted-foreground",
  skipped: "border border-dashed border-border bg-background text-muted-foreground",
};

/** An ordered list of steps or events; each state has its own marker, and its words are read with the title. */
export function Timeline({ className, ...props }: React.ComponentProps<"ol">) {
  // role="list" keeps list semantics in Safari once list styling is gone.
  return <ol data-slot="timeline" role="list" className={cn("flex min-w-0 flex-col", className)} {...props} />;
}

function Marker({ state }: { state: TimelineState }) {
  const icon =
    state === "succeeded" ? Tick02Icon : state === "attention" ? ExclamationMarkIcon : state === "failed" ? Cancel01Icon : state === "cancelled" || state === "skipped" ? MinusSignIcon : null;
  return (
    <span
      data-slot="timeline-marker"
      aria-hidden="true"
      className={cn("relative z-10 flex size-6 shrink-0 items-center justify-center rounded-full [&_svg]:size-3.5", MARKER[state])}
    >
      {state === "active" ? <Spinner role={undefined} aria-label={undefined} /> : icon ? <HugeiconsIcon icon={icon} strokeWidth={2.5} /> : null}
    </span>
  );
}

export function TimelineItem({
  state,
  title,
  description,
  time,
  children,
  className,
  ...props
}: Omit<React.ComponentProps<"li">, "title"> & { state: TimelineState; title: React.ReactNode; description?: React.ReactNode; time?: React.ReactNode }) {
  const muted = state === "pending" || state === "skipped";
  return (
    <li
      data-slot="timeline-item"
      data-state={state}
      aria-current={state === "active" ? "step" : undefined}
      className={cn("group/timeline-item relative flex min-w-0 gap-3 pb-6 last:pb-0", className)}
      {...props}
    >
      {/* The line from this marker to the next one. */}
      <span aria-hidden="true" className="absolute top-6 bottom-0 left-3 w-px -translate-x-1/2 bg-border group-last/timeline-item:hidden" />
      <Marker state={state} />
      <div className="flex min-w-0 flex-1 flex-col gap-1 pt-0.5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
          <p className={cn("min-w-0 text-sm font-medium wrap-anywhere", muted ? "text-muted-foreground" : "text-foreground")}>
            {title}
            <span className="sr-only">, {STATE_TEXT[state]}</span>
          </p>
          {time ? <span className="shrink-0 text-xs text-muted-foreground tabular-nums">{time}</span> : null}
        </div>
        {description ? <div className="text-sm text-muted-foreground">{description}</div> : null}
        {children ? <div className="mt-1 flex min-w-0 flex-col gap-2">{children}</div> : null}
      </div>
    </li>
  );
}
