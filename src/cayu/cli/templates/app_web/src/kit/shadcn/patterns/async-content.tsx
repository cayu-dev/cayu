import * as React from "react";
import { cn } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "../components/ui/empty";
import { Skeleton } from "../components/ui/skeleton";
import { Spinner } from "../components/ui/spinner";
import { useAction, type ActionResult } from "./action";

export type ContentState = "loading" | "ready" | "empty" | "no-results" | "error" | "unavailable" | "refreshing" | "refresh-error";

const TITLES: Record<ContentState, string> = {
  loading: "Loading…",
  ready: "",
  empty: "Nothing here yet",
  "no-results": "No results",
  error: "Couldn't load content",
  unavailable: "Content unavailable",
  refreshing: "Refreshing…",
  "refresh-error": "Couldn't refresh. Showing the last loaded content.",
};

/** Try again: reruns the read, says so while it runs, and keeps the failure's words if it fails again. */
export function RetryButton({ onRetry }: { onRetry: () => Promise<ActionResult> }) {
  const action = useAction();
  return (
    <div className="flex flex-col items-center gap-1.5">
      <Button variant="outline" size="sm" disabled={action.busy} onClick={() => void action.run(onRetry)}>
        {action.busy ? <Spinner data-icon="inline-start" role={undefined} aria-label={undefined} /> : null}
        Try again
      </Button>
      <p role="status" className="text-xs text-muted-foreground empty:sr-only">
        {action.busy ? "Trying again…" : action.result?.status === "error" ? action.result.message : ""}
      </p>
    </div>
  );
}

/**
 * A page or section whose content loads: a skeleton while loading, the content when
 * ready, and an empty, no-results or error state in its place otherwise. A failed
 * refresh keeps the content and offers Try again above it.
 */
export function AsyncContent({
  state,
  children,
  message,
  onRetry,
  onClearFilters,
  emptyTitle,
  emptyDescription,
  emptyAction,
  skeleton,
}: {
  state: ContentState;
  children?: React.ReactNode;
  message?: string;
  onRetry?: () => Promise<ActionResult>;
  onClearFilters?: () => void;
  emptyTitle?: string;
  emptyDescription?: React.ReactNode;
  emptyAction?: React.ReactNode;
  /** A placeholder shaped like the content, so nothing jumps when it arrives. */
  skeleton?: React.ReactNode;
}) {
  const content = state === "ready" || state === "refreshing" || state === "refresh-error";
  return (
    <section data-slot="async-content" className="flex min-w-0 flex-col gap-3">
      {content || state === "loading" ? (
        <p role="status" className="sr-only">
          {message ?? TITLES[state]}
        </p>
      ) : null}
      {state === "refresh-error" ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-dashed px-3 py-2 text-sm text-muted-foreground">
          <span>{message ?? TITLES[state]}</span>
          {onRetry ? <RetryButton onRetry={onRetry} /> : null}
        </div>
      ) : null}
      <div aria-busy={state === "loading" || state === "refreshing"} className={cn("min-w-0 transition-opacity", state === "refreshing" && "opacity-60")}>
        {content ? (
          children
        ) : state === "loading" ? (
          <div aria-hidden="true" inert>
            {skeleton ?? (
              <div className="flex flex-col gap-3">
                <Skeleton className="h-5 w-40 max-w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            )}
          </div>
        ) : (
          <Empty className="border">
            <EmptyHeader role="status">
              <EmptyTitle>{state === "empty" ? (emptyTitle ?? TITLES.empty) : TITLES[state]}</EmptyTitle>
              <EmptyDescription>
                {state === "no-results" ? "Try changing or clearing your filters." : state === "empty" ? (emptyDescription ?? message) : message}
              </EmptyDescription>
            </EmptyHeader>
            {state === "no-results" && onClearFilters ? (
              <EmptyContent>
                <Button variant="outline" size="sm" onClick={onClearFilters}>
                  Clear filters
                </Button>
              </EmptyContent>
            ) : state === "empty" && emptyAction ? (
              <EmptyContent>{emptyAction}</EmptyContent>
            ) : state === "error" && onRetry ? (
              <EmptyContent>
                <RetryButton onRetry={onRetry} />
              </EmptyContent>
            ) : null}
          </Empty>
        )}
      </div>
    </section>
  );
}
