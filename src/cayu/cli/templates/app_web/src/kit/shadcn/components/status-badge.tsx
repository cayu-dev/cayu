import type * as React from "react";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "../lib/utils";
import { Badge } from "./ui/badge";
import { Spinner } from "./ui/spinner";

/**
 * What a status means to the reader:
 * - `progress`: under way, changes on its own (spinner)
 * - `waiting`: paused until someone acts (clock)
 * - `success`, `warning`, `danger`: done, needs attention, failed (dot)
 * - `neutral`: stopped on purpose or not applicable (dot)
 */
export type StatusTone = "progress" | "waiting" | "success" | "warning" | "danger" | "neutral";

const TONE: Record<StatusTone, { variant: "secondary" | "destructive" | "outline"; className?: string }> = {
  progress: { variant: "outline", className: "border-transparent bg-info/10 text-info" },
  waiting: { variant: "outline", className: "border-transparent bg-info/10 text-info" },
  success: { variant: "outline", className: "border-transparent bg-success/10 text-success" },
  warning: { variant: "outline", className: "border-transparent bg-warning/10 text-warning" },
  danger: { variant: "destructive" },
  neutral: { variant: "secondary" },
};

/** A status as a stock Badge: the tone sets the colour and the marker, the children are the words. */
export function StatusBadge({ tone, className, children, ...props }: Omit<React.ComponentProps<typeof Badge>, "variant"> & { tone: StatusTone }) {
  const { variant, className: toneClass } = TONE[tone];
  return (
    <Badge data-tone={tone} variant={variant} className={cn("gap-1.5", toneClass, className)} {...props}>
      {tone === "progress" ? (
        <Spinner aria-hidden="true" aria-label={undefined} role={undefined} />
      ) : tone === "waiting" ? (
        <HugeiconsIcon icon={Clock01Icon} strokeWidth={2} aria-hidden="true" />
      ) : (
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-current" />
      )}
      {children}
    </Badge>
  );
}

export type StatusDefinition = { label: string; tone: StatusTone };

const humanize = (status: string) => (status.charAt(0).toUpperCase() + status.slice(1)).replace(/[_-]+/g, " ");

/**
 * A domain's statuses, declared once: `<deployStatus.Badge status={row.status} />`,
 * `deployStatus.label(status)`. Declare `defineStatuses<string>(…)` for statuses a
 * server may add later: an unknown one renders neutral, with its name humanized.
 */
export function defineStatuses<const S extends string>(definitions: Record<S, StatusDefinition>) {
  const get = (status: S): StatusDefinition => (Object.hasOwn(definitions, status) ? definitions[status] : { label: humanize(status), tone: "neutral" });
  function DefinedStatusBadge({ status, ...props }: Omit<React.ComponentProps<typeof StatusBadge>, "tone" | "children"> & { status: S }) {
    const { label, tone } = get(status);
    return (
      <StatusBadge tone={tone} {...props}>
        {label}
      </StatusBadge>
    );
  }
  return { Badge: DefinedStatusBadge, get, label: (status: S) => get(status).label, tone: (status: S) => get(status).tone };
}
