import type * as React from "react";
import { Alert02Icon, AlertCircleIcon, CheckmarkCircle02Icon, InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "../lib/utils";
import { Alert as StockAlert } from "./ui/alert";

export { AlertAction, AlertDescription, AlertTitle } from "./ui/alert";

export type AlertVariant = "default" | "info" | "success" | "warning" | "danger" | "destructive";

const TONE: Record<AlertVariant, { className?: string; icon?: typeof InformationCircleIcon }> = {
  default: {},
  info: { className: "border-info/30 bg-info/5 *:[svg]:text-info", icon: InformationCircleIcon },
  success: { className: "border-success/30 bg-success/5 *:[svg]:text-success", icon: CheckmarkCircle02Icon },
  warning: { className: "border-warning/40 bg-warning/5 *:[svg]:text-warning", icon: Alert02Icon },
  danger: { className: "border-destructive/30 bg-destructive/5", icon: AlertCircleIcon },
  destructive: { icon: AlertCircleIcon },
};

/**
 * The stock shadcn Alert with tones. `info`, `success` and `warning` tint it and add
 * their icon; `danger` is the stock `destructive`. Warning and danger are announced
 * (`role="alert"`), the others are polite (`role="status"`).
 */
export function Alert({ variant = "default", role, className, children, ...props }: React.ComponentProps<"div"> & { variant?: AlertVariant | null }) {
  const tone = variant ?? "default";
  const urgent = tone === "warning" || tone === "danger" || tone === "destructive";
  const { className: toneClass, icon } = TONE[tone];
  return (
    <StockAlert
      variant={tone === "danger" || tone === "destructive" ? "destructive" : "default"}
      data-variant={tone}
      role={role ?? (urgent ? "alert" : "status")}
      className={cn(toneClass, className)}
      {...props}
    >
      {icon ? <HugeiconsIcon icon={icon} strokeWidth={2} aria-hidden="true" /> : null}
      {children}
    </StockAlert>
  );
}
