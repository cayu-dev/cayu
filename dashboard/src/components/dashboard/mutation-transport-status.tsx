import { AlertCircle, CheckCircle, LoaderCircle, Pause, Radio, RefreshCw } from "lucide-react"
import type {
  MutationTransportPhase,
  MutationTransportSnapshot,
} from "../../lib/mutation-transport.ts"
import { cn } from "../../lib/utils"
import { Button } from "../ui/button"

export function mutationTransportErrorMessage(snapshot: MutationTransportSnapshot): string | null {
  if (snapshot.failure === null) return null
  if (snapshot.phase === "runtime_failed") {
    return `Runtime failed: ${snapshot.failure.message}`
  }
  if (snapshot.phase === "request_failed") return snapshot.failure.message
  if (snapshot.phase === "transport_failed") {
    return `Live observation could not confirm the final outcome: ${snapshot.failure.message}`
  }
  return snapshot.failure.message
}

const PHASE_PRESENTATION: Record<
  MutationTransportPhase,
  {
    label: string
    detail: string
    icon: typeof Radio
    className: string
    spin?: boolean
  }
> = {
  connecting: {
    label: "Connecting",
    detail: "Submitting the mutation and establishing its durable boundary.",
    icon: LoaderCircle,
    className: "border-chart-1/30 bg-chart-1/5 text-chart-1",
    spin: true,
  },
  streaming: {
    label: "Live stream",
    detail: "Receiving durable runtime events.",
    icon: Radio,
    className: "border-chart-2/30 bg-chart-2/5 text-chart-2",
  },
  reconnecting: {
    label: "Reconnecting",
    detail: "Recovering from the latest durable event without repeating the mutation.",
    icon: RefreshCw,
    className: "border-chart-1/30 bg-chart-1/5 text-chart-1",
    spin: true,
  },
  polling_fallback: {
    label: "Durable fallback",
    detail: "The live stream is unavailable; bounded event and state reads are tracking progress.",
    icon: RefreshCw,
    className: "border-chart-3/30 bg-chart-3/5 text-chart-3",
    spin: true,
  },
  paused: {
    label: "Paused while hidden",
    detail: "Network recovery is suspended until this tab becomes visible.",
    icon: Pause,
    className: "border-border bg-muted/40 text-muted-foreground",
  },
  terminal: {
    label: "Durable outcome confirmed",
    detail: "The session state and its terminal event agree.",
    icon: CheckCircle,
    className: "border-chart-2/30 bg-chart-2/5 text-chart-2",
  },
  request_failed: {
    label: "Request rejected",
    detail: "The server rejected the mutation before execution.",
    icon: AlertCircle,
    className: "border-destructive/30 bg-destructive/5 text-destructive",
  },
  runtime_failed: {
    label: "Runtime failed",
    detail: "Cayu reported a durable runtime failure.",
    icon: AlertCircle,
    className: "border-destructive/30 bg-destructive/5 text-destructive",
  },
  transport_failed: {
    label: "Observation incomplete",
    detail:
      "The browser could not confirm the final outcome. The detached runtime may still continue.",
    icon: AlertCircle,
    className: "border-chart-3/30 bg-chart-3/5 text-chart-3",
  },
  cancelled: {
    label: "Observation stopped",
    detail: "The browser stopped observing; this does not interrupt detached runtime work.",
    icon: Pause,
    className: "border-border bg-muted/40 text-muted-foreground",
  },
}

export function MutationTransportStatus({
  snapshot,
  className,
  onRetryObservation,
  retryingObservation = false,
}: {
  snapshot: MutationTransportSnapshot | null
  className?: string
  onRetryObservation?: () => void
  retryingObservation?: boolean
}) {
  if (snapshot === null) return null
  const presentation = PHASE_PRESENTATION[snapshot.phase]
  const Icon = presentation.icon
  const error = mutationTransportErrorMessage(snapshot)
  const truncated = snapshot.eventsTruncated || snapshot.liveTextTruncated

  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-md border px-3 py-2 text-sm",
        presentation.className,
        className,
      )}
      data-mutation-transport-phase={snapshot.phase}
      aria-live="polite"
    >
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", presentation.spin && "animate-spin")} />
      <div className="min-w-0">
        <div className="font-medium">
          {presentation.label}
          {snapshot.phase === "reconnecting" && snapshot.reconnectAttempt > 0
            ? ` · attempt ${snapshot.reconnectAttempt}`
            : ""}
        </div>
        <div className="mt-0.5 text-muted-foreground">{error ?? presentation.detail}</div>
        {truncated && (
          <div className="mt-1 text-xs text-muted-foreground">
            The live preview was truncated to its memory limit. Full durable history remains
            available on the session page.
          </div>
        )}
        {snapshot.phase === "transport_failed" && onRetryObservation !== undefined && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-2"
            disabled={retryingObservation}
            onClick={onRetryObservation}
          >
            <RefreshCw className={cn("h-3.5 w-3.5", retryingObservation && "animate-spin")} />
            {retryingObservation ? "Retrying observation..." : "Retry observation"}
          </Button>
        )}
      </div>
    </div>
  )
}
