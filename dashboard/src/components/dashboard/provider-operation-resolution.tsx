import { AlertTriangle } from "lucide-react"
import { useState } from "react"
import type { SessionState } from "../../lib/api"
import { Badge } from "../ui/badge"
import { Button } from "../ui/button"
import { Card, CardContent } from "../ui/card"
import { Textarea } from "../ui/textarea"

export function ProviderOperationResolutionBanner({
  operation,
  resolving,
  unavailableReason,
  error,
  onResolve,
}: {
  operation: SessionState["provider_operation"]
  resolving: boolean
  unavailableReason: string | null
  error: string | null
  onResolve: (action: "fallback_retry" | "fail", reason: string) => void
}) {
  const [reason, setReason] = useState("")
  const reasonReady = reason.trim().length > 0
  const targetReady = operation.stage_id !== null && operation.run_epoch !== null
  const controlsDisabled = resolving || unavailableReason !== null || !targetReady || !reasonReady
  const fallbackAllowed = operation.allowed_resolutions?.includes("fallback_retry") ?? false
  const failAllowed = operation.allowed_resolutions?.includes("fail") ?? false
  const ambiguous = operation.status === "ambiguous_submission"

  return (
    <Card
      className="border-destructive/30 bg-destructive/5"
      data-testid="provider-operation-resolution"
    >
      <CardContent className="space-y-4 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="destructive">
                <AlertTriangle className="mr-1 h-3.5 w-3.5" />
                {ambiguous ? "Provider submission is ambiguous" : "Provider operation unavailable"}
              </Badge>
              {operation.provider && <Badge variant="secondary">{operation.provider}</Badge>}
              {operation.recovery_reason && (
                <Badge variant="outline">{operation.recovery_reason.replaceAll("_", " ")}</Badge>
              )}
            </div>
            <p className="mt-2 text-sm font-medium">
              Cayu stopped instead of silently submitting a replacement model request.
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Retry starts a new model attempt from the last durable Cayu boundary. Fail closes the
              original attempt and session through the normal durable failure path.
            </p>
            {operation.duplicate_request_risk && (
              <p className="mt-2 text-sm font-medium text-destructive">
                The provider may already have accepted the original request. Retrying can create a
                duplicate provider request and cost; confirm that risk before continuing.
              </p>
            )}
            <div className="mt-2 space-y-1 break-all font-mono text-xs text-muted-foreground">
              <div>stage_id: {operation.stage_id ?? "unavailable"}</div>
              <div>run_epoch: {operation.run_epoch ?? "unavailable"}</div>
              {operation.operation_id && <div>operation_id: {operation.operation_id}</div>}
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <Button
              variant="destructive"
              disabled={controlsDisabled || !failAllowed}
              title={unavailableReason ?? undefined}
              onClick={() => onResolve("fail", reason.trim())}
            >
              Fail session
            </Button>
            <Button
              disabled={controlsDisabled || !fallbackAllowed}
              title={unavailableReason ?? undefined}
              onClick={() => onResolve("fallback_retry", reason.trim())}
            >
              Retry from Cayu boundary
            </Button>
          </div>
        </div>
        <Textarea
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          placeholder="Required operator reason for the durable audit record"
          rows={2}
          disabled={resolving || unavailableReason !== null}
          title={unavailableReason ?? undefined}
        />
        {!targetReady && (
          <p className="text-sm text-destructive">
            The exact stage or run-epoch fence is unavailable; refresh before resolving.
          </p>
        )}
        {unavailableReason && (
          <p
            className="text-sm text-muted-foreground"
            data-testid="provider-resolution-unavailable"
          >
            Provider-operation resolution is unavailable. {unavailableReason}
          </p>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
      </CardContent>
    </Card>
  )
}
