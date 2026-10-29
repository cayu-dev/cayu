/** Bars of sessions started per hour, day or week, with failures stacked on top. */
import { type Bucket, type BucketUnit, bucketLabel, bucketSpan } from "./usage-window"

export type ChartBucket = Bucket & {
  total: number | null
  failed: number
  /** Event activity tokens for the same period, distinct from sessions started. */
  detail: string | null
}

export function SessionsChart({ buckets, unit }: { buckets: ChartBucket[]; unit: BucketUnit }) {
  const max = Math.max(1, ...buckets.map((b) => b.total ?? 0))
  const every = Math.max(1, Math.ceil(buckets.length / 6))
  const anyFailed = buckets.some((b) => b.failed > 0)
  const per = unit === "hour" ? "Hour" : unit === "day" ? "Day" : "Week"

  return (
    <div>
      <div className="relative h-36">
        <div className="pointer-events-none absolute inset-0 flex flex-col justify-between">
          <div className="border-t border-dashed border-border" />
          <div className="border-t border-dashed border-border/60" />
          <div className="border-t border-border" />
        </div>
        <span className="absolute -top-2 right-0 bg-card pl-1 text-[10px] leading-4 text-muted-foreground tabular-nums">
          {max}
        </span>
        <div className="absolute inset-0 flex items-end gap-px pr-6 sm:gap-1" aria-hidden="true">
          {buckets.map((b) => {
            const total = b.total ?? 0
            const label = `${bucketSpan(b, unit)}: ${b.total === null ? "loading" : `${total} session${total === 1 ? "" : "s"}`}${b.failed ? `, ${b.failed} failed` : ""}${b.detail ? ` · ${b.detail}` : ""}`
            return (
              <div
                key={b.start}
                title={label}
                className="group flex h-full min-w-0 flex-1 flex-col justify-end"
              >
                {b.total === null ? (
                  <div className="mx-auto h-1 w-full max-w-12 animate-pulse rounded-sm bg-muted motion-reduce:animate-none" />
                ) : total === 0 ? (
                  <div className="mx-auto h-px w-full max-w-12 bg-border" />
                ) : (
                  <div
                    className="mx-auto flex w-full max-w-12 flex-col overflow-hidden rounded-t-[3px] bg-foreground/55 group-hover:bg-foreground/80"
                    style={{ height: `${Math.max(2, (total / max) * 100)}%` }}
                  >
                    {b.failed > 0 && (
                      <div
                        className="bg-destructive"
                        style={{ height: `${(b.failed / total) * 100}%` }}
                      />
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
      <div className="mt-1.5 flex gap-px pr-6 sm:gap-1" aria-hidden="true">
        {buckets.map((b, i) => (
          <div
            key={b.start}
            className="min-w-0 flex-1 overflow-visible text-[10px] whitespace-nowrap text-muted-foreground"
          >
            {i % every === 0 ? bucketLabel(b, unit) : ""}
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-foreground/55" />
          Sessions
        </span>
        {anyFailed && (
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-sm bg-destructive" />
            Failed
          </span>
        )}
      </div>
      <table className="sr-only">
        <caption>Sessions started per {per.toLowerCase()}</caption>
        <thead>
          <tr>
            <th>{per}</th>
            <th>Sessions</th>
            <th>Failed</th>
            <th>Activity</th>
          </tr>
        </thead>
        <tbody>
          {buckets.map((b) => (
            <tr key={b.start}>
              <td>{bucketSpan(b, unit)}</td>
              <td>{b.total ?? "loading"}</td>
              <td>{b.failed}</td>
              <td>{b.detail ?? "Unavailable"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
