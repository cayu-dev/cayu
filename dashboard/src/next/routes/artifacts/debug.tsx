/** The collapsed Debug block (as on the task page): IDs and raw records, out of the main view. */
import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export function DebugDetails({
  rows,
  json,
  className,
}: {
  rows: Array<[label: string, value: ReactNode, mono?: boolean]>
  json: Array<[label: string, value: unknown]>
  className?: string
}) {
  return (
    <details className={cn("rounded-xl border border-border", className)}>
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground">
        Debug
      </summary>
      <dl className="border-t border-border px-3 py-2">
        {rows.map(([label, value, mono]) => (
          <div key={label} className="grid grid-cols-[8.5rem_1fr] gap-3 py-1.5 text-sm">
            <dt className="text-muted-foreground">{label}</dt>
            <dd
              className={cn(
                "min-w-0 [overflow-wrap:anywhere]",
                mono !== false && "font-mono text-xs leading-5",
              )}
            >
              {value ?? "–"}
            </dd>
          </div>
        ))}
      </dl>
      <div className="space-y-3 border-t border-border px-3 py-3">
        {json.map(([label, value]) => (
          <div key={label}>
            <p className="mb-1 text-xs text-muted-foreground">{label}</p>
            <pre className="max-h-80 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5">
              {JSON.stringify(value, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </details>
  )
}
