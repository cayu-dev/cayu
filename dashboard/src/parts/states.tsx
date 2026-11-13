/** The empty and failed states every part shows the same way. */
import type { ReactNode } from "react"

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-border px-6 py-10 text-center">
      <p className="text-sm font-medium">{title}</p>
      {children && (
        <div className="mx-auto mt-1.5 max-w-md text-sm text-muted-foreground">{children}</div>
      )}
    </div>
  )
}

export function ErrorState({ error }: { error: unknown }) {
  return (
    <p
      role="alert"
      className="min-w-0 [overflow-wrap:anywhere] rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive"
    >
      {error instanceof Error ? error.message : "Something went wrong loading this page."}
    </p>
  )
}
