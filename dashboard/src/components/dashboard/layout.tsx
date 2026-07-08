import type { ReactNode } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

function Page({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("min-w-0 space-y-6", className)}>{children}</div>
}

function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between",
        className,
      )}
    >
      <div className="min-w-0">
        <h1 className="truncate text-2xl font-bold tracking-tight">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}

function DataCard({
  title,
  description,
  actions,
  className,
  headerClassName,
  contentClassName,
  children,
}: {
  title?: ReactNode
  description?: ReactNode
  actions?: ReactNode
  className?: string
  headerClassName?: string
  contentClassName?: string
  children: ReactNode
}) {
  return (
    <Card className={cn("min-w-0 gap-0 py-0", className)}>
      {(title || description || actions) && (
        <CardHeader className={cn("border-b border-border py-4", headerClassName)}>
          <div className="flex min-w-0 items-start justify-between gap-3">
            <div className="min-w-0">
              {title && <CardTitle className="truncate text-base">{title}</CardTitle>}
              {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
            </div>
            {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
          </div>
        </CardHeader>
      )}
      <CardContent className={cn("min-w-0 p-0", contentClassName)}>{children}</CardContent>
    </Card>
  )
}

function StateMessage({
  children,
  tone = "muted",
  className,
}: {
  children: ReactNode
  tone?: "muted" | "danger"
  className?: string
}) {
  return (
    <div
      className={cn(
        "px-4 py-8 text-center text-sm",
        tone === "danger" ? "text-destructive" : "text-muted-foreground",
        className,
      )}
    >
      {children}
    </div>
  )
}

function PayloadViewer({
  value,
  className,
  maxHeight = "max-h-56",
}: {
  value: unknown
  className?: string
  maxHeight?: string
}) {
  return (
    <pre
      className={cn(
        maxHeight,
        "overflow-auto rounded-md border border-border bg-muted/40 p-3 text-xs",
        className,
      )}
    >
      {typeof value === "string" ? value : JSON.stringify(value, null, 2)}
    </pre>
  )
}

export { DataCard, Page, PageHeader, PayloadViewer, StateMessage }
