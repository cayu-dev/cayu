import { Popover } from "@base-ui/react/popover"
import { Link, useNavigate } from "@tanstack/react-router"
import { Ellipsis, Menu, PanelLeft } from "lucide-react"
import {
  type ComponentProps,
  createContext,
  type MouseEvent,
  type ReactNode,
  type RefObject,
  useContext,
} from "react"
import { Button } from "@/components/ui/button"
import { dashboardAsset } from "@/lib/config"
import { cn } from "@/lib/utils"
import type { SessionOverview, SessionState } from "../api/types"
import { STATE_LABEL, shouldFollowRowClick } from "../app/session-model"

/**
 * Links and navigation by plain path. The packaged dashboard's router owns the typed
 * route registry until this one replaces it, so paths are passed untyped here.
 */
export function NavLink({
  to,
  search,
  ...props
}: Omit<ComponentProps<typeof Link>, "to" | "search"> & {
  to: string
  search?: Record<string, unknown>
}) {
  return <Link to={to as never} search={search as never} {...props} />
}

export function useGo() {
  const navigate = useNavigate()
  return (
    to: string,
    search?: Record<string, unknown>,
    options?: { replace?: boolean; resetScroll?: boolean },
  ) => navigate({ to: to as never, search: search as never, ...options })
}

/**
 * A click handler for a list row whose title cell is a real link: a plain click anywhere
 * on the row navigates, while cmd/ctrl/shift-clicks, middle clicks and clicks on the
 * link itself are left to the browser. Selecting text in the row doesn't navigate.
 */
export function useRowClick() {
  const go = useGo()
  return (to: string, search?: Record<string, unknown>) => (e: MouseEvent<HTMLElement>) => {
    const target = e.target instanceof Element ? e.target : null
    const onControl = !!target?.closest("a, button, input, select, textarea, label, summary")
    if (window.getSelection()?.type === "Range") return
    if (
      shouldFollowRowClick({
        button: e.button,
        metaKey: e.metaKey,
        ctrlKey: e.ctrlKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        defaultPrevented: e.defaultPrevented,
        onControl,
      })
    )
      go(to, search)
  }
}

/** Classes for a clickable list row; its title link carries keyboard focus. */
export const ROW_CLASS =
  "cursor-pointer border-b border-border last:border-0 hover:bg-muted/60 focus-within:bg-muted/60"
/** Classes for the title link inside a clickable row. */
export const ROW_LINK_CLASS = "outline-none hover:underline focus-visible:underline"

/** Marks a session that ran as an eval trial, in lists that include them. */
export function EvalTrialTag() {
  return (
    <span
      className="shrink-0 rounded border border-border px-1 text-[10px] text-muted-foreground"
      title="Run by an eval suite (cayu:eval_trial). Left out of counts by default."
    >
      Eval trial
    </span>
  )
}

export const sessionPath = (id: string) => `/sessions/${encodeURIComponent(id)}`
export const taskPath = (id: string) => `/tasks/${encodeURIComponent(id)}`

/** The Cayu logo: under the dashboard's base path when served by the runtime, next to the page otherwise. */
export const logoUrl = () =>
  window.__CAYU_DASHBOARD_CONFIG__ ? dashboardAsset("logo.svg") : "./logo.svg"

const DOT: Record<SessionState, string> = {
  running: "bg-info animate-pulse motion-reduce:animate-none",
  waiting_approval: "bg-warning",
  waiting_input: "bg-warning",
  needs_recovery: "bg-warning",
  completed: "bg-success",
  failed: "bg-destructive",
  interrupted: "bg-muted-foreground/50",
}
const TEXT: Record<SessionState, string> = {
  running: "text-info",
  waiting_approval: "text-warning",
  waiting_input: "text-warning",
  needs_recovery: "text-warning",
  completed: "text-success",
  failed: "text-destructive",
  interrupted: "text-muted-foreground",
}

/** A completed session whose agent marked its summary for attention. */
export const needsAttention = (o: SessionOverview | null | undefined) =>
  o?.state === "completed" && !!o.summary && o.summary_tone === "attention"

export function StateDot({
  state,
  attention = false,
  className,
}: {
  state: SessionState | undefined
  attention?: boolean
  className?: string
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-block size-2 shrink-0 rounded-full",
        attention ? "bg-warning" : state ? DOT[state] : "bg-border",
        className,
      )}
    />
  )
}

export function stateText(o: SessionOverview | null | undefined) {
  if (!o) return "–"
  return o.state === "completed" && o.summary ? o.summary : STATE_LABEL[o.state]
}

export function StateBadge({ overview }: { overview: SessionOverview | null | undefined }) {
  const state = overview?.state
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1.5 rounded-full border border-border px-2 text-xs font-medium whitespace-nowrap",
        needsAttention(overview) ? "text-warning" : state ? TEXT[state] : "text-muted-foreground",
      )}
      title={overview?.summary || undefined}
    >
      <StateDot state={state} attention={needsAttention(overview)} />
      {state ? STATE_LABEL[state] : "–"}
    </span>
  )
}

/** Keep authored text readable without making long summaries dominate the page. */
export function SessionSummary({
  text,
  tone,
}: {
  text: string
  tone?: "neutral" | "attention" | null
}) {
  const color = tone === "attention" ? "text-warning" : "text-muted-foreground"
  if (text.length <= 180) return <p className={cn("break-words text-sm", color)}>{text}</p>
  return (
    <details className="group">
      <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
        <span className={cn("line-clamp-2 break-words text-sm group-open:line-clamp-none", color)}>
          {text}
        </span>
        <span className="mt-1 block text-xs text-muted-foreground">
          <span className="group-open:hidden">Show full summary</span>
          <span className="hidden group-open:inline">Collapse summary</span>
        </span>
      </summary>
    </details>
  )
}

export function LabelChips({
  labels,
  className,
}: {
  labels: Record<string, string>
  className?: string
}) {
  const entries = Object.entries(labels)
  if (!entries.length) return null
  return (
    <span className={cn("flex flex-wrap gap-1", className)}>
      {entries.map(([k, v]) => (
        <span
          key={k}
          className="rounded-md border border-border px-1.5 py-px font-mono text-[11px] text-muted-foreground"
        >
          {k}={v}
        </span>
      ))}
    </span>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-border bg-background px-1 font-sans text-[10px] text-muted-foreground">
      {children}
    </kbd>
  )
}

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

/* ---------- shell state shared by the sidebar and each page's top bar ---------- */

export type Shell = {
  sidebarHidden: boolean
  setSidebarHidden: (hidden: boolean) => void
  mobileOpen: boolean
  setMobileOpen: (open: boolean) => void
  focusSearch: () => void
  sidebarToggleRef?: RefObject<HTMLButtonElement | null>
  desktopSidebarToggleRef?: RefObject<HTMLButtonElement | null>
  roleLabel?: string
  pageLayout?: boolean
}
export const ShellContext = createContext<Shell | null>(null)
export function useShell() {
  const shell = useContext(ShellContext)
  if (!shell) throw new Error("useShell must be used inside the dashboard layout.")
  return shell
}

/** The bar at the top of each page: sidebar toggles, a title, and the page's actions. */
export function TopBar({ children, actions }: { children?: ReactNode; actions?: ReactNode }) {
  const shell = useContext(ShellContext)
  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-3 sm:px-4">
      {shell && (
        <Button
          variant="ghost"
          size="icon-sm"
          className="md:hidden"
          ref={shell.sidebarToggleRef}
          aria-expanded={shell.mobileOpen}
          aria-label="Open sidebar"
          onClick={() => shell.setMobileOpen(true)}
        >
          <Menu />
        </Button>
      )}
      {shell?.sidebarHidden && (
        <Button
          variant="ghost"
          size="icon-sm"
          className="hidden md:inline-flex"
          ref={shell.desktopSidebarToggleRef}
          aria-label="Show sidebar"
          onClick={() => shell.setSidebarHidden(false)}
        >
          <PanelLeft />
        </Button>
      )}
      <h1 className="min-w-0 flex-1 truncate text-sm font-medium">{children}</h1>
      {shell?.roleLabel && (
        <span className="sr-only sm:not-sr-only sm:text-xs sm:text-muted-foreground">
          {shell.roleLabel}
        </span>
      )}
      {actions}
    </header>
  )
}

/** A scrolling page body with a centered column. */
export function PageBody({
  children,
  width = "max-w-5xl",
}: {
  children: ReactNode
  width?: string
}) {
  return (
    <div className="relative min-h-0 flex-1 overflow-y-auto">
      <div className={cn("mx-auto w-full px-4 py-5 sm:px-6", width)}>{children}</div>
    </div>
  )
}

/** Shared dismissal/focus owner for compact action disclosures. */
export function ActionsMenu({
  open,
  onOpenChange,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: ReactNode
}) {
  return (
    <Popover.Root open={open} onOpenChange={onOpenChange}>
      <Popover.Trigger render={<Button variant="ghost" size="icon-sm" aria-label="More actions" />}>
        <Ellipsis />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner align="end" sideOffset={4} className="z-50">
          <Popover.Popup
            initialFocus={false}
            className="w-60 max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-popover p-1 text-sm shadow-lg"
          >
            {children}
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}
