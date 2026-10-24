import { Outlet, useRouterState } from "@tanstack/react-router"
import {
  BookOpen,
  Bot,
  Box,
  ChartColumn,
  FileText,
  FlaskConical,
  House,
  Inbox,
  ListChecks,
  MessagesSquare,
  Monitor,
  Moon,
  PanelLeft,
  RotateCcw,
  Search,
  Settings2,
  Sun,
  Wrench,
  X,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useApp, useInbox, useSessionRows, withoutEvalTrials } from "../app/queries"
import { useServices } from "../app/services"
import { useTheme } from "../app/theme"
import { Kbd, logoUrl, NavLink, ShellContext, useGo, useShell } from "./common"

export function Layout() {
  const [sidebarHidden, setSidebarHidden] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const shell = useMemo(
    () => ({
      sidebarHidden,
      setSidebarHidden,
      mobileOpen,
      setMobileOpen,
      focusSearch: () => {
        setSidebarHidden(false)
        setMobileOpen(true)
        setTimeout(() => searchRef.current?.focus(), 0)
      },
    }),
    [sidebarHidden, mobileOpen],
  )
  // biome-ignore lint/correctness/useExhaustiveDependencies: close the mobile sidebar on navigation
  useEffect(() => setMobileOpen(false), [pathname])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        shell.focusSearch()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [shell])

  return (
    <ShellContext.Provider value={shell}>
      <div className="flex h-dvh flex-col bg-background text-foreground">
        <PrototypeBar />
        <div className="flex min-h-0 flex-1">
          {mobileOpen && (
            <button
              type="button"
              aria-label="Close sidebar"
              className="fixed inset-0 z-30 bg-black/20 md:hidden"
              onClick={() => setMobileOpen(false)}
            />
          )}
          <Sidebar searchRef={searchRef} />
          <main className="flex min-w-0 flex-1">
            <Outlet />
          </main>
        </div>
      </div>
    </ShellContext.Provider>
  )
}

function PrototypeBar() {
  const { prototype } = useServices()
  if (!prototype) return null
  const current = prototype.seeds.find((s) => s.id === prototype.seedId)
  return (
    <div className="flex h-9 min-w-0 shrink-0 items-center gap-2 border-b border-border bg-muted/60 px-3 text-xs whitespace-nowrap">
      <span className="font-medium text-muted-foreground">Prototype</span>
      <label className="flex items-center gap-1.5">
        <span className="sr-only">Agent</span>
        <select
          id="prototype-app"
          value={prototype.seedId}
          onChange={(e) => prototype.setSeed(e.target.value)}
          className="h-6 rounded-md border border-border bg-background px-1.5 text-xs"
        >
          {prototype.seeds.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
      <span className="hidden truncate text-muted-foreground sm:inline">
        {current?.description}
      </span>
      <label
        className="ml-auto flex items-center gap-1.5 text-muted-foreground"
        title="Backend item 1: agents set a title and summary. Off shows what today's runtime gives: labels, then the first message, then the ID."
      >
        <input
          id="prototype-app-titles"
          type="checkbox"
          checked={prototype.appTitles}
          onChange={(e) => prototype.setAppTitles(e.target.checked)}
        />
        <span className="hidden sm:inline">Agent-set titles</span>
        <span className="sm:hidden">Titles</span>
      </label>
      <Button variant="ghost" size="xs" onClick={prototype.reset} aria-label="Reset data">
        <RotateCcw />
        <span className="hidden sm:inline">Reset data</span>
      </Button>
    </div>
  )
}

function Sidebar({ searchRef }: { searchRef: React.RefObject<HTMLInputElement | null> }) {
  const { sidebarHidden, setSidebarHidden, mobileOpen, setMobileOpen } = useShell()
  const go = useGo()
  const [query, setQuery] = useState("")
  const app = useApp()
  const sessions = useSessionRows()
  const inbox = useInbox()
  const { theme, cycle } = useTheme()
  // The prototype sets a dashboard config for today's pages, but its logo sits next to the page.
  const logo = useServices().prototype ? "./logo.svg" : logoUrl()
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor
  const waiting = inbox.data?.items.length ?? 0
  const knowledgeWaiting = (inbox.data?.items ?? []).filter((i) => i.kind === "knowledge").length
  const running = withoutEvalTrials(sessions.data?.rows ?? []).filter(
    (r) => r.overview?.state === "running",
  ).length

  const item = (to: string, Icon: typeof Inbox, label: string, extra?: React.ReactNode) => {
    const active = to === "/" ? pathname === "/" : pathname === to || pathname.startsWith(`${to}/`)
    return (
      <NavLink
        key={to}
        to={to}
        className={cn(
          "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-sidebar-accent",
          active && "bg-sidebar-accent font-medium",
        )}
      >
        <Icon className="size-4 text-muted-foreground" aria-hidden="true" />
        <span className="flex-1">{label}</span>
        {extra}
      </NavLink>
    )
  }
  const count = (n: number | undefined, tone: "warning" | "info" | "muted" = "muted") =>
    n ? (
      <span
        className={cn(
          "rounded-full px-1.5 text-xs tabular-nums",
          tone === "warning" && "bg-warning/15 font-medium text-warning",
          tone === "info" && "bg-info/15 font-medium text-info",
          tone === "muted" && "text-muted-foreground",
        )}
      >
        {n.toLocaleString()}
      </span>
    ) : null
  const group = (label: string, children: React.ReactNode) => (
    <div className="space-y-0.5">
      <h2 className="px-2 pt-4 pb-1 text-xs font-medium text-muted-foreground">{label}</h2>
      {children}
    </div>
  )

  return (
    <aside
      aria-label="Sidebar"
      className={cn(
        "fixed inset-y-0 left-0 z-40 flex w-[248px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar-background transition-transform md:static md:translate-x-0",
        mobileOpen ? "translate-x-0" : "-translate-x-full",
        sidebarHidden && "md:hidden",
      )}
    >
      <div className="flex items-center gap-1 px-4 pt-3 pb-2">
        <NavLink to="/" className="min-w-0 flex-1" aria-label="Home">
          <span className="flex min-w-0 items-center gap-2">
            <img src={logo} alt="cayu" className="h-[18px] shrink-0 dark:invert" />
            <span
              className="min-w-0 truncate border-l border-border pl-2 text-xs font-medium text-muted-foreground"
              title="Developer dashboard"
            >
              Developer
            </span>
          </span>
        </NavLink>
        <Button
          variant="ghost"
          size="icon-sm"
          className="hidden md:inline-flex"
          aria-label="Hide sidebar"
          onClick={() => setSidebarHidden(true)}
        >
          <PanelLeft />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          className="md:hidden"
          aria-label="Close sidebar"
          onClick={() => setMobileOpen(false)}
        >
          <X />
        </Button>
      </div>

      <div className="space-y-1.5 px-3">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            go("/sessions", query.trim() ? { q: query.trim() } : {})
          }}
          className="flex h-8 items-center gap-2 rounded-lg px-2.5 text-sm text-muted-foreground focus-within:bg-sidebar-accent hover:bg-sidebar-accent"
        >
          <Search className="size-4 shrink-0" aria-hidden="true" />
          <label htmlFor="sidebar-search" className="sr-only">
            Search sessions
          </label>
          <input
            id="sidebar-search"
            ref={searchRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search sessions"
            className="grouped-field-input w-full min-w-0 bg-transparent text-foreground outline-none placeholder:text-muted-foreground"
          />
          <Kbd>⌘K</Kbd>
        </form>
      </div>

      <nav className="mt-1 min-h-0 flex-1 overflow-y-auto px-3 pb-3" aria-label="Main">
        <div className="space-y-0.5 pt-2">
          {item("/", House, "Home")}
          {item("/inbox", Inbox, "Inbox", count(waiting, "warning"))}
          {item("/sessions", MessagesSquare, "Sessions", count(running, "info"))}
          {item("/tasks", ListChecks, "Tasks")}
        </div>
        {group(
          "Improve",
          <>
            {item("/evals", FlaskConical, "Evals")}
            {item("/usage", ChartColumn, "Usage")}
          </>,
        )}
        {group(
          "Agent",
          <>
            {item("/specialists", Bot, "Specialists")}
            {item("/tools", Wrench, "Tools")}
            {item("/knowledge", BookOpen, "Knowledge", count(knowledgeWaiting, "warning"))}
            {item("/environments", Box, "Environments")}
            {item("/artifacts", FileText, "Artifacts")}
            {item("/system", Settings2, "System")}
          </>,
        )}
      </nav>

      <div className="flex items-center gap-1 border-t border-sidebar-border px-3 py-2">
        <p className="flex-1 px-2 text-[11px] text-muted-foreground">
          {app.data?.runtime_version ? `v${app.data.runtime_version}` : ""}
        </p>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={`Theme: ${theme}`}
          title={`Theme: ${theme}`}
          onClick={cycle}
        >
          <ThemeIcon />
        </Button>
      </div>
    </aside>
  )
}
