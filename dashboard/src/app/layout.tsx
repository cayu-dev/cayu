import { useQuery } from "@tanstack/react-query"
import { Link, Outlet, useNavigate, useRouterState } from "@tanstack/react-router"
import {
  AlertTriangle,
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
  Search,
  Settings2,
  Sun,
  X,
} from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { ServerContractProvider, useServerContract } from "../components/dashboard/server-contract"
import { useViewer, ViewerProvider } from "../components/dashboard/viewer"
import { Button } from "../components/ui/button"
import {
  fetchServerContract,
  isSupportedServerContract,
  SUPPORTED_SERVER_CONTRACT_VERSION,
} from "../lib/api"
import { dashboardAsset } from "../lib/config"
import { applyViewerAccess } from "../lib/dashboard-access"
import {
  DASHBOARD_ROUTE_REQUIREMENTS,
  type DashboardCapabilityRequirement,
  dashboardCapabilityUnavailableText,
  resolveDashboardCapability,
} from "../lib/dashboard-capabilities"
import { fetchDashboardIdentitySnapshot } from "../lib/dashboard-identity-query"
import { useTheme } from "../lib/theme"
import { cn } from "../lib/utils"
import { realApi } from "../next/api/real"
import { useInboxCounts } from "../next/app/queries"
import { ServicesProvider } from "../next/app/services"

/** Compatibility gates must finish before any page or authenticated identity reads mount. */
export function Layout() {
  const contract = useQuery({
    queryKey: ["server-contract"],
    queryFn: fetchServerContract,
    refetchInterval: false,
    staleTime: Number.POSITIVE_INFINITY,
  })
  const incompatible = contract.data !== undefined && !isSupportedServerContract(contract.data)
  if (contract.isLoading)
    return (
      <div role="status" className="p-6">
        Checking API contract before loading Developer dashboard data...
      </div>
    )
  if (contract.error || incompatible)
    return (
      <div
        data-testid="dashboard-contract-gate"
        role="alert"
        className="m-6 flex gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-destructive"
      >
        <AlertTriangle className="size-4 shrink-0" />
        <div className="space-y-3">
          <p>
            {incompatible
              ? `Dashboard expects CAYU server contract v${SUPPORTED_SERVER_CONTRACT_VERSION}, but the server reports v${contract.data!.contract_version}.`
              : `Could not load the CAYU server contract: ${contract.error instanceof Error ? contract.error.message : "Unknown error"}`}
          </p>
          <Button
            variant="outline"
            disabled={contract.isFetching}
            onClick={() => void contract.refetch()}
          >
            {contract.isFetching ? "Checking..." : "Retry contract check"}
          </Button>
        </div>
      </div>
    )
  if (!contract.data) return <div role="alert">The CAYU server contract returned no data.</div>
  return (
    <ServerContractProvider contract={contract.data}>
      <IdentityGate />
    </ServerContractProvider>
  )
}

function IdentityGate() {
  const serverContract = useServerContract()
  const capability = resolveDashboardCapability(serverContract.capabilities, {
    kind: "surface",
    surface: "app_info",
  })
  const identity = useQuery({
    queryKey: ["dashboard-identity"],
    queryFn: fetchDashboardIdentitySnapshot,
    enabled: capability.enabled,
    retry: false,
  })
  const unavailable = dashboardCapabilityUnavailableText(capability)
  const viewer = identity.data?.identity
  const error = identity.error ?? identity.data?.error
  if (unavailable || !viewer)
    return (
      <div
        data-testid="dashboard-identity-gate"
        role={error || unavailable ? "alert" : "status"}
        className="p-6 space-y-3"
      >
        <p>
          {unavailable ??
            (error instanceof Error ? error.message : (error ?? "Loading agent identity..."))}
        </p>
        {error && (
          <Button
            variant="outline"
            onClick={() => void identity.refetch()}
            disabled={identity.isFetching}
          >
            Retry identity check
          </Button>
        )}
      </div>
    )
  return (
    <ViewerProvider value={viewer}>
      <div
        data-testid="dashboard-identity-refresh-warning"
        role="status"
        hidden={!identity.error}
        className="px-4 py-2 text-sm"
      >
        Identity refresh is temporarily unavailable. Your current page is preserved.
      </div>
      <ServicesProvider
        value={{
          api: realApi,
          now: Date.now,
          taskIndexEnabled: serverContract.capabilities.surfaces.task_index?.read.enabled === true,
          taskCreationEnabled:
            serverContract.capabilities.surfaces.task_creation?.mutate.enabled === true,
          taskMutationsEnabled: serverContract.capabilities.mutations.task_lifecycle.enabled,
          executionEnabled: serverContract.capabilities.mutations.session_execution.enabled,
          inboxEnabled: serverContract.capabilities.surfaces.inbox_queries?.read.enabled === true,
          decisionsEnabled: serverContract.capabilities.mutations.pending_action_resolution.enabled,
          knowledgeReviewEnabled: serverContract.capabilities.mutations.knowledge_review.enabled,
        }}
      >
        <DashboardShell />
      </ServicesProvider>
    </ViewerProvider>
  )
}

const sections = [
  {
    label: null,
    items: [
      { to: "/", label: "Home", icon: House, capability: DASHBOARD_ROUTE_REQUIREMENTS["/"] },
      { to: "/inbox", label: "Inbox", icon: Inbox },
      { to: "/sessions", label: "Sessions", icon: MessagesSquare },
      {
        to: "/tasks",
        label: "Tasks",
        icon: ListChecks,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/tasks"],
      },
    ],
  },
  {
    label: "Improve",
    items: [
      {
        to: "/evals",
        label: "Evals",
        icon: FlaskConical,
        fullAccess: true,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/evals"],
      },
      {
        to: "/usage",
        label: "Usage",
        icon: ChartColumn,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/usage"],
      },
    ],
  },
  {
    label: "Agent",
    items: [
      { to: "/agents", label: "Specialists", icon: Bot },
      {
        to: "/knowledge",
        label: "Knowledge",
        icon: BookOpen,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/knowledge"],
      },
      { to: "/environments", label: "Environments", icon: Box },
      {
        to: "/artifacts",
        label: "Artifacts",
        icon: FileText,
        capability: DASHBOARD_ROUTE_REQUIREMENTS["/artifacts"],
      },
      { to: "/system", label: "System", icon: Settings2, fullAccess: true },
    ],
  },
] as const

function DashboardShell() {
  const viewer = useViewer()!
  const contract = useServerContract()
  const navigate = useNavigate()
  const location = useRouterState({ select: (s) => s.location })
  const routePath = useRouterState({ select: (s) => s.matches.at(-1)?.routeId ?? "/" })
  const [sidebarHidden, setSidebarHidden] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [query, setQuery] = useState("")
  const searchRef = useRef<HTMLInputElement>(null)
  const sidebarRef = useRef<HTMLElement>(null)
  const inboxCounts = useInboxCounts()
  const mainRef = useRef<HTMLElement>(null)
  const openRef = useRef<HTMLButtonElement>(null)
  const { theme, cycle } = useTheme()
  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor
  // Close and return focus after navigation, including a search on the current page.
  useEffect(() => {
    void location.href
    setMobileOpen(false)
  }, [location.href])
  useEffect(() => {
    if (!mobileOpen) return
    const previous =
      document.activeElement instanceof HTMLElement ? document.activeElement : openRef.current
    const main = mainRef.current
    if (main) main.inert = true
    sidebarRef.current
      ?.querySelector<HTMLButtonElement>("button[aria-label='Close sidebar']")
      ?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        setMobileOpen(false)
      }
      if (event.key !== "Tab") return
      const nodes = Array.from(
        sidebarRef.current?.querySelectorAll<HTMLElement>("a[href], button, input") ?? [],
      ).filter((node) => node.getClientRects().length > 0 && !node.hasAttribute("disabled"))
      const first = nodes[0],
        last = nodes.at(-1)
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last?.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first?.focus()
      }
    }
    const desktop = window.matchMedia("(min-width: 768px)")
    const resized = () => {
      if (desktop.matches) setMobileOpen(false)
    }
    document.addEventListener("keydown", onKey)
    desktop.addEventListener("change", resized)
    return () => {
      if (main) main.inert = false
      document.removeEventListener("keydown", onKey)
      desktop.removeEventListener("change", resized)
      previous?.focus()
    }
  }, [mobileOpen])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault()
        setSidebarHidden(false)
        setMobileOpen(window.matchMedia("(max-width: 767px)").matches)
        requestAnimationFrame(() => searchRef.current?.focus())
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
  const available = (requirement: DashboardCapabilityRequirement) =>
    applyViewerAccess(
      resolveDashboardCapability(contract.capabilities, requirement),
      requirement,
      viewer,
    ).enabled

  return (
    <div className="flex h-dvh bg-background text-foreground">
      {mobileOpen && (
        <button
          type="button"
          tabIndex={-1}
          aria-label="Dismiss sidebar"
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}
      {/* biome-ignore lint/a11y/useAriaPropsSupportedByRole: aria-modal is present only with the mobile dialog role. */}
      <aside
        ref={sidebarRef}
        aria-label="Sidebar"
        role={mobileOpen ? "dialog" : undefined}
        aria-modal={mobileOpen ? true : undefined}
        className={cn(
          "z-40 w-[248px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar-background",
          mobileOpen ? "fixed inset-y-0 left-0 flex" : sidebarHidden ? "hidden" : "hidden md:flex",
        )}
      >
        <div className="flex items-center gap-1 px-4 pt-3 pb-2">
          <Link to="/" className="min-w-0 flex-1" aria-label="Home">
            <span className="flex items-center gap-2">
              <img src={dashboardAsset("logo.svg")} alt="Cayu" className="h-[18px] dark:invert" />
              <span className="border-l border-border pl-2 text-xs font-medium text-muted-foreground">
                Developer
              </span>
            </span>
          </Link>
          <Button
            variant="ghost"
            size="icon-sm"
            className="hidden md:inline-flex"
            aria-label="Hide sidebar"
            onClick={() => {
              setSidebarHidden(true)
              requestAnimationFrame(() => openRef.current?.focus())
            }}
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
        <p className="truncate px-5 pb-2 text-sm font-medium" title={viewer.name}>
          {viewer.name}
        </p>
        <form
          className="mx-3 flex items-center gap-2 rounded-lg border border-transparent px-2 py-1.5 text-sm text-muted-foreground focus-within:border-ring"
          onSubmit={(event) => {
            event.preventDefault()
            void navigate({ to: "/sessions", search: query.trim() ? { q: query.trim() } : {} })
            setMobileOpen(false)
          }}
        >
          <Search className="size-4 shrink-0" aria-hidden="true" />
          <label htmlFor="sidebar-search" className="sr-only">
            Search sessions
          </label>
          <input
            id="sidebar-search"
            ref={searchRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search sessions"
            className="min-w-0 w-full bg-transparent text-foreground outline-none placeholder:text-muted-foreground"
          />
          <kbd className="text-xs">⌘K</kbd>
        </form>
        <nav className="min-h-0 flex-1 overflow-y-auto px-3 pb-3" aria-label="Main">
          {sections.map((section, index) => (
            <div key={section.label ?? "main"} className="space-y-0.5 pt-2">
              {section.label && (
                <h2 className="px-2 pt-2 pb-1 text-xs font-medium text-muted-foreground">
                  {section.label}
                </h2>
              )}
              {section.items
                .filter(
                  (item) =>
                    (!("fullAccess" in item) || viewer.full_access) &&
                    (!("capability" in item) || available(item.capability)),
                )
                .map(({ to, label, icon: Icon }) => {
                  const active =
                    to === "/"
                      ? routePath === "/"
                      : routePath === to || routePath.startsWith(`${to}/`)
                  return (
                    <Link
                      key={to}
                      to={to}
                      aria-label={label}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-sidebar-accent",
                        active && "bg-sidebar-accent font-medium",
                      )}
                    >
                      <Icon className="size-4 text-muted-foreground" aria-hidden="true" />
                      <span>{label}</span>
                      {to === "/inbox" &&
                        inboxCounts.data?.complete === true &&
                        (inboxCounts.data.total_count ?? 0) > 0 && (
                          <span
                            role="status"
                            aria-label="Inbox total"
                            className="ml-auto rounded-full bg-warning/15 px-1.5 text-xs tabular-nums"
                          >
                            {inboxCounts.data?.total_count}
                          </span>
                        )}
                    </Link>
                  )
                })}
              {index === 0 && available(DASHBOARD_ROUTE_REQUIREMENTS["/run"]) && (
                <Link
                  to="/run"
                  className="flex rounded-md px-2 py-1.5 text-sm text-muted-foreground hover:bg-sidebar-accent"
                >
                  Start a session
                </Link>
              )}
            </div>
          ))}
        </nav>
        <div className="flex items-center gap-1 border-t border-sidebar-border px-3 py-2">
          <p className="flex-1 px-2 text-xs text-muted-foreground">v{viewer.runtime_version}</p>
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
      <main ref={mainRef} className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border px-4">
          <Button
            ref={openRef}
            variant="ghost"
            size="icon-sm"
            className={cn(!sidebarHidden && "md:hidden")}
            aria-label="Open sidebar"
            aria-expanded={mobileOpen || !sidebarHidden}
            onClick={() => {
              setSidebarHidden(false)
              setMobileOpen(window.matchMedia("(max-width: 767px)").matches)
            }}
          >
            <PanelLeft />
          </Button>
          <span className="truncate text-sm font-medium">{viewer.name}</span>
          {!viewer.full_access && (
            <span className="ml-auto text-xs text-muted-foreground">Read and decide</span>
          )}
        </header>
        <div className="min-h-0 flex-1 overflow-auto p-4 sm:p-6 xl:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
