import { Link, Outlet, useRouterState } from "@tanstack/react-router"
import { BookOpenCheck, LayoutDashboard, List, Play } from "lucide-react"
import { cn } from "./lib/utils"

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/sessions", label: "Sessions", icon: List },
  { to: "/knowledge", label: "Knowledge", icon: BookOpenCheck },
  { to: "/run", label: "New Run", icon: Play },
] as const

export function Layout() {
  const router = useRouterState()
  const currentPath = router.location.pathname

  return (
    <div className="flex h-screen overflow-hidden">
      <nav className="w-56 flex-shrink-0 border-r border-border bg-sidebar-background flex flex-col p-4 gap-1">
        <div className="mb-6 px-3">
          <img src="/logo.svg" alt="cayu" className="h-6" />
        </div>
        {navItems.map(({ to, label, icon: Icon }) => {
          const active = to === "/" ? currentPath === "/" : currentPath.startsWith(to)
          return (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm no-underline transition-colors",
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground",
              )}
            >
              <Icon size={18} />
              {label}
            </Link>
          )
        })}
      </nav>
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
