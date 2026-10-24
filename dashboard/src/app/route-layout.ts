import "@tanstack/react-router"

declare module "@tanstack/react-router" {
  interface StaticDataRouteOption {
    /** Migrated pages own their TopBar, padding and scrolling. */
    dashboardLayout?: "page"
  }
}
