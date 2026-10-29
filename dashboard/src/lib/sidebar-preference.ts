/**
 * Whether the desktop sidebar is hidden, remembered in this browser across routes and
 * reloads, like the theme. Storage can be missing or blocked; the choice then lasts for
 * this page only.
 */
export const SIDEBAR_KEY = "cayu-dashboard-sidebar"

type PreferenceStorage = Pick<Storage, "getItem" | "setItem">

function browserStorage(): PreferenceStorage | null {
  try {
    return window.localStorage
  } catch {
    return null
  }
}

export function readSidebarHidden(storage: PreferenceStorage | null = browserStorage()): boolean {
  try {
    return storage?.getItem(SIDEBAR_KEY) === "hidden"
  } catch {
    return false
  }
}

export function writeSidebarHidden(
  hidden: boolean,
  storage: PreferenceStorage | null = browserStorage(),
): void {
  try {
    storage?.setItem(SIDEBAR_KEY, hidden ? "hidden" : "shown")
  } catch {
    // Storage can be unavailable; the choice then lasts for this page only.
  }
}
