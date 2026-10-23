import { useCallback, useEffect, useState } from "react"

export type Theme = "system" | "light" | "dark"
const KEY = "cayu-dashboard-theme"

function read(): Theme {
  try {
    const t = localStorage.getItem(KEY)
    return t === "light" || t === "dark" ? t : "system"
  } catch {
    return "system"
  }
}

function apply(theme: Theme) {
  const dark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)
  document.documentElement.classList.toggle("dark", dark)
}

/** System, light or dark; the choice is remembered in this browser. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(read)
  useEffect(() => {
    apply(theme)
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      // Storage can be unavailable; the choice then lasts for this page only.
    }
    if (theme !== "system") return
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = () => apply("system")
    media.addEventListener("change", onChange)
    return () => media.removeEventListener("change", onChange)
  }, [theme])
  const cycle = useCallback(
    () => setTheme((t) => (t === "system" ? "light" : t === "light" ? "dark" : "system")),
    [],
  )
  return { theme, cycle }
}
