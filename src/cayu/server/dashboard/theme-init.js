// Runs before the stylesheet and module. CSP permits this same-origin static file.
(() => {
  let theme = "system"
  try {
    const stored = localStorage.getItem("cayu-dashboard-theme")
    if (stored === "light" || stored === "dark") theme = stored
  } catch {
    // Private browsing or blocked storage still follows the system preference.
  }
  document.documentElement.classList.toggle(
    "dark",
    theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches),
  )
})()
