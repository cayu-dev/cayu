import * as React from "react";

export { useIsMobile } from "./use-mobile";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export type ThemeContextValue = {
  /** What the person chose. */
  theme: Theme;
  /** What applies now: `system` resolved against the OS preference. */
  resolvedTheme: ResolvedTheme;
  setTheme: (theme: Theme) => void;
  toggle: () => void;
};

const STORAGE_KEY = "theme";
const isTheme = (value: unknown): value is Theme => value === "light" || value === "dark" || value === "system";
const darkQuery = () => (typeof window !== "undefined" && "matchMedia" in window ? window.matchMedia("(prefers-color-scheme: dark)") : null);
const resolve = (theme: Theme): ResolvedTheme => (theme === "system" ? (darkQuery()?.matches ? "dark" : "light") : theme);

function readStored(fallback: Theme): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (isTheme(stored)) return stored;
  } catch {
    // Storage can be unavailable (private mode, a sandboxed frame); the default applies.
  }
  return fallback;
}

/** The `dark` class on <html> is what the shadcn tokens switch on. */
function apply(resolved: ResolvedTheme) {
  const root = document.documentElement;
  root.classList.toggle("dark", resolved === "dark");
  root.style.colorScheme = resolved;
}

const ThemeContext = React.createContext<ThemeContextValue | null>(null);

/** Light, dark or the system's preference, remembered in this browser. */
export function ThemeProvider({ children, defaultTheme = "system" }: { children: React.ReactNode; defaultTheme?: Theme }) {
  const [theme, setThemeState] = React.useState<Theme>(() => readStored(defaultTheme));
  const [resolvedTheme, setResolved] = React.useState<ResolvedTheme>(() => resolve(theme));

  React.useLayoutEffect(() => {
    const update = () => {
      const next = resolve(theme);
      apply(next);
      setResolved(next);
    };
    update();
    if (theme !== "system") return;
    const query = darkQuery();
    query?.addEventListener("change", update);
    return () => query?.removeEventListener("change", update);
  }, [theme]);

  const setTheme = React.useCallback((next: Theme) => {
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Not remembered; it still applies to this page.
    }
    setThemeState(next);
  }, []);
  const toggle = React.useCallback(() => setTheme(resolvedTheme === "dark" ? "light" : "dark"), [resolvedTheme, setTheme]);

  const value = React.useMemo(() => ({ theme, resolvedTheme, setTheme, toggle }), [theme, resolvedTheme, setTheme, toggle]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const context = React.useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used inside <ThemeProvider>");
  return context;
}
