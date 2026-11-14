import { mergeProps } from "@base-ui/react/merge-props";
import { useRender } from "@base-ui/react/use-render";
import { cn } from "../../lib/utils";

/**
 * A product's name as the sidebar brand: a home link whose text is the name,
 * truncated to the sidebar's width. `mark` puts a square monogram before it, the
 * way stock shadcn sidebar headers show a team or app logo. It renders
 * `<a href="/">`; pass `render` for the router's Link, or a `<span />` when the
 * shell already makes it a link.
 */
export function SidebarBrand({
  name,
  mark = false,
  render,
  className,
  ...props
}: Omit<useRender.ComponentProps<"a">, "children"> & { name: string; mark?: boolean }) {
  const initial = Array.from(name.replace(/^cayu-/, "").trim())[0]?.toUpperCase() ?? "?";
  return useRender({
    defaultTagName: "a",
    render,
    props: mergeProps<"a">(
      {
        href: render ? undefined : "/",
        className: cn("flex min-w-0 items-center gap-2", className),
        children: (
          <>
            {mark ? (
              <span
                aria-hidden="true"
                data-slot="sidebar-brand-mark"
                className="flex aspect-square size-8 shrink-0 items-center justify-center rounded-lg bg-sidebar-primary text-sm font-semibold text-sidebar-primary-foreground"
              >
                {initial}
              </span>
            ) : null}
            <span className="min-w-0 truncate font-semibold">{name}</span>
          </>
        ),
      },
      props,
    ),
    state: { slot: "sidebar-brand-link" },
  });
}
