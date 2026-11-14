import * as React from "react";
import { Link, Outlet, matchPath, useLocation, useParams } from "react-router";
import { UnfoldMoreIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { Avatar, AvatarFallback, AvatarImage } from "../../components/ui/avatar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuSeparator, DropdownMenuTrigger } from "../../components/ui/dropdown-menu";
import { Separator } from "../../components/ui/separator";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
  useSidebar,
} from "../../components/ui/sidebar";
import { AppPage } from "./page";
import { InsideMainLandmark } from "./status-page";

/**
 * One sidebar row. `to` is one of the app's routes, active when the location
 * matches `match` (a React Router path pattern, `"/runs/*"` for a list with detail
 * pages; defaults to `to`). `href` leaves the app, in a new tab.
 */
export type AppNavItem = { label: string; icon: IconSvgElement } & ({ to: string; match?: string } | { href: string });

/** A titled group of sidebar rows (`label` is optional for the first group). */
export type AppNavGroup = { label?: string; items: readonly AppNavItem[] };

/** Who is signed in; `children` are the profile menu's items. */
export type SidebarProfileProps = { name: string; description?: string; image?: string; children: React.ReactNode };

const initials = (name: string) =>
  name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => Array.from(part)[0]?.toUpperCase() ?? "")
    .join("") || "?";

/**
 * A routed app's frame on the stock shadcn Sidebar: the brand, the navigation
 * groups, and the profile menu at the bottom; a header with the sidebar toggle;
 * the routed page below it. On phones the sidebar is a sheet. Use it as the root
 * route's element. Moving to another record (new route params) remounts the page,
 * and a path change focuses the new page's h1.
 */
export function AppNavShell({
  name,
  home,
  brand,
  navigation,
  profile,
  footer,
  className,
  children,
}: {
  /** The app's name: the brand link's label, and the header's text on phones. */
  name: string;
  /** Where the brand links to. */
  home: string;
  /** Replaces the default brand (the name with a monogram); a `SidebarBrand`. */
  brand?: React.ReactNode;
  navigation: readonly AppNavGroup[];
  profile?: SidebarProfileProps;
  /** Above the profile at the bottom of the sidebar. */
  footer?: React.ReactNode;
  className?: string;
  /** Defaults to the routed page. */
  children?: React.ReactNode;
}) {
  const main = useFocusHeadingOnNavigate<HTMLElement>();
  const params = useParams();
  const pageKey = Object.entries(params)
    .map(([param, value]) => `${param}=${value}`)
    .join("&");
  const { pathname } = useLocation();
  const current = navigation.flatMap((group) => group.items).find((item) => "to" in item && matchPath(item.match ?? item.to, pathname) !== null);
  return (
    <SidebarProvider className={className}>
      <NavSidebar name={name} home={home} brand={brand} navigation={navigation} profile={profile} footer={footer} />
      <SidebarInset ref={main} className="min-w-0">
        <header data-slot="app-header" className="sticky top-0 z-20 flex h-12 shrink-0 items-center gap-2 border-b bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/80">
          <SidebarTrigger className="-ml-1" aria-label="Toggle navigation" />
          <Separator orientation="vertical" className="mr-1 data-[orientation=vertical]:h-4" />
          <span className="truncate text-sm font-medium">{current?.label ?? name}</span>
        </header>
        <AppPage>
          <InsideMainLandmark.Provider value>{children ?? <Outlet key={pageKey} />}</InsideMainLandmark.Provider>
        </AppPage>
      </SidebarInset>
    </SidebarProvider>
  );
}

function NavSidebar({
  name,
  home,
  brand,
  navigation,
  profile,
  footer,
}: {
  name: string;
  home: string;
  brand?: React.ReactNode;
  navigation: readonly AppNavGroup[];
  profile?: SidebarProfileProps;
  footer?: React.ReactNode;
}) {
  const { pathname } = useLocation();
  const { isMobile, setOpenMobile } = useSidebar();
  const close = () => {
    if (isMobile) setOpenMobile(false);
  };
  return (
    <Sidebar collapsible="icon" aria-label="Primary">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" tooltip={name} render={<Link to={home} aria-label={`${name} home`} onClick={close} />}>
              {brand ?? (
                <span className="flex min-w-0 items-center gap-2">
                  <span className="flex aspect-square size-8 shrink-0 items-center justify-center rounded-lg bg-sidebar-primary text-sm font-semibold text-sidebar-primary-foreground">
                    {initials(name)}
                  </span>
                  <span className="min-w-0 truncate font-semibold">{name}</span>
                </span>
              )}
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {navigation.map((group, index) => (
          <SidebarGroup key={group.label ?? index}>
            {group.label ? <SidebarGroupLabel>{group.label}</SidebarGroupLabel> : null}
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) =>
                  "to" in item ? (
                    <SidebarMenuItem key={item.to}>
                      <SidebarMenuButton
                        tooltip={item.label}
                        isActive={matchPath(item.match ?? item.to, pathname) !== null}
                        render={<Link to={item.to} onClick={close} aria-current={matchPath(item.match ?? item.to, pathname) ? "page" : undefined} />}
                      >
                        <HugeiconsIcon icon={item.icon} strokeWidth={2} />
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ) : (
                    <SidebarMenuItem key={item.href}>
                      <SidebarMenuButton tooltip={item.label} render={<a href={item.href} target="_blank" rel="noreferrer" />}>
                        <HugeiconsIcon icon={item.icon} strokeWidth={2} />
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ),
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      {footer || profile ? (
        <SidebarFooter>
          {footer}
          {profile ? <ProfileMenu {...profile} /> : null}
        </SidebarFooter>
      ) : null}
      <SidebarRail />
    </Sidebar>
  );
}

function ProfileMenu({ name, description, image, children }: SidebarProfileProps) {
  const { isMobile } = useSidebar();
  const avatar = (
    <Avatar>
      {image ? <AvatarImage src={image} alt="" /> : null}
      <AvatarFallback>{initials(name)}</AvatarFallback>
    </Avatar>
  );
  const text = (
    <span className="grid min-w-0 flex-1 text-left text-sm leading-tight">
      <span className="truncate font-medium">{name}</span>
      {description ? <span className="truncate text-xs text-muted-foreground">{description}</span> : null}
    </span>
  );
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger
            render={<SidebarMenuButton size="lg" className="data-popup-open:bg-sidebar-accent data-popup-open:text-sidebar-accent-foreground" />}
            aria-label={`${name}, open profile menu`}
          >
            {avatar}
            {text}
            <HugeiconsIcon icon={UnfoldMoreIcon} strokeWidth={2} className="ml-auto size-4" />
          </DropdownMenuTrigger>
          <DropdownMenuContent side={isMobile ? "top" : "right"} align="end" sideOffset={4} className="w-auto min-w-56">
            <div className="flex items-center gap-2 px-1.5 py-1.5 text-left text-sm">
              {avatar}
              {text}
            </div>
            <DropdownMenuSeparator />
            {children}
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}

/**
 * On a path change, focuses the new page's h1 inside the returned ref's element so
 * screen readers announce the page. Search-param changes keep focus where it is.
 */
export function useFocusHeadingOnNavigate<T extends HTMLElement = HTMLElement>() {
  const container = React.useRef<T>(null);
  const { pathname } = useLocation();
  const previous = React.useRef(pathname);
  React.useEffect(() => {
    if (previous.current === pathname) return;
    previous.current = pathname;
    const heading = container.current?.querySelector<HTMLElement>("h1");
    if (!heading) return;
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }, [pathname]);
  return container;
}
