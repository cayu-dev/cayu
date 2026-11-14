import type { RouteObject } from "react-router";
import { LoadingPage } from "@kit/layouts/app";
import type { AgentIdentity, AreaContext, AppArea, ShellParts } from "./area";
import { AppLayout } from "./layout";
import { AppNotFoundPage, AppRouteErrorPage } from "./status-pages";

/** GENERATED. Every area's routes under one layout. */
export function createRoutes(
  agent: AgentIdentity,
  areas: readonly AppArea[],
  context: Omit<AreaContext, "areas">,
  shell: ShellParts = {},
): RouteObject[] {
  return [
    {
      path: "/",
      element: <AppLayout agent={agent} areas={areas} shell={shell} />,
      ErrorBoundary: AppRouteErrorPage,
      HydrateFallback: LoadingPage,
      children: [...areas.flatMap((area) => area.routes({ ...context, areas })), { path: "*", element: <AppNotFoundPage /> }],
    },
  ];
}
