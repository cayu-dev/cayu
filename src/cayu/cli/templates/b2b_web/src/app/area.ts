import type * as React from "react";
import type { RouteObject } from "react-router";
import type { QueryClient } from "@tanstack/react-query";
import type { Requester } from "@kit/data";
import type { IconSvgElement } from "@kit/icons";

/** GENERATED. What an area's routes get: the API, the query client, and every area the app lists. */
export type AreaContext = { request: Requester; queryClient: QueryClient; areas: readonly AppArea[] };

/**
 * GENERATED. One page area of the app: its navigation entry and its routes. An area lives
 * in its own folder and is added to the app by listing it in the Agent's `src/agent/index.tsx`; the
 * navigation lists areas in that order.
 */
export type AppArea = {
  id: string;
  label: string;
  icon: IconSvgElement;
  /** Where the navigation entry goes; Overview's is `/`. */
  path: string;
  /** The routes under the app's layout, with paths relative to `/`. */
  routes: (context: AreaContext) => RouteObject[];
};

/** The area's navigation entry is active on its own path and everything under it. */
export const areaMatch = (area: AppArea) => (area.path === "/" ? "/" : `${area.path}/*`);

/** Where another area's navigation entry goes, or null when the app doesn't list that area. */
export const areaPath = (areas: readonly AppArea[], id: string) => areas.find((area) => area.id === id)?.path ?? null;

/** GENERATED. The Agent's name and what it does, for the shell and the status pages. */
export type AgentIdentity = { name: string; group: string; description: string };

/**
 * GENERATED. What the Agent adds to the shell: its own page for when nobody is signed in
 * (without one, the app asks the person to reload and sign in again), and items at the end
 * of the profile menu.
 */
export type ShellParts = { SignIn?: React.ComponentType; ProfileItems?: React.ComponentType };
