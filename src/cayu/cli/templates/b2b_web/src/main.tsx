import * as React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import { QueryClientProvider } from "@tanstack/react-query";
import "@kit/styles.css";
import { ThemeProvider } from "@kit/hooks";
import { Toaster, TooltipProvider } from "@kit/ui";
import { createQueryClient, createRequester } from "@kit/data";
import { createRoutes } from "./app/routes";
import { RequesterProvider, session } from "./app/services";
import { areas, identity, shell } from "./agent";

/**
 * GENERATED. The Agent's web app, served by `cayu serve` at / (`[tool.cayu.serve] ui`): one
 * browser router for every page, and one requester for the Agent's API under /app. Every request goes to this
 * same origin with the person's sign-in cookie; a 401 shows the sign-in page.
 */
const request = createRequester({ baseUrl: new URL("/app/", window.location.origin).href });
const queryClient = createQueryClient({ onUnauthorized: session.expire });
const router = createBrowserRouter(createRoutes(identity, areas, { request, queryClient }, shell));

const root = document.getElementById("root");
if (root === null) throw new Error("The page has no #root element.");

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <ThemeProvider defaultTheme="system">
      <TooltipProvider>
        <Toaster>
          <QueryClientProvider client={queryClient}>
            <RequesterProvider value={request}>
              <RouterProvider router={router} />
            </RequesterProvider>
          </QueryClientProvider>
        </Toaster>
      </TooltipProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
