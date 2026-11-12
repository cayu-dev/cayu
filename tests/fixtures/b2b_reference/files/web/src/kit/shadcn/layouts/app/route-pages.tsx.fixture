import type * as React from "react";
import { Link, isRouteErrorResponse, useRevalidator, useRouteError } from "react-router";
import { Button } from "../../components/ui/button";
import { SignInAgainPage, StatusPage } from "./status-page";

/** Where a status page's way out goes: usually the app's home. */
export type StatusPageLink = { to: string; label: string };

function HomeLink({ home }: { home?: StatusPageLink }) {
  return home ? <Button variant="outline" render={<Link to={home.to} />}>{home.label}</Button> : null;
}

export function NotFoundPage({ description = "The page you're looking for doesn't exist.", home }: { description?: string; home?: StatusPageLink }) {
  return (
    <StatusPage title="Page not found" description={description}>
      <HomeLink home={home} />
    </StatusPage>
  );
}

export function ForbiddenPage({ description = "Your role doesn't include this page. Ask an admin for access.", home }: { description?: string; home?: StatusPageLink }) {
  return (
    <StatusPage title="No access" description={description}>
      <HomeLink home={home} />
    </StatusPage>
  );
}

/**
 * A route's ErrorBoundary: 401, 403 and 404 responses get their own pages, and
 * anything else offers Try again. Pass the app's versions to change their words.
 */
export function RouteErrorPage({
  home,
  notFound,
  forbidden,
  signIn,
}: {
  home?: StatusPageLink;
  notFound?: React.ReactNode;
  forbidden?: React.ReactNode;
  signIn?: React.ReactNode;
}) {
  const error = useRouteError();
  const { revalidate } = useRevalidator();
  if (isRouteErrorResponse(error)) {
    if (error.status === 404) return notFound ?? <NotFoundPage home={home} />;
    if (error.status === 403) return forbidden ?? <ForbiddenPage home={home} />;
    if (error.status === 401) return signIn ?? <SignInAgainPage />;
  }
  return (
    <StatusPage title="Something went wrong" description="This page couldn't be shown. Try again, or come back later.">
      <Button onClick={() => void revalidate()}>Try again</Button>
      <HomeLink home={home} />
    </StatusPage>
  );
}
