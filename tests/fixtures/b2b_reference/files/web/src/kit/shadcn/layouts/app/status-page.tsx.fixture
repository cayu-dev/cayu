import * as React from "react";
import { Button } from "../../components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "../../components/ui/empty";
import { Spinner } from "../../components/ui/spinner";

/** True inside a layout that already renders `<main>`: a status page is then a section of it. */
export const InsideMainLandmark = React.createContext(false);

/** A full-page message with a focused h1: not found, sign in again, a failure. */
export function StatusPage({ title, description, children }: { title: string; description: string; children?: React.ReactNode }) {
  const heading = React.useRef<HTMLHeadingElement>(null);
  React.useEffect(() => heading.current?.focus(), []);
  const insideMain = React.useContext(InsideMainLandmark);
  const Landmark = insideMain ? "section" : "main";
  return (
    <Landmark
      data-slot="status-page"
      className={insideMain ? "flex min-h-[50svh] items-center justify-center" : "flex min-h-svh items-center justify-center bg-background p-6 text-foreground"}
    >
      <Empty>
        <EmptyHeader>
          <EmptyTitle className="text-xl font-semibold">
            <h1 ref={heading} tabIndex={-1} className="outline-none">
              {title}
            </h1>
          </EmptyTitle>
          <EmptyDescription>{description}</EmptyDescription>
        </EmptyHeader>
        {children ? <EmptyContent className="flex-row justify-center">{children}</EmptyContent> : null}
      </Empty>
    </Landmark>
  );
}

/** The router's `HydrateFallback` while the first route loads. */
export function LoadingPage({ label = "Loading…" }: { label?: string }) {
  return (
    <main data-slot="loading-page" className="flex min-h-svh items-center justify-center gap-2 bg-background text-sm text-muted-foreground">
      <Spinner role={undefined} aria-label={undefined} aria-hidden="true" />
      <p role="status">{label}</p>
    </main>
  );
}

/** After a 401: reloading brings back the sign-in. */
export function SignInAgainPage({
  description = "Your sign-in is no longer valid. Reload to sign in again.",
  actionLabel = "Reload and sign in",
  onSignIn = () => window.location.reload(),
}: {
  description?: string;
  actionLabel?: string;
  onSignIn?: () => void;
}) {
  return (
    <StatusPage title="Sign in again" description={description}>
      <Button onClick={onSignIn}>{actionLabel}</Button>
    </StatusPage>
  );
}
