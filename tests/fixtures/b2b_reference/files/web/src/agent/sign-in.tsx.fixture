import { useQuery, useQueryClient } from "@tanstack/react-query";
import { LoadingPage } from "@kit/layouts/app";
import { Button, Card, CardContent, DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuSeparator } from "@kit/ui";
import * as React from "react";
import { session, useAreaApi, useWhoami } from "../app/services";
import { AppSignInAgainPage } from "../app/status-pages";
import { b2bApi, messageOf } from "./api";
import { ErrorNote } from "./conversations";

/**
 * THE AGENT'S OWN. Signing in under `cayu serve --dev`: the demonstration people of
 * operations/demo_seed.py, and nobody else. Outside development `/app/dev/people` doesn't
 * exist, and the app asks the person to sign in again through Cayu Cloud's sign-in.
 */
const peopleQuery = (api: ReturnType<typeof b2bApi>) => ({
  queryKey: ["b2b", "demo-people"],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.demoPeople(signal),
  retry: false,
  staleTime: Number.POSITIVE_INFINITY,
});

function useSignInAs() {
  const api = useAreaApi(b2bApi);
  const queryClient = useQueryClient();
  return async (subject: string) => {
    await api.demoSignIn(subject);
    session.restore();
    // Everything shown was the previous person's.
    await queryClient.resetQueries();
  };
}

/** Shown while nobody is signed in. */
export function SignIn() {
  const api = useAreaApi(b2bApi);
  const people = useQuery(peopleQuery(api));
  const signInAs = useSignInAs();
  const [error, setError] = React.useState<string | null>(null);
  if (people.isPending) return <LoadingPage />;
  if (people.isError || people.data.length === 0) return <AppSignInAgainPage />;
  return (
    <main className="flex min-h-svh items-center justify-center bg-background p-6 text-foreground">
      <Card className="w-full max-w-md">
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <h1 className="text-title-sm font-semibold">Sign in</h1>
            <p className="text-body-sm text-text-secondary">Development only: choose a demonstration person.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {people.data.map((subject) => (
              <Button key={subject} variant="outline" onClick={() => signInAs(subject).catch((caught) => setError(messageOf(caught)))}>
                {subject}
              </Button>
            ))}
          </div>
          {error ? <ErrorNote message={error} /> : null}
        </CardContent>
      </Card>
    </main>
  );
}

/** In the profile menu under `cayu serve --dev`: switch to another demonstration person. */
export function ProfileItems() {
  const api = useAreaApi(b2bApi);
  const people = useQuery(peopleQuery(api));
  const { me } = useWhoami();
  const signInAs = useSignInAs();
  if (!people.data?.length || !me) return null;
  return (
    <>
      <DropdownMenuSeparator />
      <DropdownMenuRadioGroup value={me.person.subject} onValueChange={(value) => void signInAs(String(value))}>
        <DropdownMenuLabel>Demonstration person</DropdownMenuLabel>
        {people.data.map((subject) => (
          <DropdownMenuRadioItem key={subject} value={subject} closeOnClick>
            {subject}
          </DropdownMenuRadioItem>
        ))}
      </DropdownMenuRadioGroup>
    </>
  );
}
