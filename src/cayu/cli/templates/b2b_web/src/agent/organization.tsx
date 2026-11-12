import * as React from "react";
import { queryOptions, type UseQueryResult, useQuery } from "@tanstack/react-query";
import { AppPageHeader, AppPageTitle } from "@kit/layouts/app";
import { Alert, AlertDescription, AlertTitle, Button, NativeSelect, NativeSelectOption } from "@kit/ui";
import { useAreaApi } from "../app/services";
import { b2bApi, type B2bApi, type Me, type Membership, messageOf } from "./api";

/**
 * THE AGENT'S OWN. The signed-in person's memberships (`GET /app/me`) and the organization the
 * app shows. The active organization is only a filter and the default for a new conversation,
 * remembered in this browser; the server decides access either way.
 */
export const meQuery = (api: B2bApi) => queryOptions({ queryKey: ["b2b", "me"], queryFn: ({ signal }) => api.me(signal) });

const KEY = "active-organization";
const listeners = new Set<() => void>();

function readActive(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function chooseOrganization(organization: string) {
  try {
    localStorage.setItem(KEY, organization);
  } catch {
    // Not remembered; it still applies to this page.
  }
  active = organization;
  for (const listener of listeners) listener();
}

let active = readActive();

function useActiveOrganization(): string | null {
  return React.useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => active,
  );
}

/** The person's memberships and the one the app shows, once `/app/me` answers. */
export function useMembership() {
  const api = useAreaApi(b2bApi);
  const me = useQuery(meQuery(api));
  const chosen = useActiveOrganization();
  const memberships = me.data?.memberships ?? [];
  const membership: Membership | null = memberships.find((item) => item.organization === chosen) ?? memberships[0] ?? null;
  return { me, membership, memberships };
}

/** Switches the active organization; shown only to someone in more than one. */
export function OrganizationSelect() {
  const { membership, memberships } = useMembership();
  if (membership === null || memberships.length < 2) return null;
  return (
    <NativeSelect aria-label="Organization" value={membership.organization} onChange={(event) => chooseOrganization(event.target.value)}>
      {memberships.map((item) => (
        <NativeSelectOption key={item.organization} value={item.organization}>
          {item.organization}
          {item.department ? ` · ${item.department}` : ""}
        </NativeSelectOption>
      ))}
    </NativeSelect>
  );
}

/**
 * What a page shows instead of its content while it can't know the person's organization:
 * why `/app/me` failed, with a retry, or that no organization has added them. Null once it
 * knows one, and while `/app/me` is loading.
 */
export function membershipBlocker(me: UseQueryResult<Me>, membership: Membership | null): React.ReactNode {
  // Only when it never loaded: a failed refetch keeps the memberships it had, and the page.
  if (me.isError && me.data === undefined) return <MembershipFailed message={messageOf(me.error)} retry={() => void me.refetch()} />;
  if (me.isSuccess && membership === null) return <NoOrganization subject={me.data.subject} />;
  return null;
}

export function MembershipFailed({ message, retry }: { message: string; retry: () => void }) {
  return (
    <Alert variant="danger" data-origin="own">
      <AlertTitle>Couldn't load your organizations</AlertTitle>
      <AlertDescription className="flex flex-col items-start gap-2">
        {message}
        <Button variant="outline" size="sm" onClick={retry}>
          Try again
        </Button>
      </AlertDescription>
    </Alert>
  );
}

export function NoOrganization({ subject }: { subject: string }) {
  return (
    <div className="flex flex-col gap-page-section" data-origin="own">
      <AppPageHeader description={`You're signed in as ${subject}, but no organization has added you. Ask one of its admins.`}>
        <AppPageTitle>No organization yet</AppPageTitle>
      </AppPageHeader>
    </div>
  );
}
