import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { createServiceContext, createSessionStore, whoamiQuery, type Requester } from "@kit/data";
import { whoamiApi } from "../api/whoami";

/** GENERATED. The one requester for `/app`. Each area builds its own typed client from it. */
export const { Provider: RequesterProvider, useService: useRequester } = createServiceContext<Requester>("useRequester");

/** An area's client, built once per requester: `const api = useAreaApi(pullsApi)`. */
export function useAreaApi<T>(create: (request: Requester) => T): T {
  const request = useRequester();
  return React.useMemo(() => create(request), [create, request]);
}

/** A 401 marks the session expired; the layout then shows the sign-in-again page. */
export const session = createSessionStore();

export const meQuery = (request: Requester) => whoamiQuery((signal) => whoamiApi(request).me(signal));

/** Who is signed in. Until whoami answers, actions stay hidden; the server decides either way. */
export function useWhoami() {
  const me = useQuery(meQuery(useRequester()));
  return { me: me.data, mayChange: me.data?.may_change ?? false, mayDecide: me.data?.may_decide ?? false };
}
