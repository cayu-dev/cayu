import { NotFoundPage, RouteErrorPage } from "@kit/layouts/app/router";
import { SignInAgainPage } from "@kit/layouts/app";

/** GENERATED. The app's status pages, worded once. */
const home = { to: "/", label: "Go to the overview" };

export function AppNotFoundPage({ description }: { description?: string }) {
  return <NotFoundPage description={description ?? "The page you're looking for doesn't exist."} home={home} />;
}

export function AppSignInAgainPage() {
  return <SignInAgainPage description="Your sign-in is no longer valid. Reload to sign in again." />;
}

/** A route's error page; `notFound` words a missing record for its page. */
export function AppRouteErrorPage({ notFound }: { notFound?: string }) {
  return <RouteErrorPage home={home} notFound={<AppNotFoundPage description={notFound} />} signIn={<AppSignInAgainPage />} />;
}
