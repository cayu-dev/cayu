import { type APIRequestContext, expect, type Page, test } from "@playwright/test";

/**
 * THE AGENT'S OWN. The Knowledge area opens from the navigation, with its search, and says
 * plainly what the Agent knows, or that it knows nothing yet. When the app has a sign-in page
 * (organizations), a demonstration person signs in first, as in smoke.spec.ts.
 */
test("the Knowledge area shows what the Agent knows, and searches it", async ({ page, request }) => {
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(`${page.url()}: ${error.message}`));

  const navigation = await signIn(page, request);
  await navigation.getByRole("link", { name: "Knowledge" }).click();
  await expect(page).toHaveURL(/\/knowledge$/);
  await expect(page.getByRole("heading", { level: 1, name: "Knowledge" })).toBeVisible();
  const search = page.getByRole("searchbox", { name: "Search what the Agent knows" });
  await expect(search).toBeVisible();
  await expect(page.getByRole("heading", { name: "What the Agent knows" })).toBeVisible();
  await expect(page.getByText(/^The Agent hasn't kept anything yet$|statements?, latest first|The latest \d+ statements/)).toBeVisible();

  // A search goes to the server once typing pauses, and says when nothing matches.
  await search.fill("zzzz-nothing-like-this");
  await expect(page).toHaveURL(/q=zzzz-nothing-like-this/);
  await expect(page.getByText("Nothing matches “zzzz-nothing-like-this”")).toBeVisible();
  // A search with no word in it matches nothing too.
  await search.fill("?");
  await expect(page.getByText("Nothing matches “?”")).toBeVisible();
  await page.getByRole("button", { name: "Clear the search" }).click();
  await expect(page.getByRole("heading", { name: "What the Agent knows" })).toBeVisible();
  expect(failures).toEqual([]);
});

/**
 * The page reads what the Agent knows again in the background. A correction names the version
 * the person opened, so when someone else changed the statement meanwhile, the server refuses
 * it instead of the draft overwriting their change. The routes are stood in for here, so the
 * test needs no knowledge in the Agent's store.
 */
test("a correction names the version it was opened with, not a newer one", async ({ page, request }) => {
  test.setTimeout(60_000);
  let version = 1;
  const sent: { version: number }[] = [];
  const statement = () => ({
    id: "paid",
    version,
    text: version === 1 ? "Invoices are paid on the 15th." : "Invoices are paid on the 20th.",
    source: { kind: "person", label: "ada", conversation: null },
    learned_at: "2026-01-05T10:00:00+00:00",
    corrected: version === 1 ? null : { by: "someone else", at: "2026-01-06T10:00:00+00:00" },
    group: null,
    may_change: true,
    may_correct: true,
  });
  await page.route(/\/app\/knowledge(\?.*)?$/, (route) => route.fulfill({ json: { statements: [statement()], query: null, more: false } }));
  await page.route(/\/app\/knowledge\/suggestions(\?.*)?$/, (route) =>
    route.fulfill({ json: { suggestions: [], more: false, reviewed_by: "Keep or drop them in Cayu's Inbox." } }),
  );
  await page.route(/\/app\/knowledge\/correct$/, async (route) => {
    sent.push(route.request().postDataJSON() as { version: number });
    await route.fulfill({ status: 409, json: { detail: "This statement changed since you opened it. Reload it, then try again." } });
  });

  await signIn(page, request);
  await page.goto("/knowledge");
  await page.getByRole("button", { name: "Correct: Invoices are paid on the 15th." }).click();
  await page.getByRole("textbox", { name: "The corrected statement" }).fill("Invoices are paid on the 1st.");

  // Someone else corrects it while the editor is open; the next background read shows it.
  version = 2;
  await expect(page.getByText(/Someone changed this statement since you opened it/)).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Save correction" }).click();
  await expect(page.getByText("This statement changed since you opened it. Reload it, then try again.")).toBeVisible();
  expect(sent.map((body) => body.version)).toEqual([1]);
});

/** Opens the app, signing in the first demonstration person when it asks; its navigation. */
async function signIn(page: Page, request: APIRequestContext) {
  await page.goto("/");
  // The shell shows its navigation until a refused first read shows the sign-in instead.
  await page.waitForLoadState("networkidle");
  const heading = page.getByRole("heading", { name: "Sign in" });
  const navigation = page.getByLabel("Primary");
  await expect(heading.or(navigation)).toBeVisible();
  if (await heading.isVisible()) {
    const people = (await (await request.get("/app/dev/people")).json()) as string[];
    await page.getByRole("button", { name: people[0], exact: true }).click();
  }
  await expect(navigation).toBeVisible();
  return navigation;
}
