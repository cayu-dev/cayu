import { expect, test } from "@playwright/test";

/**
 * THE AGENT'S OWN. A demonstration person signs in, and every page the app's navigation links
 * to opens without an error: each area's own page, read from the navigation so an area added in
 * src/agent/index.tsx is covered too, and Cayu's Inbox and Developer dashboard, which `cayu
 * serve` serves under /cayu/. Add a test here for what an area of your own must show.
 *
 * The app reads the organizations from its own tables, so the server needs CAYU_DATABASE_URL:
 * a PostgreSQL database with `uv run --no-sync cayu storage migrate`, `uv run --no-sync alembic
 * upgrade head` and `uv run --no-sync python -m operations.demo_seed` applied.
 */
test("a demonstration person signs in, and every area in the navigation opens", async ({ page, request }) => {
  expect(
    process.env.CAYU_DATABASE_URL,
    "Set CAYU_DATABASE_URL to a migrated PostgreSQL database with the demonstration people (operations/demo_seed.py).",
  ).toBeTruthy();
  // Nobody is signed in yet, so the app's first reads are refused, and it shows the sign-in.
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
  // The demonstration login offers only operations/demo_seed.py's people.
  const people = (await (await request.get("/app/dev/people")).json()) as string[];
  expect(people.length).toBeGreaterThan(0);
  await page.getByRole("button", { name: people[0], exact: true }).click();

  const links = page.getByLabel("Primary").locator("a[href]");
  await expect(links.first()).toBeVisible();
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(`${page.url()}: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(`${page.url()}: ${message.text()}`);
  });
  const targets = [...new Set(await links.evaluateAll((items) => items.map((item) => item.getAttribute("href") ?? "")))];
  const areas = targets.filter((href) => href.startsWith("/") && !href.startsWith("/cayu"));
  const cayu = targets.filter((href) => href.startsWith("/cayu"));
  expect(areas).toContain("/");
  expect(cayu).toEqual(expect.arrayContaining(["/cayu/inbox", "/cayu/"]));

  for (const href of areas) {
    await test.step(`the area at ${href}`, async () => {
      await page.goto(href);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await page.waitForLoadState("networkidle");
      await expect(page.getByText(/Couldn't load|Content unavailable|Page not found|No access|Sign in again|Something went wrong/)).toHaveCount(0);
      await expect(page.getByRole("button", { name: "Try again" })).toHaveCount(0);
    });
  }
  for (const href of cayu) {
    await test.step(`Cayu's page at ${href}`, async () => {
      expect((await request.get(href)).status()).toBe(200);
    });
  }
  expect(failures).toEqual([]);
});
