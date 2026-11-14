import { expect, test } from "@playwright/test";

/**
 * THE AGENT'S OWN. Every page the app's navigation links to opens without an error: each
 * area's own page, read from the navigation so an area added in src/agent/index.tsx is
 * covered too, and Cayu's Inbox and Developer dashboard, which `cayu serve` serves under
 * /cayu/. Add a test here for what an area of your own must show.
 */
test("every area in the navigation opens without an error", async ({ page, request }) => {
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(`${page.url()}: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(`${page.url()}: ${message.text()}`);
  });

  await page.goto("/");
  const links = page.getByLabel("Primary").locator("a[href]");
  await expect(links.first()).toBeVisible();
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
