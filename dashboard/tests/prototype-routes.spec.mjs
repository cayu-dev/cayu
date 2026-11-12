/**
 * Opens every page of the prototype build in every seed and fails on a page error, a logged
 * error, an error screen, or an alert. Concrete pages (a session, a task, an eval run…) are
 * found by following the links each page shows. Every route in src/next/app/routes.tsx must be reached
 * in at least one seed, and every seed with sessions must open one with its Trace view.
 *
 * Run with `npm run test:prototype`, which builds the prototype first.
 */
import { readFileSync } from "node:fs"
import { expect, test } from "@playwright/test"

const ROUTES_SOURCE = readFileSync(new URL("../src/next/app/routes.tsx", import.meta.url), "utf8")
const ROUTE_PATTERNS = [
  ...new Set(
    [...ROUTES_SOURCE.matchAll(/(?:path: |page\(|moved\()"(\/[^"]*)"/g)].map((match) => match[1]),
  ),
]
const STATIC_ROUTES = ROUTE_PATTERNS.filter((pattern) => !pattern.includes("$"))
const SESSION = "/sessions/$sessionId"
const ERROR_SCREENS = ["Something went wrong!", "Something went wrong loading this page."]
/** Pages opened per route with parameters, per seed: enough to cover each kind of row. */
const PAGES_PER_ROUTE = 12

const patternOf = (path) =>
  ROUTE_PATTERNS.find((pattern) =>
    new RegExp(`^${pattern.replace(/\$[A-Za-z]+/g, "[^/]+")}$`).test(path),
  )

/**
 * Waits until the fake API's delayed answers have rendered: the page text stops changing.
 * The window restarts on each call, since hash navigation keeps the previous page's window.
 */
async function settle(page) {
  await page.evaluate(() => {
    window.__settle = undefined
  })
  await page.waitForFunction(
    () => {
      const text = document.body.innerText
      const now = performance.now()
      if (window.__settle?.text !== text) window.__settle = { text, since: now }
      return now - window.__settle.since > 300
    },
    undefined,
    { polling: 100, timeout: 15_000 },
  )
}

async function appLinks(page) {
  const hrefs = await page.$$eval("a[href]", (links) =>
    links.map((link) => link.getAttribute("href") ?? ""),
  )
  return hrefs.flatMap((href) => {
    const hash = href.indexOf("#/")
    if (hash < 0) return []
    const path = href.slice(hash + 1).split("?")[0]
    return patternOf(path) ? [path] : []
  })
}

test("every prototype page opens in every seed", async ({ page }) => {
  test.setTimeout(8 * 60_000)
  const problems = []
  let current = ""
  page.on("pageerror", (error) => problems.push(`${current}: ${error.message}`))
  page.on("console", (message) => {
    if (message.type() === "error")
      problems.push(`${current}: console: ${message.text().split("\n")[0]}`)
  })

  await page.goto("/prototype.html#/")
  await settle(page)
  const seeds = await page.$$eval("#prototype-app option", (options) =>
    options.map((option) => ({ id: option.value, label: option.textContent })),
  )
  expect(seeds.length).toBeGreaterThan(0)

  const reached = new Set()
  for (const seed of seeds) {
    await page.selectOption("#prototype-app", seed.id)
    await settle(page)

    const queue = [...STATIC_ROUTES]
    const seen = new Set(queue)
    const perRoute = new Map()
    const enqueue = (path) => {
      const pattern = patternOf(path)
      const count = perRoute.get(pattern) ?? 0
      if (seen.has(path) || count >= PAGES_PER_ROUTE) return
      seen.add(path)
      perRoute.set(pattern, count + 1)
      queue.push(path)
    }
    let sessions = 0
    let opened = 0
    while (queue.length > 0) {
      const path = queue.shift()
      const pattern = patternOf(path)
      for (const target of pattern === SESSION ? [path, `${path}?view=trace`] : [path]) {
        current = `${seed.label} ${target}`
        // Same-document hash navigation keeps the selected seed and its fake data.
        await page.goto(`/prototype.html#${target}`)
        await settle(page)
        opened += 1
        const text = await page.locator("body").innerText()
        for (const screen of ERROR_SCREENS)
          if (text.includes(screen)) problems.push(`${current}: shows "${screen}"`)
        // ErrorState shows a failed query's message in an alert instead of a fixed text.
        for (const alert of await page.getByRole("alert").filter({ visible: true }).all())
          problems.push(`${current}: alert: ${(await alert.innerText()).trim()}`)
      }
      reached.add(pattern)
      if (pattern === SESSION) {
        sessions += 1
        enqueue(`${path}/workflow`)
      }
      for (const link of await appLinks(page)) enqueue(link)
    }
    console.log(`${seed.label}: opened ${opened} pages, ${sessions} sessions`)

    await page.goto("/prototype.html#/sessions")
    await settle(page)
    const listed = (await appLinks(page)).some((link) => patternOf(link) === SESSION)
    if (listed) expect(sessions, `${seed.label}: no session opened`).toBeGreaterThan(0)
  }

  expect(problems).toEqual([])
  expect(ROUTE_PATTERNS.filter((pattern) => !reached.has(pattern))).toEqual([])
})
