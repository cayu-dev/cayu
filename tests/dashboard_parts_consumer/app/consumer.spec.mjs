/**
 * The Inbox part in an app that installed the packed package: no router, no services,
 * the generator's kit as `@kit`, and a fake host that records what the part asked of it.
 */
import { expect, test } from "@playwright/test"

const record = (page) => page.evaluate(() => window.consumer)

const errors = []
test.beforeEach(({ page }) => {
  errors.length = 0
  page.on("pageerror", (error) => errors.push(error.message))
})
test.afterEach(() => expect(errors).toEqual([]))

test("approving sends the decision and calls onSettled once", async ({ page }) => {
  await page.goto("/")
  const card = page.locator('[data-inbox-item="approval-1"]')
  await expect(card).toContainText("ap-auditor wants to run hold_payment.")
  await expect(page.getByRole("button", { name: /Approvals\s*1/ })).toBeVisible()
  // The package's @source line let Tailwind generate the part's own classes: the waiting
  // card's border-warning/50 differs from its header's border-border.
  const borders = await card.evaluate((element) => [
    getComputedStyle(element).borderTopColor,
    getComputedStyle(element.firstElementChild).borderBottomColor,
  ])
  expect(borders[0]).not.toEqual(borders[1])
  await card.getByRole("button", { name: "Approve" }).click()
  await expect(card).toContainText("Decided")
  const { settles, settled } = await record(page)
  expect(settles).toEqual([{ id: "approval-1", body: { decision: "approve", reason: null } }])
  expect(settled).toHaveLength(1)
  expect(settled[0]).toMatchObject({
    part: "inbox",
    decision: "approve",
    item: { id: "approval-1", status: "settled" },
  })
})

test("a supplied ProtectedReview replaces the decision buttons", async ({ page }) => {
  await page.goto("/?protected=1")
  const card = page.locator('[data-inbox-item="approval-1"]')
  await expect(card.getByRole("button", { name: "Review protected arguments" })).toBeVisible()
  await expect(card.getByRole("button", { name: "Approve" })).toHaveCount(0)
  await card.getByRole("button", { name: "Review protected arguments" }).click()
  await expect(card).toContainText("Decided")
  const { settles, settled } = await record(page)
  expect(settles[0].body).toMatchObject({
    decision: "approve",
    review_reference: { content_tag: "tag-1", context: { purpose: "stub" } },
  })
  expect(settled).toHaveLength(1)
})

test("decide: false shows the items read-only", async ({ page }) => {
  await page.goto("/?decide=0")
  const card = page.locator('[data-inbox-item="approval-1"]')
  await expect(card).toContainText("Decisions are unavailable for this deployment.")
  await expect(card.getByRole("button", { name: "Approve" })).toBeDisabled()
  await expect(
    page.locator('[data-inbox-item="question-1"]').getByRole("button", { name: "CC-4100" }),
  ).toBeDisabled()
  expect((await record(page)).settles).toEqual([])
})

test("InboxCard works on its own", async ({ page }) => {
  await page.goto("/?card=1")
  await expect(page.locator("[data-inbox-item]")).toHaveCount(1)
  await page.getByRole("button", { name: "Deny" }).click()
  await expect(page.locator('[data-inbox-item="approval-1"]')).toContainText("Decided")
  expect((await record(page)).settles[0].body).toEqual({ decision: "deny", reason: null })
})

test("a part without its host section names the missing key", async ({ page }) => {
  await page.goto("/?missing=1")
  await expect(page.getByRole("alert")).toContainText('host\'s "inbox" section')
})

test("ProtectedReviewPart shows the review, reads a long field and settles with its reference", async ({
  page,
}) => {
  await page.goto("/?review=1")
  const card = page.locator('[data-inbox-item="approval-1"]')
  await expect(card.getByRole("button", { name: "Approve" })).toHaveCount(0)
  await card.getByRole("button", { name: "Refresh review" }).click()
  await expect(card).toContainText("permitted: Review all calls before deciding.")
  await expect(card).toContainText("$12,400 & fees")
  await card.getByRole("button", { name: /Show all 2,400 characters of Memo/ }).click()
  await expect(card).toContainText("The whole memo.")
  await card.getByRole("textbox", { name: "Decision reason" }).fill("Checked the memo.")
  await card.getByRole("button", { name: "Approve" }).click()
  await expect(card).toContainText("Decided")
  const { settles, settled, reviews, fetches } = await record(page)
  expect(settles).toEqual([
    {
      id: "approval-1",
      body: {
        decision: "approve",
        reason: "Checked the memo.",
        review_reference: {
          content_tag: "tag-1",
          policy_version: "v1",
          context: { purpose: "consumer", recipient: "alice", tenant: null },
        },
      },
    },
  ])
  expect(settled).toHaveLength(1)
  expect(reviews).toEqual([
    ["inspect", "session-1", "consumer"],
    ["longField", "session-1", "tag-1", "memo"],
  ])
  expect(fetches).toBe(0)
})

test("ProtectedReviewPart won't decide with a review for another purpose", async ({ page }) => {
  await page.goto("/?review=1&viewPurpose=other")
  const card = page.locator('[data-inbox-item="approval-1"]')
  await card.getByRole("button", { name: "Refresh review" }).click()
  await expect(card.getByRole("alert")).toContainText("Review access was denied or is unavailable")
  await expect(card.getByRole("button", { name: "Approve" })).toHaveCount(0)
  expect((await record(page)).settles).toEqual([])
})

test("without a purpose ProtectedReviewPart shows the item read-only", async ({ page }) => {
  await page.goto("/?review=1&purpose=none")
  const card = page.locator('[data-inbox-item="approval-1"]')
  await expect(card).toContainText("names no purpose to request it with")
  await expect(card.getByRole("button", { name: "Refresh review" })).toHaveCount(0)
  await expect(card.getByRole("button", { name: "Approve" })).toHaveCount(0)
  expect((await record(page)).reviews).toEqual([])
})
