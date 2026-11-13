import assert from "node:assert/strict"
import { readdirSync, readFileSync } from "node:fs"
import { dirname, join, relative, resolve } from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { inboxKeys, refreshAfterChange } from "../src/parts/inbox/keys.ts"
import { partKeys } from "../src/parts/keys.ts"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const parts = join(root, "src/parts")
const sources = (directory) =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) return sources(path)
    return /\.tsx?$/.test(entry.name) ? [path] : []
  })

test("every key a part uses starts with cayu", () => {
  const keys = [
    partKeys.all,
    partKeys.knowledgeEntry("k1", 2),
    partKeys.sessionRecoveries(),
    partKeys.sessionRecoveries("s1"),
    partKeys.sessionRecovery("s1", "a1"),
    inboxKeys.all,
    inboxKeys.list({ q: "x" }),
    inboxKeys.list({ q: "x" }, null),
    inboxKeys.list({ q: "x" }, "c1"),
    inboxKeys.counts(),
    inboxKeys.counts({ q: "x" }),
    inboxKeys.sessionTitles(["s1"]),
  ]
  for (const key of keys) assert.equal(key[0], "cayu", JSON.stringify(key))
  for (const key of keys.slice(6)) assert.deepEqual(key.slice(0, 2), inboxKeys.all)
})

test("prefix keys match the exact keys under them", () => {
  const page = inboxKeys.list({ kind: "approval" }, "c1")
  assert.deepEqual(page.slice(0, -1), inboxKeys.list({ kind: "approval" }))
  assert.deepEqual(inboxKeys.counts({ q: "x" }).slice(0, -1), inboxKeys.counts())
  assert.deepEqual(
    partKeys.sessionRecovery("s1", "a1").slice(0, 3),
    partKeys.sessionRecoveries("s1"),
  )
})

test("parts build every query key from partKeys or inboxKeys", () => {
  for (const path of sources(parts)) {
    const text = readFileSync(path, "utf8")
    for (const match of text.matchAll(/queryKey:\s*([^,}\n]+)/g))
      assert.match(
        match[1],
        /^(inboxKeys\.|partKeys\.|\[\.\.\.key\b|key\b)/,
        `${relative(root, path)}: queryKey: ${match[1]}`,
      )
    if (/queryKey:\s*(\[\.\.\.key\b|key\b)/.test(text))
      assert.match(
        text,
        /const key = useMemo\(\s*\(\) =>\s*inboxKeys\.list\(/,
        relative(root, path),
      )
  }
})

test("after a settle the part refreshes only inboxKeys.all, then the host once", async () => {
  const calls = []
  const client = {
    invalidateQueries: async (filters) => {
      calls.push(["invalidate", filters])
    },
  }
  const change = {
    part: "inbox",
    item: { id: "i1", kind: "approval", status: "settled" },
    decision: "approve",
  }
  await refreshAfterChange(
    client,
    { onSettled: async (got) => calls.push(["onSettled", got]) },
    change,
  )
  assert.deepEqual(calls, [
    ["invalidate", { queryKey: inboxKeys.all }],
    ["onSettled", change],
  ])
  calls.length = 0
  await refreshAfterChange(client, {}, change)
  assert.deepEqual(calls, [["invalidate", { queryKey: inboxKeys.all }]])
})

test("only the shared refresh and the watch invalidate from inside parts", () => {
  const invalidating = sources(parts)
    .filter((path) => /invalidateQueries/.test(readFileSync(path, "utf8")))
    .map((path) => relative(parts, path))
    .sort()
  assert.deepEqual(invalidating, ["inbox/keys.ts", "inbox/queries.ts"])
  const card = readFileSync(join(parts, "inbox/inbox-card.tsx"), "utf8")
  assert.equal(card.match(/onSuccess:/g)?.length, 2)
  assert.equal(card.match(/refreshAfterChange\(/g)?.length, 1)
  assert.doesNotMatch(card, /onSettled/)
})
