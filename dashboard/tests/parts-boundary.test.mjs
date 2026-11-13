import assert from "node:assert/strict"
import { readdirSync, readFileSync } from "node:fs"
import { dirname, join, relative, resolve } from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"

// Parts ship to apps (src/parts), so they may import only what every host provides:
// react, react-query, the @kit contract, the server's generated types and each other.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const parts = join(root, "src/parts")
const PACKAGES = new Set(["react", "@tanstack/react-query", "@kit/ui", "@kit/lib", "@kit/icons"])
const SHARED = ["src/lib/generated/server-api", "src/lib/release-metadata.ts"]

const sources = (directory) =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) return sources(path)
    return /\.tsx?$/.test(entry.name) ? [path] : []
  })

const withoutComments = (text) => text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "")
const specifiers = (text) =>
  [...withoutComments(text).matchAll(/(?:\bfrom|\bimport)\s*\(?\s*["']([^"']+)["']/g)].map(
    (match) => match[1],
  )

test("the parts folder has files to check", () => {
  const names = sources(parts).map((path) => relative(parts, path))
  for (const name of ["host.tsx", "index.ts", "inbox/index.ts", "inbox/inbox-card.tsx"])
    assert.ok(names.includes(name), name)
})

test("parts import only react, react-query, @kit, the generated types and each other", () => {
  for (const path of sources(parts)) {
    for (const specifier of specifiers(readFileSync(path, "utf8"))) {
      const where = `${relative(root, path)} imports ${specifier}`
      if (!specifier.startsWith(".")) {
        assert.ok(PACKAGES.has(specifier), where)
        continue
      }
      const target = relative(root, resolve(dirname(path), specifier))
      assert.ok(
        target.startsWith("src/parts/") || SHARED.includes(target),
        `${where}, outside src/parts`,
      )
    }
  }
})

test("parts read no window, router, lucide or dashboard alias", () => {
  for (const path of sources(parts)) {
    const text = withoutComments(readFileSync(path, "utf8"))
    const where = relative(root, path)
    assert.doesNotMatch(text, /\bwindow\b/, `${where} reads window`)
    assert.doesNotMatch(text, /@tanstack\/react-router/, `${where} imports the router`)
    assert.doesNotMatch(text, /lucide-react/, `${where} imports lucide-react`)
    assert.doesNotMatch(text, /["']@\//, `${where} imports through @/`)
  }
})

test("the dashboard's @kit names only its own components", () => {
  for (const name of ["ui.ts", "lib.ts", "icons.ts"]) {
    const text = readFileSync(join(root, "src/kit", name), "utf8")
    for (const specifier of specifiers(text))
      assert.ok(
        specifier.startsWith("@/components/ui/") ||
          specifier === "@/lib/utils" ||
          ["lucide-react", "react"].includes(specifier),
        `src/kit/${name} imports ${specifier}`,
      )
  }
})
