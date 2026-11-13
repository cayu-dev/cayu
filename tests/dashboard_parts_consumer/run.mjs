/**
 * Prove @cayu-tech/dashboard-parts in an app, as an app gets it:
 *
 *   node tests/dashboard_parts_consumer/run.mjs [TARBALL] [--keep]
 *
 * Builds a throwaway app from ./app and the generator's web template (its package.json,
 * lockfile and shadcn kit, read from src/cayu/cli/templates rather than copied here),
 * installs TARBALL (default: the one in src/cayu/data/dashboard_parts) as a real copy in
 * node_modules, then typechecks, builds with no dedupe or path maps besides `@kit`, and
 * runs consumer.spec.mjs against the build in one Chromium worker.
 */
import { execFileSync } from "node:child_process"
import { cpSync, lstatSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, "../..")
const template = join(repo, "src/cayu/cli/templates/b2b_web")
const args = process.argv.slice(2)
const keep = args.includes("--keep")
const tarball = resolve(
  args.find((arg) => !arg.startsWith("--")) ??
    (() => {
      const folder = join(repo, "src/cayu/data/dashboard_parts")
      const names = readdirSync(folder).filter((name) => name.endsWith(".tgz"))
      if (names.length !== 1) throw new Error(`Expected one tarball in ${folder}.`)
      return join(folder, names[0])
    })(),
)
const playwright = JSON.parse(readFileSync(join(repo, "dashboard/package.json"), "utf8"))
  .devDependencies["@playwright/test"]

const app = mkdtempSync(join(tmpdir(), "cayu-parts-consumer-"))
const run = (command, rest) => {
  console.log(`$ ${command} ${rest.join(" ")}`)
  execFileSync(command, rest, { cwd: app, stdio: "inherit" })
}
try {
  cpSync(join(template, "package.json"), join(app, "package.json"))
  cpSync(join(template, "package-lock.json"), join(app, "package-lock.json"))
  cpSync(join(template, "src/kit/shadcn"), join(app, "src/kit/shadcn"), { recursive: true })
  cpSync(join(here, "app"), app, { recursive: true })
  run("npm", ["ci", "--no-audit", "--no-fund"])
  run("npm", ["install", "--no-save", "--no-audit", "--no-fund", tarball, `@playwright/test@${playwright}`])
  const installed = join(app, "node_modules/@cayu-tech/dashboard-parts")
  if (lstatSync(installed).isSymbolicLink()) throw new Error("npm linked the package instead of copying it.")
  run("npx", ["tsc", "-p", "tsconfig.json"])
  run("npx", ["vite", "build", "--logLevel", "warn"])
  run("npx", ["playwright", "test", "--workers=1"])
} finally {
  if (keep) console.log(`Kept the consumer app in ${app}`)
  else rmSync(app, { recursive: true, force: true })
}
