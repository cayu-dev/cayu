import { readFile } from "node:fs/promises"
import path from "node:path"
import process from "node:process"
import { fileURLToPath } from "node:url"

const dashboardRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const outputArgument = process.argv[2]
if (outputArgument === undefined) {
  throw new Error("usage: check-third-party-licenses.mjs <vite-output-directory>")
}

const baseline = await readFile(path.join(dashboardRoot, "THIRD_PARTY_LICENSES.md"))
const emitted = await readFile(
  path.join(path.resolve(dashboardRoot, outputArgument), "THIRD_PARTY_LICENSES.md"),
)
if (!baseline.equals(emitted)) {
  throw new Error(
    "THIRD_PARTY_LICENSES.md is stale; copy the finalized production-build inventory " +
      "to dashboard/THIRD_PARTY_LICENSES.md after reviewing dependency license changes",
  )
}
