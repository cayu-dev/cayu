import assert from "node:assert/strict"
import { spawnSync } from "node:child_process"
import { test } from "vite-plus/test"
import { fileURLToPath } from "node:url"

// Typechecking the whole dashboard takes longer than a unit test.
test(
  "navigation rejects destinations, parameters, and searches outside the registered contract",
  { timeout: 120_000 },
  () => {
    const result = spawnSync(
      process.execPath,
      [
        fileURLToPath(new URL("../node_modules/typescript/bin/tsc", import.meta.url)),
        "--project",
        fileURLToPath(new URL("./typecheck/tsconfig.json", import.meta.url)),
      ],
      { encoding: "utf8" },
    )
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
  },
)
