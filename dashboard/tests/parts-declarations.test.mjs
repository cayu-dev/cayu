import assert from "node:assert/strict"
import path from "node:path"
import test from "node:test"

import { declarationName, shipsDeclaration } from "../scripts/parts-declarations.mjs"

test("declaration names use / on Windows too, so the parts' declarations ship", () => {
  const types = "C:\\work\\dashboard\\dist\\parts-types"
  const name = declarationName(types, `${types}\\parts\\inbox\\index.d.ts`, path.win32)

  assert.equal(name, "parts/inbox/index.d.ts")
  assert.equal(shipsDeclaration(name), true)
  assert.equal(
    declarationName(types, `${types}\\lib\\release-metadata.d.ts`, path.win32),
    "lib/release-metadata.d.ts",
  )
})

test("declaration names are unchanged on POSIX", () => {
  const types = "/work/dashboard/dist/parts-types"

  assert.equal(declarationName(types, `${types}/parts/index.d.ts`, path.posix), "parts/index.d.ts")
})

test("only the parts and the release metadata ship", () => {
  assert.equal(shipsDeclaration("parts/index.d.ts"), true)
  assert.equal(shipsDeclaration("lib/release-metadata.d.ts"), true)
  assert.equal(shipsDeclaration("kit/ui.d.ts"), false)
  assert.equal(shipsDeclaration("lib/generated/server-api/types.gen.d.ts"), false)
  assert.equal(shipsDeclaration("partsy/index.d.ts"), false)
})
