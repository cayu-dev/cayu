import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import vm from "node:vm"

const script = readFileSync(new URL("../public/theme-init.js", import.meta.url), "utf8")
test("static theme init applies saved preference before rendering, or system on invalid/blocked storage", () => {
  for (const stored of ["light", "dark", "system", "invalid", null, new Error("blocked")]) {
    for (const systemDark of [true, false]) {
      let dark
      vm.runInNewContext(script, {
        localStorage: {
          getItem: () => {
            if (stored instanceof Error) throw stored
            return stored
          },
        },
        matchMedia: () => ({ matches: systemDark }),
        document: {
          documentElement: {
            classList: {
              toggle: (name, enabled) => {
                assert.equal(name, "dark")
                dark = enabled
              },
            },
          },
        },
      })
      assert.equal(dark, stored === "dark" || (stored !== "light" && systemDark))
    }
  }
})
