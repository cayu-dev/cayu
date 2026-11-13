/**
 * Build and pack @cayu-tech/dashboard-parts (`npm run build:parts -- [OUT_DIR]`).
 *
 * The JavaScript comes from vite.parts.config.ts and the types from tsconfig.parts.json.
 * The package's version is the Cayu version in src/lib/release-metadata.ts, and the
 * tarball is written as OUT_DIR/cayu-dashboard-parts-<version>.tgz (default: dist/).
 * npm pack fixes file times and order, so the same sources and npm give the same bytes.
 */
import { execFileSync } from "node:child_process"
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import { dirname, join, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import ts from "typescript"
import { declarationName, shipsDeclaration } from "./parts-declarations.mjs"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const out = resolve(process.argv[2] ?? join(root, "dist"))
const stage = join(root, "dist/parts-package")
const types = join(root, "dist/parts-types")

const metadata = readFileSync(join(root, "src/lib/release-metadata.ts"), "utf8")
const version = /DASHBOARD_SOURCE_CAYU_VERSION = "([^"]+)"/.exec(metadata)?.[1]
if (!version) throw new Error("src/lib/release-metadata.ts names no Cayu version.")

const run = (command, args) =>
  execFileSync(command, args, { cwd: root, stdio: ["ignore", "inherit", "inherit"] })

rmSync(stage, { recursive: true, force: true })
rmSync(types, { recursive: true, force: true })
run("npx", ["vite", "build", "--config", "vite.parts.config.ts", "--logLevel", "warn"])
run("npx", ["tsc", "-p", "tsconfig.parts.json"])

// The parts' declarations and the release metadata they name, with their relative
// imports pointing at the shipped .js names. The dashboard's own kit stays out: the app
// brings its kit.
const declarations = (directory) =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    return entry.isDirectory() ? declarations(path) : path.endsWith(".d.ts") ? [path] : []
  })
const shipped = new Map()
for (const path of declarations(types)) {
  const name = declarationName(types, path)
  if (shipsDeclaration(name))
    shipped.set(
      name,
      readFileSync(path, "utf8").replace(/(["']\.{1,2}\/[^"']*)\.tsx?(["'])/g, "$1.js$2"),
    )
}
// Of the server's generated contract types, only those the parts reach.
const contract = contractTypes(
  readFileSync(join(types, "lib/generated/server-api/types.gen.d.ts"), "utf8"),
  [...shipped.values()].join("\n"),
)
shipped.set("lib/generated/server-api/types.gen.d.ts", contract)
shipped.set("lib/generated/server-api/index.d.ts", 'export type * from "./types.gen.js";\n')
for (const [name, text] of shipped) {
  mkdirSync(dirname(join(stage, "dist", name)), { recursive: true })
  writeFileSync(join(stage, "dist", name), text)
}

const manifest = JSON.parse(readFileSync(join(root, "parts/package.json"), "utf8"))
writeFileSync(join(stage, "package.json"), `${JSON.stringify({ ...manifest, version }, null, 2)}\n`)
cpSync(join(root, "parts/styles.css"), join(stage, "styles.css"))
cpSync(join(root, "parts/README.md"), join(stage, "README.md"))
// An ejected dashboard has its own LICENSE and NOTICE; the repository keeps them at its root.
for (const name of ["LICENSE", "NOTICE"]) {
  const source = [join(root, name), join(root, "..", name)].find((path) => existsSync(path))
  if (!source) throw new Error(`No ${name} to ship with the package.`)
  cpSync(source, join(stage, name))
}

mkdirSync(out, { recursive: true })
const packed = JSON.parse(
  execFileSync("npm", ["pack", "--json", "--pack-destination", out], {
    cwd: stage,
    encoding: "utf8",
  }),
)[0].filename
const name = `cayu-dashboard-parts-${version}.tgz`
for (const entry of readdirSync(out))
  if (/^cayu-dashboard-parts-.+\.tgz$/.test(entry) && entry !== name) rmSync(join(out, entry))
renameSync(join(out, packed), join(out, name))
console.log(`Packed ${manifest.name}@${version} into ${relative(process.cwd(), join(out, name))}`)

/** The declarations in `source` that `uses` names, with everything they name in turn. */
function contractTypes(source, uses) {
  const file = ts.createSourceFile("types.gen.d.ts", source, ts.ScriptTarget.Latest, true)
  const declared = new Map()
  for (const statement of file.statements)
    if (ts.isTypeAliasDeclaration(statement) || ts.isInterfaceDeclaration(statement))
      declared.set(statement.name.text, statement)
  const names = (node, found = new Set()) => {
    if (ts.isIdentifier(node) && declared.has(node.text)) found.add(node.text)
    // forEachChild stops at the first callback that returns a value.
    ts.forEachChild(node, (child) => {
      names(child, found)
    })
    return found
  }
  const kept = new Set()
  const pending = [...new Set(uses.match(/\b[A-Z]\w*\b/g))].filter((name) => declared.has(name))
  while (pending.length) {
    const name = pending.pop()
    if (kept.has(name)) continue
    kept.add(name)
    pending.push(...names(declared.get(name)))
  }
  const text = file.statements
    .filter(
      (statement) =>
        kept.has(statement.name?.text) && declared.get(statement.name.text) === statement,
    )
    .map((statement) => statement.getFullText(file).trim())
  const header = "// The parts' share of the server contract types in src/lib/generated."
  return `${header}\n${text.join("\n\n")}\n`
}
