import path from "path"
import { defineConfig } from "vite-plus"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// CAYU_PROTOTYPE=1 builds the reworked dashboard prototype (prototype.html, fake data)
// into dist/prototype. It never ships in the packaged dashboard. CAYU_PROTOTYPE_SEEDS can
// point at a JSON array of extra seeds kept outside the repository (real snapshots).
const PROTOTYPE = process.env.CAYU_PROTOTYPE === "1"
const PROTOTYPE_SEEDS =
  process.env.CAYU_PROTOTYPE_SEEDS ?? path.resolve(__dirname, "./src/next/mock/seeds/none.json")

export default defineConfig(({ command }) => ({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: [
      // Legacy pages use relative links imports. Only the prototype build/dev
      // server needs hash-aware reads; the packaged production build is unchanged.
      ...(PROTOTYPE || command === "serve"
        ? [
            {
              find: /^\.\.\/lib\/links(?:\.ts)?$/,
              replacement: path.resolve(__dirname, "./src/next/legacy-links.ts"),
              customResolver(source: string, importer: string | undefined) {
                if (importer?.endsWith("/next/legacy-links.ts"))
                  return path.resolve(__dirname, "./src/lib/links.ts")
                return source
              },
            },
          ]
        : []),
      // The parts in src/parts import components through `@kit`; here that is the
      // dashboard's own (src/kit). An app importing the parts maps `@kit` to its kit.
      { find: /^@kit\/(.*)$/, replacement: path.resolve(__dirname, "./src/kit/$1") },
      { find: "@", replacement: path.resolve(__dirname, "./src") },
      { find: "@prototype-seeds", replacement: PROTOTYPE_SEEDS },
    ],
  },
  build: PROTOTYPE
    ? {
        outDir: "dist/prototype",
        emptyOutDir: true,
        cssCodeSplit: false,
        rolldownOptions: { input: path.resolve(__dirname, "prototype.html") },
      }
    : {
        modulePreload: {
          // Traversal order can differ across build platforms. Preload order does
          // not control module evaluation; keep the packaged bytes reproducible.
          resolveDependencies: (_filename, dependencies) => [...dependencies].sort(),
        },
        // Emit the licenses for modules that actually survive bundling. The
        // post-build finalizer adds CSS and registry-generated source that Vite's
        // JavaScript module graph cannot discover.
        license: {
          fileName: "THIRD_PARTY_LICENSES.md",
        },
        rolldownOptions: {
          output: {
            // Shared route chunks can otherwise evaluate a circular dependency
            // before its class exports are initialized.
            strictExecutionOrder: true,
            chunkFileNames: (chunk) => {
              // Rolldown can name the application chunk after the source
              // directory, which changes when users eject the dashboard.
              const name = chunk.moduleIds.includes(path.resolve(__dirname, "src/main.tsx"))
                ? "dashboard"
                : chunk.name.startsWith("shared~")
                  ? "shared"
                  : chunk.name
              return `assets/${name}-[hash].js`
            },
            assetFileNames: (asset) =>
              asset.names.some((name) => name.endsWith(".css"))
                ? "assets/dashboard-[hash][extname]"
                : "assets/[name]-[hash][extname]",
            codeSplitting: {
              groups: [
                {
                  name: "shared",
                  minShareCount: 2,
                  entriesAware: true,
                  entriesAwareMergeThreshold: 16 * 1024,
                },
              ],
            },
          },
        },
      },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  // `npm test`: the unit tests run in Node; none needs a DOM.
  test: {
    include: ["tests/**/*.test.{mjs,ts}"],
    restoreMocks: true,
  },
  lint: {
    plugins: ["eslint", "typescript", "unicorn", "oxc", "react", "jsx-a11y", "vitest"],
    jsPlugins: [{ name: "vite-plus", specifier: "vite-plus/oxlint-plugin" }],
    rules: {
      "vite-plus/prefer-vite-plus-imports": "error",
      // role="status" on the element we style is equivalent ARIA; <output> would add
      // form semantics we don't want.
      "jsx-a11y/prefer-tag-over-role": "off",
      // Base UI's render prop passes placeholder elements whose accessible name arrives
      // through merged props, which the rule can't see.
      "jsx-a11y/control-has-associated-label": "off",
      // The React Compiler's rules. The dashboard isn't compiled with it, and what they flag
      // (refs read while rendering, state set from an effect) is deliberate here; rewriting it
      // would change behavior, not style.
      "react/refs": "off",
      "react/set-state-in-effect": "off",
      "react/preserve-manual-memoization": "off",
      // Event payloads are typed `unknown` and shown as they come; the type-aware checks on
      // turning them into text would need a formatter for each field.
      "typescript/no-base-to-string": "off",
      "typescript/restrict-template-expressions": "off",
      // Paths, scopes and digits are split into code points on purpose.
      "typescript/no-misused-spread": "off",
      // The host and client interfaces declare plain functions with method syntax; none uses `this`.
      "typescript/unbound-method": "off",
      // `_` marks a parameter or binding that is unused on purpose, such as a key dropped by
      // destructuring the rest.
      "eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", ignoreRestSiblings: true },
      ],
      // A row keyed by its position must say why that is safe.
      "react/no-array-index-key": "warn",
    },
    // Not linted: node_modules, which an ejected dashboard has no .gitignore to keep out; the
    // shadcn/ui components, copied from its registry as they are; and src/lib/generated,
    // written by scripts/generate-api-types.py.
    ignorePatterns: ["node_modules/**", "dist/**", "src/components/ui/**", "src/lib/generated/**"],
    // Type-aware rules; `npm run typecheck` (tsc) checks the types themselves.
    options: { typeAware: true },
  },
  // Double quotes, no semicolons, 100 columns: the style the dashboard was written in.
  fmt: {
    printWidth: 100,
    semi: false,
    sortPackageJson: false,
    // public/ is served as it is.
    ignorePatterns: [
      "node_modules/**",
      "dist/**",
      "public/**",
      "src/components/ui/**",
      "src/lib/generated/**",
      "server-openapi.json",
      "THIRD_PARTY_LICENSES.md",
    ],
  },
}))
