import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite-plus";

/*
 * GENERATED. The app imports every component from `@kit/...`; this alias decides which kit
 * that is. The default, src/kit/shadcn, is stock shadcn components (Base UI, base-nova) plus
 * the parts built from them that the app uses. Another kit is a folder next to it with the same
 * entries (ui, data, lib, hooks, icons, patterns, layouts/app, layouts/app/router and
 * styles.css); point this alias and `@kit/*` in tsconfig.json at it, and no page changes.
 *
 * `npm run build` writes ./static, which `cayu serve` serves at / (`[tool.cayu.serve] ui`;
 * committed, so a deployment needs no Node). `npm run dev` serves the app on its own port and
 * sends the API under /app, and Cayu's /api and /cayu, to `cayu serve --dev` on :8000.
 */
const kit = "shadcn";

export default defineConfig({
  // Absolute, so a deep link such as /conversations/ID loads the same files.
  base: "/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: [{ find: /^@kit\/(.*)$/, replacement: path.resolve(import.meta.dirname, `src/kit/${kit}/$1`) }],
  },
  build: {
    outDir: process.env.WEB_OUT_DIR ?? "static",
    emptyOutDir: true,
    // One file for the whole app, about 320 kB gzipped.
    chunkSizeWarningLimit: 1500,
    license: { fileName: "THIRD_PARTY_LICENSES.md" },
  },
  server: { proxy: { "^/(app|api|cayu)(/|$)": "http://127.0.0.1:8000" } },
  // `npm run lint` (Oxlint) and `npm run format` (Oxfmt). The kit's shadcn components stay as
  // the registry writes them, and ./static is the build.
  lint: {
    plugins: ["eslint", "typescript", "unicorn", "oxc", "react", "jsx-a11y"],
    jsPlugins: [{ name: "vite-plus", specifier: "vite-plus/oxlint-plugin" }],
    rules: {
      "vite-plus/prefer-vite-plus-imports": "error",
      // role="status" on the element we style is equivalent ARIA; <output> would add form semantics.
      "jsx-a11y/prefer-tag-over-role": "off",
      // Base UI's render prop passes placeholder elements whose accessible name arrives through merged props.
      "jsx-a11y/control-has-associated-label": "off",
      // `_` marks a parameter or binding that is unused on purpose, such as a prop dropped by
      // destructuring the rest.
      "eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_", ignoreRestSiblings: true }],
      // A row keyed by its position must say why that is safe.
      "react/no-array-index-key": "warn",
    },
    ignorePatterns: ["static/**", "src/kit/**"],
    options: { typeAware: true },
  },
  fmt: {
    printWidth: 160,
    sortPackageJson: false,
    ignorePatterns: ["static/**", "src/kit/**", "index.html"],
  },
});
