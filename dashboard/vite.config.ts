import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// CAYU_PROTOTYPE=1 builds the reworked dashboard prototype (prototype.html, fake data)
// into dist/prototype. It never ships in the packaged dashboard. CAYU_PROTOTYPE_SEEDS can
// point at a JSON array of extra seeds kept outside the repository (real snapshots).
const PROTOTYPE = process.env.CAYU_PROTOTYPE === "1";
const PROTOTYPE_SEEDS =
  process.env.CAYU_PROTOTYPE_SEEDS ?? path.resolve(__dirname, "./src/next/mock/seeds/none.json");

export default defineConfig(({ command }) => ({
  base: "./",
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: [
      // Legacy pages use relative links imports. Only the prototype build/dev
      // server needs hash-aware reads; the packaged production build is unchanged.
      ...((PROTOTYPE || command === "serve") ? [{
        find: /^\.\.\/lib\/links(?:\.ts)?$/,
        replacement: path.resolve(__dirname, "./src/next/legacy-links.ts"),
        customResolver(source: string, importer: string | undefined) {
          if (importer?.endsWith("/next/legacy-links.ts")) return path.resolve(__dirname, "./src/lib/links.ts");
          return source;
        },
      }] : []),
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
              const name = chunk.name.startsWith("shared~") ? "shared" : chunk.name;
              return `assets/${name}-[hash].js`;
            },
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
}));
