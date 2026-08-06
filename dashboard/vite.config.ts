import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "./",
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
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
});
