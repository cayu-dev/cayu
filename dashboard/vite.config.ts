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
    rolldownOptions: {
      output: {
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
              // Merge transport-inefficient microchunks without pulling
              // route-only code into the always-loaded dashboard shell.
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
