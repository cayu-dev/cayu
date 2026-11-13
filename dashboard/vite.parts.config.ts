import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The parts in src/parts as an ES module library (`npm run build:parts`): one file per
// source module, with react, react-query and the app's `@kit` left for the app to provide.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist/parts-package/dist",
    emptyOutDir: true,
    minify: false,
    copyPublicDir: false,
    lib: {
      entry: {
        "parts/index": path.resolve(import.meta.dirname, "src/parts/index.ts"),
        "parts/inbox/index": path.resolve(import.meta.dirname, "src/parts/inbox/index.ts"),
      },
      formats: ["es"],
    },
    rolldownOptions: {
      external: [
        "react",
        "react-dom",
        "react/jsx-runtime",
        "@tanstack/react-query",
        /^@kit\//,
      ],
      output: {
        preserveModules: true,
        preserveModulesRoot: path.resolve(import.meta.dirname, "src"),
        entryFileNames: "[name].js",
      },
    },
  },
});
