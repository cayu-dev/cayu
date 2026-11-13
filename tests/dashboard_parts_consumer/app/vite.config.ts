import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// `@kit` is the app's own kit; nothing maps the package, react or react-query.
const kit = path.resolve(import.meta.dirname, "src/kit/shadcn")
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: [
      { find: "@kit/icons", replacement: path.resolve(import.meta.dirname, "src/kit-icons.ts") },
      { find: /^@kit\/(.*)$/, replacement: `${kit}/$1` },
    ],
  },
  // Keep Vite from pre-bundling a second copy of the app's kit into the package.
  optimizeDeps: { exclude: ["@cayu-tech/dashboard-parts"] },
  build: { outDir: "dist", emptyOutDir: true },
})
