import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run build` writes ./static, which `cayu serve` serves at /app/ui/. `npm run dev`
// serves the interface on its own port and sends /app to `cayu serve --dev` on :8000.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  build: {
    outDir: "static",
    emptyOutDir: true,
    license: { fileName: "THIRD_PARTY_LICENSES.md" },
  },
  server: { proxy: { "/app": "http://127.0.0.1:8000" } },
});
