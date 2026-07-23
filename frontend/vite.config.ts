import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Standard Vite output: dist/. Dockerfile copies this to nginx.
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // Split heavy deps into their own chunks. Without this the
        // single `index-<hash>.js` exceeds Vite's import-analysis
        // parser buffer once panzoom + hyparquet + uplot + xyflow + elk
        // are all in one file (~3 MB minified).
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          panzoom: ["panzoom"],
          uplot: ["uplot"],
          parquet: ["hyparquet"],
          flow: ["@xyflow/react", "elkjs"],
        },
      },
    },
    // Suppress the "chunk > 500 kB" warning — we already split deliberately.
    chunkSizeWarningLimit: 2000,
  },
});
