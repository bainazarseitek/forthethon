import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/app": {
          target: env.API_ORIGIN || "http://127.0.0.1:8000",
          changeOrigin: false,
          ws: true,
        },
      },
    },
    build: {
      outDir: "../app/static",
      emptyOutDir: true,
      sourcemap: false,
    },
  };
});
