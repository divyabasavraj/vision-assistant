// Vitest config. We only test pure logic files (no browser), so no DOM environment is needed.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
