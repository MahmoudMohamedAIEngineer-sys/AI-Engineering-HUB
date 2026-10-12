import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The lesson and knowledge files are read from the repository at render time, so they must not
  // be traced away from the project root by the bundler.
  outputFileTracingRoot: projectRoot,
  // The platform serves this app from a proxied preview host, so the dev server must accept
  // requests for that origin rather than refusing them as cross-origin.
  allowedDevOrigins: ["*.e2b.app", "*.arena.ai", "localhost"],
};

export default nextConfig;
