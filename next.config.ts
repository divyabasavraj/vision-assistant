import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Phone testing over the LAN (npm run dev -- --hostname 0.0.0.0 ...): the dev server only serves
  // its JS to origins listed here. Dev-only; has no effect on the production build.
  allowedDevOrigins: ["192.168.0.*", "10.*", "172.16.*"],
};

export default nextConfig;
