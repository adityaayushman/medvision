/** @type {import('next').NextConfig} */

// In dev, proxy API + static image requests to the FastAPI backend so the
// browser talks to a single origin (no CORS juggling). Override with
// BACKEND_URL when the backend runs elsewhere.
const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/static/:path*", destination: `${BACKEND}/static/:path*` },
      { source: "/health", destination: `${BACKEND}/health` },
    ];
  },
};

export default nextConfig;
