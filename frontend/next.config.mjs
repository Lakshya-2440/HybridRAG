const backendUrl =
  process.env.NODE_ENV === "development"
    ? "http://127.0.0.1:8000"
    : process.env.NEXT_PUBLIC_API_URL || "https://rag-backend.onrender.com";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
