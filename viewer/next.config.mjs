/** Static export: the report must stay up even when no server does. */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};
export default nextConfig;
