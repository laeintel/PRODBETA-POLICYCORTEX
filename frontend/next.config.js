/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Only use standalone for Docker builds
  output: process.env.DOCKER === 'true' ? 'standalone' : undefined,
  images: {
    domains: ['localhost'],
  },
  compiler: {
    // Remove console logs in production builds except warn/error
    removeConsole: process.env.NODE_ENV === 'production' ? { exclude: ['error', 'warn'] } : false,
  },
  async headers() {
    // Security headers are now handled by middleware.ts for better nonce support
    // This empty headers function is kept for Next.js compatibility
    return []
  },
  async rewrites() {
    const isDocker = process.env.IN_DOCKER === 'true' || process.env.DOCKER === 'true';
    // In local docker-compose, core service is named 'core'; in production compose it's 'backend'
    const backendService = process.env.BACKEND_SERVICE_NAME || (isDocker ? 'core' : 'localhost');
    const backendPort = process.env.BACKEND_SERVICE_PORT || (isDocker ? '8080' : '8080');
    const backendHost = isDocker ? `http://${backendService}:${backendPort}` : 'http://localhost:8080';
    const graphqlHost = isDocker ? 'http://graphql:4000' : (process.env.NEXT_PUBLIC_GRAPHQL_URL || 'http://localhost:4000');
    return [
      // Health passthrough so client-side calls to /health work in dev
      { source: '/health', destination: `${backendHost}/health` },
      { source: '/api/v1/:path*', destination: `${backendHost}/api/v1/:path*` },
      { source: '/api/:path*', destination: `${backendHost}/api/:path*` },
      { source: '/actions/:path*', destination: `${backendHost}/api/v1/actions/:path*` },
      { source: '/api/deep/:path*', destination: `${backendHost}/api/v1/:path*` },
      { source: '/graphql', destination: `${graphqlHost}/graphql` },
    ];
  },
  async redirects() {
    return [
      { source: '/dashboard', destination: '/tactical', permanent: false },
      { source: '/costs', destination: '/tactical/cost-governance', permanent: false },
      { source: '/devops', destination: '/tactical/devops', permanent: false },
      { source: '/monitoring', destination: '/tactical/monitoring-overview', permanent: false },
      // Ensure old /policies links route to the governance hub
      { source: '/policies', destination: '/governance', permanent: false },
      { source: '/operations', destination: '/tactical', permanent: false },
    ]
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        fs: false,
        net: false,
        tls: false,
      };
    }
    return config;
  },
};

module.exports = nextConfig;