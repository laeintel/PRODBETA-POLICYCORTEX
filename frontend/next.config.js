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
    const isProd = process.env.NODE_ENV === 'production'
    const useWs = (process.env.NEXT_PUBLIC_USE_WS || '').toLowerCase() === 'true'

    const allowedConnect = new Set(["'self'", 'https:'])
    if (!isProd) {
      allowedConnect.add('http://localhost:8080')
      allowedConnect.add('http://localhost:4000')
    }
    if (process.env.NEXT_PUBLIC_API_URL) allowedConnect.add(process.env.NEXT_PUBLIC_API_URL)
    if (process.env.NEXT_PUBLIC_WS_URL) allowedConnect.add(process.env.NEXT_PUBLIC_WS_URL)
    if (useWs) { allowedConnect.add('wss:'); if (!isProd) allowedConnect.add('ws:') }

    const scriptSrc = isProd ? "script-src 'self'" : "script-src 'self' 'unsafe-inline' 'unsafe-eval'"
    const styleSrc = isProd ? "style-src 'self'" : "style-src 'self' 'unsafe-inline'"

    const ContentSecurityPolicy = [
      "default-src 'self'",
      scriptSrc,
      styleSrc,
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      `connect-src ${Array.from(allowedConnect).join(' ')}`,
      "frame-ancestors 'none'",
      "object-src 'none'",
      "base-uri 'self'",
    ].join('; ')

    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(self), geolocation=(), payment=()' },
          { key: 'Content-Security-Policy', value: ContentSecurityPolicy },
        ],
      },
    ]
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
      { source: '/governance', destination: '/policies', permanent: false },
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