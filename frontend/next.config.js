/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  images: {
    domains: ['localhost'],
  },
  compiler: {
    // Remove console logs in production builds except warn/error
    removeConsole: process.env.NODE_ENV === 'production' ? { exclude: ['error', 'warn'] } : false,
  },
  async headers() {
    // Security headers suitable for demo/prod; adjust CSP as needed for integrations
    const ContentSecurityPolicy = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "connect-src 'self' https: wss: ws:",
      "frame-ancestors 'none'",
      "object-src 'none'",
      "base-uri 'self'",
    ].join('; ');

    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=(), payment=()' },
          { key: 'Content-Security-Policy', value: ContentSecurityPolicy },
        ],
      },
    ];
  },
  async rewrites() {
    const isDocker = process.env.IN_DOCKER === 'true' || process.env.DOCKER === 'true';
    // In local docker-compose, core service is named 'core'; in production compose it's 'backend'
    const backendService = process.env.BACKEND_SERVICE_NAME || (isDocker ? 'core' : 'localhost');
    const backendPort = process.env.BACKEND_SERVICE_PORT || (isDocker ? '8080' : '8080');
    const backendHost = isDocker ? `http://${backendService}:${backendPort}` : 'http://localhost:8080';
    const graphqlHost = isDocker ? 'http://graphql:4000' : 'http://localhost:4000';
    return [
      { source: '/api/v1/:path*', destination: `${backendHost}/api/v1/:path*` },
      { source: '/api/:path*', destination: `${backendHost}/api/:path*` },
      { source: '/actions/:path*', destination: `${backendHost}/api/v1/actions/:path*` },
      { source: '/api/deep/:path*', destination: `${backendHost}/api/v1/:path*` },
      { source: '/graphql', destination: `${graphqlHost}/graphql` },
    ];
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