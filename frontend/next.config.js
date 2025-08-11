/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    const isDocker = process.env.IN_DOCKER === 'true' || process.env.DOCKER === 'true';
    const backendHost = isDocker ? 'http://backend:8080' : 'http://localhost:8080';
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