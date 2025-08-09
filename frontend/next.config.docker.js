/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  images: {
    domains: ['localhost', 'backend', 'frontend'],
  },
  async rewrites() {
    return [
      // In Docker, use container names for internal communication
      {
        source: '/api/v1/:path*',
        destination: 'http://backend:8080/api/v1/:path*',
      },
      {
        source: '/api/:path*',
        destination: 'http://backend:8080/api/:path*',
      },
      {
        source: '/actions/:path*',
        destination: 'http://backend:8080/api/v1/actions/:path*',
      },
      {
        source: '/graphql',
        destination: 'http://graphql:4000/graphql',
      },
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