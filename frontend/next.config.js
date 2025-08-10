/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      // Route all API v1 calls to the Rust backend
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8080/api/v1/:path*',
      },
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/api/:path*',
      },
      {
        source: '/actions/:path*',
        destination: 'http://localhost:8090/api/v1/actions/:path*',
      },
      {
        source: '/api/deep/:path*',
        destination: 'http://localhost:8090/api/v1/:path*',
      },
      {
        source: '/graphql',
        destination: 'http://localhost:4000/graphql',
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