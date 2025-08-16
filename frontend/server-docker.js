/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

const { createServer } = require('http')
const { parse } = require('url')
const next = require('next')
const httpProxy = require('http-proxy-middleware')

const dev = false
const hostname = '0.0.0.0'
const port = 3000

const app = next({ dev, hostname, port })
const handle = app.getRequestHandler()

// Create proxy middleware
const apiProxy = httpProxy.createProxyMiddleware({
  target: 'http://backend:8080',
  changeOrigin: true,
  onError: (err, req, res) => {
    console.error('Proxy error:', err)
    res.status(503).json({ error: 'Service unavailable' })
  }
})

const graphqlProxy = httpProxy.createProxyMiddleware({
  target: 'http://graphql:4000',
  changeOrigin: true,
  onError: (err, req, res) => {
    console.error('GraphQL proxy error:', err)
    res.status(503).json({ error: 'GraphQL service unavailable' })
  }
})

app.prepare().then(() => {
  createServer(async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true)
      const { pathname } = parsedUrl

      // Proxy API requests to backend
      if (pathname.startsWith('/api/') || pathname === '/health') {
        return apiProxy(req, res)
      }

      // Proxy actions to backend
      if (pathname.startsWith('/actions/')) {
        req.url = req.url.replace('/actions/', '/api/v1/actions/')
        return apiProxy(req, res)
      }

      // Proxy GraphQL requests
      if (pathname === '/graphql') {
        return graphqlProxy(req, res)
      }

      // Handle Next.js requests
      await handle(req, res, parsedUrl)
    } catch (err) {
      console.error('Error occurred handling', req.url, err)
      res.statusCode = 500
      res.end('Internal server error')
    }
  })
    .once('error', (err) => {
      console.error(err)
      process.exit(1)
    })
    .listen(port, hostname, () => {
      console.log(`> Ready on http://${hostname}:${port}`)
    })
})