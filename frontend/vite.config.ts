import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,json,vue,txt,woff2}']
      },
      manifest: {
        name: 'PolicyCortex',
        short_name: 'PolicyCortex',
        description: 'AI-Powered Azure Governance Intelligence Platform',
        theme_color: '#1976d2',
        background_color: '#ffffff',
        display: 'standalone',
        icons: [
          {
            src: 'vite.svg',
            sizes: '192x192',
            type: 'image/svg+xml'
          },
          {
            src: 'vite.svg',
            sizes: '512x512',
            type: 'image/svg+xml'
          }
        ]
      }
    })
  ],
  resolve: {
    alias: [
      {
        find: '@/config',
        replacement: path.resolve(__dirname, './src/config')
      },
      {
        find: '@/components',
        replacement: path.resolve(__dirname, './src/components')
      },
      {
        find: '@/pages',
        replacement: path.resolve(__dirname, './src/pages')
      },
      {
        find: '@/hooks',
        replacement: path.resolve(__dirname, './src/hooks')
      },
      {
        find: '@/utils',
        replacement: path.resolve(__dirname, './src/utils')
      },
      {
        find: '@/services',
        replacement: path.resolve(__dirname, './src/services')
      },
      {
        find: '@/types',
        replacement: path.resolve(__dirname, './src/types')
      },
      {
        find: '@/store',
        replacement: path.resolve(__dirname, './src/store')
      },
      {
        find: '@/assets',
        replacement: path.resolve(__dirname, './src/assets')
      },
      {
        find: '@/providers',
        replacement: path.resolve(__dirname, './src/providers')
      },
      {
        find: '@/routes',
        replacement: path.resolve(__dirname, './src/routes')
      },
      {
        find: '@',
        replacement: path.resolve(__dirname, './src')
      }
    ]
  },
  server: {
    port: 3000,
    strictPort: true,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          mui: ['@mui/material', '@mui/icons-material'],
          router: ['react-router-dom'],
          query: ['@tanstack/react-query']
        }
      }
    }
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true
  }
})