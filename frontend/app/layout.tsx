/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import type { Metadata, Viewport } from 'next'
import { Inter, Orbitron, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import AppShell from '../components/AppShell'
import GlobalOmnibar from '../components/GlobalOmnibar'
import { headers } from 'next/headers'
import Script from 'next/script'
import { PerformanceMonitor } from '../components/PerformanceMonitor'
import { AccessibilityChecker } from '../components/AccessibilityChecker'

const inter = Inter({ subsets: ['latin'] })
const orbitron = Orbitron({ 
  subsets: ['latin'],
  variable: '--font-display',
})
const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-mono',
})

export const metadata: Metadata = {
  title: 'PolicyCortex - AI-Powered Azure Governance',
  description: 'Transform Azure governance with AI-powered insights and automation',
  manifest: '/manifest.json',
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: '#3b82f6',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Get nonce from headers set by middleware
  const nonce = headers().get('x-nonce') || ''
  
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <Script
          id="theme-script"
          nonce={nonce}
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                const theme = localStorage.getItem('theme');
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                const selectedTheme = theme || (prefersDark ? 'dark' : 'light');
                
                if (selectedTheme === 'dark') {
                  document.documentElement.classList.add('dark');
                  document.documentElement.style.colorScheme = 'dark';
                } else {
                  document.documentElement.classList.remove('dark');
                  document.documentElement.style.colorScheme = 'light';
                }
              })();
            `,
          }}
        />
      </head>
      <body className={`${inter.className} ${orbitron.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
        {/* Skip to main content link for accessibility */}
        <a 
          href="#main-content" 
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[10000] focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
        >
          Skip to main content
        </a>
        <Providers>
          <GlobalOmnibar />
          <AppShell>{children}</AppShell>
        </Providers>
        {/* Development tools */}
        {process.env.NODE_ENV === 'development' && (
          <>
            <PerformanceMonitor />
            <AccessibilityChecker />
          </>
        )}
      </body>
    </html>
  )
}