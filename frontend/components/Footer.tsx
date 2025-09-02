'use client'

import Link from 'next/link'
import { Shield, Github, Twitter, Linkedin, Mail, Phone, MapPin } from 'lucide-react'

export default function Footer() {
  const currentYear = new Date().getFullYear()
  
  const footerLinks = {
    solutions: [
      { name: 'PolicyCortex Platform', href: '/platform' },
      { name: 'Cloud Governance', href: '/governance' },
      { name: 'Security & Compliance', href: '/security' },
      { name: 'Cost Optimization', href: '/finops' },
    ],
    resources: [
      { name: 'Documentation', href: '/docs' },
      { name: 'API Reference', href: '/api' },
      { name: 'Case Studies', href: '/case-studies' },
      { name: 'Blog', href: '/blog' },
    ],
    company: [
      { name: 'About Us', href: '/about' },
      { name: 'Careers', href: '/careers' },
      { name: 'Partners', href: '/partners' },
      { name: 'Contact', href: '/contact' },
    ],
    legal: [
      { name: 'Privacy Policy', href: '/privacy' },
      { name: 'Terms of Service', href: '/terms' },
      { name: 'Cookie Policy', href: '/cookies' },
      { name: 'Security', href: '/security-policy' },
    ],
  }
  
  return (
    <footer className="border-t border-border dark:border-gray-800 bg-background dark:bg-gray-900 safe-bottom">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Top section with logo and description */}
        <div className="mb-8 sm:mb-12">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-primary" />
              <div>
                <h3 className="text-lg font-bold text-foreground dark:text-white">PolicyCortex</h3>
                <p className="text-sm text-muted-foreground dark:text-gray-400">AI-Powered Cloud Governance</p>
              </div>
            </div>
            
            {/* Social links */}
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/policycortex"
                className="p-2 rounded-lg hover:bg-muted dark:hover:bg-gray-800 transition-colors touch-target"
                aria-label="GitHub"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Github className="w-5 h-5 text-muted-foreground dark:text-gray-400" />
              </a>
              <a
                href="https://twitter.com/policycortex"
                className="p-2 rounded-lg hover:bg-muted dark:hover:bg-gray-800 transition-colors touch-target"
                aria-label="Twitter"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Twitter className="w-5 h-5 text-muted-foreground dark:text-gray-400" />
              </a>
              <a
                href="https://linkedin.com/company/policycortex"
                className="p-2 rounded-lg hover:bg-muted dark:hover:bg-gray-800 transition-colors touch-target"
                aria-label="LinkedIn"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Linkedin className="w-5 h-5 text-muted-foreground dark:text-gray-400" />
              </a>
            </div>
          </div>
        </div>
        
        {/* Links grid - responsive columns */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8 mb-8 sm:mb-12">
          {/* Solutions column */}
          <div>
            <h4 className="text-sm font-semibold text-foreground dark:text-white mb-3">Solutions</h4>
            <ul className="space-y-2">
              {footerLinks.solutions.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Resources column */}
          <div>
            <h4 className="text-sm font-semibold text-foreground dark:text-white mb-3">Resources</h4>
            <ul className="space-y-2">
              {footerLinks.resources.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Company column */}
          <div>
            <h4 className="text-sm font-semibold text-foreground dark:text-white mb-3">Company</h4>
            <ul className="space-y-2">
              {footerLinks.company.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Contact column */}
          <div>
            <h4 className="text-sm font-semibold text-foreground dark:text-white mb-3">Contact</h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <MapPin className="w-4 h-4 text-muted-foreground dark:text-gray-400 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-muted-foreground dark:text-gray-400">
                  123 Cloud Street<br />
                  San Francisco, CA 94105
                </span>
              </li>
              <li className="flex items-center gap-2">
                <Phone className="w-4 h-4 text-muted-foreground dark:text-gray-400 flex-shrink-0" />
                <a
                  href="tel:+1-555-CORTEX"
                  className="text-sm text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors"
                >
                  +1 (555) CORTEX
                </a>
              </li>
              <li className="flex items-center gap-2">
                <Mail className="w-4 h-4 text-muted-foreground dark:text-gray-400 flex-shrink-0" />
                <a
                  href="mailto:info@policycortex.com"
                  className="text-sm text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors"
                >
                  info@policycortex.com
                </a>
              </li>
            </ul>
          </div>
        </div>
        
        {/* Divider */}
        <div className="border-t border-border dark:border-gray-800 pt-6 sm:pt-8">
          {/* Bottom section with copyright and legal links */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <p className="text-sm text-muted-foreground dark:text-gray-400">
              © {currentYear} PolicyCortex Inc. All rights reserved.
            </p>
            
            {/* Legal links - mobile responsive */}
            <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-sm">
              {footerLinks.legal.map((link, index) => (
                <div key={link.href} className="flex items-center gap-2 sm:gap-4">
                  <Link
                    href={link.href}
                    className="text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white transition-colors"
                  >
                    {link.name}
                  </Link>
                  {index < footerLinks.legal.length - 1 && (
                    <span className="text-muted-foreground dark:text-gray-600" aria-hidden="true">•</span>
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Compliance badges */}
          <div className="mt-6 flex flex-wrap items-center gap-4 text-xs text-muted-foreground dark:text-gray-500">
            <span>SOC 2 Type II</span>
            <span aria-hidden="true">•</span>
            <span>ISO 27001</span>
            <span aria-hidden="true">•</span>
            <span>GDPR Compliant</span>
            <span aria-hidden="true">•</span>
            <span>HIPAA Ready</span>
          </div>
        </div>
      </div>
    </footer>
  )
}