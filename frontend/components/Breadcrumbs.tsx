'use client'

import Link from 'next/link'

export interface Crumb {
  href: string
  label: string
}

export default function Breadcrumbs({ items }: { items: Crumb[] }) {
  return (
    <nav aria-label="Breadcrumb" className="text-sm text-gray-400 mb-4">
      {items.map((item, idx) => (
        <span key={item.href}>
          {idx > 0 && <span className="mx-2">/</span>}
          {idx < items.length - 1 ? (
            <Link href={item.href} className="hover:text-white">{item.label}</Link>
          ) : (
            <span className="text-white">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}


