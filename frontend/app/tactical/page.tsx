'use client';

import Link from 'next/link';
import { 
  Shield, FileCheck, Box, BarChart3, Brain, DollarSign, Settings, MessageSquare,
  Users, Network, Lock, GitBranch
} from 'lucide-react';

export default function TacticalLanding() {
  const sections: Array<{
    id: string;
    title: string;
    icon: React.ElementType;
    description: string;
    links: Array<{ label: string; href: string }>;
  }> = [
    {
      id: 'security',
      title: 'Security & Compliance',
      icon: Shield,
      description: 'Posture, detection, identity, policies, and audits',
      links: [
        { label: 'Security Overview', href: '/security/overview' },
        { label: 'Threat Detection', href: '/security/threat-detection' },
        { label: 'Vulnerability Scan', href: '/security/vulnerability-scan' },
        { label: 'Access Control', href: '/security/access-control' },
        { label: 'Identity Management', href: '/security/identity-management' },
        { label: 'Compliance Hub', href: '/security/compliance-hub' },
        { label: 'Policy Engine', href: '/security/policy-engine' },
        { label: 'Audit Trail', href: '/security/audit-trail' },
        { label: 'Encryption Keys', href: '/security/encryption-keys' },
        { label: 'Certificates', href: '/security/certificates' },
        { label: 'Security Groups', href: '/security/security-groups' },
        { label: 'Firewall Rules', href: '/security/firewall-rules' }
      ]
    },
    {
      id: 'governance',
      title: 'Governance & Policy',
      icon: FileCheck,
      description: 'Standards, exceptions, and insights',
      links: [
        { label: 'Policies', href: '/policies' },
        { label: 'Violations', href: '/policies/violations' },
        { label: 'AI Insights', href: '/policies/insights' },
        { label: 'Compliance', href: '/compliance' }
      ]
    },
    {
      id: 'infrastructure',
      title: 'Infrastructure',
      icon: Box,
      description: 'Resources, topology, and platforms',
      links: [
        { label: 'Compute Resources', href: '/tactical/compute' },
        { label: 'Storage Systems', href: '/tactical/storage' }
      ]
    },
    {
      id: 'monitoring',
      title: 'Monitoring & Analytics',
      icon: BarChart3,
      description: 'Observability, SLOs, and telemetry',
      links: [
        { label: 'Monitoring Overview', href: '/tactical/monitoring-overview' }
      ]
    },
    {
      id: 'ai',
      title: 'AI & Intelligence',
      icon: Brain,
      description: 'Predictions, correlations, and copilots',
      links: [
        { label: 'Predictions', href: '/predictions' },
        { label: 'Correlations', href: '/correlations' },
        { label: 'AI Expert', href: '/ai-expert' }
      ]
    },
    {
      id: 'finops',
      title: 'Financial Management',
      icon: DollarSign,
      description: 'Costs, budgets, and optimizations',
      links: [
        { label: 'Cost Overview', href: '/tactical/cost-governance' },
        { label: 'Cost Governance', href: '/tactical/cost-governance' },
        { label: 'Budgets', href: '/tactical/budgets' }
      ]
    },
    {
      id: 'devops',
      title: 'DevOps & CI/CD',
      icon: Settings,
      description: 'Pipelines, releases, and artifacts',
      links: [
        { label: 'DevOps', href: '/tactical/devops' },
        { label: 'Pipelines', href: '/tactical/pipelines' },
        { label: 'Releases', href: '/tactical/releases' }
      ]
    },
    {
      id: 'communication',
      title: 'Communication',
      icon: MessageSquare,
      description: 'Notifications and channels',
      links: [
        { label: 'Notifications', href: '/tactical/notifications' },
        { label: 'Emails', href: '/tactical/emails' },
        { label: 'SMS', href: '/tactical/sms' },
        { label: 'Slack', href: '/tactical/slack' }
      ]
    },
    {
      id: 'admin',
      title: 'Administration',
      icon: Users,
      description: 'Access, teams, and integrations',
      links: [
        { label: 'Users', href: '/tactical/users' },
        { label: 'Roles', href: '/tactical/roles' },
        { label: 'Teams', href: '/tactical/teams' },
        { label: 'Integrations', href: '/tactical/integrations' }
      ]
    }
  ];

  return (
    <main className="min-h-screen bg-gray-900">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <header className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Tactical Command</h1>
            <p className="text-sm text-gray-400">Operational dashboard across security, governance, infra, and finops</p>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-gray-800/60 border border-gray-800 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-400">Health</p>
              <p className="text-sm text-green-400 font-semibold">98%</p>
            </div>
            <div className="bg-gray-800/60 border border-gray-800 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-400">Issues</p>
              <p className="text-sm text-yellow-400 font-semibold">23</p>
            </div>
            <div className="bg-gray-800/60 border border-gray-800 rounded-lg px-3 py-2">
              <p className="text-[10px] text-gray-400">Spend/Day</p>
              <p className="text-sm text-white font-semibold">$4.2K</p>
            </div>
          </div>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Policy Compliance', value: '84.4%', badge: '↑2.3%', href: '/policies' },
            { label: 'Monthly Spend', value: '$127k', badge: '↓3.2%', href: '/tactical/cost-governance' },
            { label: 'Risk Score', value: '72.0', badge: '↓5.1%', href: '/security/overview' },
            { label: 'Resource Utilization', value: '63.5%', badge: '↑5.2%', href: '/tactical/compute' },
          ].map(k => (
            <Link key={k.label} href={k.href} className="block bg-black border border-gray-800 rounded-xl p-4 hover:border-gray-700">
              <p className="text-xs text-gray-400">{k.label}</p>
              <div className="flex items-baseline justify-between mt-1">
                <p className="text-2xl font-bold text-white">{k.value}</p>
                <span className={`text-xs px-2 py-0.5 rounded-full ${k.badge.includes('↑') ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>{k.badge}</span>
              </div>
            </Link>
          ))}
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {sections.map(({ id, title, icon: Icon, description, links }) => (
            <div key={id} className="bg-black border border-gray-800 rounded-xl p-5">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-gray-800">
                  <Icon className="w-5 h-5 text-gray-200" />
                </div>
                <h2 className="text-sm font-semibold text-white">{title}</h2>
              </div>
              <p className="text-xs text-gray-400 mb-3">{description}</p>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {links.map((l) => (
                  <li key={l.href}>
                    <Link href={l.href} className="block px-3 py-2 rounded-lg border border-gray-800 hover:bg-gray-800 text-xs text-gray-200">
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
      </div>
    </main>
  );
}