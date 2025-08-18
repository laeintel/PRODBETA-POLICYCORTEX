'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { GitBranch } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Trace Analysis" subtitle="Trace Analysis Operations Center" icon={GitBranch} />;
}