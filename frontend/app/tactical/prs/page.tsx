'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { GitBranch } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Pull Requests" subtitle="Pull Requests Operations Center" icon={GitBranch} />;
}