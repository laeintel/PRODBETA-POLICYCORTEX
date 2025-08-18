'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { GitBranch } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Branch Policies" subtitle="Branch Policies Operations Center" icon={GitBranch} />;
}