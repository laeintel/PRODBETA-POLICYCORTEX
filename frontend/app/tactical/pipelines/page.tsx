'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { GitBranch } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Pipeline Dashboard" subtitle="Pipeline Dashboard Operations Center" icon={GitBranch} />;
}