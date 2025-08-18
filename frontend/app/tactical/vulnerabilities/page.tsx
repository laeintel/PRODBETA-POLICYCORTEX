'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { ScanLine } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Vulnerability Scan" subtitle="Vulnerability Scan Operations Center" icon={ScanLine} />;
}