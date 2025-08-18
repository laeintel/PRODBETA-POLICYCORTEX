'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { AlertTriangle } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Incident Response" subtitle="Incident Response Operations Center" icon={AlertTriangle} />;
}