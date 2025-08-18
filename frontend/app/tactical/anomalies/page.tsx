'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { AlertCircle } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Anomaly Detection" subtitle="Anomaly Detection Operations Center" icon={AlertCircle} />;
}