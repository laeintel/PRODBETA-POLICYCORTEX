'use client';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { AlertTriangle } from 'lucide-react';

export default function Page() {
  return <TacticalPageTemplate title="Anomaly Alerts" subtitle="Anomaly Alerts Operations Center" icon={AlertTriangle} />;
}