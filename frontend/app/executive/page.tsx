'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Shield,
  Users,
  AlertTriangle,
  CheckCircle,
  Target,
  BarChart3,
  PieChart,
  Activity,
  Building,
  Briefcase,
  FileText,
  Download,
  Calendar,
  ArrowUp,
  ArrowDown,
  Brain
} from 'lucide-react';
import { MLPredictionEngine } from '@/lib/ml-predictions';

interface BusinessKPI {
  id: string;
  name: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  target: string | number;
  status: 'on-track' | 'at-risk' | 'off-track';
  businessImpact: string;
}

interface RiskItem {
  id: string;
  category: 'financial' | 'operational' | 'security' | 'compliance';
  title: string;
  impact: string;
  likelihood: 'high' | 'medium' | 'low';
  mitigation: string;
  owner: string;
}

export default function ExecutiveDashboard() {
  const router = useRouter();
  const [kpis, setKpis] = useState<BusinessKPI[]>([]);
  const [risks, setRisks] = useState<RiskItem[]>([]);
  const [roiMetrics, setRoiMetrics] = useState<any>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load business-focused metrics
    const loadExecutiveData = async () => {
      try {
        // Fetch ROI data from API
        const roiRes = await fetch('/api/v1/executive/roi', { cache: 'no-store' });
        if (roiRes.ok) {
          const roiData = await roiRes.json();
          setRoiMetrics(roiData);
          
          // Transform API data to KPIs if available
          if (roiData.metrics) {
            const apiKpis = [
              {
                id: 'cloud-roi',
                name: 'Cloud ROI',
                value: roiData.metrics.roi_percentage || '287%',
                change: roiData.metrics.roi_change || 23,
                trend: (roiData.metrics.roi_trend || 'up') as 'up' | 'down' | 'stable',
                target: roiData.metrics.roi_target || '250%',
                status: (roiData.metrics.roi_status || 'on-track') as 'on-track' | 'at-risk' | 'off-track',
                businessImpact: roiData.metrics.roi_impact || 'Every $1 invested returns $2.87 in business value'
              },
              {
                id: 'cost-savings',
                name: 'Annual Cost Savings',
                value: roiData.metrics.savings || '$3.2M',
                change: roiData.metrics.savings_change || 45,
                trend: (roiData.metrics.savings_trend || 'up') as 'up' | 'down' | 'stable',
                target: roiData.metrics.savings_target || '$2.5M',
                status: (roiData.metrics.savings_status || 'on-track') as 'on-track' | 'at-risk' | 'off-track',
                businessImpact: roiData.metrics.savings_impact || '28% reduction in operational expenses'
              },
              ...roiData.metrics.additional_kpis || []
            ];
            setKpis(apiKpis);
          } else {
            // Use default KPIs if API doesn't have metrics
            setDefaultKpis();
          }
        } else {
          // Fallback to default KPIs
          setDefaultKpis();
        }
      } catch (error) {
        console.error('Failed to load executive data:', error);
        setDefaultKpis();
      }
      
      function setDefaultKpis() {
        setKpis([
        {
          id: 'cloud-roi',
          name: 'Cloud ROI',
          value: '287%',
          change: 23,
          trend: 'up',
          target: '250%',
          status: 'on-track',
          businessImpact: 'Every $1 invested returns $2.87 in business value'
        },
        {
          id: 'cost-savings',
          name: 'Annual Cost Savings',
          value: '$3.2M',
          change: 45,
          trend: 'up',
          target: '$2.5M',
          status: 'on-track',
          businessImpact: '28% reduction in operational expenses'
        },
        {
          id: 'compliance-score',
          name: 'Compliance Score',
          value: '94%',
          change: -2,
          trend: 'down',
          target: '95%',
          status: 'at-risk',
          businessImpact: 'Minor gap could impact SOC2 certification'
        },
        {
          id: 'security-posture',
          name: 'Security Risk Score',
          value: 'Low',
          change: -15,
          trend: 'down',
          target: 'Low',
          status: 'on-track',
          businessImpact: '73% reduction in security incidents YoY'
        },
        {
          id: 'time-to-market',
          name: 'Deployment Velocity',
          value: '3.2 days',
          change: -40,
          trend: 'down',
          target: '5 days',
          status: 'on-track',
          businessImpact: 'Features reach customers 40% faster'
        },
        {
          id: 'availability',
          name: 'Service Availability',
          value: '99.98%',
          change: 0.05,
          trend: 'up',
          target: '99.95%',
          status: 'on-track',
          businessImpact: 'Exceeding SLA commitments to enterprise clients'
        }
      ]);
      }

      setRisks([
        {
          id: 'risk-1',
          category: 'financial',
          title: 'Q2 Cloud Budget Overrun Risk',
          impact: '$450K potential overrun',
          likelihood: 'high',
          mitigation: 'Implement cost controls and reserved instances',
          owner: 'CFO'
        },
        {
          id: 'risk-2',
          category: 'compliance',
          title: 'GDPR Audit Finding',
          impact: 'Potential €2M fine',
          likelihood: 'medium',
          mitigation: 'Data retention policy update in progress',
          owner: 'Chief Compliance Officer'
        },
        {
          id: 'risk-3',
          category: 'operational',
          title: 'Single Point of Failure in Payment System',
          impact: '$100K/hour during outage',
          likelihood: 'low',
          mitigation: 'Multi-region failover being implemented',
          owner: 'CTO'
        }
      ]);

      setRoiMetrics({
        totalInvestment: 1120000,
        totalReturns: 3214400,
        paybackPeriod: '8 months',
        netPresentValue: 2094400,
        internalRateOfReturn: '187%'
      });

      setLoading(false);
    };

    loadExecutiveData();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'on-track': return 'text-green-600 bg-green-50 dark:bg-green-900/20';
      case 'at-risk': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20';
      case 'off-track': return 'text-red-600 bg-red-50 dark:bg-red-900/20';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20';
    }
  };

  const getRiskColor = (likelihood: string) => {
    switch (likelihood) {
      case 'high': return 'bg-red-100 dark:bg-red-900/30 border-red-300 dark:border-red-700';
      case 'medium': return 'bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700';
      case 'low': return 'bg-green-100 dark:bg-green-900/30 border-green-300 dark:border-green-700';
      default: return 'bg-gray-100 dark:bg-gray-900/30 border-gray-300 dark:border-gray-700';
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Executive Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Briefcase className="h-10 w-10 text-indigo-600" />
            Executive Intelligence Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Business-focused insights and ROI metrics for C-suite decision making
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => router.push('/executive/reports')}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2"
          >
            <FileText className="h-5 w-5" />
            Generate Board Report
          </button>
          <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
            <Calendar className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Executive Summary */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-6 mb-8 text-white">
        <h2 className="text-2xl font-bold mb-4">Executive Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-indigo-100 text-sm">Governance ROI</p>
            <p className="text-3xl font-bold">{roiMetrics.internalRateOfReturn}</p>
            <p className="text-indigo-100 text-sm mt-1">
              PolicyCortex delivering exceptional returns
            </p>
          </div>
          <div>
            <p className="text-indigo-100 text-sm">Risk Reduction</p>
            <p className="text-3xl font-bold">73%</p>
            <p className="text-indigo-100 text-sm mt-1">
              Critical incidents reduced year-over-year
            </p>
          </div>
          <div>
            <p className="text-indigo-100 text-sm">Operational Efficiency</p>
            <p className="text-3xl font-bold">42%</p>
            <p className="text-indigo-100 text-sm mt-1">
              Faster time-to-market for new features
            </p>
          </div>
        </div>
      </div>

      {/* Business KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {kpis.map((kpi) => (
          <div key={kpi.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {kpi.name}
              </h3>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(kpi.status)}`}>
                {kpi.status.replace('-', ' ').toUpperCase()}
              </span>
            </div>
            <div className="flex items-end justify-between mb-2">
              <p className="text-3xl font-bold">{kpi.value}</p>
              <div className="flex items-center gap-1">
                {kpi.trend === 'up' ? (
                  <ArrowUp className="h-4 w-4 text-green-500" />
                ) : kpi.trend === 'down' ? (
                  <ArrowDown className="h-4 w-4 text-red-500" />
                ) : null}
                <span className={`text-sm font-medium ${
                  kpi.trend === 'up' ? 'text-green-600' : 
                  kpi.trend === 'down' ? 'text-red-600' : 
                  'text-gray-600'
                }`}>
                  {Math.abs(kpi.change)}%
                </span>
              </div>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              Target: {kpi.target}
            </p>
            <div className="pt-3 border-t dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                <span className="font-medium">Business Impact:</span> {kpi.businessImpact}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* ROI Calculator Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* ROI Metrics */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="h-6 w-6 text-green-600" />
            Governance ROI Calculator
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Total Investment</span>
              <span className="font-semibold text-lg">
                ${roiMetrics.totalInvestment?.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Total Returns</span>
              <span className="font-semibold text-lg text-green-600">
                ${roiMetrics.totalReturns?.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Net Present Value</span>
              <span className="font-semibold text-lg text-green-600">
                ${roiMetrics.netPresentValue?.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Payback Period</span>
              <span className="font-semibold text-lg">{roiMetrics.paybackPeriod}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">Internal Rate of Return</span>
              <span className="font-semibold text-lg text-green-600">
                {roiMetrics.internalRateOfReturn}
              </span>
            </div>
          </div>
          <button
            onClick={() => router.push('/executive/roi')}
            className="mt-6 w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            View Detailed ROI Analysis
          </button>
        </div>

        {/* Risk to Revenue Map */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="h-6 w-6 text-red-600" />
            Risk to Revenue Heat Map
          </h2>
          <div className="space-y-3">
            {risks.map((risk) => (
              <div
                key={risk.id}
                className={`p-4 rounded-lg border ${getRiskColor(risk.likelihood)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold">{risk.title}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Impact: <span className="font-medium">{risk.impact}</span>
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Mitigation: {risk.mitigation}
                    </p>
                    <p className="text-xs text-gray-500 mt-2">
                      Owner: {risk.owner}
                    </p>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    risk.likelihood === 'high' ? 'bg-red-600 text-white' :
                    risk.likelihood === 'medium' ? 'bg-yellow-600 text-white' :
                    'bg-green-600 text-white'
                  }`}>
                    {risk.likelihood.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => router.push('/executive/risk-map')}
            className="mt-4 w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            View Complete Risk Assessment
          </button>
        </div>
      </div>

      {/* Department Performance */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Building className="h-6 w-6 text-blue-600" />
          Department Cloud Governance Performance
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { dept: 'Engineering', score: 92, cost: '$487K', compliance: 96 },
            { dept: 'Sales', score: 88, cost: '$123K', compliance: 94 },
            { dept: 'Marketing', score: 85, cost: '$89K', compliance: 91 },
            { dept: 'Operations', score: 94, cost: '$234K', compliance: 98 }
          ].map((dept) => (
            <div key={dept.dept} className="p-4 border dark:border-gray-700 rounded-lg">
              <h3 className="font-semibold mb-2">{dept.dept}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Governance Score</span>
                  <span className={`font-medium ${
                    dept.score >= 90 ? 'text-green-600' : 
                    dept.score >= 80 ? 'text-yellow-600' : 
                    'text-red-600'
                  }`}>
                    {dept.score}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Monthly Spend</span>
                  <span className="font-medium">{dept.cost}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Compliance</span>
                  <span className="font-medium">{dept.compliance}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions for Executives */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <button
          onClick={() => router.push('/executive/dashboard')}
          className="p-4 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 text-center"
        >
          <BarChart3 className="h-8 w-8 mx-auto mb-2" />
          <p className="font-semibold">View KPI Details</p>
        </button>
        <button
          onClick={() => router.push('/executive/roi')}
          className="p-4 bg-green-600 text-white rounded-lg hover:bg-green-700 text-center"
        >
          <Target className="h-8 w-8 mx-auto mb-2" />
          <p className="font-semibold">ROI Analysis</p>
        </button>
        <button
          onClick={() => router.push('/executive/risk-map')}
          className="p-4 bg-red-600 text-white rounded-lg hover:bg-red-700 text-center"
        >
          <Shield className="h-8 w-8 mx-auto mb-2" />
          <p className="font-semibold">Risk Dashboard</p>
        </button>
        <button
          onClick={() => router.push('/executive/reports')}
          className="p-4 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-center"
        >
          <FileText className="h-8 w-8 mx-auto mb-2" />
          <p className="font-semibold">Board Reports</p>
        </button>
      </div>
    </div>
  );
}