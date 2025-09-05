/**
 * Governance P&L - The CFO Moat
 * Shows $ impact per policy + 90-day forecast
 */

import { real } from '@/lib/real';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, TrendingDown, DollarSign, AlertCircle } from 'lucide-react';

interface PolicyPnL {
  policy: string;
  policyType: string;
  resourcesAffected: number;
  savings: number;
  forecast: number;
  trend: 'up' | 'down' | 'stable';
  complianceRate: number;
  riskReduction: number;
}

interface GovernancePnLData {
  items: PolicyPnL[];
  totalSavings: number;
  totalForecast: number;
  totalPolicies: number;
  complianceAverage: number;
}

export default async function GovernancePnL() {
  try {
    const data = await real<GovernancePnLData>('/api/v1/costs/pnl');
    
    return (
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Governance P&L</h1>
          <p className="mt-2 text-gray-600">Per-policy financial impact and 90-day forecast</p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Total Savings MTD</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <DollarSign className="h-4 w-4 text-green-600 mr-1" />
                <span className="text-2xl font-bold text-green-600">
                  ${data.totalSavings.toLocaleString()}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">90-Day Forecast</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <TrendingUp className="h-4 w-4 text-blue-600 mr-1" />
                <span className="text-2xl font-bold text-blue-600">
                  ${data.totalForecast.toLocaleString()}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Active Policies</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-2xl font-bold">{data.totalPolicies}</span>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Compliance Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-2xl font-bold">{data.complianceAverage}%</span>
            </CardContent>
          </Card>
        </div>

        {/* Policy Impact Table */}
        <Card>
          <CardHeader>
            <CardTitle>Policy-Level Impact Analysis</CardTitle>
            <CardDescription>
              Financial impact breakdown by individual governance policy
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-4">Policy</th>
                    <th className="text-left py-3 px-4">Type</th>
                    <th className="text-right py-3 px-4">Resources</th>
                    <th className="text-right py-3 px-4">Savings MTD</th>
                    <th className="text-right py-3 px-4">90-day Forecast</th>
                    <th className="text-center py-3 px-4">Trend</th>
                    <th className="text-right py-3 px-4">Compliance</th>
                    <th className="text-right py-3 px-4">Risk Reduction</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((item) => (
                    <tr key={item.policy} className="border-b hover:bg-gray-50">
                      <td className="py-3 px-4">
                        <div className="font-medium">{item.policy}</div>
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {item.policyType}
                        </span>
                      </td>
                      <td className="text-right py-3 px-4">{item.resourcesAffected}</td>
                      <td className="text-right py-3 px-4">
                        <span className={item.savings > 0 ? 'text-green-600 font-medium' : ''}>
                          ${item.savings.toLocaleString()}
                        </span>
                      </td>
                      <td className="text-right py-3 px-4">
                        <span className="font-medium">${item.forecast.toLocaleString()}</span>
                      </td>
                      <td className="text-center py-3 px-4">
                        {item.trend === 'up' && <TrendingUp className="h-4 w-4 text-green-600 mx-auto" />}
                        {item.trend === 'down' && <TrendingDown className="h-4 w-4 text-red-600 mx-auto" />}
                        {item.trend === 'stable' && <span className="text-gray-400">—</span>}
                      </td>
                      <td className="text-right py-3 px-4">
                        <span className={item.complianceRate >= 80 ? 'text-green-600' : 'text-amber-600'}>
                          {item.complianceRate}%
                        </span>
                      </td>
                      <td className="text-right py-3 px-4">
                        <span className="text-blue-600">-{item.riskReduction}%</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* ROI Explanation */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertCircle className="h-5 w-5 mr-2 text-blue-600" />
              How We Calculate ROI
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm max-w-none text-gray-600">
              <p>
                Our Governance P&L tracks the direct financial impact of each policy enforcement:
              </p>
              <ul>
                <li><strong>Savings MTD:</strong> Actual cost reduction from resource optimization, rightsizing, and waste elimination this month</li>
                <li><strong>90-day Forecast:</strong> Projected savings based on current enforcement rate and resource growth patterns</li>
                <li><strong>Risk Reduction:</strong> Percentage decrease in security incidents and compliance violations</li>
                <li><strong>Compliance Rate:</strong> Percentage of resources meeting policy requirements</li>
              </ul>
              <p className="mt-4">
                Data sourced from Azure Cost Management API with real-time policy enforcement metrics.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  } catch (error) {
    return (
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <CardContent className="py-8">
            <div className="text-center">
              <AlertCircle className="h-12 w-12 text-amber-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Unable to Load P&L Data</h3>
              <p className="text-gray-600">
                Please ensure the Azure Cost Management API is configured and accessible.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }
}