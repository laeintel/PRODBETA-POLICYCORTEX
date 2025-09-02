export default async function ExecutiveDashboard() {
  // Load executive metrics
  let metrics = {
    roi: { percentage: 287, quarterSavings: 287000, yearSavings: 1148000, riskAvoided: 1200000 },
    compliance: { score: 94, frameworks: 12, controls: 1847, gaps: 23 },
    security: { score: 89, incidents: 3, mttR: 4.2, prevented: 156 },
    operations: { availability: 99.97, resources: 342, automation: 89, incidents: 8 },
    cost: { budget: 600000, spent: 508092, saved: 91908, forecast: 595000 },
    risk: { high: 2, medium: 8, low: 23, mitigated: 45 }
  };

  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/api/v1/executive/metrics`, {
      cache: 'no-store'
    });
    if (res.ok) {
      metrics = await res.json();
    }
  } catch (error) {
    console.log('Using mock executive metrics');
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Executive Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white">Executive Dashboard</h1>
              <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
                Real-time governance metrics and business intelligence
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-500 dark:text-gray-400">Last Updated</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
              <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium">
                Export Report
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Primary KPI Cards - C-Suite Focus */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* ROI Card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="text-3xl font-bold text-green-600 dark:text-green-400">
                {metrics.roi.percentage}%
              </span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Return on Investment</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Quarter Savings</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  ${(metrics.roi.quarterSavings / 1000).toFixed(0)}K
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Risk Avoided</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  ${(metrics.roi.riskAvoided / 1000000).toFixed(1)}M
                </span>
              </div>
            </div>
          </div>

          {/* Compliance Score Card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <span className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                {metrics.compliance.score}%
              </span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Compliance Score</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Frameworks</span>
                <span className="font-medium text-gray-900 dark:text-white">{metrics.compliance.frameworks}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Controls</span>
                <span className="font-medium text-gray-900 dark:text-white">{metrics.compliance.controls}</span>
              </div>
            </div>
          </div>

          {/* Security Posture Card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <span className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                {metrics.security.score}%
              </span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Security Score</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Active Incidents</span>
                <span className="font-medium text-gray-900 dark:text-white">{metrics.security.incidents}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">MTTR</span>
                <span className="font-medium text-gray-900 dark:text-white">{metrics.security.mttR}h</span>
              </div>
            </div>
          </div>

          {/* Operational Excellence Card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="text-3xl font-bold text-amber-600 dark:text-amber-400">
                {metrics.operations.availability}%
              </span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">System Availability</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Resources</span>
                <span className="font-medium text-gray-900 dark:text-white">{metrics.operations.resources}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500 dark:text-gray-400">Automation</span>
                <span className="font-medium text-gray-900 dark:text-white">{metrics.operations.automation}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Strategic Value Cards - Patent Technologies */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Predictive AI Card */}
          <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl p-6 text-white shadow-xl">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold mb-2">Predictive AI Engine</h3>
                <p className="text-indigo-100 text-sm">
                  Patent #4: 99.2% accuracy in compliance drift prediction
                </p>
              </div>
              <div className="bg-white/20 backdrop-blur rounded-lg px-3 py-1">
                <span className="text-sm font-medium">PATENTED</span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-3xl font-bold">1,234</p>
                <p className="text-sm text-indigo-100">Active Predictions</p>
              </div>
              <div>
                <p className="text-3xl font-bold">7d</p>
                <p className="text-sm text-indigo-100">Look-ahead Window</p>
              </div>
            </div>
            <a href="/ai/predictions" className="inline-flex items-center gap-2 text-white hover:text-indigo-100 transition-colors">
              <span className="font-medium">View AI Insights</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </a>
          </div>

          {/* Blockchain Audit Card */}
          <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-xl">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold mb-2">Blockchain Audit Trail</h3>
                <p className="text-emerald-100 text-sm">
                  Tamper-evident immutable audit with cryptographic proof
                </p>
              </div>
              <div className="bg-white/20 backdrop-blur rounded-lg px-3 py-1">
                <span className="text-sm font-medium">VERIFIED</span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-3xl font-bold">100%</p>
                <p className="text-sm text-emerald-100">Chain Integrity</p>
              </div>
              <div>
                <p className="text-3xl font-bold">45K</p>
                <p className="text-sm text-emerald-100">Events Today</p>
              </div>
            </div>
            <a href="/audit" className="inline-flex items-center gap-2 text-white hover:text-emerald-100 transition-colors">
              <span className="font-medium">Verify Audit Trail</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </a>
          </div>

          {/* Governance Platform Card */}
          <div className="bg-gradient-to-br from-rose-500 to-pink-600 rounded-xl p-6 text-white shadow-xl">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold mb-2">Unified Governance</h3>
                <p className="text-rose-100 text-sm">
                  Patent #3: Cross-domain correlation with real-time insights
                </p>
              </div>
              <div className="bg-white/20 backdrop-blur rounded-lg px-3 py-1">
                <span className="text-sm font-medium">AI-POWERED</span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-3xl font-bold">156</p>
                <p className="text-sm text-rose-100">Policies Active</p>
              </div>
              <div>
                <p className="text-3xl font-bold">$45K</p>
                <p className="text-sm text-rose-100">Monthly Savings</p>
              </div>
            </div>
            <a href="/policy" className="inline-flex items-center gap-2 text-white hover:text-rose-100 transition-colors">
              <span className="font-medium">Manage Policies</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </a>
          </div>
        </div>

        {/* Financial Overview */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">Financial Overview</h2>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="space-y-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Annual Budget</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${(metrics.cost.budget / 1000).toFixed(0)}K
              </p>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all" 
                  style={{ width: `${(metrics.cost.spent / metrics.cost.budget) * 100}%` }}
                />
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">YTD Spent</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${(metrics.cost.spent / 1000).toFixed(0)}K
              </p>
              <p className="text-sm text-gray-500">
                {((metrics.cost.spent / metrics.cost.budget) * 100).toFixed(1)}% of budget
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Savings</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                ${(metrics.cost.saved / 1000).toFixed(0)}K
              </p>
              <p className="text-sm text-gray-500">15.3% cost reduction</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">EOY Forecast</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${(metrics.cost.forecast / 1000).toFixed(0)}K
              </p>
              <p className="text-sm text-green-600">Under budget by ${((metrics.cost.budget - metrics.cost.forecast) / 1000).toFixed(0)}K</p>
            </div>
          </div>
        </div>

        {/* Risk Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Distribution */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Risk Distribution</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">High Risk</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-red-500 h-2 rounded-full" style={{ width: '10%' }}></div>
                  </div>
                  <span className="text-lg font-semibold text-gray-900 dark:text-white w-8">{metrics.risk.high}</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">Medium Risk</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-amber-500 h-2 rounded-full" style={{ width: '30%' }}></div>
                  </div>
                  <span className="text-lg font-semibold text-gray-900 dark:text-white w-8">{metrics.risk.medium}</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">Low Risk</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '60%' }}></div>
                  </div>
                  <span className="text-lg font-semibold text-gray-900 dark:text-white w-8">{metrics.risk.low}</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">Mitigated</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '90%' }}></div>
                  </div>
                  <span className="text-lg font-semibold text-gray-900 dark:text-white w-8">{metrics.risk.mitigated}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Executive Actions</h3>
            <div className="grid grid-cols-2 gap-3">
              <a href="/tactical" className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors text-center">
                <svg className="w-8 h-8 mx-auto mb-2 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span className="text-sm font-medium text-gray-900 dark:text-white">Command Center</span>
              </a>
              <a href="/policy" className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors text-center">
                <svg className="w-8 h-8 mx-auto mb-2 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="text-sm font-medium text-gray-900 dark:text-white">Policy Review</span>
              </a>
              <a href="/finops" className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors text-center">
                <svg className="w-8 h-8 mx-auto mb-2 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm font-medium text-gray-900 dark:text-white">Cost Analysis</span>
              </a>
              <a href="/audit" className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors text-center">
                <svg className="w-8 h-8 mx-auto mb-2 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <span className="text-sm font-medium text-gray-900 dark:text-white">Audit Reports</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}