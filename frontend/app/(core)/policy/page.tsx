export default async function PolicyHub() {
  // Load policy metrics
  let metrics = {
    coverage: { accounts: 12, resources: 87, percentage: 94 },
    predictedDrift: { count: 23, timeframe: '7 days', severity: 'medium' },
    violationsPrevented: { count: 156, timeframe: '30 days', savings: 45000 },
    costImpact: { savings: 125000, avoidance: 78000, timeframe: 'quarter' }
  };

  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/api/v1/policy/metrics`, {
      cache: 'no-store'
    });
    if (res.ok) {
      metrics = await res.json();
    }
  } catch (error) {
    console.log('Using mock policy metrics');
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-700 pb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Policy</h1>
        <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
          Define, simulate, enforce, and evidence cloud guardrails
        </p>
      </header>

      {/* Hero Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Coverage Card */}
        <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Coverage</h3>
            <span className="text-3xl font-bold">{metrics.coverage.percentage}%</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-blue-100">Accounts in scope</span>
              <span className="font-medium">{metrics.coverage.accounts}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-100">Resources covered</span>
              <span className="font-medium">{metrics.coverage.resources}%</span>
            </div>
          </div>
        </div>

        {/* Predicted Drift Card */}
        <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Predicted Drift</h3>
            <span className="text-3xl font-bold">{metrics.predictedDrift.count}</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-purple-100">Timeframe</span>
              <span className="font-medium">{metrics.predictedDrift.timeframe}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-purple-100">Severity</span>
              <span className="font-medium capitalize">{metrics.predictedDrift.severity}</span>
            </div>
          </div>
        </div>

        {/* Violations Prevented Card */}
        <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Violations Prevented</h3>
            <span className="text-3xl font-bold">{metrics.violationsPrevented.count}</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-green-100">Last {metrics.violationsPrevented.timeframe}</span>
              <span className="font-medium">${(metrics.violationsPrevented.savings / 1000).toFixed(0)}K saved</span>
            </div>
          </div>
        </div>

        {/* Cost Impact Card */}
        <div className="bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Cost Impact</h3>
            <span className="text-3xl font-bold">${((metrics.costImpact.savings + metrics.costImpact.avoidance) / 1000).toFixed(0)}K</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-amber-100">Savings</span>
              <span className="font-medium">${(metrics.costImpact.savings / 1000).toFixed(0)}K</span>
            </div>
            <div className="flex justify-between">
              <span className="text-amber-100">Avoidance</span>
              <span className="font-medium">${(metrics.costImpact.avoidance / 1000).toFixed(0)}K</span>
            </div>
          </div>
        </div>
      </div>

      {/* Primary Actions */}
      <div className="flex flex-wrap gap-4">
        <a
          href="/policy/composer"
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Policy
        </a>
        <a
          href="/policy/packs"
          className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-medium flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
          </svg>
          Install Policy Pack
        </a>
        <button className="px-6 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors font-medium flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          Run Simulation
        </button>
      </div>

      {/* Lists Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Policies */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Active Policies</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div>
                <p className="font-medium text-gray-900 dark:text-white">NIST 800-53 Baseline</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">156 controls • Auto-fix enabled</p>
              </div>
              <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded">Active</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div>
                <p className="font-medium text-gray-900 dark:text-white">Cost Optimization Pack</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">42 controls • PR-Block mode</p>
              </div>
              <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded">Active</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div>
                <p className="font-medium text-gray-900 dark:text-white">Data Residency Policy</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">8 controls • Observe mode</p>
              </div>
              <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 rounded">Testing</span>
            </div>
          </div>
        </div>

        {/* Exceptions Expiring Soon */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Exceptions Expiring Soon</h3>
          <div className="space-y-3">
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="flex items-center justify-between">
                <p className="font-medium text-gray-900 dark:text-white">Legacy DB Access</p>
                <span className="text-sm text-red-600 dark:text-red-400 font-medium">2 days</span>
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Production database exception</p>
            </div>
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="flex items-center justify-between">
                <p className="font-medium text-gray-900 dark:text-white">Public S3 Bucket</p>
                <span className="text-sm text-amber-600 dark:text-amber-400 font-medium">5 days</span>
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Marketing assets exception</p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between">
                <p className="font-medium text-gray-900 dark:text-white">Non-MFA Admin</p>
                <span className="text-sm text-gray-600 dark:text-gray-400 font-medium">14 days</span>
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Service account exception</p>
            </div>
          </div>
        </div>

        {/* Recent Prevention Events */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Recent Prevention Events</h3>
          <div className="space-y-3">
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <p className="font-medium text-gray-900 dark:text-white">Blocked: Public RDS Instance</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">Prevented at CI gate • 10 min ago</p>
            </div>
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <p className="font-medium text-gray-900 dark:text-white">Auto-fixed: Unencrypted Volume</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">Applied encryption • 2 hours ago</p>
            </div>
            <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
              <p className="font-medium text-gray-900 dark:text-white">Warning: Oversized Instance</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">PR created for right-sizing • 5 hours ago</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}