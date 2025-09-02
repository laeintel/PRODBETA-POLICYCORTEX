export default function PolicyPacks() {
  const packs = [
    { name: 'NIST 800-53 Moderate', controls: 325, driftReduction: '87%', impact: '$234K/yr', status: 'installed' },
    { name: 'CIS AWS Benchmark', controls: 195, driftReduction: '72%', impact: '$156K/yr', status: 'available' },
    { name: 'CIS Azure Benchmark', controls: 208, driftReduction: '75%', impact: '$189K/yr', status: 'available' },
    { name: 'SOC2 Type II', controls: 64, driftReduction: '91%', impact: '$89K/yr', status: 'installed' },
    { name: 'HIPAA', controls: 78, driftReduction: '94%', impact: '$112K/yr', status: 'available' },
    { name: 'FedRAMP High', controls: 421, driftReduction: '96%', impact: '$456K/yr', status: 'available' },
  ];

  return (
    <div className="space-y-6">
      <header className="border-b border-gray-200 dark:border-gray-700 pb-4">
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Policy Packs</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Curated baselines for compliance frameworks and best practices
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {packs.map((pack) => (
          <div key={pack.name} className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{pack.name}</h3>
              {pack.status === 'installed' ? (
                <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded">Installed</span>
              ) : (
                <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">Install</button>
              )}
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Controls</span>
                <span className="font-medium text-gray-900 dark:text-white">{pack.controls}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Drift Reduction</span>
                <span className="font-medium text-green-600 dark:text-green-400">{pack.driftReduction}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Est. Impact</span>
                <span className="font-medium text-blue-600 dark:text-blue-400">{pack.impact}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}