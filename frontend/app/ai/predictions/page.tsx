import { PredictionCard } from '@/components/PredictionCard'
import { Brain, TrendingUp, AlertTriangle, DollarSign, Shield } from 'lucide-react'

async function getPredictions() {
  try {
    // Use environment variable for base URL, fallback to empty string for same-origin requests
    const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || ''
    const res = await fetch(`${baseUrl}/api/v1/predictions`, {
      cache: 'no-store'
    })
    if (res.ok) {
      const data = await res.json()
      // Transform the API response to match our component structure
      if (data.predictions) {
        return data.predictions.map((p: any) => ({
          id: p.id,
          title: p.resource,
          kind: p.type,
          confidence: p.confidence / 100, // Convert percentage to decimal
          explanation: p.prediction,
          eta: p.timeframe,
          impact: p.impact,
          recommendation: p.recommendation,
          riskLevel: p.impact === 'High' ? 'HIGH' : p.impact === 'Medium' ? 'MEDIUM' : 'LOW',
          category: p.type.includes('Security') ? 'Security' : 
                    p.type.includes('Cost') ? 'FinOps' : 
                    p.type.includes('Compliance') ? 'Compliance' : 'Operations'
        }))
      }
      return data
    }
  } catch (error) {
    console.error('Failed to fetch predictions:', error)
  }
  
  // Return mock data as fallback
  return [
    {
      id: '1',
      title: 'Storage Account Public Access Risk',
      kind: 'COMPLIANCE_DRIFT',
      confidence: 0.92,
      explanation: 'Multiple storage accounts are predicted to have public access enabled within 3 days based on recent configuration patterns.',
      eta: '3 days',
      impact: 'High - Potential data exposure',
      recommendation: 'Enable Azure Policy to deny public access on storage accounts',
      riskLevel: 'HIGH',
      category: 'Security'
    },
    {
      id: '2',
      title: 'Cost Anomaly Detection',
      kind: 'COST_ANOMALY',
      confidence: 0.87,
      explanation: 'Unusual spending pattern detected in compute resources, projected to exceed budget by 23%.',
      eta: '5 days',
      impact: '$12,500 over budget',
      recommendation: 'Review and rightsize VM instances in production resource group',
      riskLevel: 'MEDIUM',
      category: 'FinOps'
    },
    {
      id: '3',
      title: 'Network Security Group Drift',
      kind: 'SECURITY_RISK',
      confidence: 0.78,
      explanation: 'NSG rules likely to be modified allowing broader access than security baseline.',
      eta: '7 days',
      impact: 'Medium - Increased attack surface',
      recommendation: 'Implement NSG flow logs and restrict inbound rules',
      riskLevel: 'MEDIUM',
      category: 'Security'
    }
  ]
}

export default async function PredictionsPage() {
  const predictions = await getPredictions()
  
  // Group predictions by category
  const categories = predictions.reduce((acc: any, p: any) => {
    const cat = p.category || 'Other'
    if (!acc[cat]) acc[cat] = []
    acc[cat].push(p)
    return acc
  }, {})
  
  // Calculate stats
  const stats = {
    total: predictions.length,
    high: predictions.filter((p: any) => p.riskLevel === 'HIGH').length,
    medium: predictions.filter((p: any) => p.riskLevel === 'MEDIUM').length,
    low: predictions.filter((p: any) => p.riskLevel === 'LOW').length,
    avgConfidence: predictions.reduce((sum: number, p: any) => sum + (p.confidence || 0), 0) / predictions.length
  }

  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Brain className="w-8 h-8 text-primary dark:text-blue-400" />
          <h1 className="text-3xl font-bold text-foreground dark:text-white">
            Predictions
          </h1>
        </div>
        <p className="text-muted-foreground dark:text-gray-400">
          AI-powered predictions with 7-day look-ahead and automated remediation
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground dark:text-gray-400">Total Predictions</p>
              <p className="text-2xl font-bold text-foreground dark:text-white">{stats.total}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground dark:text-gray-400">High Risk</p>
              <p className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.high}</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground dark:text-gray-400">Medium Risk</p>
              <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{stats.medium}</p>
            </div>
            <Shield className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
          </div>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground dark:text-gray-400">Low Risk</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.low}</p>
            </div>
            <Shield className="w-8 h-8 text-green-600 dark:text-green-400" />
          </div>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground dark:text-gray-400">Avg Confidence</p>
              <p className="text-2xl font-bold text-foreground dark:text-white">
                {Math.round(stats.avgConfidence * 100)}%
              </p>
            </div>
            <Brain className="w-8 h-8 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
      </div>

      {/* Segmented Controls */}
      <div className="flex gap-2 mb-6">
        <button className="px-4 py-2 bg-primary dark:bg-blue-600 text-primary-foreground dark:text-white rounded-lg">
          All
        </button>
        <button className="px-4 py-2 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700">
          Compliance Drift
        </button>
        <button className="px-4 py-2 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700">
          Security Risk
        </button>
        <button className="px-4 py-2 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700">
          Cost Anomaly
        </button>
      </div>

      {/* Predictions Grid */}
      <div className="space-y-6">
        {Object.entries(categories).map(([category, items]: [string, any]) => (
          <div key={category}>
            <h2 className="text-lg font-semibold text-foreground dark:text-white mb-3">
              {category}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {items.map((p: any) => (
                <PredictionCard key={p.id} p={p} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {predictions.length === 0 && (
        <div className="text-center py-12">
          <Brain className="w-16 h-16 text-muted-foreground dark:text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground dark:text-white mb-2">
            No predictions available
          </h3>
          <p className="text-sm text-muted-foreground dark:text-gray-400">
            The AI engine is analyzing your environment. Check back soon.
          </p>
        </div>
      )}
    </div>
  )
}