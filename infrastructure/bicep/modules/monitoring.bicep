// Monitoring module
param environment string
param location string
param tags object = {}
param workspaceId string
param criticalAlertEmails array = []
param warningAlertEmails array = []
param budgetAlertEmails array = []
param monthlyBudgetAmount int = 1000

// Action Groups
resource criticalActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = if (length(criticalAlertEmails) > 0) {
  name: 'ag-policycortex-critical-${environment}'
  location: 'Global'
  tags: tags
  properties: {
    groupShortName: 'Critical'
    enabled: true
    emailReceivers: [for email in criticalAlertEmails: {
      name: replace(email, '@', '-at-')
      emailAddress: email
      useCommonAlertSchema: true
    }]
  }
}

resource warningActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = if (length(warningAlertEmails) > 0) {
  name: 'ag-policycortex-warning-${environment}'
  location: 'Global'
  tags: tags
  properties: {
    groupShortName: 'Warning'
    enabled: true
    emailReceivers: [for email in warningAlertEmails: {
      name: replace(email, '@', '-at-')
      emailAddress: email
      useCommonAlertSchema: true
    }]
  }
}

// Metric Alerts
resource highCpuAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = if (length(criticalAlertEmails) > 0) {
  name: 'alert-high-cpu-${environment}'
  location: 'Global'
  tags: tags
  properties: {
    description: 'High CPU usage alert for Container Apps'
    severity: 1
    enabled: true
    scopes: [
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}'
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'High CPU'
          metricName: 'CpuPercentage'
          operator: 'GreaterThan'
          threshold: 80
          timeAggregation: 'Average'
          metricNamespace: 'Microsoft.App/containerApps'
        }
      ]
    }
    actions: [
      {
        actionGroupId: criticalActionGroup.id
      }
    ]
  }
}

resource highMemoryAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = if (length(criticalAlertEmails) > 0) {
  name: 'alert-high-memory-${environment}'
  location: 'Global'
  tags: tags
  properties: {
    description: 'High memory usage alert for Container Apps'
    severity: 1
    enabled: true
    scopes: [
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}'
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'High Memory'
          metricName: 'MemoryPercentage'
          operator: 'GreaterThan'
          threshold: 85
          timeAggregation: 'Average'
          metricNamespace: 'Microsoft.App/containerApps'
        }
      ]
    }
    actions: [
      {
        actionGroupId: criticalActionGroup.id
      }
    ]
  }
}

// Scheduled Query Rules
resource authenticationFailuresAlert 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = if (length(warningAlertEmails) > 0) {
  name: 'alert-auth-failures-${environment}'
  location: location
  tags: tags
  properties: {
    displayName: 'Authentication Failures Alert'
    description: 'Alert when authentication failures exceed threshold'
    severity: 2
    enabled: true
    evaluationFrequency: 'PT5M'
    scopes: [workspaceId]
    windowSize: 'PT15M'
    criteria: {
      allOf: [
        {
          query: '''
            AppTraces
            | where TimeGenerated > ago(15m)
            | where Message contains "authentication" and Message contains "failed"
            | summarize FailureCount = count() by bin(TimeGenerated, 5m)
            | where FailureCount > 20
          '''
          threshold: 0
          timeAggregation: 'Maximum'
          failingPeriods: {
            numberOfEvaluationPeriods: 3
            minFailingPeriodsToTriggerAlert: 1
          }
        }
      ]
    }
    actions: {
      actionGroups: [
        warningActionGroup.id
      ]
    }
  }
}

// Budget (if emails provided)
resource budget 'Microsoft.Consumption/budgets@2023-05-01' = if (length(budgetAlertEmails) > 0) {
  name: 'budget-policycortex-${environment}'
  scope: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}'
  properties: {
    timePeriod: {
      startDate: '2024-01-01T00:00:00Z'
    }
    timeGrain: 'Monthly'
    amount: monthlyBudgetAmount
    category: 'Cost'
    notifications: {
      actual_GreaterThan_80_Percent: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        contactEmails: budgetAlertEmails
        contactGroups: []
        contactRoles: []
        thresholdType: 'Actual'
      }
      forecasted_GreaterThan_100_Percent: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        contactEmails: budgetAlertEmails
        contactGroups: []
        contactRoles: []
        thresholdType: 'Forecasted'
      }
    }
  }
}

// Portal Dashboard
resource portalDashboard 'Microsoft.Portal/dashboards@2020-09-01-preview' = {
  name: guid('dashboard-policycortex-${environment}')
  location: location
  tags: union(tags, {
    'hidden-title': 'PolicyCortex ${environment} Dashboard'
  })
  properties: {
    lenses: [
      {
        order: 0
        parts: [
          {
            position: {
              x: 0
              y: 0
              rowSpan: 4
              colSpan: 6
            }
            metadata: {
              inputs: [
                {
                  name: 'resourceIds'
                  value: [workspaceId]
                }
                {
                  name: 'query'
                  value: 'AppTraces | summarize count() by bin(TimeGenerated, 1h) | render timechart'
                }
              ]
              type: 'Extension/Microsoft_OperationsManagementSuite_Workspace/PartType/LogsDashboardPart'
            }
          }
        ]
      }
    ]
    metadata: {
      model: {
        timeRange: {
          value: {
            relative: {
              duration: 24
              timeUnit: 1
            }
          }
          type: 'MsPortalFx.Composition.Configuration.ValueTypes.TimeRange'
        }
        filterLocale: {
          value: 'en-us'
        }
        filters: {
          value: {
            MsPortalFx_TimeRange: {
              model: {
                format: 'utc'
                granularity: 'auto'
                relative: '24h'
              }
              displayCache: {
                name: 'UTC Time'
                value: 'Past 24 hours'
              }
              filteredPartIds: []
            }
          }
        }
      }
    }
  }
}

// Outputs
output criticalActionGroupId string = length(criticalAlertEmails) > 0 ? criticalActionGroup.id : ''
output warningActionGroupId string = length(warningAlertEmails) > 0 ? warningActionGroup.id : ''
output dashboardId string = portalDashboard.id