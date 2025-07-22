// Monitoring module - Minimal for initial deployment
param environment string
param location string
param tags object = {}
param workspaceId string
param criticalAlertEmails array = []
param warningAlertEmails array = []
param budgetAlertEmails array = []
param monthlyBudgetAmount int = 1000

// Action Groups - Only create if emails are provided
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

// Outputs
output criticalActionGroupId string = length(criticalAlertEmails) > 0 ? criticalActionGroup.id : ''
output warningActionGroupId string = length(warningAlertEmails) > 0 ? warningActionGroup.id : ''
output dashboardId string = ''  // Dashboard removed for initial deployment