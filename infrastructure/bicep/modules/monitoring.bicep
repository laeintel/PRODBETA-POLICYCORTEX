// Monitoring module - Minimal for initial deployment
param environment string
param tags object = {}
param criticalAlertEmails array = []
param warningAlertEmails array = []
param budgetAlertEmails array = []
param monthlyBudgetAmount int = 1000
param resourceGroupName string
@description('Budget start date in format yyyy-MM-dd')
param budgetStartDate string = '${utcNow('yyyy-MM')}-01'

// Action Groups - Only create if emails are provided
resource criticalActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = if (length(criticalAlertEmails) > 0) {
  name: 'ag-policortex001-critical-${environment}'
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
  name: 'ag-policortex001-warning-${environment}'
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

// Budget Alert - Commented out due to RBAC permissions required
// Budgets require Cost Management Contributor role at subscription level
// To enable budgets, ensure the service principal has the necessary permissions
// resource budget 'Microsoft.Consumption/budgets@2021-10-01' = if (length(budgetAlertEmails) > 0) {
//   name: 'budget-policortex001-${environment}'
//   properties: {
//     ...
//   }
// }

// Outputs
output criticalActionGroupId string = length(criticalAlertEmails) > 0 ? criticalActionGroup.id : ''
output warningActionGroupId string = length(warningAlertEmails) > 0 ? warningActionGroup.id : ''
output dashboardId string = ''  // Dashboard removed for initial deployment
output budgetId string = ''  // Budget removed due to RBAC permissions