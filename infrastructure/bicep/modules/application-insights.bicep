// Application Insights module
param appInsightsName string
param location string
param tags object = {}
param workspaceResourceId string

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Flow_Type: 'Redfield'
    Request_Source: 'IbizaAIExtension'
    WorkspaceResourceId: workspaceResourceId
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    // Set sampling percentage to 10% as requested
    SamplingPercentage: 10
  }
}

output appInsightsId string = applicationInsights.id
output appInsightsName string = applicationInsights.name
output instrumentationKey string = applicationInsights.properties.InstrumentationKey
output connectionString string = applicationInsights.properties.ConnectionString