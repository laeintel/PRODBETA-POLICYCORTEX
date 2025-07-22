// Staging environment parameters
using '../main.bicep'

// Basic Configuration
param environment = 'staging'
param location = 'East US'
param owner = 'AeoliTech'
param allowedIps = []

// Access Policy Configuration
param createTerraformAccessPolicy = true

// Container Apps deployment
param deployContainerApps = true

// Data Services Configuration
param deploySqlServer = true
param sqlAdminUsername = 'sqladmin'
param sqlAzureadAdminLogin = 'admin@yourdomain.com'
param sqlAzureadAdminObjectId = '00000000-0000-0000-0000-000000000000'
param sqlDatabaseSku = 'GP_S_Gen5_2'
param sqlDatabaseMaxSizeGB = 50
param cosmosConsistencyLevel = 'Session'
param cosmosFailoverLocation = 'West US 2'
param cosmosMaxThroughput = 2000
param redisCapacity = 2
param redisSKUName = 'Standard'

// AI Services Configuration
param deployMLWorkspace = true
param createMLContainerRegistry = false
param trainingClusterVMSize = 'Standard_DS3_v2'
param trainingClusterMaxNodes = 3
param computeInstanceVMSize = 'Standard_DS3_v2'
param cognitiveServicesSku = 'S0'
param deployOpenAI = true
param openAISku = 'S0'

// Monitoring Configuration
param criticalAlertEmails = ['admin@company.com']
param warningAlertEmails = ['devops@company.com']
param budgetAlertEmails = ['finance@company.com']
param monthlyBudgetAmount = 1500

// JWT Secret Key (will be overridden in pipeline)
param jwtSecretKey = 'staging-secret-key-change-in-production'