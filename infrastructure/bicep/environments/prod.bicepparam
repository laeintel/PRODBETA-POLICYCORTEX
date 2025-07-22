// Production environment parameters
using '../main.bicep'

// Basic Configuration
param environment = 'prod'
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
param sqlDatabaseSku = 'GP_S_Gen5_4'  // Larger for production
param sqlDatabaseMaxSizeGB = 100       // Larger for production
param cosmosConsistencyLevel = 'Strong' // Stronger consistency for production
param cosmosFailoverLocation = 'West US 2'
param cosmosMaxThroughput = 8000       // Higher throughput for production
param redisCapacity = 4                // Larger for production
param redisSKUName = 'Premium'         // Premium for production

// AI Services Configuration
param deployMLWorkspace = true
param createMLContainerRegistry = false
param trainingClusterVMSize = 'Standard_DS4_v2'  // Larger for production
param trainingClusterMaxNodes = 10               // More nodes for production
param computeInstanceVMSize = 'Standard_DS4_v2'  // Larger for production
param cognitiveServicesSku = 'S0'
param deployOpenAI = true
param openAISku = 'S0'

// Monitoring Configuration
param criticalAlertEmails = ['admin@company.com', 'oncall@company.com']
param warningAlertEmails = ['devops@company.com', 'monitoring@company.com']
param budgetAlertEmails = ['finance@company.com', 'admin@company.com']
param monthlyBudgetAmount = 5000  // Higher budget for production

// JWT Secret Key (will be overridden in pipeline)
param jwtSecretKey = 'production-secret-key-change-in-production'