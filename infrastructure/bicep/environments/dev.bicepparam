// Development environment parameters
using '../main.bicep'

// Basic Configuration
param environment = 'dev'
param location = 'East US'
param owner = 'AeoliTech'
param allowedIps = []

// Set to false if Key Vault access policy already exists
param createTerraformAccessPolicy = false

// Container Apps deployment (ENABLED for simple infrastructure)
param deployContainerApps = true

// Data Services Configuration
param deploySqlServer = false  // SQL Server provisioning restricted in East US
param sqlAdminUsername = 'sqladmin'
param sqlAzureadAdminLogin = 'admin@yourdomain.com'
param sqlAzureadAdminObjectId = '00000000-0000-0000-0000-000000000000'
param sqlDatabaseSku = 'GP_S_Gen5_1'  // Smaller for dev
param sqlDatabaseMaxSizeGB = 10      // Smaller for dev
param cosmosConsistencyLevel = 'Session'
param cosmosFailoverLocation = 'West US 2'
param cosmosMaxThroughput = 1000       // Smaller for dev
param redisCapacity = 1                 // Smaller for dev
param redisSKUName = 'Basic'           // Basic for dev

// AI Services Configuration
param deployMLWorkspace = false  // ML workspace has soft-delete conflict
param createMLContainerRegistry = false  // Use main ACR
param trainingClusterVMSize = 'Standard_DS2_v2'  // Smaller for dev
param trainingClusterMaxNodes = 2             // Fewer nodes for dev
param computeInstanceVMSize = 'Standard_DS2_v2' // Smaller for dev
param cognitiveServicesSku = 'S0'              // Standard tier (F0 not supported for CognitiveServices)
param deployOpenAI = false                      // Skip OpenAI for dev
param openAISku = 'S0'

// Monitoring Configuration
param criticalAlertEmails = ['admin@company.com']
param warningAlertEmails = ['devops@company.com']
param budgetAlertEmails = ['finance@company.com']
param monthlyBudgetAmount = 500  // Lower budget for dev

// JWT Secret Key (will be overridden in pipeline)
param jwtSecretKey = 'development-secret-key-change-in-production'