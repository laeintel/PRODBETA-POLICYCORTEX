// Data Services module
param environment string
param location string
param tags object = {}
param keyVaultName string
param subnetId string
param privateEndpointsSubnetId string
param privateDnsZones object
param deploySqlServer bool = true
param sqlAdminUsername string = 'sqladmin'
param sqlAzureadAdminLogin string = 'admin@yourdomain.com'
param sqlAzureadAdminObjectId string = '00000000-0000-0000-0000-000000000000'
param sqlDatabaseSku string = 'GP_S_Gen5_2'
param sqlDatabaseMaxSizeGB int = 32
param cosmosConsistencyLevel string = 'Session'
param cosmosFailoverLocation string = 'West US 2'
param cosmosMaxThroughput int = 4000
param redisCapacity int = 2
param redisSKUName string = 'Standard'


// Cosmos DB Account
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: 'policortex001-cosmos-${environment}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: {
      defaultConsistencyLevel: cosmosConsistencyLevel
      maxIntervalInSeconds: 300
      maxStalenessPrefix: 100000
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
      {
        locationName: cosmosFailoverLocation
        failoverPriority: 1
        isZoneRedundant: false
      }
    ]
    databaseAccountOfferType: 'Standard'
    enableAutomaticFailover: true
    enableMultipleWriteLocations: false
    isVirtualNetworkFilterEnabled: true
    virtualNetworkRules: [
      {
        id: subnetId
        ignoreMissingVNetServiceEndpoint: false
      }
    ]
    disableKeyBasedMetadataWriteAccess: false
    enableFreeTier: environment == 'dev'
    enableAnalyticalStorage: false
    analyticalStorageConfiguration: {
      schemaType: 'WellDefined'
    }
    createMode: 'Default'
    backupPolicy: {
      type: 'Periodic'
      periodicModeProperties: {
        backupIntervalInMinutes: 240
        backupRetentionIntervalInHours: 8
        backupStorageRedundancy: 'Geo'
      }
    }
    networkAclBypass: 'AzureServices'
    publicNetworkAccess: 'Enabled'
  }
}

// Cosmos DB Database
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosAccount
  name: 'policortexDB'
  properties: {
    resource: {
      id: 'policortexDB'
    }
    options: {
      autoscaleSettings: {
        maxThroughput: cosmosMaxThroughput
      }
    }
  }
}

// Cosmos DB Containers
var containerConfigs = [
  { name: 'policies', partitionKey: '/policyId' }
  { name: 'compliance', partitionKey: '/complianceId' }
  { name: 'audit', partitionKey: '/auditId' }
  { name: 'users', partitionKey: '/userId' }
  { name: 'sessions', partitionKey: '/sessionId' }
]

resource cosmosContainers 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = [for config in containerConfigs: {
  parent: cosmosDatabase
  name: config.name
  properties: {
    resource: {
      id: config.name
      partitionKey: {
        paths: [config.partitionKey]
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/"_etag"/?'
          }
        ]
      }
      defaultTtl: 86400
    }
    options: {}
  }
}]

// Redis Cache
resource redisCache 'Microsoft.Cache/redis@2023-08-01' = {
  name: 'policortex001-redis-${environment}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: redisSKUName
      family: redisSKUName == 'Premium' ? 'P' : 'C'
      capacity: redisCapacity
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    redisConfiguration: redisSKUName == 'Premium' ? {
      'aof-backup-enabled': 'false'
      'maxfragmentationmemory-reserved': '30'
      'maxmemory-delta': '30'
      'maxmemory-reserved': '30'
      'notify-keyspace-events': ''
      'rdb-backup-enabled': 'false'
    } : {
      'maxfragmentationmemory-reserved': '30'
      'maxmemory-delta': '30'
      'maxmemory-reserved': '30'
      'notify-keyspace-events': ''
    }
    redisVersion: '6'
  }
}

// SQL Server (conditional)
resource sqlServer 'Microsoft.Sql/servers@2023-05-01-preview' = if (deploySqlServer) {
  name: 'policortex001-sql-${environment}'
  location: location
  tags: tags
  properties: {
    administratorLogin: sqlAdminUsername
    administratorLoginPassword: '${uniqueString(resourceGroup().id, environment, 'sql')}!${toUpper(uniqueString(subscription().id, environment))}'
    version: '12.0'
    minimalTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    administrators: {
      administratorType: 'ActiveDirectory'
      principalType: 'User'
      login: sqlAzureadAdminLogin
      sid: sqlAzureadAdminObjectId
      tenantId: tenant().tenantId
      azureADOnlyAuthentication: false
    }
  }
}

// SQL Database (conditional)
resource sqlDatabase 'Microsoft.Sql/servers/databases@2023-05-01-preview' = if (deploySqlServer) {
  parent: sqlServer
  name: 'policortexDB'
  location: location
  tags: tags
  sku: {
    name: sqlDatabaseSku
    tier: startsWith(sqlDatabaseSku, 'GP_S') ? 'GeneralPurpose' : 'GeneralPurpose'
  }
  properties: {
    maxSizeBytes: sqlDatabaseMaxSizeGB * 1024 * 1024 * 1024
    collation: 'SQL_Latin1_General_CP1_CI_AS'
    catalogCollation: 'SQL_Latin1_General_CP1_CI_AS'
    zoneRedundant: false
    readScale: 'Disabled'
    requestedBackupStorageRedundancy: 'Geo'
    autoPauseDelay: environment == 'dev' ? 60 : -1
    minCapacity: startsWith(sqlDatabaseSku, 'GP_S') ? json('0.5') : json('1')
  }
}

// SQL Firewall Rules (conditional)
resource sqlFirewallAzure 'Microsoft.Sql/servers/firewallRules@2023-05-01-preview' = if (deploySqlServer) {
  parent: sqlServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Private Endpoints
resource cosmosPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'policortex001-cosmos-pe-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'cosmos-connection'
        properties: {
          privateLinkServiceId: cosmosAccount.id
          groupIds: ['Sql']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'Cosmos DB private endpoint'
          }
        }
      }
    ]
  }
}

resource redisPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'policortex001-redis-pe-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'redis-connection'
        properties: {
          privateLinkServiceId: redisCache.id
          groupIds: ['redisCache']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'Redis Cache private endpoint'
          }
        }
      }
    ]
  }
}

resource sqlPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = if (deploySqlServer) {
  name: 'policortex001-sql-pe-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'sql-connection'
        properties: {
          privateLinkServiceId: sqlServer.id
          groupIds: ['sqlServer']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'SQL Server private endpoint'
          }
        }
      }
    ]
  }
}

// Private DNS Zone Groups
resource cosmosDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = {
  parent: cosmosPrivateEndpoint
  name: 'cosmos-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'cosmos-config'
        properties: {
          privateDnsZoneId: privateDnsZones.cosmos
        }
      }
    ]
  }
}

resource redisDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = {
  parent: redisPrivateEndpoint
  name: 'redis-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'redis-config'
        properties: {
          privateDnsZoneId: privateDnsZones.redis
        }
      }
    ]
  }
}

resource sqlDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = if (deploySqlServer) {
  parent: sqlPrivateEndpoint
  name: 'sql-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'sql-config'
        properties: {
          privateDnsZoneId: privateDnsZones.sql
        }
      }
    ]
  }
}

// Key Vault Secrets for data services
resource sqlPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (deploySqlServer) {
  name: '${keyVaultName}/sql-admin-password'
  properties: {
    value: uniqueString(resourceGroup().id, environment, 'sql')
    contentType: 'text/plain'
  }
}

resource redisConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/redis-connection-string'
  properties: {
    value: '${redisCache.properties.hostName}:${redisCache.properties.sslPort},password=${redisCache.listKeys().primaryKey},ssl=True,abortConnect=False'
    contentType: 'text/plain'
  }
}

resource cosmosConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cosmos-connection-string'
  properties: {
    value: cosmosAccount.listConnectionStrings().connectionStrings[0].connectionString
    contentType: 'text/plain'
  }
}

// Outputs
output cosmosAccountId string = cosmosAccount.id
output cosmosAccountName string = cosmosAccount.name
output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output redisCacheId string = redisCache.id
output redisCacheName string = redisCache.name
output redisHostName string = redisCache.properties.hostName
output sqlServerId string = deploySqlServer ? sqlServer.id : ''
output sqlServerName string = deploySqlServer ? sqlServer.name : ''
output sqlDatabaseId string = deploySqlServer ? sqlDatabase.id : ''