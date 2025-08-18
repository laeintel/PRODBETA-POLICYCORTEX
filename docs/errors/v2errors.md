azurerm_cosmosdb_account.main: Still creating... [9m10s elapsed]
╷
│ Warning: Argument is deprecated
│ 
│   with azurerm_container_registry.main,
│   on main.tf line 335, in resource "azurerm_container_registry" "main":
│  335:     virtual_network {
│ 
│ The property `virtual_network` is deprecated since this is used exclusively
│ for service endpoints which are being deprecated. Users are expected to use
│ Private Endpoints instead. This property will be removed in v4.0 of the
│ AzureRM Provider.
╵
╷
│ Error: A resource with the ID "/subscriptions/***/resourceGroups/rg-tfstate-cortex-dev" already exists - to be managed via Terraform this resource needs to be imported into the State. Please see the resource documentation for "azurerm_resource_group" for more information.
│ 
│   with azurerm_resource_group.tfstate,
│   on main.tf line 113, in resource "azurerm_resource_group" "tfstate":
│  113: resource "azurerm_resource_group" "tfstate" {
│ 
╵
╷
│ Error: creating Registry (Subscription: "***"
│ Resource Group Name: "rg-cortex-dev"
│ Registry Name: "crcortexdev3p0bata"): performing Create: unexpected status 400 (400 Bad Request) with error: NetworkRuleValidationError: Could not validate network rule - Subnets snet-container-apps of virtual network /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev do not have ServiceEndpoints for Microsoft.ContainerRegistry resources configured. Add Microsoft.ContainerRegistry to subnet's ServiceEndpoints collection before trying to ACL Microsoft.ContainerRegistry resources to these subnets.
│ 
│   with azurerm_container_registry.main,
│   on main.tf line 325, in resource "azurerm_container_registry" "main":
│  325: resource "azurerm_container_registry" "main" {
│ 
╵
╷
│ Error: creating Storage Account (Subscription: "***"
│ Resource Group Name: "rg-cortex-dev"
│ Storage Account Name: "stcortexdev3p0bata"): polling after Create: polling failed: the Azure API returned the following error:
│ 
│ Status: "NetworkAclsValidationFailure"
│ Code: ""
│ Message: "Validation of network acls failure: SubnetsHaveNoServiceEndpointsConfigured:Subnets snet-container-apps of virtual network /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev do not have ServiceEndpoints for Microsoft.Storage resources configured. Add Microsoft.Storage to subnet's ServiceEndpoints collection before trying to ACL Microsoft.Storage resources to these subnets.."
│ Activity Id: ""
│ 
│ ---
│ 
│ API Response:
│ 
│ ----[start]----
│ {"status":"Failed","error":{"code":"NetworkAclsValidationFailure","message":"Validation of network acls failure: SubnetsHaveNoServiceEndpointsConfigured:Subnets snet-container-apps of virtual network /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev do not have ServiceEndpoints for Microsoft.Storage resources configured. Add Microsoft.Storage to subnet's ServiceEndpoints collection before trying to ACL Microsoft.Storage resources to these subnets.."}}
│ -----[end]-----
│ 
│ 
│   with azurerm_storage_account.main,
│   on main.tf line 369, in resource "azurerm_storage_account" "main":
│  369: resource "azurerm_storage_account" "main" {
│ 
╵
╷
│ Error: creating Key Vault (Subscription: "***"
│ Resource Group Name: "rg-cortex-dev"
│ Key Vault Name: "kv-cortex-dev-3p0bata"): performing CreateOrUpdate: vaults.VaultsClient#CreateOrUpdate: Failure sending request: StatusCode=0 -- Original Error: Code="VirtualNetworkNotValid" Message="Operation on Virtual Network could not be performed. StatusCode: 400 (BadRequest). Error Code: SubnetsHaveNoServiceEndpointsConfigured. Error Message: Subnets snet-container-apps of virtual network /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev do not have ServiceEndpoints for Microsoft.KeyVault resources configured. Add Microsoft.KeyVault to subnet's ServiceEndpoints collection before trying to ACL Microsoft.KeyVault resources to these subnets.."
│ 
│   with azurerm_key_vault.main,
│   on main.tf line 409, in resource "azurerm_key_vault" "main":
│  409: resource "azurerm_key_vault" "main" {
│ 
╵
╷
│ Error: creating Database Account (Subscription: "***"
│ Resource Group Name: "rg-cortex-dev"
│ Database Account Name: "cosmos-cortex-dev-3p0bata"): creating/updating CosmosDB Account "cosmos-cortex-dev-3p0bata" (Resource Group "rg-cortex-dev"): polling after DatabaseAccountsCreateOrUpdate: polling failed: the Azure API returned the following error:
│ 
│ Status: "Conflict"
│ Code: ""
│ Message: "Database account creation failed. Operation Id: dad3f68b-38e5-41ff-8420-e44827392ead, Error : Error encountered calling networking RP. StatusCode: BadRequest, ErrorCode: SubnetsHaveNoServiceEndpointsConfigured, ErrorMessage: NRP ARM exception, ErrorCode: SubnetsHaveNoServiceEndpointsConfigured, ErrorMessage: Subnets snet-container-apps of virtual network /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev do not have ServiceEndpoints for Microsoft.AzureCosmosDB resources configured. Add Microsoft.AzureCosmosDB to subnet's ServiceEndpoints collection before trying to ACL Microsoft.AzureCosmosDB resources to these subnets.\r\nActivityId: d43bdb5b-2dc3-476e-8cc6-32288c64d49e, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure
│ Activity Id: ""
│ 
│ ---
│ 
│ API Response:
│ 
│ ----[start]----
│ {"status":"Failed","error":{"code":"Conflict","message":"Database account creation failed. Operation Id: dad3f68b-38e5-41ff-8420-e44827392ead, Error : Error encountered calling networking RP. StatusCode: BadRequest, ErrorCode: SubnetsHaveNoServiceEndpointsConfigured, ErrorMessage: NRP ARM exception, ErrorCode: SubnetsHaveNoServiceEndpointsConfigured, ErrorMessage: Subnets snet-container-apps of virtual network /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev do not have ServiceEndpoints for Microsoft.AzureCosmosDB resources configured. Add Microsoft.AzureCosmosDB to subnet's ServiceEndpoints collection before trying to ACL Microsoft.AzureCosmosDB resources to these subnets.\r\nActivityId: d43bdb5b-2dc3-476e-8cc6-32288c64d49e, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft.Azure.Documents.Common/2.14.0, Microsoft
│ -----[end]-----
│ 
│ 
│   with azurerm_cosmosdb_account.main,
│   on main.tf line 513, in resource "azurerm_cosmosdb_account" "main":
│  513: resource "azurerm_cosmosdb_account" "main" {
│ 
╵
╷
│ Error: creating Account (Subscription: "***"
│ Resource Group Name: "rg-cortex-dev"
│ Account Name: "cogao-cortex-dev"): unexpected status 409 (409 Conflict) with error: FlagMustBeSetForRestore: An existing resource with ID '/subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.CognitiveServices/accounts/cogao-cortex-dev' has been soft-deleted. To restore the resource, you must specify 'restore' to be 'true' in the property. If you don't want to restore existing resource, please purge it first.
│ 
│   with azurerm_cognitive_account.openai,
│   on main.tf line 567, in resource "azurerm_cognitive_account" "openai":
│  567: resource "azurerm_cognitive_account" "openai" {
│ 
╵
╷
│ Error: creating Managed Environment (Subscription: "***"
│ Resource Group Name: "rg-cortex-dev"
│ Managed Environment Name: "cae-cortex-dev"): polling after CreateOrUpdate: polling failed: the Azure API returned the following error:
│ 
│ Status: "Failed"
│ Code: "ManagedEnvironmentSubnetIsDelegated"
│ Message: "AgentPoolProfile subnet with id /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps cannot be used as it's a delegated subnet. Please check https://aka.ms/adv-network-prerequest for more details.\r\nStatus: 400 (Bad Request)\r\nErrorCode: SubnetIsDelegated\r\n\r\nContent:\r\n{\n  \"code\": \"SubnetIsDelegated\",\n  \"details\": null,\n  \"message\": \"AgentPoolProfile subnet with id /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps cannot be used as it's a delegated subnet. Please check https://aka.ms/adv-network-prerequest for more details.\",\n  \"subcode\": \"\"\n }\r\n\r\nHeaders:\r\nCache-Control: no-cache\r\nPragma: no-cache\r\nx-ms-operation-identifier: REDACTED\r\nx-ms-correlation-request-id: REDACTED\r\nx-ms-request-id: 88a302d1-c6b1-4f9c-b7bd-9a42244af605\r\nStrict-Transport-Security: REDACTED\r\nx-ms-thrott
│ Activity Id: ""
│ 
│ ---
│ 
│ API Response:
│ 
│ ----[start]----
│ {"id":"/subscriptions/***/providers/Microsoft.App/locations/eastus/managedEnvironmentOperationStatuses/9362d7cf-cca0-4d62-8b86-167ba6e20a7f","name":"9362d7cf-cca0-4d62-8b86-167ba6e20a7f","status":"Failed","error":{"code":"ManagedEnvironmentSubnetIsDelegated","message":"AgentPoolProfile subnet with id /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps cannot be used as it's a delegated subnet. Please check https://aka.ms/adv-network-prerequest for more details.\r\nStatus: 400 (Bad Request)\r\nErrorCode: SubnetIsDelegated\r\n\r\nContent:\r\n{\n  \"code\": \"SubnetIsDelegated\",\n  \"details\": null,\n  \"message\": \"AgentPoolProfile subnet with id /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps cannot be used as it's a delegated subnet. Please check https://aka.ms/adv-network-prerequest for more details.\",\n  \"subcode\": \"
│ -----[end]-----
│ 
│ 
│   with azurerm_container_app_environment.main,
│   on main.tf line 651, in resource "azurerm_container_app_environment" "main":
│  651: resource "azurerm_container_app_environment" "main" {
│ 
│ creating Managed Environment (Subscription:
│ "***"
│ Resource Group Name: "rg-cortex-dev"
│ Managed Environment Name: "cae-cortex-dev"): polling after CreateOrUpdate:
│ polling failed: the Azure API returned the following error:
│ 
│ Status: "Failed"
│ Code: "ManagedEnvironmentSubnetIsDelegated"
│ Message: "AgentPoolProfile subnet with id
│ /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps
│ cannot be used as it's a delegated subnet. Please check
│ https://aka.ms/adv-network-prerequest for more details.\r\nStatus: 400 (Bad
│ Request)\r\nErrorCode: SubnetIsDelegated\r\n\r\nContent:\r\n{\n  \"code\":
│ \"SubnetIsDelegated\",\n  \"details\": null,\n  \"message\":
│ \"AgentPoolProfile subnet with id
│ /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps
│ cannot be used as it's a delegated subnet. Please check
│ https://aka.ms/adv-network-prerequest for more details.\",\n  \"subcode\":
│ \"\"\n }\r\n\r\nHeaders:\r\nCache-Control: no-cache\r\nPragma:
│ no-cache\r\nx-ms-operation-identifier:
│ REDACTED\r\nx-ms-correlation-request-id: REDACTED\r\nx-ms-request-id:
│ 88a302d1-c6b1-4f9c-b7bd-9a42244af605\r\nStrict-Transport-Security:
│ REDACTED\r\nx-ms-throttling-version:
│ REDACTED\r\nx-ms-ratelimit-remaining-subscription-writes:
│ REDACTED\r\nx-ms-routing-request-id: REDACTED\r\nX-Content-Type-Options:
│ REDACTED\r\nX-Cache: REDACTED\r\nX-MSEdge-Ref: REDACTED\r\nDate: Sun, 17
│ Aug 2025 16:33:35 GMT\r\nContent-Length: 399\r\nContent-Type:
│ application/json\r\nExpires: -1\r\n"
│ Activity Id: ""
│ 
│ ---
│ 
│ API Response:
│ 
│ ----[start]----
│ {"id":"/subscriptions/***/providers/Microsoft.App/locations/eastus/managedEnvironmentOperationStatuses/9362d7cf-cca0-4d62-8b86-167ba6e20a7f","name":"9362d7cf-cca0-4d62-8b86-167ba6e20a7f","status":"Failed","error":{"code":"ManagedEnvironmentSubnetIsDelegated","message":"AgentPoolProfile
│ subnet with id
│ /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps
│ cannot be used as it's a delegated subnet. Please check
│ https://aka.ms/adv-network-prerequest for more details.\r\nStatus: 400 (Bad
│ Request)\r\nErrorCode: SubnetIsDelegated\r\n\r\nContent:\r\n{\n  \"code\":
│ \"SubnetIsDelegated\",\n  \"details\": null,\n  \"message\":
│ \"AgentPoolProfile subnet with id
│ /subscriptions/***/resourceGroups/rg-cortex-dev/providers/Microsoft.Network/virtualNetworks/vnet-cortex-dev/subnets/snet-container-apps
│ cannot be used as it's a delegated subnet. Please check
│ https://aka.ms/adv-network-prerequest for more details.\",\n  \"subcode\":
│ \"\"\n }\r\n\r\nHeaders:\r\nCache-Control: no-cache\r\nPragma:
│ no-cache\r\nx-ms-operation-identifier:
│ REDACTED\r\nx-ms-correlation-request-id: REDACTED\r\nx-ms-request-id:
│ 88a302d1-c6b1-4f9c-b7bd-9a42244af605\r\nStrict-Transport-Security:
│ REDACTED\r\nx-ms-throttling-version:
│ REDACTED\r\nx-ms-ratelimit-remaining-subscription-writes:
│ REDACTED\r\nx-ms-routing-request-id: REDACTED\r\nX-Content-Type-Options:
│ REDACTED\r\nX-Cache: REDACTED\r\nX-MSEdge-Ref: REDACTED\r\nDate: Sun, 17
│ Aug 2025 16:33:35 GMT\r\nContent-Length: 399\r\nContent-Type:
│ application/json\r\nExpires:
│ -1\r\n"},"startTime":"2025-08-17T16:32:10.0578692"}
│ -----[end]-----
│ 
╵
Error: Terraform exited with code 1.
Terraform apply completed with some warnings