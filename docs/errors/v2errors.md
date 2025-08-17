Run terraform init -input=false \

Initializing the backend...

Successfully configured the backend "azurerm"! Terraform will automatically
use this backend unless the backend configuration changes.
Terraform encountered problems during initialisation, including problems
with the configuration, described below.

The Terraform configuration must be valid before initialization so that
Terraform can determine which modules and providers need to be installed.
╷
│ Error: Duplicate required providers configuration
│ 
│   on main-complete.tf line 6, in terraform:
│    6:   required_providers {
│ 
│ A module may have only one required providers configuration. The required
│ providers were previously configured at main-clean.tf:6,3-21.
╵

╷
│ Error: Duplicate provider configuration
│ 
│   on main-complete.tf line 18:
│   18: provider "azurerm" {
│ 
│ A default (non-aliased) provider configuration for "azurerm" was already
│ given at main-clean.tf:22,1-19. If multiple configurations are required,
│ set the "alias" argument for alternative configurations.
╵

╷
│ Error: Duplicate data "azurerm_client_config" configuration
│ 
│   on main-complete.tf line 34:
│   34: data "azurerm_client_config" "current" {}
│ 
│ A azurerm_client_config data resource named "current" was already declared
│ at main-clean.tf:36,1-39. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_resource_group" configuration
│ 
│   on main-complete.tf line 140:
│  140: resource "azurerm_resource_group" "main" {
│ 
│ A azurerm_resource_group resource named "main" was already declared at
│ main-clean.tf:39,1-41. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_virtual_network" configuration
│ 
│   on main-complete.tf line 150:
│  150: resource "azurerm_virtual_network" "main" {
│ 
│ A azurerm_virtual_network resource named "main" was already declared at
│ main-clean.tf:108,1-42. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_subnet" configuration
│ 
│   on main-complete.tf line 159:
│  159: resource "azurerm_subnet" "container_apps" {
│ 
│ A azurerm_subnet resource named "container_apps" was already declared at
│ main-clean.tf:117,1-43. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_log_analytics_workspace" configuration
│ 
│   on main-complete.tf line 284:
│  284: resource "azurerm_log_analytics_workspace" "main" {
│ 
│ A azurerm_log_analytics_workspace resource named "main" was already
│ declared at main-clean.tf:88,1-50. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_application_insights" configuration
│ 
│   on main-complete.tf line 293:
│  293: resource "azurerm_application_insights" "main" {
│ 
│ A azurerm_application_insights resource named "main" was already declared
│ at main-clean.tf:98,1-47. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_registry" configuration
│ 
│   on main-complete.tf line 322:
│  322: resource "azurerm_container_registry" "main" {
│ 
│ A azurerm_container_registry resource named "main" was already declared at
│ main-clean.tf:46,1-45. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_storage_account" configuration
│ 
│   on main-complete.tf line 366:
│  366: resource "azurerm_storage_account" "main" {
│ 
│ A azurerm_storage_account resource named "main" was already declared at
│ main-clean.tf:56,1-42. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_key_vault" configuration
│ 
│   on main-complete.tf line 406:
│  406: resource "azurerm_key_vault" "main" {
│ 
│ A azurerm_key_vault resource named "main" was already declared at
│ main-clean.tf:66,1-36. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "random_password" configuration
│ 
│   on main-complete.tf line 462:
│  462: resource "random_password" "db_password" {
│ 
│ A random_password resource named "db_password" was already declared at
│ main-clean.tf:353,1-41. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_postgresql_flexible_server" configuration
│ 
│   on main-complete.tf line 467:
│  467: resource "azurerm_postgresql_flexible_server" "main" {
│ 
│ A azurerm_postgresql_flexible_server resource named "main" was already
│ declared at main-clean.tf:150,1-53. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_postgresql_flexible_server_database" configuration
│ 
│   on main-complete.tf line 489:
│  489: resource "azurerm_postgresql_flexible_server_database" "main" {
│ 
│ A azurerm_postgresql_flexible_server_database resource named "main" was
│ already declared at main-clean.tf:164,1-62. Resource names must be unique
│ per type in each module.
╵

╷
│ Error: Duplicate resource "azurerm_key_vault_secret" configuration
│ 
│   on main-complete.tf line 497:
│  497: resource "azurerm_key_vault_secret" "db_password" {
│ 
│ A azurerm_key_vault_secret resource named "db_password" was already
│ declared at main-clean.tf:359,1-50. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_cosmosdb_account" configuration
│ 
│   on main-complete.tf line 507:
│  507: resource "azurerm_cosmosdb_account" "main" {
│ 
│ A azurerm_cosmosdb_account resource named "main" was already declared at
│ main-clean.tf:180,1-43. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app_environment" configuration
│ 
│   on main-complete.tf line 645:
│  645: resource "azurerm_container_app_environment" "main" {
│ 
│ A azurerm_container_app_environment resource named "main" was already
│ declared at main-clean.tf:125,1-52. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_user_assigned_identity" configuration
│ 
│   on main-complete.tf line 659:
│  659: resource "azurerm_user_assigned_identity" "container_apps" {
│ 
│ A azurerm_user_assigned_identity resource named "container_apps" was
│ already declared at main-clean.tf:135,1-59. Resource names must be unique
│ per type in each module.
╵

╷
│ Error: Duplicate resource "azurerm_role_assignment" configuration
│ 
│   on main-complete.tf line 667:
│  667: resource "azurerm_role_assignment" "acr_pull" {
│ 
│ A azurerm_role_assignment resource named "acr_pull" was already declared at
│ main-clean.tf:143,1-46. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app" configuration
│ 
│   on main-complete.tf line 683:
│  683: resource "azurerm_container_app" "core" {
│ 
│ A azurerm_container_app resource named "core" was already declared at
│ main-clean.tf:205,1-40. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app" configuration
│ 
│   on main-complete.tf line 737:
│  737: resource "azurerm_container_app" "frontend" {
│ 
│ A azurerm_container_app resource named "frontend" was already declared at
│ main-clean.tf:252,1-44. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app" configuration
│ 
│   on main-complete.tf line 791:
│  791: resource "azurerm_container_app" "graphql" {
│ 
│ A azurerm_container_app resource named "graphql" was already declared at
│ main-clean.tf:305,1-43. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 844:
│  844: output "resource_group_name" {
│ 
│ An output named "resource_group_name" was already defined at
│ main-clean.tf:366,1-29. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 856:
│  856: output "container_registry_name" {
│ 
│ An output named "container_registry_name" was already defined at
│ main-clean.tf:370,1-33. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 860:
│  860: output "container_registry_url" {
│ 
│ An output named "container_registry_url" was already defined at
│ main-clean.tf:374,1-32. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 864:
│  864: output "container_apps_environment_id" {
│ 
│ An output named "container_apps_environment_id" was already defined at
│ main-clean.tf:378,1-39. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 868:
│  868: output "core_app_url" {
│ 
│ An output named "core_app_url" was already defined at
│ main-clean.tf:382,1-22. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 872:
│  872: output "frontend_app_url" {
│ 
│ An output named "frontend_app_url" was already defined at
│ main-clean.tf:386,1-26. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 876:
│  876: output "graphql_app_url" {
│ 
│ An output named "graphql_app_url" was already defined at
│ main-clean.tf:390,1-25. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 880:
│  880: output "key_vault_name" {
│ 
│ An output named "key_vault_name" was already defined at
│ main-clean.tf:394,1-24. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 884:
│  884: output "postgresql_server_name" {
│ 
│ An output named "postgresql_server_name" was already defined at
│ main-clean.tf:398,1-32. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 888:
│  888: output "cosmos_db_endpoint" {
│ 
│ An output named "cosmos_db_endpoint" was already defined at
│ main-clean.tf:402,1-28. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main-complete.tf line 901:
│  901: output "backend_config" {
│ 
│ An output named "backend_config" was already defined at
│ backend-config.tf:23,1-24. Output names must be unique within a module.
╵

╷
│ Error: Duplicate required providers configuration
│ 
│   on main.tf line 4, in terraform:
│    4:   required_providers {
│ 
│ A module may have only one required providers configuration. The required
│ providers were previously configured at main-clean.tf:6,3-21.
╵

╷
│ Error: Duplicate backend configuration
│ 
│   on main.tf line 15, in terraform:
│   15:   backend "azurerm" {
│ 
│ A module may have only one backend configuration. The backend was
│ previously configured at main-clean.tf:17,3-20.
╵

╷
│ Error: Duplicate provider configuration
│ 
│   on main.tf line 21:
│   21: provider "azurerm" {
│ 
│ A default (non-aliased) provider configuration for "azurerm" was already
│ given at main-clean.tf:22,1-19. If multiple configurations are required,
│ set the "alias" argument for alternative configurations.
╵

╷
│ Error: Duplicate data "azurerm_client_config" configuration
│ 
│   on main.tf line 35:
│   35: data "azurerm_client_config" "current" {}
│ 
│ A azurerm_client_config data resource named "current" was already declared
│ at main-clean.tf:36,1-39. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate variable declaration
│ 
│   on main.tf line 44:
│   44: variable "environment" {
│ 
│ A variable named "environment" was already declared at
│ main-complete.tf:38,1-23. Variable names must be unique within a module.
╵

╷
│ Error: Duplicate variable declaration
│ 
│   on main.tf line 50:
│   50: variable "location" {
│ 
│ A variable named "location" was already declared at
│ main-complete.tf:44,1-20. Variable names must be unique within a module.
╵

╷
│ Error: Duplicate local value definition
│ 
│   on main.tf line 68, in locals:
│   68:   common_tags = merge(var.tags, {
│   69:     Environment = var.environment
│   70:     CreatedDate = timestamp()
│   71:   })
│ 
│ A local value named "common_tags" was already defined at
│ main-complete.tf:98,3-103,4. Local value names must be unique within a
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_resource_group" configuration
│ 
│   on main.tf line 79:
│   79: resource "azurerm_resource_group" "main" {
│ 
│ A azurerm_resource_group resource named "main" was already declared at
│ main-clean.tf:39,1-41. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_storage_account" configuration
│ 
│   on main.tf line 92:
│   92: resource "azurerm_storage_account" "main" {
│ 
│ A azurerm_storage_account resource named "main" was already declared at
│ main-clean.tf:56,1-42. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_cognitive_account" configuration
│ 
│   on main.tf line 213:
│  213: resource "azurerm_cognitive_account" "openai" {
│ 
│ A azurerm_cognitive_account resource named "openai" was already declared at
│ main-complete.tf:561,1-46. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_postgresql_flexible_server" configuration
│ 
│   on main.tf line 264:
│  264: resource "azurerm_postgresql_flexible_server" "main" {
│ 
│ A azurerm_postgresql_flexible_server resource named "main" was already
│ declared at main-clean.tf:150,1-53. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_postgresql_flexible_server_database" configuration
│ 
│   on main.tf line 288:
│  288: resource "azurerm_postgresql_flexible_server_database" "main" {
│ 
│ A azurerm_postgresql_flexible_server_database resource named "main" was
│ already declared at main-clean.tf:164,1-62. Resource names must be unique
│ per type in each module.
╵

╷
│ Error: Duplicate resource "azurerm_cosmosdb_account" configuration
│ 
│   on main.tf line 305:
│  305: resource "azurerm_cosmosdb_account" "main" {
│ 
│ A azurerm_cosmosdb_account resource named "main" was already declared at
│ main-clean.tf:180,1-43. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_virtual_network" configuration
│ 
│   on main.tf line 344:
│  344: resource "azurerm_virtual_network" "main" {
│ 
│ A azurerm_virtual_network resource named "main" was already declared at
│ main-clean.tf:108,1-42. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_registry" configuration
│ 
│   on main.tf line 362:
│  362: resource "azurerm_container_registry" "main" {
│ 
│ A azurerm_container_registry resource named "main" was already declared at
│ main-clean.tf:46,1-45. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_key_vault" configuration
│ 
│   on main.tf line 373:
│  373: resource "azurerm_key_vault" "main" {
│ 
│ A azurerm_key_vault resource named "main" was already declared at
│ main-clean.tf:66,1-36. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_log_analytics_workspace" configuration
│ 
│   on main.tf line 390:
│  390: resource "azurerm_log_analytics_workspace" "main" {
│ 
│ A azurerm_log_analytics_workspace resource named "main" was already
│ declared at main-clean.tf:88,1-50. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_application_insights" configuration
│ 
│   on main.tf line 401:
│  401: resource "azurerm_application_insights" "main" {
│ 
│ A azurerm_application_insights resource named "main" was already declared
│ at main-clean.tf:98,1-47. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app_environment" configuration
│ 
│   on main.tf line 418:
│  418: resource "azurerm_container_app_environment" "main" {
│ 
│ A azurerm_container_app_environment resource named "main" was already
│ declared at main-clean.tf:125,1-52. Resource names must be unique per type
│ in each module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app" configuration
│ 
│   on main.tf line 440:
│  440: resource "azurerm_container_app" "core" {
│ 
│ A azurerm_container_app resource named "core" was already declared at
│ main-clean.tf:205,1-40. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate resource "azurerm_container_app" configuration
│ 
│   on main.tf line 511:
│  511: resource "azurerm_container_app" "frontend" {
│ 
│ A azurerm_container_app resource named "frontend" was already declared at
│ main-clean.tf:252,1-44. Resource names must be unique per type in each
│ module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main.tf line 634:
│  634: output "resource_group_name" {
│ 
│ An output named "resource_group_name" was already defined at
│ main-complete.tf:844,1-29. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main.tf line 644:
│  644: output "postgresql_server_name" {
│ 
│ An output named "postgresql_server_name" was already defined at
│ main-complete.tf:884,1-32. Output names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on main.tf line 659:
│  659: output "openai_endpoint" {
│ 
│ An output named "openai_endpoint" was already defined at
│ main-complete.tf:892,1-25. Output names must be unique within a module.
╵

╷
│ Error: Duplicate variable declaration
│ 
│   on variables.tf line 4:
│    4: variable "environment" {
│ 
│ A variable named "environment" was already declared at main.tf:44,1-23.
│ Variable names must be unique within a module.
╵

╷
│ Error: Duplicate variable declaration
│ 
│   on variables.tf line 13:
│   13: variable "location" {
│ 
│ A variable named "location" was already declared at main.tf:50,1-20.
│ Variable names must be unique within a module.
╵

╷
│ Error: Duplicate local value definition
│ 
│   on variables.tf line 29, in locals:
│   29:   hash_suffix = substr(md5(local.hash_input), 0, 6)
│ 
│ A local value named "hash_suffix" was already defined at
│ main-complete.tf:61,3-26. Local value names must be unique within a module.
╵

╷
│ Error: Duplicate local value definition
│ 
│   on variables.tf line 32, in locals:
│   32:   env_suffix = var.environment
│ 
│ A local value named "env_suffix" was already defined at main.tf:74,3-34.
│ Local value names must be unique within a module.
╵

╷
│ Error: Duplicate local value definition
│ 
│   on variables.tf line 35, in locals:
│   35:   resource_names = {
│   36:     # Resource Group
│   37:     resource_group = "rg-${var.project_name}-${local.env_suffix}"
│   38:     # Container Apps
│   39:     container_env = "cae-${var.project_name}-${local.env_suffix}"
│   40:     core_app      = "ca-${var.project_name}-core-${local.env_suffix}"
│   41:     frontend_app  = "ca-${var.project_name}-frontend-${local.env_suffix}"
│   42:     graphql_app   = "ca-${var.project_name}-graphql-${local.env_suffix}"
│   43:     # Container Registry (global, needs unique name)
│   44:     container_registry = "cr${var.project_name}${local.env_suffix}${local.hash_suffix}"
│   45:     # Storage Account (global, needs unique name, no hyphens)
│   46:     storage_account = "st${var.project_name}${local.env_suffix}${local.hash_suffix}"
│   47:     # Key Vault
│   48:     key_vault = "kv-${var.project_name}-${local.env_suffix}-${local.hash_suffix}"
│   49:     # Database
│   50:     postgresql = "psql-${var.project_name}-${local.env_suffix}"
│   51:     cosmos_db  = "cosmos-${var.project_name}-${local.env_suffix}-${local.hash_suffix}"
│   52:     # Monitoring
│   53:     log_workspace = "log-${var.project_name}-${local.env_suffix}"
│   54:     app_insights  = "appi-${var.project_name}-${local.env_suffix}"
│   55:     # Networking
│   56:     vnet = "vnet-${var.project_name}-${local.env_suffix}"
│   57:     # AI Services
│   58:     openai = "cogao-${var.project_name}-${local.env_suffix}"
│   59:   }
│ 
│ A local value named "resource_names" was already defined at
│ main-complete.tf:64,3-96,4. Local value names must be unique within a
│ module.
╵

╷
│ Error: Duplicate local value definition
│ 
│   on variables.tf line 70, in locals:
│   70:   common_tags = {
│   71:     Environment = var.environment
│   72:     Project     = "PolicyCortex"
│   73:     ManagedBy   = "Terraform"
│   74:     Repository  = "github.com/laeintel/policycortex"
│   75:     CostCenter  = var.environment == "prod" ? "Production" : "Development"
│   76:   }
│ 
│ A local value named "common_tags" was already defined at main.tf:68,3-71,5.
│ Local value names must be unique within a module.
╵

╷
│ Error: Duplicate output definition
│ 
│   on variables.tf line 85:
│   85: output "container_registry_url" {
│ 
│ An output named "container_registry_url" was already defined at
│ main-complete.tf:860,1-32. Output names must be unique within a module.
╵

Error: Terraform exited with code 1.
Error: Process completed with exit code 1.