# Development environment configuration
environment   = "dev"
location      = "East US"
project_name  = "policycortex"
owner         = "AeoliTech"
allowed_ips   = []  # Add your IP addresses here if needed

# Set to false if Key Vault access policy already exists
create_terraform_access_policy = false

# Container Apps deployment
deploy_container_apps = true

# Data Services Configuration
sql_admin_username = "sqladmin"
sql_azuread_admin_login = "admin@yourdomain.com"
sql_azuread_admin_object_id = "00000000-0000-0000-0000-000000000000"
sql_database_sku = "GP_S_Gen5_1"  # Smaller for dev
sql_database_max_size_gb = 10      # Smaller for dev
cosmos_consistency_level = "Session"
cosmos_failover_location = "West US 2"
cosmos_max_throughput = 1000       # Smaller for dev
redis_capacity = 1                 # Smaller for dev
redis_sku_name = "Basic"           # Basic for dev

# AI Services Configuration
create_ml_container_registry = false  # Use main ACR
training_cluster_vm_size = "Standard_DS2_v2"  # Smaller for dev
training_cluster_max_nodes = 2             # Fewer nodes for dev
compute_instance_vm_size = "Standard_DS2_v2" # Smaller for dev
cognitive_services_sku = "F0"              # Free tier for dev
deploy_openai = false                      # Skip OpenAI for dev
openai_sku = "S0"