# Development environment configuration
environment   = "dev"
location      = "East US"
project_name  = "policycortex"
owner         = "AeoliTech"
allowed_ips   = []  # Add your IP addresses here if needed

# Set to false if Key Vault access policy already exists
create_terraform_access_policy = false

# Container Apps deployment (disabled for AKS deployment)
deploy_container_apps = false

# Kubernetes/AKS deployment (enable for AKS deployment)
deploy_kubernetes = true
kubernetes_version = "1.28.3"
kubernetes_node_count = 3
kubernetes_node_vm_size = "Standard_D4s_v3"
kubernetes_enable_auto_scaling = true
kubernetes_min_node_count = 2
kubernetes_max_node_count = 10
kubernetes_enable_ai_node_pool = false  # Enable if you need GPU nodes for AI workloads
kubernetes_ai_node_vm_size = "Standard_NC6s_v3"
kubernetes_ai_node_count = 1

# Application Gateway for ingress (optional)
enable_application_gateway = false

# Data Services Configuration
deploy_sql_server = false  # SQL Server provisioning restricted in East US
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
deploy_ml_workspace = false  # ML workspace has soft-delete conflict
create_ml_container_registry = false  # Use main ACR
training_cluster_vm_size = "Standard_DS2_v2"  # Smaller for dev
training_cluster_max_nodes = 2             # Fewer nodes for dev
compute_instance_vm_size = "Standard_DS2_v2" # Smaller for dev
cognitive_services_sku = "S0"              # Standard tier (F0 not supported for CognitiveServices)
deploy_openai = false                      # Skip OpenAI for dev
openai_sku = "S0"

# Monitoring Configuration
critical_alert_emails = ["admin@company.com"]
warning_alert_emails = ["devops@company.com"]
budget_alert_emails = ["finance@company.com"]
monthly_budget_amount = 500  # Lower budget for dev