# Production environment configuration
environment   = "prod"
location      = "East US"
project_name  = "policycortex"
owner         = "AeoliTech"
allowed_ips   = []

# Key Vault access
create_terraform_access_policy = false

# Container Apps deployment
deploy_container_apps = true

# Data Services Configuration - Production
sql_admin_username = "sqladmin"
sql_azuread_admin_login = "admin@yourdomain.com"
sql_azuread_admin_object_id = "00000000-0000-0000-0000-000000000000"
sql_database_sku = "GP_Gen5_4"  # Larger for prod
sql_database_max_size_gb = 100   # Larger for prod
cosmos_consistency_level = "Session"
cosmos_failover_location = "West US 2"
cosmos_max_throughput = 10000    # Larger for prod
redis_capacity = 6               # Larger for prod
redis_sku_name = "Premium"       # Premium for prod

# AI Services Configuration - Production
create_ml_container_registry = true   # Separate ACR for prod
training_cluster_vm_size = "Standard_DS4_v2"  # Larger for prod
training_cluster_max_nodes = 10            # More nodes for prod
compute_instance_vm_size = "Standard_DS4_v2" # Larger for prod
cognitive_services_sku = "S0"              # Standard for prod
deploy_openai = true                       # Enable OpenAI for prod
openai_sku = "S0"

# Monitoring Configuration
critical_alert_emails = ["admin@company.com", "security@company.com"]
warning_alert_emails = ["devops@company.com", "monitoring@company.com"]
budget_alert_emails = ["finance@company.com", "management@company.com"]
monthly_budget_amount = 5000  # Higher budget for prod