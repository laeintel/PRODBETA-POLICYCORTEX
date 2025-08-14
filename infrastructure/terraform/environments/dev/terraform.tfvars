# Development Environment Configuration - Optimized for Free Tier
environment = "dev"
location    = "eastus" # Use consistent lowercase format

tags = {
  Owner       = "AeoliTech"
  Project     = "PolicyCortex"
  ManagedBy   = "Terraform"
  Environment = "Development"
  CostCenter  = "FreeTier"
}

# Enable Azure OpenAI deployments via Terraform
enable_openai_deployments = true

# Provide the exact version string from the Azure portal for your region
# Example: "2025-01-01-preview" (replace with the actual shown version)
openai_deployments = [
  {
    deploy_name   = "gpt-4o"
    model_name    = "gpt-4o"
    model_version = "2024-11-20"
    format        = "OpenAI"
    scale_type    = "Standard"
  }
]

# Frontend env wiring (literal defaults; CI will override via TF_VAR_*)
frontend_next_public_azure_client_id               = ""
frontend_next_public_azure_tenant_id               = ""
frontend_next_public_msal_redirect_uri             = "http://localhost:3000"
frontend_next_public_msal_post_logout_redirect_uri = "http://localhost:3000"
frontend_next_public_aoai_endpoint                 = ""
frontend_next_public_aoai_api_version              = "2024-08-01-preview"
frontend_next_public_aoai_chat_deployment          = "gpt-4o"
frontend_next_public_aoai_api_key                  = ""
frontend_next_public_demo_mode                     = true