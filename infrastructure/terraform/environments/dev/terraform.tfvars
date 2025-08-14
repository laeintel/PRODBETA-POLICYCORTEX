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

# Frontend env wiring (optional for local; used in Container App)
frontend_next_public_azure_client_id              = "${env("NEXT_PUBLIC_AZURE_CLIENT_ID", "")}"
frontend_next_public_azure_tenant_id              = "${env("NEXT_PUBLIC_AZURE_TENANT_ID", "")}"
frontend_next_public_msal_redirect_uri            = "http://localhost:3000"
frontend_next_public_msal_post_logout_redirect_uri = "http://localhost:3000"
frontend_next_public_aoai_endpoint                = "${env("NEXT_PUBLIC_AOAI_ENDPOINT", "")}"
frontend_next_public_aoai_api_version             = "${env("NEXT_PUBLIC_AOAI_API_VERSION", "2024-08-01-preview")}"
frontend_next_public_aoai_chat_deployment         = "${env("NEXT_PUBLIC_AOAI_CHAT_DEPLOYMENT", "gpt-4o")}"
frontend_next_public_aoai_api_key                 = "${env("NEXT_PUBLIC_AOAI_API_KEY", "")}"
frontend_next_public_demo_mode                    = true