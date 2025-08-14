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
    model_version = "2025-01-01-preview" # TODO: replace with exact version from portal
    format        = "OpenAI"
    scale_type    = "Standard"
  }
]