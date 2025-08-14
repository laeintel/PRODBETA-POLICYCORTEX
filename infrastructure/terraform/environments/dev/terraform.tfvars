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

# Enable Azure OpenAI deployments via Terraform (temporarily false until exact version is provided)
enable_openai_deployments = false

# Provide the exact version string from the Azure portal for your region
# Example: "2025-01-01-preview" (replace with the actual shown version)
openai_deployments = []