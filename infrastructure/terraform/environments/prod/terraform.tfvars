# Production Environment Configuration - Optimized for Free Tier
environment = "prod"
location    = "eastus" # Use consistent lowercase format

tags = {
  Owner       = "AeoliTech"
  Project     = "PolicyCortex"
  ManagedBy   = "Terraform"
  Environment = "Production"
  CostCenter  = "FreeTier"
  Compliance  = "Required"
}