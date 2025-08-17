# Import blocks for existing Azure resources
# These import blocks allow Terraform to adopt existing resources
# that were created outside of Terraform

# Import existing resource group if it exists
# This will be skipped if the resource doesn't exist
import {
  to = azurerm_resource_group.main
  id = "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78/resourceGroups/rg-cortex-dev"
}

# Note: Additional import blocks can be added here for other existing resources
# The import blocks will only be processed if the resources exist in Azure
# If resources don't exist, Terraform will create them normally

# Example patterns for other resources that might need importing:
# 
# Container Apps Environment:
# import {
#   to = azurerm_container_app_environment.main
#   id = "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78/resourceGroups/rg-cortex-dev/providers/Microsoft.App/managedEnvironments/cae-cortex-dev"
# }
#
# Container Apps:
# import {
#   to = azurerm_container_app.core
#   id = "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78/resourceGroups/rg-cortex-dev/providers/Microsoft.App/containerApps/ca-cortex-core-dev"
# }
#
# import {
#   to = azurerm_container_app.frontend
#   id = "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78/resourceGroups/rg-cortex-dev/providers/Microsoft.App/containerApps/ca-cortex-frontend-dev"
# }