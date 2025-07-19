# Moved blocks to handle transition from non-count to count resources
# This tells Terraform that resources without [0] should be treated as [0]
moved {
  from = azurerm_container_app.api_gateway
  to   = azurerm_container_app.api_gateway[0]
}

moved {
  from = azurerm_container_app.azure_integration
  to   = azurerm_container_app.azure_integration[0]
}

moved {
  from = azurerm_container_app.ai_engine
  to   = azurerm_container_app.ai_engine[0]
}

moved {
  from = azurerm_container_app.data_processing
  to   = azurerm_container_app.data_processing[0]
}

moved {
  from = azurerm_container_app.conversation
  to   = azurerm_container_app.conversation[0]
}

moved {
  from = azurerm_container_app.notification
  to   = azurerm_container_app.notification[0]
}

moved {
  from = azurerm_container_app.frontend
  to   = azurerm_container_app.frontend[0]
}