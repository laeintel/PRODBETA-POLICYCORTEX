# Monitoring Module for PolicyCortex
# Implements comprehensive monitoring, alerting, and dashboards

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Data sources for existing resources
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

data "azurerm_log_analytics_workspace" "main" {
  name                = var.log_analytics_workspace_name
  resource_group_name = var.resource_group_name
}

data "azurerm_application_insights" "main" {
  name                = var.application_insights_name
  resource_group_name = var.resource_group_name
}

# Action Group for critical alerts
resource "azurerm_monitor_action_group" "critical" {
  name                = "${var.project_name}-critical-alerts-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  short_name          = "critical"

  # Email notifications
  dynamic "email_receiver" {
    for_each = var.critical_alert_emails
    content {
      name          = "email-${email_receiver.key}"
      email_address = email_receiver.value
    }
  }

  # SMS notifications for critical alerts
  dynamic "sms_receiver" {
    for_each = var.critical_alert_sms
    content {
      name         = "sms-${sms_receiver.key}"
      country_code = sms_receiver.value.country_code
      phone_number = sms_receiver.value.phone_number
    }
  }

  # Webhook for integration with external systems
  dynamic "webhook_receiver" {
    for_each = var.webhook_urls
    content {
      name        = "webhook-${webhook_receiver.key}"
      service_uri = webhook_receiver.value
    }
  }

  tags = var.tags
}

# Action Group for warning alerts
resource "azurerm_monitor_action_group" "warning" {
  name                = "${var.project_name}-warning-alerts-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  short_name          = "warning"

  # Email notifications only for warnings
  dynamic "email_receiver" {
    for_each = var.warning_alert_emails
    content {
      name          = "email-${email_receiver.key}"
      email_address = email_receiver.value
    }
  }

  tags = var.tags
}

# Activity Log Alert for Resource Group changes
resource "azurerm_monitor_activity_log_alert" "resource_group_changes" {
  name                = "${var.project_name}-rg-changes-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [data.azurerm_resource_group.main.id]
  description         = "Alert when resources are modified in PolicyCortex resource group"

  criteria {
    category       = "Administrative"
    operation_name = "Microsoft.Resources/subscriptions/resourceGroups/write"
  }

  action {
    action_group_id = azurerm_monitor_action_group.warning.id
  }

  tags = var.tags
}

# Metric Alert for Container Apps CPU usage
resource "azurerm_monitor_metric_alert" "container_apps_cpu" {
  count               = length(var.container_app_names)
  name                = "${var.project_name}-${var.container_app_names[count.index]}-cpu-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [var.container_app_resource_ids[count.index]]
  description         = "Alert when ${var.container_app_names[count.index]} CPU usage is high"
  severity            = 2
  frequency           = "PT1M"
  window_size         = "PT5M"

  criteria {
    metric_namespace = "Microsoft.App/containerApps"
    metric_name      = "CpuPercentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = var.cpu_threshold_percentage
  }

  action {
    action_group_id = azurerm_monitor_action_group.warning.id
  }

  tags = var.tags
}

# Metric Alert for Container Apps Memory usage
resource "azurerm_monitor_metric_alert" "container_apps_memory" {
  count               = length(var.container_app_names)
  name                = "${var.project_name}-${var.container_app_names[count.index]}-memory-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [var.container_app_resource_ids[count.index]]
  description         = "Alert when ${var.container_app_names[count.index]} memory usage is high"
  severity            = 2
  frequency           = "PT1M"
  window_size         = "PT5M"

  criteria {
    metric_namespace = "Microsoft.App/containerApps"
    metric_name      = "MemoryPercentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = var.memory_threshold_percentage
  }

  action {
    action_group_id = azurerm_monitor_action_group.warning.id
  }

  tags = var.tags
}

# Metric Alert for Container Apps Restart count
resource "azurerm_monitor_metric_alert" "container_apps_restarts" {
  count               = length(var.container_app_names)
  name                = "${var.project_name}-${var.container_app_names[count.index]}-restarts-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [var.container_app_resource_ids[count.index]]
  description         = "Alert when ${var.container_app_names[count.index]} has high restart count"
  severity            = 1
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.App/containerApps"
    metric_name      = "RestartCount"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = var.restart_threshold_count
  }

  action {
    action_group_id = azurerm_monitor_action_group.critical.id
  }

  tags = var.tags
}

# Log Analytics Alert for Application Errors
resource "azurerm_monitor_scheduled_query_rules_alert_v2" "application_errors" {
  name                = "${var.project_name}-app-errors-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  
  evaluation_frequency = "PT5M"
  window_duration      = "PT15M"
  scopes               = [data.azurerm_log_analytics_workspace.main.id]
  severity             = 1
  description          = "Alert when application error rate is high"

  criteria {
    query                   = <<-QUERY
      AppTraces
      | where TimeGenerated > ago(15m)
      | where SeverityLevel >= 3
      | summarize ErrorCount = count() by bin(TimeGenerated, 5m)
      | where ErrorCount > ${var.error_threshold_count}
    QUERY
    time_aggregation_method = "Maximum"
    threshold               = 0
    operator                = "GreaterThan"
    metric_measure_column   = "ErrorCount"

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 3
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.critical.id]
  }

  tags = var.tags
}

# Log Analytics Alert for Failed Authentications
resource "azurerm_monitor_scheduled_query_rules_alert_v2" "auth_failures" {
  name                = "${var.project_name}-auth-failures-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  
  evaluation_frequency = "PT5M"
  window_duration      = "PT15M"
  scopes               = [data.azurerm_log_analytics_workspace.main.id]
  severity             = 2
  description          = "Alert when authentication failure rate is high"

  criteria {
    query                   = <<-QUERY
      AppTraces
      | where TimeGenerated > ago(15m)
      | where Message contains "authentication" and Message contains "failed"
      | summarize FailureCount = count() by bin(TimeGenerated, 5m)
      | where FailureCount > ${var.auth_failure_threshold}
    QUERY
    time_aggregation_method = "Maximum"
    threshold               = 0
    operator                = "GreaterThan"
    metric_measure_column   = "FailureCount"

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 3
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.warning.id]
  }

  tags = var.tags
}

# Cosmos DB Alert for High RU Consumption
resource "azurerm_monitor_metric_alert" "cosmos_ru_consumption" {
  count               = var.deploy_cosmos_monitoring ? 1 : 0
  name                = "${var.project_name}-cosmos-ru-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [var.cosmos_db_account_id]
  description         = "Alert when Cosmos DB RU consumption is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.DocumentDB/databaseAccounts"
    metric_name      = "TotalRequestUnits"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = var.cosmos_ru_threshold
  }

  action {
    action_group_id = azurerm_monitor_action_group.warning.id
  }

  tags = var.tags
}

# SQL Database Alert for High DTU
resource "azurerm_monitor_metric_alert" "sql_dtu_consumption" {
  count               = var.deploy_sql_monitoring ? 1 : 0
  name                = "${var.project_name}-sql-dtu-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [var.sql_database_id]
  description         = "Alert when SQL Database DTU consumption is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.Sql/servers/databases"
    metric_name      = "dtu_consumption_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = var.sql_dtu_threshold_percentage
  }

  action {
    action_group_id = azurerm_monitor_action_group.warning.id
  }

  tags = var.tags
}

# Storage Account Alert for High Transactions
resource "azurerm_monitor_metric_alert" "storage_transactions" {
  count               = var.deploy_storage_monitoring ? 1 : 0
  name                = "${var.project_name}-storage-transactions-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [var.storage_account_id]
  description         = "Alert when Storage Account transaction count is high"
  severity            = 3
  frequency           = "PT15M"
  window_size         = "PT1H"

  criteria {
    metric_namespace = "Microsoft.Storage/storageAccounts"
    metric_name      = "Transactions"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = var.storage_transaction_threshold
  }

  action {
    action_group_id = azurerm_monitor_action_group.warning.id
  }

  tags = var.tags
}

# Azure Dashboard for PolicyCortex Monitoring
resource "azurerm_portal_dashboard" "main" {
  name                = "${var.project_name}-dashboard-${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  
  dashboard_properties = templatefile("${path.module}/dashboard.json", {
    subscription_id                = var.subscription_id
    resource_group_name           = data.azurerm_resource_group.main.name
    log_analytics_workspace_id    = data.azurerm_log_analytics_workspace.main.id
    application_insights_id       = data.azurerm_application_insights.main.id
    container_app_environment_id  = var.container_app_environment_id
    cosmos_db_account_id         = var.cosmos_db_account_id
    sql_database_id              = var.sql_database_id
    storage_account_id           = var.storage_account_id
    project_name                 = var.project_name
    environment                  = var.environment
  })

  tags = var.tags
}

# Budget Alert for Resource Group
resource "azurerm_consumption_budget_resource_group" "main" {
  name              = "${var.project_name}-budget-${var.environment}"
  resource_group_id = data.azurerm_resource_group.main.id

  amount     = var.monthly_budget_amount
  time_grain = "Monthly"

  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00'Z'", timestamp())
  }

  filter {
    dimension {
      name = "ResourceGroupName"
      values = [data.azurerm_resource_group.main.name]
    }
  }

  notification {
    enabled   = true
    threshold = 80
    operator  = "GreaterThan"

    contact_emails = var.budget_alert_emails
  }

  notification {
    enabled   = true
    threshold = 100
    operator  = "GreaterThan"

    contact_emails = var.budget_alert_emails
  }

  notification {
    enabled   = true
    threshold = 120
    operator  = "GreaterThan"

    contact_emails = var.budget_alert_emails
  }
}

# Workbook for detailed analytics
resource "azurerm_application_insights_workbook" "main" {
  name                = uuidv5("dns", "${var.project_name}-workbook-${var.environment}")
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  display_name        = "PolicyCortex Analytics Workbook - ${upper(var.environment)}"
  
  data_json = templatefile("${path.module}/workbook.json", {
    subscription_id                = var.subscription_id
    resource_group_name           = data.azurerm_resource_group.main.name
    log_analytics_workspace_id    = data.azurerm_log_analytics_workspace.main.id
    application_insights_id       = data.azurerm_application_insights.main.id
    project_name                 = var.project_name
    environment                  = var.environment
  })

  tags = var.tags
}