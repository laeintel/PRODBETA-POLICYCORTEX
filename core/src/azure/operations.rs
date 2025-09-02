// Azure Operations Integration
// Provides automation, notifications, and operational data

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use super::client::AzureClient;
use super::api_versions;

/// Automation Account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationAccount {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub account_type: String,
    pub location: String,
    pub properties: AutomationAccountProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationAccountProperties {
    pub sku: AutomationSku,
    pub state: Option<String>,
    #[serde(rename = "creationTime")]
    pub creation_time: Option<DateTime<Utc>>,
    #[serde(rename = "lastModifiedTime")]
    pub last_modified_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationSku {
    pub name: String,
    pub family: Option<String>,
    pub capacity: Option<i32>,
}

/// Automation Runbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Runbook {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub runbook_type: String,
    pub location: String,
    pub properties: RunbookProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookProperties {
    #[serde(rename = "runbookType")]
    pub runbook_type: String,
    pub state: String,
    #[serde(rename = "logVerbose")]
    pub log_verbose: bool,
    #[serde(rename = "logProgress")]
    pub log_progress: bool,
    #[serde(rename = "logActivityTrace")]
    pub log_activity_trace: Option<i32>,
    #[serde(rename = "jobCount")]
    pub job_count: Option<i32>,
    pub parameters: Option<HashMap<String, RunbookParameter>>,
    #[serde(rename = "creationTime")]
    pub creation_time: Option<DateTime<Utc>>,
    #[serde(rename = "lastModifiedTime")]
    pub last_modified_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookParameter {
    #[serde(rename = "type")]
    pub param_type: String,
    #[serde(rename = "isMandatory")]
    pub is_mandatory: bool,
    pub position: Option<i32>,
    #[serde(rename = "defaultValue")]
    pub default_value: Option<String>,
}

/// Automation Job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationJob {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub job_type: String,
    pub properties: JobProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProperties {
    pub runbook: RunbookReference,
    #[serde(rename = "startedBy")]
    pub started_by: Option<String>,
    #[serde(rename = "runOn")]
    pub run_on: Option<String>,
    #[serde(rename = "jobId")]
    pub job_id: String,
    #[serde(rename = "creationTime")]
    pub creation_time: DateTime<Utc>,
    pub status: String,
    #[serde(rename = "statusDetails")]
    pub status_details: Option<String>,
    #[serde(rename = "startTime")]
    pub start_time: Option<DateTime<Utc>>,
    #[serde(rename = "endTime")]
    pub end_time: Option<DateTime<Utc>>,
    pub exception: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookReference {
    pub name: String,
}

/// Action Group (for notifications)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGroup {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub group_type: String,
    pub location: String,
    pub properties: ActionGroupProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGroupProperties {
    #[serde(rename = "groupShortName")]
    pub group_short_name: String,
    pub enabled: bool,
    #[serde(rename = "emailReceivers")]
    pub email_receivers: Option<Vec<EmailReceiver>>,
    #[serde(rename = "smsReceivers")]
    pub sms_receivers: Option<Vec<SmsReceiver>>,
    #[serde(rename = "webhookReceivers")]
    pub webhook_receivers: Option<Vec<WebhookReceiver>>,
    #[serde(rename = "azureAppPushReceivers")]
    pub azure_app_push_receivers: Option<Vec<AzureAppPushReceiver>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailReceiver {
    pub name: String,
    #[serde(rename = "emailAddress")]
    pub email_address: String,
    #[serde(rename = "useCommonAlertSchema")]
    pub use_common_alert_schema: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsReceiver {
    pub name: String,
    #[serde(rename = "countryCode")]
    pub country_code: String,
    #[serde(rename = "phoneNumber")]
    pub phone_number: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookReceiver {
    pub name: String,
    #[serde(rename = "serviceUri")]
    pub service_uri: String,
    #[serde(rename = "useCommonAlertSchema")]
    pub use_common_alert_schema: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureAppPushReceiver {
    pub name: String,
    #[serde(rename = "emailAddress")]
    pub email_address: String,
}

/// Azure Operations service
pub struct OperationsService {
    client: AzureClient,
}

impl OperationsService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get all automation accounts
    pub async fn get_automation_accounts(&self) -> Result<Vec<AutomationAccount>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Automation/automationAccounts",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::AUTOMATION).await
    }

    /// Get runbooks for an automation account
    pub async fn get_runbooks(&self, account_name: &str, resource_group: &str) -> Result<Vec<Runbook>> {
        let path = format!(
            "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Automation/automationAccounts/{}/runbooks",
            self.client.config.subscription_id, resource_group, account_name
        );

        self.client.get_all_pages(&path, api_versions::AUTOMATION).await
    }

    /// Get jobs for an automation account
    pub async fn get_jobs(&self, account_name: &str, resource_group: &str) -> Result<Vec<AutomationJob>> {
        let path = format!(
            "/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Automation/automationAccounts/{}/jobs",
            self.client.config.subscription_id, resource_group, account_name
        );

        self.client.get_all_pages(&path, api_versions::AUTOMATION).await
    }

    /// Get all action groups
    pub async fn get_action_groups(&self) -> Result<Vec<ActionGroup>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Insights/actionGroups",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::MONITOR).await
    }

    /// Get automation workflows summary
    pub async fn get_automation_workflows(&self) -> Result<AutomationSummary> {
        info!("Fetching automation workflows from Azure");

        let accounts = self.get_automation_accounts().await.unwrap_or_default();
        let action_groups = self.get_action_groups().await.unwrap_or_default();

        let mut total_runbooks = 0;
        let mut total_jobs = 0;
        let mut active_jobs = 0;
        let mut failed_jobs = 0;

        // Get runbooks and jobs for each automation account
        for account in &accounts {
            // Extract resource group from account ID
            if let Some(rg) = extract_resource_group(&account.id) {
                if let Ok(runbooks) = self.get_runbooks(&account.name, &rg).await {
                    total_runbooks += runbooks.len();
                }

                if let Ok(jobs) = self.get_jobs(&account.name, &rg).await {
                    total_jobs += jobs.len();
                    active_jobs += jobs.iter()
                        .filter(|j| j.properties.status == "Running" || j.properties.status == "Activating")
                        .count();
                    failed_jobs += jobs.iter()
                        .filter(|j| j.properties.status == "Failed")
                        .count();
                }
            }
        }

        let enabled_action_groups = action_groups.iter()
            .filter(|ag| ag.properties.enabled)
            .count();

        Ok(AutomationSummary {
            total_automation_accounts: accounts.len(),
            total_runbooks,
            total_jobs,
            active_jobs,
            failed_jobs,
            completed_jobs: total_jobs - active_jobs - failed_jobs,
            total_action_groups: action_groups.len(),
            enabled_action_groups,
        })
    }

    /// Get notification configurations
    pub async fn get_notifications(&self) -> Result<Vec<NotificationConfig>> {
        let action_groups = self.get_action_groups().await?;

        let notifications = action_groups.into_iter()
            .map(|ag| {
                let email_count = ag.properties.email_receivers.as_ref().map_or(0, |e| e.len());
                let sms_count = ag.properties.sms_receivers.as_ref().map_or(0, |s| s.len());
                let webhook_count = ag.properties.webhook_receivers.as_ref().map_or(0, |w| w.len());

                NotificationConfig {
                    id: ag.id,
                    name: ag.name,
                    short_name: ag.properties.group_short_name,
                    enabled: ag.properties.enabled,
                    email_receivers: email_count,
                    sms_receivers: sms_count,
                    webhook_receivers: webhook_count,
                    total_receivers: email_count + sms_count + webhook_count,
                }
            })
            .collect();

        Ok(notifications)
    }
}

#[derive(Debug, Serialize)]
pub struct AutomationSummary {
    pub total_automation_accounts: usize,
    pub total_runbooks: usize,
    pub total_jobs: usize,
    pub active_jobs: usize,
    pub failed_jobs: usize,
    pub completed_jobs: usize,
    pub total_action_groups: usize,
    pub enabled_action_groups: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct NotificationConfig {
    pub id: String,
    pub name: String,
    pub short_name: String,
    pub enabled: bool,
    pub email_receivers: usize,
    pub sms_receivers: usize,
    pub webhook_receivers: usize,
    pub total_receivers: usize,
}

fn extract_resource_group(resource_id: &str) -> Option<String> {
    let parts: Vec<&str> = resource_id.split('/').collect();
    if let Some(index) = parts.iter().position(|&p| p == "resourceGroups") {
        if index + 1 < parts.len() {
            return Some(parts[index + 1].to_string());
        }
    }
    None
}