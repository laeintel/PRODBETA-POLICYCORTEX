// Azure Activity Log Integration
// Provides activity and audit log data

use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};

use super::client::AzureClient;
use super::api_versions;

/// Activity Log Entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityLogEntry {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub entry_type: String,
    pub value: Vec<ActivityEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEvent {
    #[serde(rename = "eventName")]
    pub event_name: EventName,
    pub id: String,
    pub level: String,
    #[serde(rename = "resourceGroupName")]
    pub resource_group_name: Option<String>,
    #[serde(rename = "resourceProviderName")]
    pub resource_provider_name: EventName,
    #[serde(rename = "resourceId")]
    pub resource_id: Option<String>,
    #[serde(rename = "resourceType")]
    pub resource_type: Option<EventName>,
    #[serde(rename = "operationName")]
    pub operation_name: EventName,
    #[serde(rename = "operationId")]
    pub operation_id: String,
    pub status: EventName,
    #[serde(rename = "eventTimestamp")]
    pub event_timestamp: DateTime<Utc>,
    #[serde(rename = "submissionTimestamp")]
    pub submission_timestamp: DateTime<Utc>,
    #[serde(rename = "subscriptionId")]
    pub subscription_id: String,
    pub properties: Option<ActivityProperties>,
    #[serde(rename = "relatedEvents")]
    pub related_events: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventName {
    pub value: String,
    #[serde(rename = "localizedValue")]
    pub localized_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityProperties {
    #[serde(rename = "statusCode")]
    pub status_code: Option<String>,
    #[serde(rename = "statusMessage")]
    pub status_message: Option<String>,
    #[serde(rename = "eventCategory")]
    pub event_category: Option<String>,
    pub entity: Option<String>,
    pub message: Option<String>,
}

/// Activity summary
#[derive(Debug, Serialize)]
pub struct ActivitySummary {
    pub total_activities: usize,
    pub activities_by_level: HashMap<String, usize>,
    pub activities_by_status: HashMap<String, usize>,
    pub activities_by_operation: HashMap<String, usize>,
    pub recent_activities: Vec<SimpleActivity>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimpleActivity {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub resource_id: Option<String>,
    pub level: String,
    pub status: String,
    pub message: Option<String>,
}

/// Azure Activity Log service
pub struct ActivityService {
    client: AzureClient,
}

impl ActivityService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get activity log entries for a time range
    pub async fn get_activity_logs(
        &self,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<ActivityEvent>> {
        info!("Fetching activity logs from Azure");

        let start = start_time.unwrap_or_else(|| Utc::now() - Duration::hours(24));
        let end = end_time.unwrap_or_else(|| Utc::now());

        let filter = format!(
            "eventTimestamp ge '{}' and eventTimestamp le '{}'",
            start.to_rfc3339(),
            end.to_rfc3339()
        );

        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Insights/eventtypes/management/values?$filter={}",
            self.client.config.subscription_id,
            urlencoding::encode(&filter)
        );

        let response: ActivityLogEntry = self.client
            .get_management(&path, api_versions::ACTIVITY_LOG)
            .await?;

        Ok(response.value)
    }

    /// Get recent activities (last 24 hours)
    pub async fn get_recent_activities(&self) -> Result<Vec<SimpleActivity>> {
        let activities = self.get_activity_logs(None, None).await?;

        let simple_activities = activities
            .into_iter()
            .map(|a| SimpleActivity {
                id: a.id,
                timestamp: a.event_timestamp,
                operation: a.operation_name.value,
                resource_id: a.resource_id,
                level: a.level,
                status: a.status.value,
                message: a.properties.and_then(|p| p.message),
            })
            .collect();

        Ok(simple_activities)
    }

    /// Get activity summary
    pub async fn get_activity_summary(&self) -> Result<ActivitySummary> {
        let activities = self.get_activity_logs(None, None).await?;

        let mut activities_by_level = HashMap::new();
        let mut activities_by_status = HashMap::new();
        let mut activities_by_operation = HashMap::new();

        for activity in &activities {
            *activities_by_level.entry(activity.level.clone()).or_insert(0) += 1;
            *activities_by_status.entry(activity.status.value.clone()).or_insert(0) += 1;
            *activities_by_operation.entry(activity.operation_name.value.clone()).or_insert(0) += 1;
        }

        let recent_activities = activities
            .iter()
            .take(20)
            .map(|a| SimpleActivity {
                id: a.id.clone(),
                timestamp: a.event_timestamp,
                operation: a.operation_name.value.clone(),
                resource_id: a.resource_id.clone(),
                level: a.level.clone(),
                status: a.status.value.clone(),
                message: a.properties.as_ref().and_then(|p| p.message.clone()),
            })
            .collect();

        Ok(ActivitySummary {
            total_activities: activities.len(),
            activities_by_level,
            activities_by_status,
            activities_by_operation,
            recent_activities,
        })
    }

    /// Get administrative activities
    pub async fn get_administrative_activities(&self) -> Result<Vec<ActivityEvent>> {
        let activities = self.get_activity_logs(None, None).await?;

        let admin_activities = activities
            .into_iter()
            .filter(|a| {
                a.properties.as_ref()
                    .and_then(|p| p.event_category.as_ref())
                    .map_or(false, |c| c == "Administrative")
            })
            .collect();

        Ok(admin_activities)
    }

    /// Get security-related activities
    pub async fn get_security_activities(&self) -> Result<Vec<ActivityEvent>> {
        let activities = self.get_activity_logs(None, None).await?;

        let security_activities = activities
            .into_iter()
            .filter(|a| {
                let op_name = a.operation_name.value.to_lowercase();
                op_name.contains("security") ||
                op_name.contains("role") ||
                op_name.contains("permission") ||
                op_name.contains("access") ||
                op_name.contains("authentication") ||
                op_name.contains("authorization")
            })
            .collect();

        Ok(security_activities)
    }

    /// Get resource modification activities
    pub async fn get_resource_modifications(&self) -> Result<Vec<ActivityEvent>> {
        let activities = self.get_activity_logs(None, None).await?;

        let modifications = activities
            .into_iter()
            .filter(|a| {
                let op_name = a.operation_name.value.to_lowercase();
                op_name.contains("write") ||
                op_name.contains("delete") ||
                op_name.contains("create") ||
                op_name.contains("update") ||
                op_name.contains("modify")
            })
            .collect();

        Ok(modifications)
    }

    /// Get failed operations
    pub async fn get_failed_operations(&self) -> Result<Vec<ActivityEvent>> {
        let activities = self.get_activity_logs(None, None).await?;

        let failed = activities
            .into_iter()
            .filter(|a| {
                a.status.value.to_lowercase().contains("failed") ||
                a.level == "Error" ||
                a.level == "Critical"
            })
            .collect();

        Ok(failed)
    }
}

/// Helper module for URL encoding
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                ' ' => "%20".to_string(),
                '\'' => "%27".to_string(),
                _ if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' => c.to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}