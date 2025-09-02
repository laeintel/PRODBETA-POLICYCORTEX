// Azure Monitor Integration
// Provides metrics, alerts, and monitoring data

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use super::client::AzureClient;
use super::api_versions;

/// Azure Monitor metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub id: String,
    pub name: MetricName,
    pub unit: String,
    pub timeseries: Vec<TimeSeries>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricName {
    pub value: String,
    #[serde(rename = "localizedValue")]
    pub localized_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub data: Vec<MetricData>,
    #[serde(rename = "metadatavalues")]
    pub metadata_values: Option<Vec<MetadataValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricData {
    #[serde(rename = "timeStamp")]
    pub timestamp: DateTime<Utc>,
    pub average: Option<f64>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub total: Option<f64>,
    pub count: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataValue {
    pub name: MetricName,
    pub value: String,
}

/// Azure Monitor alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub alert_type: String,
    pub location: String,
    pub properties: AlertProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertProperties {
    pub severity: String,
    pub description: Option<String>,
    pub enabled: bool,
    #[serde(rename = "scopes")]
    pub scopes: Vec<String>,
    #[serde(rename = "evaluationFrequency")]
    pub evaluation_frequency: Option<String>,
    #[serde(rename = "windowSize")]
    pub window_size: Option<String>,
    #[serde(rename = "targetResourceType")]
    pub target_resource_type: Option<String>,
    #[serde(rename = "lastUpdatedTime")]
    pub last_updated_time: Option<DateTime<Utc>>,
}

/// Alert incident
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertIncident {
    pub name: String,
    #[serde(rename = "ruleName")]
    pub rule_name: String,
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    #[serde(rename = "isActive")]
    pub is_active: bool,
    #[serde(rename = "activatedTime")]
    pub activated_time: DateTime<Utc>,
    #[serde(rename = "resolvedTime")]
    pub resolved_time: Option<DateTime<Utc>>,
}

/// Azure Monitor service
pub struct MonitorService {
    client: AzureClient,
}

impl MonitorService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get metrics for a resource
    pub async fn get_metrics(
        &self,
        resource_id: &str,
        metric_names: &[&str],
        time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    ) -> Result<Vec<Metric>> {
        let (start, end) = time_range.unwrap_or_else(|| {
            let now = Utc::now();
            (now - Duration::hours(1), now)
        });

        let metric_names_str = metric_names.join(",");
        let timespan = format!("{}/{}", start.to_rfc3339(), end.to_rfc3339());
        
        let path = format!(
            "{}/providers/Microsoft.Insights/metrics?metricnames={}&timespan={}&aggregation=Average,Total,Maximum,Minimum",
            resource_id, metric_names_str, timespan
        );

        let response: MetricsResponse = self.client
            .get_management(&path, api_versions::MONITOR)
            .await?;

        Ok(response.value)
    }

    /// Get all metric alerts
    pub async fn get_metric_alerts(&self) -> Result<Vec<Alert>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Insights/metricAlerts",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::MONITOR).await
    }

    /// Get active alert incidents
    pub async fn get_alert_incidents(&self) -> Result<Vec<AlertIncident>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.AlertsManagement/alerts",
            self.client.config.subscription_id
        );

        let response: AlertIncidentsResponse = self.client
            .get_management(&path, "2019-05-05-preview")
            .await?;

        Ok(response.value.into_iter()
            .map(|a| AlertIncident {
                name: a.name,
                rule_name: a.properties.essentials.alert_rule.unwrap_or_default(),
                rule_id: a.properties.essentials.alert_id.unwrap_or_default(),
                is_active: a.properties.essentials.monitor_condition == "Fired",
                activated_time: a.properties.essentials.start_date_time,
                resolved_time: a.properties.essentials.monitor_condition_resolved_date_time,
            })
            .collect())
    }

    /// Get system health metrics
    pub async fn get_system_health(&self) -> Result<SystemHealth> {
        info!("Fetching system health metrics from Azure Monitor");

        // Get subscription-level metrics
        let resource_id = format!("/subscriptions/{}", self.client.config.subscription_id);
        
        // Try to get some basic metrics (these might not all be available at subscription level)
        let metrics = self.get_metrics(
            &resource_id,
            &["Percentage CPU", "Available Memory Bytes", "Network In Total", "Network Out Total"],
            None
        ).await.unwrap_or_default();

        // Get alerts
        let alerts = self.get_metric_alerts().await.unwrap_or_default();
        let incidents = self.get_alert_incidents().await.unwrap_or_default();

        let active_alerts = incidents.iter().filter(|i| i.is_active).count();
        let total_alerts = alerts.len();

        Ok(SystemHealth {
            cpu_usage: extract_latest_metric_value(&metrics, "Percentage CPU").unwrap_or(0.0),
            memory_usage: extract_latest_metric_value(&metrics, "Available Memory Bytes")
                .map(|v| 100.0 - (v / 1_073_741_824.0 * 100.0)) // Convert to GB and percentage
                .unwrap_or(0.0),
            network_in: extract_latest_metric_value(&metrics, "Network In Total").unwrap_or(0.0),
            network_out: extract_latest_metric_value(&metrics, "Network Out Total").unwrap_or(0.0),
            active_alerts,
            total_alerts,
            alert_severity_distribution: calculate_alert_severity_distribution(&alerts),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct SystemHealth {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_in: f64,
    pub network_out: f64,
    pub active_alerts: usize,
    pub total_alerts: usize,
    pub alert_severity_distribution: HashMap<String, usize>,
}

#[derive(Debug, Deserialize)]
struct MetricsResponse {
    pub value: Vec<Metric>,
}

#[derive(Debug, Deserialize)]
struct AlertIncidentsResponse {
    pub value: Vec<AlertIncidentDetail>,
}

#[derive(Debug, Deserialize)]
struct AlertIncidentDetail {
    pub name: String,
    pub properties: AlertIncidentProperties,
}

#[derive(Debug, Deserialize)]
struct AlertIncidentProperties {
    pub essentials: AlertEssentials,
}

#[derive(Debug, Deserialize)]
struct AlertEssentials {
    #[serde(rename = "alertRule")]
    pub alert_rule: Option<String>,
    #[serde(rename = "alertId")]
    pub alert_id: Option<String>,
    #[serde(rename = "monitorCondition")]
    pub monitor_condition: String,
    #[serde(rename = "startDateTime")]
    pub start_date_time: DateTime<Utc>,
    #[serde(rename = "monitorConditionResolvedDateTime")]
    pub monitor_condition_resolved_date_time: Option<DateTime<Utc>>,
}

fn extract_latest_metric_value(metrics: &[Metric], metric_name: &str) -> Option<f64> {
    metrics.iter()
        .find(|m| m.name.value.contains(metric_name))
        .and_then(|m| m.timeseries.first())
        .and_then(|ts| ts.data.last())
        .and_then(|d| d.average.or(d.total).or(d.maximum))
}

fn calculate_alert_severity_distribution(alerts: &[Alert]) -> HashMap<String, usize> {
    let mut distribution = HashMap::new();
    for alert in alerts {
        *distribution.entry(alert.properties.severity.clone()).or_insert(0) += 1;
    }
    distribution
}