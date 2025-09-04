// PolicyCortex PREVENT Pillar - Configuration Drift Detector
// Monitors Azure resource configurations and detects drift patterns
// Publishes drift events to prediction engine for proactive violation prevention

use crate::azure_client_async::{AsyncAzureClient, AzureClientConfig};
use crate::api::{ResourceData, PolicyViolation};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Drift event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DriftType {
    ConfigurationChange,
    PolicyDrift,
    ComplianceDrift,
    SecurityPosture,
    CostOptimization,
    TagCompliance,
    AccessControl,
    NetworkSecurity,
}

/// Drift severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum DriftSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Configuration snapshot for drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub resource_id: String,
    pub resource_type: String,
    pub subscription_id: String,
    pub configuration: serde_json::Value,
    pub tags: HashMap<String, String>,
    pub policies: Vec<String>,
    pub compliance_state: String,
    pub security_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Drift event that gets published to prediction engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftEvent {
    pub id: String,
    pub drift_type: DriftType,
    pub severity: DriftSeverity,
    pub resource_id: String,
    pub resource_type: String,
    pub subscription_id: String,
    pub drift_details: DriftDetails,
    pub velocity: f64,
    pub acceleration: f64,
    pub predicted_violation_eta: Option<i64>, // Hours until predicted violation
    pub detected_at: DateTime<Utc>,
}

/// Detailed drift information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetails {
    pub changed_properties: Vec<String>,
    pub previous_value: Option<serde_json::Value>,
    pub current_value: Option<serde_json::Value>,
    pub drift_percentage: f64,
    pub drift_direction: String, // "improving", "degrading", "neutral"
    pub related_policies: Vec<String>,
}

/// Drift metrics for tracking velocity and acceleration
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    pub resource_id: String,
    pub drift_history: VecDeque<DriftEvent>,
    pub total_drifts: usize,
    pub last_drift: Option<DateTime<Utc>>,
    pub velocity_per_day: f64,
    pub acceleration_per_day: f64,
    pub risk_score: f64,
}

/// Drift detector configuration
#[derive(Debug, Clone)]
pub struct DriftDetectorConfig {
    pub scan_interval_seconds: u64,
    pub history_retention_days: u32,
    pub velocity_threshold: f64,
    pub acceleration_threshold: f64,
    pub batch_size: usize,
    pub prediction_api_url: String,
}

impl Default for DriftDetectorConfig {
    fn default() -> Self {
        Self {
            scan_interval_seconds: 300, // 5 minutes
            history_retention_days: 30,
            velocity_threshold: 0.2,     // 20% drift per day triggers alert
            acceleration_threshold: 0.05, // 5% acceleration per day
            batch_size: 50,
            prediction_api_url: "http://localhost:8001".to_string(),
        }
    }
}

/// Main drift detector service
pub struct DriftDetector {
    azure_client: Arc<AsyncAzureClient>,
    config: DriftDetectorConfig,
    resource_snapshots: Arc<RwLock<HashMap<String, ConfigSnapshot>>>,
    drift_metrics: Arc<RwLock<HashMap<String, DriftMetrics>>>,
    event_queue: Arc<Mutex<VecDeque<DriftEvent>>>,
}

impl DriftDetector {
    /// Create new drift detector instance
    pub fn new(azure_client: Arc<AsyncAzureClient>, config: DriftDetectorConfig) -> Self {
        Self {
            azure_client,
            config,
            resource_snapshots: Arc::new(RwLock::new(HashMap::new())),
            drift_metrics: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Start the drift detection service
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting drift detector service");
        
        // Start periodic scanning
        let detector = Arc::new(self.clone());
        
        // Spawn drift detection task
        let scan_detector = detector.clone();
        tokio::spawn(async move {
            let mut scan_interval = interval(
                std::time::Duration::from_secs(scan_detector.config.scan_interval_seconds)
            );
            
            loop {
                scan_interval.tick().await;
                if let Err(e) = scan_detector.scan_for_drift().await {
                    error!("Drift scan failed: {}", e);
                }
            }
        });
        
        // Spawn event publishing task
        let publish_detector = detector.clone();
        tokio::spawn(async move {
            let mut publish_interval = interval(std::time::Duration::from_secs(30));
            
            loop {
                publish_interval.tick().await;
                if let Err(e) = publish_detector.publish_drift_events().await {
                    error!("Failed to publish drift events: {}", e);
                }
            }
        });
        
        // Spawn metrics calculation task
        let metrics_detector = detector.clone();
        tokio::spawn(async move {
            let mut metrics_interval = interval(std::time::Duration::from_secs(60));
            
            loop {
                metrics_interval.tick().await;
                if let Err(e) = metrics_detector.calculate_drift_metrics().await {
                    error!("Failed to calculate drift metrics: {}", e);
                }
            }
        });
        
        Ok(())
    }

    /// Scan Azure resources for configuration drift
    pub async fn scan_for_drift(&self) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Starting drift scan");
        
        // Get all subscriptions to monitor
        let subscriptions = self.azure_client.get_subscriptions().await?;
        
        for subscription in subscriptions {
            // Get resources in subscription
            let resources = self.azure_client
                .get_resources_in_subscription(&subscription.id)
                .await?;
            
            // Process resources in batches
            for chunk in resources.chunks(self.config.batch_size) {
                for resource in chunk {
                    if let Err(e) = self.check_resource_drift(resource).await {
                        warn!("Failed to check drift for resource {}: {}", resource.id, e);
                    }
                }
            }
        }
        
        info!("Drift scan completed");
        Ok(())
    }

    /// Check individual resource for drift
    async fn check_resource_drift(&self, resource: &ResourceData) -> Result<(), Box<dyn std::error::Error>> {
        let resource_id = &resource.id;
        
        // Get current configuration
        let current_snapshot = self.create_snapshot(resource).await?;
        
        // Compare with previous snapshot
        let mut snapshots = self.resource_snapshots.write().await;
        
        if let Some(previous_snapshot) = snapshots.get(resource_id) {
            // Detect drift
            if let Some(drift_event) = self.detect_drift(previous_snapshot, &current_snapshot).await? {
                // Add to event queue
                let mut queue = self.event_queue.lock().await;
                queue.push_back(drift_event);
                
                debug!("Drift detected for resource: {}", resource_id);
            }
        }
        
        // Update snapshot
        snapshots.insert(resource_id.clone(), current_snapshot);
        
        Ok(())
    }

    /// Create configuration snapshot
    async fn create_snapshot(&self, resource: &ResourceData) -> Result<ConfigSnapshot, Box<dyn std::error::Error>> {
        // Get detailed resource configuration
        let configuration = self.azure_client
            .get_resource_configuration(&resource.id)
            .await?;
        
        // Get compliance state
        let compliance_state = self.azure_client
            .get_resource_compliance(&resource.id)
            .await?;
        
        // Calculate security score (simplified)
        let security_score = self.calculate_security_score(&configuration).await;
        
        Ok(ConfigSnapshot {
            resource_id: resource.id.clone(),
            resource_type: resource.resource_type.clone(),
            subscription_id: resource.subscription_id.clone(),
            configuration,
            tags: resource.tags.clone(),
            policies: resource.applied_policies.clone(),
            compliance_state,
            security_score,
            timestamp: Utc::now(),
        })
    }

    /// Detect drift between snapshots
    async fn detect_drift(
        &self,
        previous: &ConfigSnapshot,
        current: &ConfigSnapshot,
    ) -> Result<Option<DriftEvent>, Box<dyn std::error::Error>> {
        // Compare configurations
        let config_diff = self.compare_configurations(&previous.configuration, &current.configuration);
        
        if config_diff.is_empty() && previous.compliance_state == current.compliance_state {
            return Ok(None);
        }
        
        // Determine drift type and severity
        let (drift_type, severity) = self.categorize_drift(&config_diff, previous, current);
        
        // Calculate drift metrics
        let drift_percentage = self.calculate_drift_percentage(&config_diff);
        let drift_direction = self.determine_drift_direction(previous, current);
        
        // Get drift velocity and acceleration
        let metrics = self.get_resource_metrics(&current.resource_id).await;
        
        // Predict violation ETA based on drift velocity
        let predicted_violation_eta = self.predict_violation_eta(
            drift_percentage,
            metrics.velocity_per_day,
            metrics.acceleration_per_day,
        );
        
        Ok(Some(DriftEvent {
            id: format!("drift-{}-{}", current.resource_id, Utc::now().timestamp()),
            drift_type,
            severity,
            resource_id: current.resource_id.clone(),
            resource_type: current.resource_type.clone(),
            subscription_id: current.subscription_id.clone(),
            drift_details: DriftDetails {
                changed_properties: config_diff,
                previous_value: Some(previous.configuration.clone()),
                current_value: Some(current.configuration.clone()),
                drift_percentage,
                drift_direction,
                related_policies: current.policies.clone(),
            },
            velocity: metrics.velocity_per_day,
            acceleration: metrics.acceleration_per_day,
            predicted_violation_eta,
            detected_at: Utc::now(),
        }))
    }

    /// Compare configurations and return changed properties
    fn compare_configurations(
        &self,
        previous: &serde_json::Value,
        current: &serde_json::Value,
    ) -> Vec<String> {
        let mut changed_properties = Vec::new();
        
        if let (Some(prev_obj), Some(curr_obj)) = (previous.as_object(), current.as_object()) {
            for (key, curr_value) in curr_obj {
                if let Some(prev_value) = prev_obj.get(key) {
                    if prev_value != curr_value {
                        changed_properties.push(key.clone());
                    }
                } else {
                    changed_properties.push(format!("added:{}", key));
                }
            }
            
            for key in prev_obj.keys() {
                if !curr_obj.contains_key(key) {
                    changed_properties.push(format!("removed:{}", key));
                }
            }
        }
        
        changed_properties
    }

    /// Categorize drift type and severity
    fn categorize_drift(
        &self,
        changed_props: &[String],
        previous: &ConfigSnapshot,
        current: &ConfigSnapshot,
    ) -> (DriftType, DriftSeverity) {
        // Check for security-related changes
        let security_props = ["networkSecurityGroup", "firewall", "encryption", "authentication"];
        let has_security_changes = changed_props.iter()
            .any(|p| security_props.iter().any(|sp| p.contains(sp)));
        
        if has_security_changes {
            let severity = if current.security_score < previous.security_score - 0.2 {
                DriftSeverity::Critical
            } else if current.security_score < previous.security_score {
                DriftSeverity::High
            } else {
                DriftSeverity::Medium
            };
            return (DriftType::SecurityPosture, severity);
        }
        
        // Check for compliance changes
        if previous.compliance_state != current.compliance_state {
            return (
                DriftType::ComplianceDrift,
                if current.compliance_state == "NonCompliant" {
                    DriftSeverity::High
                } else {
                    DriftSeverity::Medium
                },
            );
        }
        
        // Check for policy changes
        if previous.policies != current.policies {
            return (DriftType::PolicyDrift, DriftSeverity::Medium);
        }
        
        // Default to configuration change
        let severity = match changed_props.len() {
            1..=2 => DriftSeverity::Low,
            3..=5 => DriftSeverity::Medium,
            _ => DriftSeverity::High,
        };
        
        (DriftType::ConfigurationChange, severity)
    }

    /// Calculate drift percentage
    fn calculate_drift_percentage(&self, changed_props: &[String]) -> f64 {
        // Simplified calculation based on number of changes
        // In production, weight changes by importance
        let base_percentage = (changed_props.len() as f64) * 5.0;
        f64::min(base_percentage, 100.0)
    }

    /// Determine drift direction
    fn determine_drift_direction(&self, previous: &ConfigSnapshot, current: &ConfigSnapshot) -> String {
        if current.security_score > previous.security_score + 0.05 {
            "improving".to_string()
        } else if current.security_score < previous.security_score - 0.05 {
            "degrading".to_string()
        } else {
            "neutral".to_string()
        }
    }

    /// Calculate security score for configuration
    async fn calculate_security_score(&self, configuration: &serde_json::Value) -> f64 {
        let mut score = 1.0;
        
        // Check for security features (simplified)
        if let Some(obj) = configuration.as_object() {
            // Check encryption
            if !obj.contains_key("encryption") || !obj["encryption"].as_bool().unwrap_or(false) {
                score -= 0.2;
            }
            
            // Check network security
            if !obj.contains_key("networkSecurityGroup") {
                score -= 0.15;
            }
            
            // Check authentication
            if let Some(auth) = obj.get("authentication") {
                if !auth["mfaEnabled"].as_bool().unwrap_or(false) {
                    score -= 0.1;
                }
            }
            
            // Check backup
            if !obj.contains_key("backupEnabled") || !obj["backupEnabled"].as_bool().unwrap_or(false) {
                score -= 0.1;
            }
        }
        
        f64::max(score, 0.0)
    }

    /// Get or create metrics for resource
    async fn get_resource_metrics(&self, resource_id: &str) -> DriftMetrics {
        let mut metrics_map = self.drift_metrics.write().await;
        
        metrics_map.entry(resource_id.to_string())
            .or_insert_with(|| DriftMetrics {
                resource_id: resource_id.to_string(),
                drift_history: VecDeque::new(),
                total_drifts: 0,
                last_drift: None,
                velocity_per_day: 0.0,
                acceleration_per_day: 0.0,
                risk_score: 0.0,
            })
            .clone()
    }

    /// Predict time until violation based on drift metrics
    fn predict_violation_eta(
        &self,
        drift_percentage: f64,
        velocity: f64,
        acceleration: f64,
    ) -> Option<i64> {
        if velocity <= 0.0 {
            return None;
        }
        
        // Simple physics model: distance = velocity * time + 0.5 * acceleration * time^2
        // Solve for time when drift reaches violation threshold (e.g., 80%)
        let violation_threshold = 80.0;
        let remaining_drift = violation_threshold - drift_percentage;
        
        if remaining_drift <= 0.0 {
            return Some(0); // Already in violation state
        }
        
        // Simplified calculation
        let hours_until_violation = (remaining_drift / (velocity * 24.0)) as i64;
        
        Some(hours_until_violation)
    }

    /// Calculate drift metrics for all resources
    async fn calculate_drift_metrics(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut metrics_map = self.drift_metrics.write().await;
        
        for (resource_id, metrics) in metrics_map.iter_mut() {
            if metrics.drift_history.len() < 2 {
                continue;
            }
            
            // Calculate velocity (drift events per day)
            let time_span = metrics.drift_history.back().unwrap().detected_at
                - metrics.drift_history.front().unwrap().detected_at;
            let days = time_span.num_days() as f64;
            
            if days > 0.0 {
                metrics.velocity_per_day = metrics.drift_history.len() as f64 / days;
                
                // Calculate acceleration (change in velocity)
                if metrics.drift_history.len() > 3 {
                    let mid_point = metrics.drift_history.len() / 2;
                    let first_half_velocity = mid_point as f64 / (days / 2.0);
                    let second_half_velocity = (metrics.drift_history.len() - mid_point) as f64 / (days / 2.0);
                    metrics.acceleration_per_day = (second_half_velocity - first_half_velocity) / days;
                }
                
                // Calculate risk score
                metrics.risk_score = (metrics.velocity_per_day * 0.6 + 
                                     metrics.acceleration_per_day.abs() * 0.4)
                                     .min(1.0);
            }
        }
        
        Ok(())
    }

    /// Publish drift events to prediction engine
    async fn publish_drift_events(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut queue = self.event_queue.lock().await;
        
        if queue.is_empty() {
            return Ok(());
        }
        
        let events_to_publish: Vec<DriftEvent> = queue.drain(..).collect();
        drop(queue); // Release lock early
        
        // Send events to prediction API
        let client = reqwest::Client::new();
        let endpoint = format!("{}/api/v1/drift/events", self.config.prediction_api_url);
        
        for event in events_to_publish {
            match client.post(&endpoint)
                .json(&event)
                .send()
                .await {
                Ok(response) => {
                    if response.status().is_success() {
                        debug!("Published drift event: {}", event.id);
                    } else {
                        warn!("Failed to publish drift event: {}", response.status());
                    }
                }
                Err(e) => {
                    error!("Error publishing drift event: {}", e);
                }
            }
            
            // Update metrics
            let mut metrics_map = self.drift_metrics.write().await;
            if let Some(metrics) = metrics_map.get_mut(&event.resource_id) {
                metrics.drift_history.push_back(event.clone());
                metrics.total_drifts += 1;
                metrics.last_drift = Some(event.detected_at);
                
                // Keep history size limited
                while metrics.drift_history.len() > 100 {
                    metrics.drift_history.pop_front();
                }
            }
        }
        
        Ok(())
    }

    /// Get drift statistics for monitoring
    pub async fn get_drift_stats(&self) -> DriftStatistics {
        let metrics = self.drift_metrics.read().await;
        let snapshots = self.resource_snapshots.read().await;
        
        let total_resources = snapshots.len();
        let drifting_resources = metrics.values()
            .filter(|m| m.velocity_per_day > self.config.velocity_threshold)
            .count();
        
        let high_risk_resources = metrics.values()
            .filter(|m| m.risk_score > 0.7)
            .count();
        
        let average_velocity = if !metrics.is_empty() {
            metrics.values().map(|m| m.velocity_per_day).sum::<f64>() / metrics.len() as f64
        } else {
            0.0
        };
        
        DriftStatistics {
            total_resources_monitored: total_resources,
            actively_drifting: drifting_resources,
            high_risk_count: high_risk_resources,
            average_drift_velocity: average_velocity,
            last_scan: Utc::now(),
        }
    }
}

/// Drift statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftStatistics {
    pub total_resources_monitored: usize,
    pub actively_drifting: usize,
    pub high_risk_count: usize,
    pub average_drift_velocity: f64,
    pub last_scan: DateTime<Utc>,
}

// Implement Clone for DriftDetector (required for Arc)
impl Clone for DriftDetector {
    fn clone(&self) -> Self {
        Self {
            azure_client: self.azure_client.clone(),
            config: self.config.clone(),
            resource_snapshots: self.resource_snapshots.clone(),
            drift_metrics: self.drift_metrics.clone(),
            event_queue: self.event_queue.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_drift_detection() {
        // Test drift detection logic
        let config = DriftDetectorConfig::default();
        // Add test implementation
    }
    
    #[tokio::test]
    async fn test_drift_metrics_calculation() {
        // Test metrics calculation
        // Add test implementation
    }
}