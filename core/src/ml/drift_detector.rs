// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use super::*;
use std::collections::HashMap;
use chrono::{DateTime, Duration, Utc};

pub struct DriftDetector {
    baseline_configurations: HashMap<String, BaselineConfig>,
    drift_thresholds: DriftThresholds,
    drift_history: HashMap<String, Vec<DriftEvent>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineConfig {
    resource_id: String,
    configuration: serde_json::Value,
    established_at: DateTime<Utc>,
    compliance_state: ComplianceState,
    critical_properties: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriftThresholds {
    critical_drift: f64,
    major_drift: f64,
    minor_drift: f64,
    time_window: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriftEvent {
    timestamp: DateTime<Utc>,
    property: String,
    old_value: serde_json::Value,
    new_value: serde_json::Value,
    drift_score: f64,
    impact: DriftImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ComplianceState {
    Compliant,
    NonCompliant,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum DriftImpact {
    Critical,
    Major,
    Minor,
    Negligible,
}

impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            critical_drift: 0.8,
            major_drift: 0.6,
            minor_drift: 0.3,
            time_window: Duration::hours(24),
        }
    }
}

impl DriftDetector {
    pub fn new() -> Self {
        Self {
            baseline_configurations: HashMap::new(),
            drift_thresholds: DriftThresholds::default(),
            drift_history: HashMap::new(),
        }
    }

    pub fn establish_baseline(&mut self, resource_id: String, configuration: serde_json::Value) {
        let critical_properties = self.identify_critical_properties(&configuration);
        
        let baseline = BaselineConfig {
            resource_id: resource_id.clone(),
            configuration: configuration.clone(),
            established_at: Utc::now(),
            compliance_state: self.evaluate_compliance(&configuration),
            critical_properties,
        };
        
        self.baseline_configurations.insert(resource_id, baseline);
    }

    fn identify_critical_properties(&self, configuration: &serde_json::Value) -> Vec<String> {
        // Properties that directly affect compliance
        vec![
            "properties.encryption".to_string(),
            "properties.publicNetworkAccess".to_string(),
            "properties.minimumTlsVersion".to_string(),
            "properties.firewallRules".to_string(),
            "properties.networkAcls".to_string(),
            "properties.authentication".to_string(),
            "properties.authorization".to_string(),
            "properties.audit".to_string(),
            "tags.environment".to_string(),
            "tags.compliance".to_string(),
        ]
    }

    fn evaluate_compliance(&self, configuration: &serde_json::Value) -> ComplianceState {
        // Simplified compliance check
        let mut compliant = true;
        
        // Check encryption
        if let Some(encryption) = configuration.pointer("/properties/encryption/status") {
            if encryption == "Disabled" {
                compliant = false;
            }
        }
        
        // Check public network access
        if let Some(public_access) = configuration.pointer("/properties/publicNetworkAccess") {
            if public_access == "Enabled" {
                compliant = false;
            }
        }
        
        // Check TLS version
        if let Some(tls_version) = configuration.pointer("/properties/minimumTlsVersion") {
            if tls_version.as_str().unwrap_or("") < "TLS1_2" {
                compliant = false;
            }
        }
        
        if compliant {
            ComplianceState::Compliant
        } else {
            ComplianceState::NonCompliant
        }
    }

    pub fn detect_drift(&mut self, resource_id: &str, current_config: &serde_json::Value) -> DriftAnalysis {
        let baseline = match self.baseline_configurations.get(resource_id) {
            Some(b) => b,
            None => {
                // No baseline, establish one
                self.establish_baseline(resource_id.to_string(), current_config.clone());
                return DriftAnalysis {
                    resource_id: resource_id.to_string(),
                    total_drift_score: 0.0,
                    drift_velocity: 0.0,
                    critical_drifts: vec![],
                    time_to_violation: None,
                    recommended_actions: vec!["Baseline established. Monitoring for drift.".to_string()],
                };
            }
        };
        
        let drift_events = self.calculate_drift(&baseline.configuration, current_config, &baseline.critical_properties);
        
        // Store drift events
        self.drift_history
            .entry(resource_id.to_string())
            .or_insert_with(Vec::new)
            .extend(drift_events.clone());
        
        // Calculate drift metrics
        let total_drift_score = self.calculate_total_drift_score(&drift_events);
        let drift_velocity = self.calculate_drift_velocity(resource_id);
        let critical_drifts = self.identify_critical_drifts(&drift_events);
        let time_to_violation = self.predict_time_to_violation(total_drift_score, drift_velocity);
        let recommended_actions = self.generate_recommendations(&drift_events, total_drift_score);
        
        DriftAnalysis {
            resource_id: resource_id.to_string(),
            total_drift_score,
            drift_velocity,
            critical_drifts,
            time_to_violation,
            recommended_actions,
        }
    }

    fn calculate_drift(
        &self,
        baseline: &serde_json::Value,
        current: &serde_json::Value,
        critical_properties: &[String],
    ) -> Vec<DriftEvent> {
        let mut drift_events = Vec::new();
        
        for property in critical_properties {
            let baseline_value = baseline.pointer(property);
            let current_value = current.pointer(property);
            
            if baseline_value != current_value {
                let drift_score = self.calculate_property_drift_score(property, baseline_value, current_value);
                let impact = self.determine_impact(drift_score);
                
                drift_events.push(DriftEvent {
                    timestamp: Utc::now(),
                    property: property.clone(),
                    old_value: baseline_value.cloned().unwrap_or(serde_json::Value::Null),
                    new_value: current_value.cloned().unwrap_or(serde_json::Value::Null),
                    drift_score,
                    impact,
                });
            }
        }
        
        drift_events
    }

    fn calculate_property_drift_score(
        &self,
        property: &str,
        baseline_value: Option<&serde_json::Value>,
        current_value: Option<&serde_json::Value>,
    ) -> f64 {
        // Higher scores for more critical properties
        let base_score = match property {
            p if p.contains("encryption") => 0.9,
            p if p.contains("publicNetworkAccess") => 0.85,
            p if p.contains("authentication") => 0.8,
            p if p.contains("minimumTlsVersion") => 0.75,
            p if p.contains("firewallRules") => 0.7,
            p if p.contains("networkAcls") => 0.65,
            _ => 0.5,
        };
        
        // Adjust based on the nature of the change
        let score = match (baseline_value, current_value) {
            (Some(b), Some(c)) if b == "Enabled" && c == "Disabled" => base_score * 1.2,
            (Some(b), Some(c)) if b == "Disabled" && c == "Enabled" => base_score * 0.8,
            (Some(_), None) | (None, Some(_)) => base_score * 1.1,
            _ => base_score,
        };
        f64::min(score, 1.0)
    }

    fn determine_impact(&self, drift_score: f64) -> DriftImpact {
        if drift_score >= self.drift_thresholds.critical_drift {
            DriftImpact::Critical
        } else if drift_score >= self.drift_thresholds.major_drift {
            DriftImpact::Major
        } else if drift_score >= self.drift_thresholds.minor_drift {
            DriftImpact::Minor
        } else {
            DriftImpact::Negligible
        }
    }

    fn calculate_total_drift_score(&self, drift_events: &[DriftEvent]) -> f64 {
        if drift_events.is_empty() {
            return 0.0;
        }
        
        // Weighted sum with emphasis on critical drifts
        let sum: f64 = drift_events.iter().map(|e| {
            match e.impact {
                DriftImpact::Critical => e.drift_score * 2.0,
                DriftImpact::Major => e.drift_score * 1.5,
                DriftImpact::Minor => e.drift_score * 1.0,
                DriftImpact::Negligible => e.drift_score * 0.5,
            }
        }).sum();
        
        (sum / drift_events.len() as f64).min(1.0)
    }

    fn calculate_drift_velocity(&self, resource_id: &str) -> f64 {
        let history = match self.drift_history.get(resource_id) {
            Some(h) => h,
            None => return 0.0,
        };
        
        if history.len() < 2 {
            return 0.0;
        }
        
        // Calculate rate of change over time window
        let now = Utc::now();
        let window_start = now - self.drift_thresholds.time_window;
        
        let recent_events: Vec<&DriftEvent> = history
            .iter()
            .filter(|e| e.timestamp > window_start)
            .collect();
        
        if recent_events.len() < 2 {
            return 0.0;
        }
        
        // Calculate average drift score change per hour
        let time_span = (recent_events.last().unwrap().timestamp - recent_events.first().unwrap().timestamp)
            .num_hours() as f64;
        
        if time_span == 0.0 {
            return 0.0;
        }
        
        let score_change: f64 = recent_events.iter().map(|e| e.drift_score).sum::<f64>() / recent_events.len() as f64;
        
        score_change / time_span
    }

    fn identify_critical_drifts(&self, drift_events: &[DriftEvent]) -> Vec<CriticalDrift> {
        drift_events
            .iter()
            .filter(|e| matches!(e.impact, DriftImpact::Critical | DriftImpact::Major))
            .map(|e| CriticalDrift {
                property: e.property.clone(),
                current_value: e.new_value.clone(),
                required_value: e.old_value.clone(),
                impact_description: self.describe_impact(&e.property, &e.old_value, &e.new_value),
            })
            .collect()
    }

    fn describe_impact(
        &self,
        property: &str,
        old_value: &serde_json::Value,
        new_value: &serde_json::Value,
    ) -> String {
        match property {
            p if p.contains("encryption") => {
                format!("Encryption changed from {} to {}. Data at risk.", old_value, new_value)
            },
            p if p.contains("publicNetworkAccess") => {
                format!("Public network access changed from {} to {}. Security exposure risk.", old_value, new_value)
            },
            p if p.contains("authentication") => {
                format!("Authentication settings modified. Access control may be compromised.", )
            },
            _ => format!("Configuration drift detected in {}", property),
        }
    }

    fn predict_time_to_violation(&self, drift_score: f64, velocity: f64) -> Option<i64> {
        if velocity <= 0.0 {
            return None;
        }
        
        // Calculate time until drift score reaches critical threshold
        let remaining_drift = self.drift_thresholds.critical_drift - drift_score;
        
        if remaining_drift <= 0.0 {
            return Some(0); // Already in violation
        }
        
        let hours_to_violation = (remaining_drift / velocity) as i64;
        
        Some(hours_to_violation.max(0))
    }

    fn generate_recommendations(&self, drift_events: &[DriftEvent], total_drift_score: f64) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if total_drift_score > self.drift_thresholds.critical_drift {
            recommendations.push("URGENT: Critical configuration drift detected. Immediate remediation required.".to_string());
        } else if total_drift_score > self.drift_thresholds.major_drift {
            recommendations.push("WARNING: Significant configuration drift detected. Review and remediate within 24 hours.".to_string());
        }
        
        for event in drift_events {
            match event.property.as_str() {
                p if p.contains("encryption") => {
                    recommendations.push("Enable encryption to maintain compliance.".to_string());
                },
                p if p.contains("publicNetworkAccess") => {
                    recommendations.push("Disable public network access and configure private endpoints.".to_string());
                },
                p if p.contains("minimumTlsVersion") => {
                    recommendations.push("Update TLS version to 1.2 or higher.".to_string());
                },
                _ => {}
            }
        }
        
        if recommendations.is_empty() {
            recommendations.push("Monitor configuration for continued compliance.".to_string());
        }
        
        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAnalysis {
    pub resource_id: String,
    pub total_drift_score: f64,
    pub drift_velocity: f64,
    pub critical_drifts: Vec<CriticalDrift>,
    pub time_to_violation: Option<i64>,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalDrift {
    pub property: String,
    pub current_value: serde_json::Value,
    pub required_value: serde_json::Value,
    pub impact_description: String,
}