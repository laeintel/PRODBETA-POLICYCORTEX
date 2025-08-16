// Pattern Library for Common Violations
// Pre-built patterns for detecting and predicting common compliance violations

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Violation pattern library
pub struct ViolationPatternLibrary {
    patterns: HashMap<String, ViolationPattern>,
    pattern_matchers: Vec<Box<dyn PatternMatcher>>,
}

impl ViolationPatternLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            patterns: HashMap::new(),
            pattern_matchers: Vec::new(),
        };
        
        library.initialize_patterns();
        library
    }
    
    fn initialize_patterns(&mut self) {
        // Encryption drift pattern
        self.patterns.insert(
            "encryption_drift".to_string(),
            ViolationPattern {
                id: "pattern_001".to_string(),
                name: "Encryption Drift".to_string(),
                category: PatternCategory::Security,
                indicators: vec![
                    "encryption.enabled changing from true to false".to_string(),
                    "encryption.keySource modified".to_string(),
                    "supportsHttpsTrafficOnly set to false".to_string(),
                ],
                time_to_violation_hours: 24,
                confidence: 0.92,
                severity: ViolationSeverity::High,
                remediation_template: Some("enable-storage-encryption".to_string()),
                detection_rules: vec![
                    DetectionRule {
                        condition: "encryption.enabled == false && previous.encryption.enabled == true".to_string(),
                        weight: 0.9,
                    },
                ],
                historical_frequency: 0.15,
                false_positive_rate: 0.05,
            }
        );
        
        // Network exposure pattern
        self.patterns.insert(
            "network_exposure".to_string(),
            ViolationPattern {
                id: "pattern_002".to_string(),
                name: "Network Exposure".to_string(),
                category: PatternCategory::Network,
                indicators: vec![
                    "networkAcls.defaultAction changed to Allow".to_string(),
                    "ipRules array emptied".to_string(),
                    "publicNetworkAccess enabled".to_string(),
                ],
                time_to_violation_hours: 12,
                confidence: 0.88,
                severity: ViolationSeverity::Critical,
                remediation_template: Some("secure-network-access".to_string()),
                detection_rules: vec![
                    DetectionRule {
                        condition: "networkAcls.defaultAction == 'Allow'".to_string(),
                        weight: 0.8,
                    },
                ],
                historical_frequency: 0.12,
                false_positive_rate: 0.08,
            }
        );
        
        // Backup configuration drift
        self.patterns.insert(
            "backup_drift".to_string(),
            ViolationPattern {
                id: "pattern_003".to_string(),
                name: "Backup Configuration Drift".to_string(),
                category: PatternCategory::DataProtection,
                indicators: vec![
                    "backup.enabled changed to false".to_string(),
                    "backup.retentionDays reduced".to_string(),
                    "backup.frequency changed from daily to weekly".to_string(),
                ],
                time_to_violation_hours: 48,
                confidence: 0.85,
                severity: ViolationSeverity::Medium,
                remediation_template: Some("configure-backup".to_string()),
                detection_rules: vec![
                    DetectionRule {
                        condition: "backup.enabled == false".to_string(),
                        weight: 0.9,
                    },
                    DetectionRule {
                        condition: "backup.retentionDays < 30".to_string(),
                        weight: 0.6,
                    },
                ],
                historical_frequency: 0.20,
                false_positive_rate: 0.10,
            }
        );
        
        // Tag compliance drift
        self.patterns.insert(
            "tag_compliance".to_string(),
            ViolationPattern {
                id: "pattern_004".to_string(),
                name: "Tag Compliance Drift".to_string(),
                category: PatternCategory::Governance,
                indicators: vec![
                    "Required tags missing".to_string(),
                    "Tag values modified incorrectly".to_string(),
                    "Cost center tags removed".to_string(),
                ],
                time_to_violation_hours: 72,
                confidence: 0.90,
                severity: ViolationSeverity::Low,
                remediation_template: Some("apply-required-tags".to_string()),
                detection_rules: vec![
                    DetectionRule {
                        condition: "!tags.contains('Environment')".to_string(),
                        weight: 0.5,
                    },
                    DetectionRule {
                        condition: "!tags.contains('CostCenter')".to_string(),
                        weight: 0.5,
                    },
                ],
                historical_frequency: 0.30,
                false_positive_rate: 0.03,
            }
        );
        
        // Access control weakening
        self.patterns.insert(
            "access_control_weakening".to_string(),
            ViolationPattern {
                id: "pattern_005".to_string(),
                name: "Access Control Weakening".to_string(),
                category: PatternCategory::Identity,
                indicators: vec![
                    "RBAC permissions expanded".to_string(),
                    "Service principal granted owner role".to_string(),
                    "Anonymous access enabled".to_string(),
                ],
                time_to_violation_hours: 6,
                confidence: 0.95,
                severity: ViolationSeverity::Critical,
                remediation_template: Some("restrict-access-control".to_string()),
                detection_rules: vec![
                    DetectionRule {
                        condition: "rbac.role == 'Owner' && rbac.principalType == 'ServicePrincipal'".to_string(),
                        weight: 0.95,
                    },
                ],
                historical_frequency: 0.08,
                false_positive_rate: 0.02,
            }
        );
    }
    
    /// Match patterns against resource configuration
    pub fn match_patterns(&self, resource_config: &ResourceConfiguration) -> Vec<PatternMatch> {
        let mut matches = Vec::new();
        
        for (pattern_id, pattern) in &self.patterns {
            let match_score = self.calculate_match_score(resource_config, pattern);
            
            if match_score > 0.7 {
                matches.push(PatternMatch {
                    pattern_id: pattern_id.clone(),
                    pattern_name: pattern.name.clone(),
                    match_score,
                    confidence: pattern.confidence * match_score,
                    predicted_violation_time: Utc::now() + chrono::Duration::hours(pattern.time_to_violation_hours as i64),
                    matched_indicators: self.get_matched_indicators(resource_config, pattern),
                    recommended_action: pattern.remediation_template.clone(),
                });
            }
        }
        
        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        matches
    }
    
    /// Calculate how well a resource matches a pattern
    fn calculate_match_score(&self, config: &ResourceConfiguration, pattern: &ViolationPattern) -> f64 {
        let mut total_weight = 0.0;
        let mut matched_weight = 0.0;
        
        for rule in &pattern.detection_rules {
            total_weight += rule.weight;
            
            // Simplified rule evaluation (in production, use proper expression evaluator)
            if self.evaluate_rule(config, &rule.condition) {
                matched_weight += rule.weight;
            }
        }
        
        if total_weight > 0.0 {
            matched_weight / total_weight
        } else {
            0.0
        }
    }
    
    /// Simple rule evaluation (placeholder)
    fn evaluate_rule(&self, config: &ResourceConfiguration, condition: &str) -> bool {
        // In production, implement proper expression evaluation
        // For now, simple checks
        if condition.contains("encryption.enabled == false") {
            return !config.properties.get("encryption_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
        }
        
        if condition.contains("networkAcls.defaultAction == 'Allow'") {
            return config.properties.get("network_default_action")
                .and_then(|v| v.as_str())
                .map(|s| s == "Allow")
                .unwrap_or(false);
        }
        
        false
    }
    
    /// Get indicators that matched for a pattern
    fn get_matched_indicators(&self, config: &ResourceConfiguration, pattern: &ViolationPattern) -> Vec<String> {
        let mut matched = Vec::new();
        
        for indicator in &pattern.indicators {
            // Simplified indicator matching
            if indicator.contains("encryption") && 
               !config.properties.get("encryption_enabled").and_then(|v| v.as_bool()).unwrap_or(true) {
                matched.push(indicator.clone());
            }
        }
        
        matched
    }
    
    /// Get pattern by ID
    pub fn get_pattern(&self, pattern_id: &str) -> Option<&ViolationPattern> {
        self.patterns.get(pattern_id)
    }
    
    /// Get patterns by category
    pub fn get_patterns_by_category(&self, category: PatternCategory) -> Vec<&ViolationPattern> {
        self.patterns.values()
            .filter(|p| p.category == category)
            .collect()
    }
    
    /// Get high-confidence patterns
    pub fn get_high_confidence_patterns(&self) -> Vec<&ViolationPattern> {
        self.patterns.values()
            .filter(|p| p.confidence >= 0.9)
            .collect()
    }
    
    /// Update pattern confidence based on feedback
    pub fn update_pattern_confidence(&mut self, pattern_id: &str, was_correct: bool) {
        if let Some(pattern) = self.patterns.get_mut(pattern_id) {
            // Simple confidence update using exponential moving average
            let alpha = 0.1; // Learning rate
            if was_correct {
                pattern.confidence = pattern.confidence * (1.0 - alpha) + alpha;
            } else {
                pattern.confidence = pattern.confidence * (1.0 - alpha);
            }
            
            // Update false positive rate
            if !was_correct {
                pattern.false_positive_rate = pattern.false_positive_rate * (1.0 - alpha) + alpha;
            }
        }
    }
}

/// Violation pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationPattern {
    pub id: String,
    pub name: String,
    pub category: PatternCategory,
    pub indicators: Vec<String>,
    pub time_to_violation_hours: u32,
    pub confidence: f64,
    pub severity: ViolationSeverity,
    pub remediation_template: Option<String>,
    pub detection_rules: Vec<DetectionRule>,
    pub historical_frequency: f64,
    pub false_positive_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternCategory {
    Security,
    Network,
    DataProtection,
    Governance,
    Identity,
    Cost,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionRule {
    pub condition: String,
    pub weight: f64,
}

/// Pattern match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: String,
    pub pattern_name: String,
    pub match_score: f64,
    pub confidence: f64,
    pub predicted_violation_time: DateTime<Utc>,
    pub matched_indicators: Vec<String>,
    pub recommended_action: Option<String>,
}

/// Resource configuration for pattern matching
#[derive(Debug, Clone)]
pub struct ResourceConfiguration {
    pub resource_id: String,
    pub resource_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub tags: HashMap<String, String>,
    pub last_modified: DateTime<Utc>,
}

/// Trait for custom pattern matchers
pub trait PatternMatcher: Send + Sync {
    fn match_pattern(&self, config: &ResourceConfiguration) -> Option<PatternMatch>;
    fn get_pattern_id(&self) -> String;
}