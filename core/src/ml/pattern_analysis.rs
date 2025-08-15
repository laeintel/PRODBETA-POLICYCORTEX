use super::*;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

pub struct PatternAnalyzer {
    pattern_library: HashMap<String, ViolationPattern>,
    time_series_buffer: HashMap<String, VecDeque<TimeSeriesPoint>>,
    anomaly_detector: AnomalyDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub feature_signature: Vec<f64>,
    pub temporal_sequence: Vec<TemporalEvent>,
    pub confidence_threshold: f64,
    pub avg_time_to_violation: i64,
    pub occurrence_count: u32,
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    ConfigurationDrift,
    PeriodicViolation,
    CascadingFailure,
    ResourceExhaustion,
    PolicyConflict,
    ComplianceDecay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimeSeriesPoint {
    timestamp: DateTime<Utc>,
    value: f64,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemporalEvent {
    event_type: String,
    relative_time: i64, // seconds from pattern start
    probability: f64,
    indicators: Vec<String>,
}

struct AnomalyDetector {
    sensitivity: f64,
    window_size: usize,
    z_score_threshold: f64,
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            pattern_library: Self::initialize_pattern_library(),
            time_series_buffer: HashMap::new(),
            anomaly_detector: AnomalyDetector {
                sensitivity: 0.85,
                window_size: 100,
                z_score_threshold: 3.0,
            },
        }
    }

    fn initialize_pattern_library() -> HashMap<String, ViolationPattern> {
        let mut library = HashMap::new();
        
        // Configuration Drift Pattern
        library.insert("drift_001".to_string(), ViolationPattern {
            pattern_id: "drift_001".to_string(),
            pattern_type: PatternType::ConfigurationDrift,
            feature_signature: vec![0.8, 0.6, 0.4, 0.7, 0.9],
            temporal_sequence: vec![
                TemporalEvent {
                    event_type: "initial_change".to_string(),
                    relative_time: 0,
                    probability: 1.0,
                    indicators: vec!["config_modified".to_string()],
                },
                TemporalEvent {
                    event_type: "drift_acceleration".to_string(),
                    relative_time: 3600 * 6, // 6 hours
                    probability: 0.7,
                    indicators: vec!["multiple_changes".to_string()],
                },
                TemporalEvent {
                    event_type: "violation".to_string(),
                    relative_time: 3600 * 24, // 24 hours
                    probability: 0.85,
                    indicators: vec!["policy_breach".to_string()],
                },
            ],
            confidence_threshold: 0.75,
            avg_time_to_violation: 24 * 3600,
            occurrence_count: 156,
            success_rate: 0.89,
        });

        // Periodic Violation Pattern (e.g., certificate expiry)
        library.insert("periodic_001".to_string(), ViolationPattern {
            pattern_id: "periodic_001".to_string(),
            pattern_type: PatternType::PeriodicViolation,
            feature_signature: vec![0.9, 0.2, 0.1, 0.95, 0.3],
            temporal_sequence: vec![
                TemporalEvent {
                    event_type: "approaching_expiry".to_string(),
                    relative_time: -7 * 24 * 3600, // 7 days before
                    probability: 1.0,
                    indicators: vec!["days_to_expiry".to_string()],
                },
                TemporalEvent {
                    event_type: "warning_threshold".to_string(),
                    relative_time: -3 * 24 * 3600, // 3 days before
                    probability: 0.95,
                    indicators: vec!["expiry_warning".to_string()],
                },
                TemporalEvent {
                    event_type: "violation".to_string(),
                    relative_time: 0,
                    probability: 1.0,
                    indicators: vec!["expired".to_string()],
                },
            ],
            confidence_threshold: 0.95,
            avg_time_to_violation: 3 * 24 * 3600,
            occurrence_count: 89,
            success_rate: 0.98,
        });

        // Cascading Failure Pattern
        library.insert("cascade_001".to_string(), ViolationPattern {
            pattern_id: "cascade_001".to_string(),
            pattern_type: PatternType::CascadingFailure,
            feature_signature: vec![0.5, 0.7, 0.9, 0.8, 0.6],
            temporal_sequence: vec![
                TemporalEvent {
                    event_type: "initial_failure".to_string(),
                    relative_time: 0,
                    probability: 1.0,
                    indicators: vec!["service_degraded".to_string()],
                },
                TemporalEvent {
                    event_type: "dependency_impact".to_string(),
                    relative_time: 1800, // 30 minutes
                    probability: 0.8,
                    indicators: vec!["dependent_services_affected".to_string()],
                },
                TemporalEvent {
                    event_type: "system_violation".to_string(),
                    relative_time: 7200, // 2 hours
                    probability: 0.75,
                    indicators: vec!["multiple_policy_breaches".to_string()],
                },
            ],
            confidence_threshold: 0.7,
            avg_time_to_violation: 2 * 3600,
            occurrence_count: 43,
            success_rate: 0.86,
        });

        library
    }

    pub fn detect_patterns(&self, resource_id: &str, time_window: Duration) -> Vec<DetectedPattern> {
        let mut detected_patterns = Vec::new();
        
        // Get time series data for the resource
        if let Some(time_series) = self.time_series_buffer.get(resource_id) {
            // Check each pattern in the library
            for (pattern_id, pattern) in &self.pattern_library {
                if let Some(detection) = self.match_pattern(time_series, pattern, time_window) {
                    detected_patterns.push(detection);
                }
            }
        }
        
        // Sort by confidence
        detected_patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        detected_patterns
    }

    fn match_pattern(
        &self,
        time_series: &VecDeque<TimeSeriesPoint>,
        pattern: &ViolationPattern,
        time_window: Duration,
    ) -> Option<DetectedPattern> {
        // Extract features from time series
        let features = self.extract_features(time_series);
        
        // Calculate similarity with pattern signature
        let similarity = self.calculate_similarity(&features, &pattern.feature_signature);
        
        if similarity > pattern.confidence_threshold {
            // Check temporal sequence
            let sequence_match = self.check_temporal_sequence(time_series, &pattern.temporal_sequence);
            
            if sequence_match > 0.6 {
                let confidence = (similarity + sequence_match) / 2.0;
                
                return Some(DetectedPattern {
                    pattern_id: pattern.pattern_id.clone(),
                    pattern_type: pattern.pattern_type.clone(),
                    confidence,
                    estimated_time_to_violation: pattern.avg_time_to_violation,
                    matched_events: self.get_matched_events(time_series, &pattern.temporal_sequence),
                    recommendation: self.generate_recommendation(pattern),
                });
            }
        }
        
        None
    }

    fn extract_features(&self, time_series: &VecDeque<TimeSeriesPoint>) -> Vec<f64> {
        if time_series.is_empty() {
            return vec![0.0; 5];
        }
        
        let values: Vec<f64> = time_series.iter().map(|p| p.value).collect();
        
        // Calculate statistical features
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        // Calculate trend
        let trend = if values.len() > 1 {
            (values.last().unwrap() - values.first().unwrap()) / values.len() as f64
        } else {
            0.0
        };
        
        // Calculate volatility
        let volatility = if values.len() > 1 {
            let diffs: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
            diffs.iter().sum::<f64>() / diffs.len() as f64
        } else {
            0.0
        };
        
        vec![
            mean.min(1.0).max(0.0),
            std_dev.min(1.0).max(0.0),
            trend.abs().min(1.0),
            volatility.min(1.0).max(0.0),
            (values.len() as f64 / 100.0).min(1.0),
        ]
    }

    fn calculate_similarity(&self, features1: &[f64], features2: &[f64]) -> f64 {
        if features1.len() != features2.len() {
            return 0.0;
        }
        
        // Cosine similarity
        let dot_product: f64 = features1.iter().zip(features2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = features1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = features2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1 * norm2)
    }

    fn check_temporal_sequence(
        &self,
        time_series: &VecDeque<TimeSeriesPoint>,
        sequence: &[TemporalEvent],
    ) -> f64 {
        // Simplified temporal matching
        // In production, this would use dynamic time warping or similar
        0.75 // Placeholder
    }

    fn get_matched_events(
        &self,
        time_series: &VecDeque<TimeSeriesPoint>,
        sequence: &[TemporalEvent],
    ) -> Vec<String> {
        sequence.iter().map(|e| e.event_type.clone()).collect()
    }

    fn generate_recommendation(&self, pattern: &ViolationPattern) -> String {
        match pattern.pattern_type {
            PatternType::ConfigurationDrift => {
                "Configuration is drifting from compliant state. Review and lock down configuration settings.".to_string()
            },
            PatternType::PeriodicViolation => {
                "Periodic violation pattern detected. Set up automated renewal or monitoring.".to_string()
            },
            PatternType::CascadingFailure => {
                "Cascading failure risk detected. Implement circuit breakers and failover mechanisms.".to_string()
            },
            PatternType::ResourceExhaustion => {
                "Resource exhaustion pattern detected. Scale resources or implement usage limits.".to_string()
            },
            PatternType::PolicyConflict => {
                "Policy conflict pattern detected. Review and reconcile conflicting policies.".to_string()
            },
            PatternType::ComplianceDecay => {
                "Compliance decay pattern detected. Implement continuous compliance monitoring.".to_string()
            },
        }
    }

    pub fn add_data_point(&mut self, resource_id: String, value: f64, metadata: HashMap<String, String>) {
        let point = TimeSeriesPoint {
            timestamp: Utc::now(),
            value,
            metadata,
        };
        
        self.time_series_buffer
            .entry(resource_id.clone())
            .or_insert_with(|| VecDeque::with_capacity(1000))
            .push_back(point);
        
        // Keep buffer size limited
        if let Some(buffer) = self.time_series_buffer.get_mut(&resource_id) {
            while buffer.len() > 1000 {
                buffer.pop_front();
            }
        }
    }

    pub fn detect_anomalies(&self, resource_id: &str) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();
        
        if let Some(time_series) = self.time_series_buffer.get(resource_id) {
            let values: Vec<f64> = time_series.iter().map(|p| p.value).collect();
            
            if values.len() >= self.anomaly_detector.window_size {
                // Calculate rolling statistics
                for i in self.anomaly_detector.window_size..values.len() {
                    let window = &values[i - self.anomaly_detector.window_size..i];
                    let mean = window.iter().sum::<f64>() / window.len() as f64;
                    let std_dev = (window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window.len() as f64).sqrt();
                    
                    // Calculate z-score for current value
                    let z_score = if std_dev > 0.0 {
                        (values[i] - mean).abs() / std_dev
                    } else {
                        0.0
                    };
                    
                    if z_score > self.anomaly_detector.z_score_threshold {
                        anomalies.push(Anomaly {
                            timestamp: time_series[i].timestamp,
                            value: values[i],
                            z_score,
                            severity: if z_score > 5.0 { "Critical" } else if z_score > 4.0 { "High" } else { "Medium" }.to_string(),
                        });
                    }
                }
            }
        }
        
        anomalies
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub estimated_time_to_violation: i64,
    pub matched_events: Vec<String>,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub z_score: f64,
    pub severity: String,
}