// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Anomaly Detection System
// Detects unusual patterns in resource configurations and usage

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Anomaly detection system using multiple techniques
pub struct AnomalyDetector {
    isolation_forest: IsolationForest,
    statistical_detector: StatisticalDetector,
    pattern_detector: PatternBasedDetector,
    threshold: f64,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            isolation_forest: IsolationForest::new(100, 256),
            statistical_detector: StatisticalDetector::new(),
            pattern_detector: PatternBasedDetector::new(),
            threshold: 0.7,
        }
    }
    
    /// Detect anomalies in resource metrics
    pub fn detect_anomalies(&self, metrics: &[ResourceMetrics]) -> Vec<AnomalyResult> {
        let mut results = Vec::new();
        
        for metric in metrics {
            // Convert metrics to feature vector
            let features = self.extract_features(metric);
            
            // Run multiple detection methods
            let isolation_score = self.isolation_forest.anomaly_score(&features);
            let statistical_score = self.statistical_detector.detect(&features);
            let pattern_score = self.pattern_detector.detect(metric);
            
            // Combine scores (weighted average)
            let combined_score = (isolation_score * 0.4) + 
                               (statistical_score * 0.3) + 
                               (pattern_score * 0.3);
            
            if combined_score > self.threshold {
                results.push(AnomalyResult {
                    resource_id: metric.resource_id.clone(),
                    anomaly_type: self.classify_anomaly(metric, combined_score),
                    score: combined_score,
                    confidence: self.calculate_confidence(isolation_score, statistical_score, pattern_score),
                    detected_at: Utc::now(),
                    description: self.generate_description(metric, combined_score),
                    severity: self.calculate_severity(combined_score),
                    recommended_action: self.recommend_action(metric, combined_score),
                });
            }
        }
        
        results
    }
    
    fn extract_features(&self, metrics: &ResourceMetrics) -> Vec<f64> {
        vec![
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.disk_io,
            metrics.network_in,
            metrics.network_out,
            metrics.request_rate,
            metrics.error_rate,
            metrics.latency,
        ]
    }
    
    fn classify_anomaly(&self, metrics: &ResourceMetrics, score: f64) -> AnomalyType {
        if metrics.error_rate > 0.1 {
            AnomalyType::ErrorSpike
        } else if metrics.cpu_usage > 90.0 || metrics.memory_usage > 90.0 {
            AnomalyType::ResourceExhaustion
        } else if metrics.request_rate > metrics.baseline_request_rate * 3.0 {
            AnomalyType::TrafficSpike
        } else if score > 0.9 {
            AnomalyType::ConfigurationDrift
        } else {
            AnomalyType::Unknown
        }
    }
    
    fn calculate_confidence(&self, iso_score: f64, stat_score: f64, pattern_score: f64) -> f64 {
        // Higher confidence when multiple detectors agree
        let agreement = ((iso_score - stat_score).abs() < 0.2) as u32 +
                       ((stat_score - pattern_score).abs() < 0.2) as u32 +
                       ((iso_score - pattern_score).abs() < 0.2) as u32;
        
        match agreement {
            3 => 0.95, // All detectors strongly agree
            2 => 0.85, // Two detectors agree
            1 => 0.70, // Some agreement
            _ => 0.60, // Detectors disagree
        }
    }
    
    fn generate_description(&self, metrics: &ResourceMetrics, score: f64) -> String {
        if metrics.error_rate > 0.1 {
            format!("High error rate detected: {:.1}%", metrics.error_rate * 100.0)
        } else if metrics.cpu_usage > 90.0 {
            format!("CPU usage critically high: {:.1}%", metrics.cpu_usage)
        } else if score > 0.9 {
            "Significant deviation from normal behavior patterns detected".to_string()
        } else {
            "Unusual activity pattern detected".to_string()
        }
    }
    
    fn calculate_severity(&self, score: f64) -> Severity {
        if score > 0.9 {
            Severity::Critical
        } else if score > 0.8 {
            Severity::High
        } else if score > 0.7 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }
    
    fn recommend_action(&self, metrics: &ResourceMetrics, score: f64) -> String {
        if metrics.error_rate > 0.1 {
            "Investigate application errors and check logs for root cause".to_string()
        } else if metrics.cpu_usage > 90.0 || metrics.memory_usage > 90.0 {
            "Consider scaling up resources or optimizing application".to_string()
        } else if score > 0.9 {
            "Review recent configuration changes and validate against baseline".to_string()
        } else {
            "Monitor closely and investigate if pattern persists".to_string()
        }
    }
}

/// Isolation Forest for anomaly detection
pub struct IsolationForest {
    num_trees: usize,
    sample_size: usize,
    trees: Vec<IsolationTree>,
}

impl IsolationForest {
    pub fn new(num_trees: usize, sample_size: usize) -> Self {
        Self {
            num_trees,
            sample_size,
            trees: Vec::new(),
        }
    }
    
    pub fn anomaly_score(&self, features: &[f64]) -> f64 {
        // Simplified isolation forest scoring
        // In production, would use actual tree traversal
        let mut score: f64 = 0.0;
        
        for feature in features {
            if *feature > 80.0 || *feature < 5.0 {
                score += 0.2;
            }
        }
        
        score.min(1.0)
    }
}

/// Isolation tree node
struct IsolationTree {
    split_feature: usize,
    split_value: f64,
    left: Option<Box<IsolationTree>>,
    right: Option<Box<IsolationTree>>,
}

/// Statistical anomaly detector
pub struct StatisticalDetector {
    mean: Vec<f64>,
    std_dev: Vec<f64>,
    z_threshold: f64,
}

impl StatisticalDetector {
    pub fn new() -> Self {
        Self {
            mean: vec![50.0, 50.0, 30.0, 100.0, 100.0, 1000.0, 0.01, 50.0],
            std_dev: vec![20.0, 20.0, 15.0, 50.0, 50.0, 500.0, 0.02, 25.0],
            z_threshold: 3.0,
        }
    }
    
    pub fn detect(&self, features: &[f64]) -> f64 {
        let mut max_z_score: f64 = 0.0;
        
        for (i, &value) in features.iter().enumerate() {
            if i < self.mean.len() {
                let z_score = ((value - self.mean[i]) / self.std_dev[i]).abs();
                max_z_score = max_z_score.max(z_score);
            }
        }
        
        (max_z_score / self.z_threshold).min(1.0)
    }
    
    pub fn update_statistics(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }
        
        let num_features = data[0].len();
        self.mean = vec![0.0; num_features];
        self.std_dev = vec![0.0; num_features];
        
        // Calculate mean
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                self.mean[i] += value;
            }
        }
        
        for i in 0..num_features {
            self.mean[i] /= data.len() as f64;
        }
        
        // Calculate standard deviation
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                self.std_dev[i] += (value - self.mean[i]).powi(2);
            }
        }
        
        for i in 0..num_features {
            self.std_dev[i] = (self.std_dev[i] / data.len() as f64).sqrt();
        }
    }
}

/// Pattern-based anomaly detector
pub struct PatternBasedDetector {
    known_patterns: Vec<Pattern>,
}

impl PatternBasedDetector {
    pub fn new() -> Self {
        Self {
            known_patterns: Self::initialize_patterns(),
        }
    }
    
    fn initialize_patterns() -> Vec<Pattern> {
        vec![
            Pattern {
                name: "DDoS Attack".to_string(),
                indicators: vec![
                    Indicator::HighRequestRate(10000.0),
                    Indicator::HighErrorRate(0.3),
                ],
            },
            Pattern {
                name: "Memory Leak".to_string(),
                indicators: vec![
                    Indicator::IncreasingMemory,
                    Indicator::LowCPU,
                ],
            },
            Pattern {
                name: "Resource Starvation".to_string(),
                indicators: vec![
                    Indicator::HighCPU,
                    Indicator::HighMemory,
                    Indicator::HighLatency,
                ],
            },
        ]
    }
    
    pub fn detect(&self, metrics: &ResourceMetrics) -> f64 {
        let mut max_score: f64 = 0.0;
        
        for pattern in &self.known_patterns {
            let score = pattern.match_score(metrics);
            max_score = max_score.max(score);
        }
        
        max_score
    }
}

/// Pattern definition
struct Pattern {
    name: String,
    indicators: Vec<Indicator>,
}

impl Pattern {
    fn match_score(&self, metrics: &ResourceMetrics) -> f64 {
        let mut matched = 0;
        
        for indicator in &self.indicators {
            if indicator.matches(metrics) {
                matched += 1;
            }
        }
        
        matched as f64 / self.indicators.len() as f64
    }
}

/// Pattern indicator
enum Indicator {
    HighCPU,
    HighMemory,
    HighLatency,
    HighRequestRate(f64),
    HighErrorRate(f64),
    IncreasingMemory,
    LowCPU,
}

impl Indicator {
    fn matches(&self, metrics: &ResourceMetrics) -> bool {
        match self {
            Indicator::HighCPU => metrics.cpu_usage > 80.0,
            Indicator::HighMemory => metrics.memory_usage > 80.0,
            Indicator::HighLatency => metrics.latency > 1000.0,
            Indicator::HighRequestRate(threshold) => metrics.request_rate > *threshold,
            Indicator::HighErrorRate(threshold) => metrics.error_rate > *threshold,
            Indicator::IncreasingMemory => metrics.memory_trend > 0.1,
            Indicator::LowCPU => metrics.cpu_usage < 20.0,
        }
    }
}

/// Resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub resource_id: String,
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_io: f64,
    pub network_in: f64,
    pub network_out: f64,
    pub request_rate: f64,
    pub error_rate: f64,
    pub latency: f64,
    pub baseline_request_rate: f64,
    pub memory_trend: f64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub resource_id: String,
    pub anomaly_type: AnomalyType,
    pub score: f64,
    pub confidence: f64,
    pub detected_at: DateTime<Utc>,
    pub description: String,
    pub severity: Severity,
    pub recommended_action: String,
}

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    ErrorSpike,
    ResourceExhaustion,
    TrafficSpike,
    ConfigurationDrift,
    SecurityThreat,
    PerformanceDegradation,
    Unknown,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

/// Anomaly statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyStatistics {
    pub total_anomalies: usize,
    pub by_type: HashMap<String, usize>,
    pub by_severity: HashMap<String, usize>,
    pub trend: String,
    pub most_affected_resources: Vec<String>,
}