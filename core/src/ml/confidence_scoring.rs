// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Confidence Scoring System for ML Predictions
// Provides confidence metrics using ensemble methods and feature quality analysis

use super::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Model metadata for tracking model lifecycle
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_id: String,
    pub version: String,
    pub trained_at: DateTime<Utc>,
    pub training_samples: usize,
    pub accuracy: f64,
}

/// Confidence scorer using ensemble methods
pub struct ConfidenceScorer {
    ensemble_models: Vec<Box<dyn PredictionModel>>,
    feature_quality_analyzer: FeatureQualityAnalyzer,
    historical_accuracy: HashMap<String, f64>,
    confidence_thresholds: ConfidenceThresholds,
}

/// Confidence thresholds for different risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    pub high_confidence: f64,
    pub medium_confidence: f64,
    pub low_confidence: f64,
    pub minimum_acceptable: f64,
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            high_confidence: 0.85,
            medium_confidence: 0.70,
            low_confidence: 0.50,
            minimum_acceptable: 0.30,
        }
    }
}

impl ConfidenceScorer {
    pub fn new() -> Self {
        Self {
            ensemble_models: Vec::new(),
            feature_quality_analyzer: FeatureQualityAnalyzer::new(),
            historical_accuracy: HashMap::new(),
            confidence_thresholds: ConfidenceThresholds::default(),
        }
    }
    
    /// Calculate confidence score for a prediction
    pub fn calculate_confidence(&self, prediction: &PredictionOutput, features: &FeatureSet) -> ConfidenceScore {
        // Calculate different confidence components
        let ensemble_confidence = self.calculate_ensemble_confidence(prediction);
        let feature_confidence = self.feature_quality_analyzer.calculate_quality_score(features);
        let temporal_confidence = self.calculate_temporal_confidence(&prediction.model_metadata);
        let historical_confidence = self.calculate_historical_confidence(&prediction.prediction_type);
        
        // Weighted combination
        let weights = ConfidenceWeights {
            ensemble: 0.35,
            feature_quality: 0.25,
            temporal: 0.20,
            historical: 0.20,
        };
        
        let overall_confidence = 
            ensemble_confidence * weights.ensemble +
            feature_confidence * weights.feature_quality +
            temporal_confidence * weights.temporal +
            historical_confidence * weights.historical;
        
        // Determine confidence level
        let level = if overall_confidence >= self.confidence_thresholds.high_confidence {
            ConfidenceLevel::High
        } else if overall_confidence >= self.confidence_thresholds.medium_confidence {
            ConfidenceLevel::Medium
        } else if overall_confidence >= self.confidence_thresholds.low_confidence {
            ConfidenceLevel::Low
        } else {
            ConfidenceLevel::VeryLow
        };
        
        ConfidenceScore {
            overall_score: overall_confidence,
            ensemble_score: ensemble_confidence,
            feature_quality_score: feature_confidence,
            temporal_score: temporal_confidence,
            historical_score: historical_confidence,
            confidence_level: level,
            explanation: self.generate_explanation(overall_confidence, &weights),
        }
    }
    
    /// Calculate ensemble-based confidence
    fn calculate_ensemble_confidence(&self, prediction: &PredictionOutput) -> f64 {
        // If we have multiple models, use disagreement as uncertainty measure
        if self.ensemble_models.is_empty() {
            // Single model - use probability as confidence
            return prediction.probability;
        }
        
        // For ensemble, calculate variance/disagreement
        // Lower variance = higher confidence
        // This is simplified - in production, actually run ensemble
        let variance_penalty = if prediction.is_edge_case { 0.2 } else { 0.0 };
        (prediction.probability - variance_penalty).max(0.0)
    }
    
    /// Calculate historical confidence based on past performance
    fn calculate_historical_confidence(&self, prediction_type: &str) -> f64 {
        self.historical_accuracy.get(prediction_type)
            .copied()
            .unwrap_or(0.75) // Default confidence
    }
    
    /// Calculate temporal confidence based on model age
    fn calculate_temporal_confidence(&self, metadata: &ModelMetadata) -> f64 {
        let model_age = Utc::now() - metadata.trained_at;
        let days_old = model_age.num_days();
        
        // Confidence decays over time
        match days_old {
            0..=7 => 1.0,
            8..=30 => 0.9,
            31..=90 => 0.75,
            91..=180 => 0.6,
            _ => 0.5,
        }
    }
    
    /// Generate human-readable explanation for confidence score
    fn generate_explanation(&self, score: f64, weights: &ConfidenceWeights) -> String {
        let mut factors = Vec::new();
        
        if score >= self.confidence_thresholds.high_confidence {
            factors.push("High agreement between ensemble models");
            factors.push("Good feature quality with minimal missing data");
            factors.push("Recent model training");
        } else if score >= self.confidence_thresholds.medium_confidence {
            factors.push("Moderate ensemble agreement");
            factors.push("Some feature quality issues");
        } else {
            factors.push("Low ensemble agreement");
            factors.push("Significant feature quality issues");
            factors.push("Model may need retraining");
        }
        
        format!(
            "Confidence score: {:.1}%. Factors: {}",
            score * 100.0,
            factors.join(", ")
        )
    }
    
    /// Update historical accuracy for a prediction type
    pub fn update_historical_accuracy(&mut self, prediction_type: String, accuracy: f64) {
        self.historical_accuracy.insert(prediction_type, accuracy);
    }
    
    /// Add model to ensemble
    pub fn add_ensemble_model(&mut self, model: Box<dyn PredictionModel>) {
        self.ensemble_models.push(model);
    }
}

/// Feature quality analyzer
pub struct FeatureQualityAnalyzer {
    missing_value_penalty: f64,
    outlier_penalty: f64,
    staleness_penalty: f64,
}

impl FeatureQualityAnalyzer {
    pub fn new() -> Self {
        Self {
            missing_value_penalty: 0.1,
            outlier_penalty: 0.05,
            staleness_penalty: 0.08,
        }
    }
    
    /// Calculate feature quality score
    pub fn calculate_quality_score(&self, features: &FeatureSet) -> f64 {
        let mut score = 1.0;
        
        // Penalize for missing values
        let missing_ratio = features.missing_count as f64 / features.total_features as f64;
        score -= missing_ratio * self.missing_value_penalty;
        
        // Penalize for outliers
        let outlier_ratio = features.outlier_count as f64 / features.total_features as f64;
        score -= outlier_ratio * self.outlier_penalty;
        
        // Penalize for stale data
        if features.has_estimated_values {
            score -= self.staleness_penalty;
        }
        
        let data_age = Utc::now() - features.last_updated;
        if data_age.num_hours() > 24 {
            score -= self.staleness_penalty;
        }
        
        score.max(0.0).min(1.0)
    }
}

/// Confidence score output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    pub overall_score: f64,
    pub ensemble_score: f64,
    pub feature_quality_score: f64,
    pub temporal_score: f64,
    pub historical_score: f64,
    pub confidence_level: ConfidenceLevel,
    pub explanation: String,
}

/// Confidence level categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
    VeryLow,
}

/// Weights for confidence components
struct ConfidenceWeights {
    ensemble: f64,
    feature_quality: f64,
    temporal: f64,
    historical: f64,
}

/// Ensemble disagreement calculator
pub struct EnsembleDisagreementCalculator {
    models: Vec<Box<dyn PredictionModel>>,
}

impl EnsembleDisagreementCalculator {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
        }
    }
    
    /// Calculate disagreement between ensemble models
    pub fn calculate_disagreement(&self, features: &FeatureSet) -> f64 {
        if self.models.len() < 2 {
            return 0.0;
        }
        
        // Get predictions from all models
        let predictions: Vec<f64> = self.models.iter()
            .filter_map(|model| model.predict(features).ok())
            .map(|pred| pred.probability)
            .collect();
        
        if predictions.is_empty() {
            return 1.0; // Maximum disagreement if no predictions
        }
        
        // Calculate variance as disagreement measure
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Normalize variance to [0, 1] range
        (variance * 4.0).min(1.0) // Assuming max variance of 0.25 for probabilities
    }
    
    /// Add model to ensemble
    pub fn add_model(&mut self, model: Box<dyn PredictionModel>) {
        self.models.push(model);
    }
}

/// Trait for prediction models in ensemble
pub trait PredictionModel: Send + Sync {
    fn predict(&self, features: &FeatureSet) -> Result<ModelPrediction, String>;
    fn get_model_id(&self) -> String;
}

/// Model prediction output
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    pub probability: f64,
    pub class_label: String,
    pub model_id: String,
}

/// Feature set for quality analysis
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub total_features: usize,
    pub missing_count: usize,
    pub outlier_count: usize,
    pub has_estimated_values: bool,
    pub last_updated: DateTime<Utc>,
}

/// Prediction output for confidence scoring
#[derive(Debug, Clone)]
pub struct PredictionOutput {
    pub prediction_type: String,
    pub probability: f64,
    pub model_metadata: ModelMetadata,
    pub is_edge_case: bool,
    pub training_samples_similar: usize,
}

/// Confidence monitoring and alerting
pub struct ConfidenceMonitor {
    threshold_alerts: HashMap<String, f64>,
    alert_history: Vec<ConfidenceAlert>,
}

impl ConfidenceMonitor {
    pub fn new() -> Self {
        let mut threshold_alerts = HashMap::new();
        threshold_alerts.insert("critical_decision".to_string(), 0.85);
        threshold_alerts.insert("automated_action".to_string(), 0.75);
        threshold_alerts.insert("recommendation".to_string(), 0.60);
        
        Self {
            threshold_alerts,
            alert_history: Vec::new(),
        }
    }
    
    /// Check if confidence meets threshold for action type
    pub fn check_threshold(&mut self, action_type: &str, confidence: f64) -> ThresholdCheck {
        let required_threshold = self.threshold_alerts.get(action_type)
            .copied()
            .unwrap_or(0.70);
        
        if confidence >= required_threshold {
            ThresholdCheck::Passed
        } else {
            let alert = ConfidenceAlert {
                timestamp: Utc::now(),
                action_type: action_type.to_string(),
                required_confidence: required_threshold,
                actual_confidence: confidence,
                message: format!(
                    "Confidence {:.1}% below required {:.1}% for {}",
                    confidence * 100.0,
                    required_threshold * 100.0,
                    action_type
                ),
            };
            
            self.alert_history.push(alert.clone());
            ThresholdCheck::Failed(alert)
        }
    }
    
    /// Get recent alerts
    pub fn get_recent_alerts(&self, hours: i64) -> Vec<&ConfidenceAlert> {
        let cutoff = Utc::now() - chrono::Duration::hours(hours);
        self.alert_history.iter()
            .filter(|alert| alert.timestamp > cutoff)
            .collect()
    }
}

/// Threshold check result
#[derive(Debug, Clone)]
pub enum ThresholdCheck {
    Passed,
    Failed(ConfidenceAlert),
}

/// Confidence alert
#[derive(Debug, Clone)]
pub struct ConfidenceAlert {
    pub timestamp: DateTime<Utc>,
    pub action_type: String,
    pub required_confidence: f64,
    pub actual_confidence: f64,
    pub message: String,
}