// Confidence Scoring System for ML Predictions
// Provides confidence metrics using ensemble methods and feature quality analysis

use super::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

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
            minimum_acceptable: 0.40,
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
    pub fn calculate_confidence(
        &self,
        prediction: &PredictionOutput,
        features: &FeatureSet,
    ) -> ConfidenceScore {
        // Calculate ensemble agreement
        let ensemble_confidence = self.calculate_ensemble_agreement(prediction, features);
        
        // Assess feature quality
        let feature_confidence = self.feature_quality_analyzer.assess_quality(features);
        
        // Check historical accuracy for this prediction type
        let historical_confidence = self.get_historical_confidence(&prediction.prediction_type);
        
        // Calculate temporal confidence (how recent is the model)
        let temporal_confidence = self.calculate_temporal_confidence(&prediction.model_metadata);
        
        // Calculate data completeness confidence
        let completeness_confidence = self.calculate_completeness_confidence(features);
        
        // Weighted combination of confidence factors
        let weights = ConfidenceWeights::default();
        let overall_confidence = 
            weights.ensemble * ensemble_confidence +
            weights.feature_quality * feature_confidence +
            weights.historical * historical_confidence +
            weights.temporal * temporal_confidence +
            weights.completeness * completeness_confidence;
        
        // Determine confidence level
        let confidence_level = self.determine_confidence_level(overall_confidence);
        
        // Identify factors affecting confidence
        let confidence_factors = self.identify_confidence_factors(
            ensemble_confidence,
            feature_confidence,
            historical_confidence,
            temporal_confidence,
            completeness_confidence,
        );
        
        ConfidenceScore {
            overall_score: overall_confidence,
            confidence_level,
            ensemble_agreement: ensemble_confidence,
            feature_quality: feature_confidence,
            historical_accuracy: historical_confidence,
            temporal_relevance: temporal_confidence,
            data_completeness: completeness_confidence,
            confidence_factors,
            uncertainty_sources: self.identify_uncertainty_sources(features, prediction),
            recommendation: self.generate_confidence_recommendation(overall_confidence),
        }
    }
    
    /// Calculate agreement among ensemble models
    fn calculate_ensemble_agreement(
        &self,
        prediction: &PredictionOutput,
        features: &FeatureSet,
    ) -> f64 {
        if self.ensemble_models.is_empty() {
            return 0.5; // Default if no ensemble
        }
        
        // Get predictions from all models (simulated)
        let predictions: Vec<f64> = (0..5).map(|i| {
            prediction.probability + (i as f64 - 2.5) * 0.05 // Simulate variance
        }).collect();
        
        // Calculate variance
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Convert variance to confidence (lower variance = higher confidence)
        1.0 - variance.min(1.0)
    }
    
    /// Get historical confidence for prediction type
    fn get_historical_confidence(&self, prediction_type: &str) -> f64 {
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
    
    /// Calculate confidence based on data completeness
    fn calculate_completeness_confidence(&self, features: &FeatureSet) -> f64 {
        let total_features = features.expected_features;
        let available_features = features.available_features;
        let missing_critical = features.missing_critical_features;
        
        let completeness_ratio = available_features as f64 / total_features as f64;
        
        // Penalize heavily for missing critical features
        if missing_critical > 0 {
            completeness_ratio * 0.5
        } else {
            completeness_ratio
        }
    }
    
    /// Determine confidence level from score
    fn determine_confidence_level(&self, score: f64) -> ConfidenceLevel {
        if score >= self.confidence_thresholds.high_confidence {
            ConfidenceLevel::High
        } else if score >= self.confidence_thresholds.medium_confidence {
            ConfidenceLevel::Medium
        } else if score >= self.confidence_thresholds.low_confidence {
            ConfidenceLevel::Low
        } else {
            ConfidenceLevel::VeryLow
        }
    }
    
    /// Identify factors affecting confidence
    fn identify_confidence_factors(
        &self,
        ensemble: f64,
        feature: f64,
        historical: f64,
        temporal: f64,
        completeness: f64,
    ) -> Vec<ConfidenceFactor> {
        let mut factors = Vec::new();
        
        if ensemble > 0.8 {
            factors.push(ConfidenceFactor {
                factor_type: "Ensemble Agreement".to_string(),
                impact: FactorImpact::Positive,
                score: ensemble,
                description: "High agreement among ensemble models".to_string(),
            });
        } else if ensemble < 0.5 {
            factors.push(ConfidenceFactor {
                factor_type: "Ensemble Disagreement".to_string(),
                impact: FactorImpact::Negative,
                score: ensemble,
                description: "Significant disagreement among models".to_string(),
            });
        }
        
        if feature < 0.6 {
            factors.push(ConfidenceFactor {
                factor_type: "Feature Quality".to_string(),
                impact: FactorImpact::Negative,
                score: feature,
                description: "Low quality or noisy input features".to_string(),
            });
        }
        
        if temporal < 0.7 {
            factors.push(ConfidenceFactor {
                factor_type: "Model Age".to_string(),
                impact: FactorImpact::Negative,
                score: temporal,
                description: "Model may be outdated".to_string(),
            });
        }
        
        if completeness < 0.8 {
            factors.push(ConfidenceFactor {
                factor_type: "Data Completeness".to_string(),
                impact: FactorImpact::Negative,
                score: completeness,
                description: "Missing important features".to_string(),
            });
        }
        
        factors
    }
    
    /// Identify sources of uncertainty
    fn identify_uncertainty_sources(
        &self,
        features: &FeatureSet,
        prediction: &PredictionOutput,
    ) -> Vec<UncertaintySource> {
        let mut sources = Vec::new();
        
        // Check for feature uncertainty
        if features.has_estimated_values {
            sources.push(UncertaintySource {
                source_type: UncertaintyType::EstimatedFeatures,
                severity: UncertaintySeverity::Medium,
                description: "Some feature values were estimated".to_string(),
                mitigation: "Collect actual feature values".to_string(),
            });
        }
        
        // Check for edge cases
        if prediction.is_edge_case {
            sources.push(UncertaintySource {
                source_type: UncertaintyType::EdgeCase,
                severity: UncertaintySeverity::High,
                description: "Prediction is for an edge case scenario".to_string(),
                mitigation: "Manual review recommended".to_string(),
            });
        }
        
        // Check for limited training data
        if prediction.training_samples_similar < 100 {
            sources.push(UncertaintySource {
                source_type: UncertaintyType::LimitedTrainingData,
                severity: UncertaintySeverity::Medium,
                description: "Limited similar examples in training data".to_string(),
                mitigation: "Collect more training data for this scenario".to_string(),
            });
        }
        
        sources
    }
    
    /// Generate recommendation based on confidence
    fn generate_confidence_recommendation(&self, confidence: f64) -> String {
        match self.determine_confidence_level(confidence) {
            ConfidenceLevel::High => {
                "High confidence prediction. Safe to automate actions.".to_string()
            }
            ConfidenceLevel::Medium => {
                "Medium confidence. Consider manual review for critical resources.".to_string()
            }
            ConfidenceLevel::Low => {
                "Low confidence. Manual review recommended before taking action.".to_string()
            }
            ConfidenceLevel::VeryLow => {
                "Very low confidence. Do not automate. Requires manual investigation.".to_string()
            }
        }
    }
}

/// Feature quality analyzer
pub struct FeatureQualityAnalyzer {
    quality_metrics: HashMap<String, QualityMetric>,
}

impl FeatureQualityAnalyzer {
    pub fn new() -> Self {
        Self {
            quality_metrics: HashMap::new(),
        }
    }
    
    pub fn assess_quality(&self, features: &FeatureSet) -> f64 {
        let mut quality_scores = Vec::new();
        
        // Check for outliers
        let outlier_score = 1.0 - (features.outlier_count as f64 / features.available_features as f64);
        quality_scores.push(outlier_score);
        
        // Check for missing values
        let completeness_score = features.available_features as f64 / features.expected_features as f64;
        quality_scores.push(completeness_score);
        
        // Check for data freshness
        let freshness_score = self.calculate_freshness_score(&features.last_updated);
        quality_scores.push(freshness_score);
        
        // Average quality score
        quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
    }
    
    fn calculate_freshness_score(&self, last_updated: &DateTime<Utc>) -> f64 {
        let age = Utc::now() - *last_updated;
        let minutes_old = age.num_minutes();
        
        match minutes_old {
            0..=5 => 1.0,
            6..=60 => 0.9,
            61..=1440 => 0.75, // Up to 24 hours
            _ => 0.5,
        }
    }
}

/// Confidence score output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    pub overall_score: f64,
    pub confidence_level: ConfidenceLevel,
    pub ensemble_agreement: f64,
    pub feature_quality: f64,
    pub historical_accuracy: f64,
    pub temporal_relevance: f64,
    pub data_completeness: f64,
    pub confidence_factors: Vec<ConfidenceFactor>,
    pub uncertainty_sources: Vec<UncertaintySource>,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
    VeryLow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceFactor {
    pub factor_type: String,
    pub impact: FactorImpact,
    pub score: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorImpact {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintySource {
    pub source_type: UncertaintyType,
    pub severity: UncertaintySeverity,
    pub description: String,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyType {
    EstimatedFeatures,
    EdgeCase,
    LimitedTrainingData,
    ModelDrift,
    DataQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintySeverity {
    High,
    Medium,
    Low,
}

/// Confidence calculation weights
#[derive(Debug, Clone)]
struct ConfidenceWeights {
    pub ensemble: f64,
    pub feature_quality: f64,
    pub historical: f64,
    pub temporal: f64,
    pub completeness: f64,
}

impl Default for ConfidenceWeights {
    fn default() -> Self {
        Self {
            ensemble: 0.3,
            feature_quality: 0.25,
            historical: 0.2,
            temporal: 0.15,
            completeness: 0.1,
        }
    }
}

/// Feature set for confidence calculation
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub expected_features: usize,
    pub available_features: usize,
    pub missing_critical_features: usize,
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

/// Quality metric for features
#[derive(Debug, Clone)]
struct QualityMetric {
    pub name: String,
    pub score: f64,
    pub last_calculated: DateTime<Utc>,
}

/// Trait for prediction models (used in ensemble)
pub trait PredictionModel: Send + Sync {
    fn predict(&self, features: &[f64]) -> Result<f64, String>;
    fn get_model_id(&self) -> String;
}