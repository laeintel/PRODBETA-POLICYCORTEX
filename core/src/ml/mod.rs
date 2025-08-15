// ML Module for Predictive Compliance and Risk Analysis
pub mod predictive_compliance;
pub mod risk_scoring;
pub mod pattern_analysis;
pub mod drift_detector;
pub mod natural_language;
pub mod graph_neural_network;
pub mod correlation_engine;

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationPrediction {
    pub id: Uuid,
    pub resource_id: String,
    pub resource_type: String,
    pub policy_id: String,
    pub policy_name: String,
    pub prediction_time: DateTime<Utc>,
    pub violation_time: DateTime<Utc>,
    pub confidence_score: f64,
    pub risk_level: RiskLevel,
    pub business_impact: BusinessImpact,
    pub remediation_suggestions: Vec<RemediationSuggestion>,
    pub drift_indicators: Vec<DriftIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    pub financial_impact: f64,
    pub compliance_impact: String,
    pub operational_impact: String,
    pub security_impact: String,
    pub affected_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationSuggestion {
    pub action: String,
    pub description: String,
    pub automated: bool,
    pub arm_template: Option<String>,
    pub estimated_time: String,
    pub success_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftIndicator {
    pub property: String,
    pub current_value: String,
    pub expected_value: String,
    pub drift_rate: f64,
    pub time_to_violation: i64, // hours
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_type: String,
    pub version: String,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub training_date: DateTime<Utc>,
    pub evaluation_date: DateTime<Utc>,
}

pub trait PredictiveModel {
    fn predict(&self, resource: &serde_json::Value) -> Result<ViolationPrediction, String>;
    fn train(&mut self, training_data: Vec<serde_json::Value>) -> Result<ModelMetrics, String>;
    fn evaluate(&self, test_data: Vec<serde_json::Value>) -> Result<ModelMetrics, String>;
}