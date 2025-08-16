// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use super::*;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

pub struct RiskScoringEngine {
    weights: RiskWeights,
    thresholds: RiskThresholds,
    historical_data: HashMap<String, Vec<RiskEvent>>,
}

#[derive(Debug, Clone)]
struct RiskWeights {
    compliance_weight: f64,
    security_weight: f64,
    cost_weight: f64,
    operational_weight: f64,
    time_decay_factor: f64,
}

#[derive(Debug, Clone)]
struct RiskThresholds {
    critical: f64,
    high: f64,
    medium: f64,
    low: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RiskEvent {
    resource_id: String,
    event_type: String,
    severity: f64,
    timestamp: DateTime<Utc>,
    resolved: bool,
}

impl Default for RiskWeights {
    fn default() -> Self {
        Self {
            compliance_weight: 0.35,
            security_weight: 0.35,
            cost_weight: 0.15,
            operational_weight: 0.15,
            time_decay_factor: 0.95,
        }
    }
}

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            critical: 0.8,
            high: 0.6,
            medium: 0.4,
            low: 0.2,
        }
    }
}

impl RiskScoringEngine {
    pub fn new() -> Self {
        Self {
            weights: RiskWeights::default(),
            thresholds: RiskThresholds::default(),
            historical_data: HashMap::new(),
        }
    }

    pub fn calculate_risk_score(&self, prediction: &ViolationPrediction) -> (f64, RiskLevel) {
        let mut score = 0.0;
        
        // Base score from confidence
        score += prediction.confidence_score * 0.3;
        
        // Compliance risk component
        let compliance_risk = self.calculate_compliance_risk(prediction);
        score += compliance_risk * self.weights.compliance_weight;
        
        // Security risk component
        let security_risk = self.calculate_security_risk(prediction);
        score += security_risk * self.weights.security_weight;
        
        // Cost risk component
        let cost_risk = self.calculate_cost_risk(prediction);
        score += cost_risk * self.weights.cost_weight;
        
        // Operational risk component
        let operational_risk = self.calculate_operational_risk(prediction);
        score += operational_risk * self.weights.operational_weight;
        
        // Apply time decay (urgency factor)
        let hours_to_violation = (prediction.violation_time - prediction.prediction_time).num_hours() as f64;
        let urgency_factor = 1.0 / (1.0 + (hours_to_violation / 24.0).exp());
        score *= (1.0 + urgency_factor);
        
        // Normalize score to 0-1 range
        score = score.min(1.0).max(0.0);
        
        // Determine risk level
        let risk_level = if score >= self.thresholds.critical {
            RiskLevel::Critical
        } else if score >= self.thresholds.high {
            RiskLevel::High
        } else if score >= self.thresholds.medium {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        (score, risk_level)
    }

    fn calculate_compliance_risk(&self, prediction: &ViolationPrediction) -> f64 {
        // Evaluate compliance impact
        let mut risk = 0.0;
        
        // Check for regulatory framework impacts
        if prediction.business_impact.compliance_impact.contains("SOC2") {
            risk += 0.8;
        }
        if prediction.business_impact.compliance_impact.contains("HIPAA") {
            risk += 0.9;
        }
        if prediction.business_impact.compliance_impact.contains("GDPR") {
            risk += 0.85;
        }
        if prediction.business_impact.compliance_impact.contains("ISO") {
            risk += 0.7;
        }
        
        // Consider financial penalties
        if prediction.business_impact.financial_impact > 100000.0 {
            risk += 0.9;
        } else if prediction.business_impact.financial_impact > 50000.0 {
            risk += 0.7;
        } else if prediction.business_impact.financial_impact > 10000.0 {
            risk += 0.5;
        }
        
        f64::min(risk, 1.0)
    }

    fn calculate_security_risk(&self, prediction: &ViolationPrediction) -> f64 {
        let mut risk = 0.0;
        
        // Parse security impact
        if prediction.business_impact.security_impact.contains("Critical") {
            risk += 0.95;
        } else if prediction.business_impact.security_impact.contains("High") {
            risk += 0.75;
        } else if prediction.business_impact.security_impact.contains("Medium") {
            risk += 0.5;
        } else if prediction.business_impact.security_impact.contains("Low") {
            risk += 0.25;
        }
        
        // Check for specific security concerns
        for indicator in &prediction.drift_indicators {
            match indicator.property.as_str() {
                "encryption.status" | "encryption" => {
                    if indicator.current_value.contains("Disabled") {
                        risk += 0.8;
                    }
                },
                "publicNetworkAccess" | "networkAccess" => {
                    if indicator.current_value.contains("Enabled") {
                        risk += 0.6;
                    }
                },
                "authentication" | "mfa" => {
                    if indicator.current_value.contains("Disabled") {
                        risk += 0.7;
                    }
                },
                _ => {}
            }
        }
        
        f64::min(risk, 1.0)
    }

    fn calculate_cost_risk(&self, prediction: &ViolationPrediction) -> f64 {
        let mut risk = 0.0;
        
        // Direct financial impact
        let financial_impact = prediction.business_impact.financial_impact;
        if financial_impact > 500000.0 {
            risk = 1.0;
        } else if financial_impact > 100000.0 {
            risk = 0.8;
        } else if financial_impact > 50000.0 {
            risk = 0.6;
        } else if financial_impact > 10000.0 {
            risk = 0.4;
        } else {
            risk = financial_impact / 25000.0; // Linear scale for smaller amounts
        }
        
        f64::min(risk, 1.0)
    }

    fn calculate_operational_risk(&self, prediction: &ViolationPrediction) -> f64 {
        let mut risk = 0.0;
        
        // Parse operational impact
        if prediction.business_impact.operational_impact.contains("Critical") {
            risk += 0.9;
        } else if prediction.business_impact.operational_impact.contains("High") {
            risk += 0.7;
        } else if prediction.business_impact.operational_impact.contains("Medium") {
            risk += 0.5;
        } else if prediction.business_impact.operational_impact.contains("Low") {
            risk += 0.3;
        }
        
        // Consider number of affected resources
        let affected_count = prediction.business_impact.affected_resources.len();
        if affected_count > 100 {
            risk += 0.8;
        } else if affected_count > 50 {
            risk += 0.6;
        } else if affected_count > 10 {
            risk += 0.4;
        } else if affected_count > 5 {
            risk += 0.2;
        }
        
        f64::min(risk, 1.0)
    }

    pub fn prioritize_predictions(&self, predictions: &mut Vec<ViolationPrediction>) {
        // Calculate risk scores for all predictions
        let mut scored_predictions: Vec<(f64, usize)> = predictions
            .iter()
            .enumerate()
            .map(|(idx, pred)| {
                let (score, _) = self.calculate_risk_score(pred);
                (score, idx)
            })
            .collect();
        
        // Sort by risk score (highest first)
        scored_predictions.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        // Reorder predictions based on risk score
        let mut prioritized = Vec::new();
        for (_, idx) in scored_predictions {
            prioritized.push(predictions[idx].clone());
        }
        
        *predictions = prioritized;
    }

    pub fn get_risk_summary(&self, predictions: &[ViolationPrediction]) -> RiskSummary {
        let mut critical_count = 0;
        let mut high_count = 0;
        let mut medium_count = 0;
        let mut low_count = 0;
        let mut total_financial_impact = 0.0;
        
        for prediction in predictions {
            let (_, risk_level) = self.calculate_risk_score(prediction);
            match risk_level {
                RiskLevel::Critical => critical_count += 1,
                RiskLevel::High => high_count += 1,
                RiskLevel::Medium => medium_count += 1,
                RiskLevel::Low => low_count += 1,
            }
            total_financial_impact += prediction.business_impact.financial_impact;
        }
        
        RiskSummary {
            critical_risks: critical_count,
            high_risks: high_count,
            medium_risks: medium_count,
            low_risks: low_count,
            total_financial_impact,
            risk_score: self.calculate_overall_risk_score(predictions),
        }
    }

    fn calculate_overall_risk_score(&self, predictions: &[ViolationPrediction]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }
        
        let total_score: f64 = predictions
            .iter()
            .map(|p| self.calculate_risk_score(p).0)
            .sum();
        
        // Use weighted average with emphasis on highest risks
        let max_score = predictions
            .iter()
            .map(|p| self.calculate_risk_score(p).0)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        (total_score / predictions.len() as f64) * 0.6 + max_score * 0.4
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSummary {
    pub critical_risks: usize,
    pub high_risks: usize,
    pub medium_risks: usize,
    pub low_risks: usize,
    pub total_financial_impact: f64,
    pub risk_score: f64,
}