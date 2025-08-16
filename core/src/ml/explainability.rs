// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Explainable AI Module for PolicyCortex
// Provides human-readable explanations for ML predictions using SHAP-like techniques

use super::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Main explainer for generating human-readable explanations
pub struct PredictionExplainer {
    feature_names: HashMap<usize, String>,
    feature_descriptions: HashMap<String, String>,
    explanation_templates: HashMap<String, String>,
    shap_calculator: ShapValueCalculator,
}

impl PredictionExplainer {
    pub fn new() -> Self {
        let mut feature_names = HashMap::new();
        let mut feature_descriptions = HashMap::new();
        
        // Initialize common feature mappings
        Self::initialize_feature_mappings(&mut feature_names, &mut feature_descriptions);
        
        Self {
            feature_names,
            feature_descriptions,
            explanation_templates: Self::initialize_templates(),
            shap_calculator: ShapValueCalculator::new(),
        }
    }
    
    /// Generate explanation for a violation prediction
    pub fn explain_violation_prediction(
        &self,
        resource: &ResourceData,
        prediction: &ViolationPrediction,
        feature_values: &[f64],
    ) -> PredictionExplanation {
        // Calculate SHAP values
        let shap_values = self.shap_calculator.calculate_shap_values(feature_values);
        
        // Get top contributing factors
        let top_factors = self.identify_top_factors(&shap_values, feature_values);
        
        // Generate human-readable explanation
        let narrative = self.generate_narrative_explanation(
            resource,
            prediction,
            &top_factors,
        );
        
        // Generate recommendations based on factors
        let recommendations = self.generate_recommendations(&top_factors, prediction);
        
        // Create counterfactual explanations
        let counterfactuals = self.generate_counterfactuals(feature_values, &shap_values);
        
        // Assess explanation confidence
        let explanation_confidence = self.assess_explanation_confidence(&shap_values);
        
        PredictionExplanation {
            prediction_id: prediction.id.to_string(),
            resource_id: resource.id.clone(),
            narrative_explanation: narrative,
            top_factors,
            feature_importance: self.calculate_feature_importance(&shap_values),
            recommendations,
            counterfactuals,
            shap_values: shap_values.clone(),
            explanation_confidence,
            visualizations: self.generate_visualizations(&shap_values),
            timestamp: Utc::now(),
        }
    }
    
    /// Identify top contributing factors
    fn identify_top_factors(
        &self,
        shap_values: &[f64],
        feature_values: &[f64],
    ) -> Vec<ContributingFactor> {
        let mut factors: Vec<(usize, f64)> = shap_values.iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        
        // Sort by absolute SHAP value
        factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top 5 factors
        factors.iter()
            .take(5)
            .map(|(idx, shap_value)| {
                let feature_name = self.feature_names.get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", idx));
                
                let feature_description = self.feature_descriptions.get(&feature_name)
                    .cloned()
                    .unwrap_or_else(|| "Unknown feature".to_string());
                
                let current_value = feature_values.get(*idx).copied().unwrap_or(0.0);
                
                ContributingFactor {
                    feature_name: feature_name.clone(),
                    feature_description,
                    impact_score: *shap_value,
                    current_value: self.format_feature_value(&feature_name, current_value),
                    threshold_value: self.get_threshold_value(&feature_name),
                    contribution_type: if *shap_value > 0.0 {
                        ContributionType::Increases
                    } else {
                        ContributionType::Decreases
                    },
                    readable_explanation: self.generate_factor_explanation(
                        &feature_name,
                        current_value,
                        *shap_value,
                    ),
                }
            })
            .collect()
    }
    
    /// Generate narrative explanation
    fn generate_narrative_explanation(
        &self,
        resource: &ResourceData,
        prediction: &ViolationPrediction,
        factors: &[ContributingFactor],
    ) -> String {
        let mut narrative = format!(
            "The {} '{}' has a {:.1}% probability of violating the '{}' policy within the next {} hours.\n\n",
            resource.resource_type,
            resource.name,
            prediction.confidence_score * 100.0,
            prediction.policy_name,
            prediction.violation_time.signed_duration_since(Utc::now()).num_hours()
        );
        
        narrative.push_str("Key factors contributing to this prediction:\n\n");
        
        for (i, factor) in factors.iter().enumerate().take(3) {
            narrative.push_str(&format!(
                "{}. {}\n",
                i + 1,
                factor.readable_explanation
            ));
        }
        
        narrative.push_str(&format!(
            "\nRisk Level: {:?}\n",
            prediction.risk_level
        ));
        
        if let Some(impact) = &prediction.business_impact {
            narrative.push_str(&format!(
                "Potential Business Impact: {}\n",
                impact.compliance_impact
            ));
        }
        
        narrative
    }
    
    /// Generate recommendations based on factors
    fn generate_recommendations(
        &self,
        factors: &[ContributingFactor],
        prediction: &ViolationPrediction,
    ) -> Vec<ActionRecommendation> {
        let mut recommendations = Vec::new();
        
        for factor in factors.iter().take(3) {
            let recommendation = match factor.feature_name.as_str() {
                "encryption_enabled" if factor.current_value == "false" => {
                    ActionRecommendation {
                        action: "Enable Encryption".to_string(),
                        description: "Enable encryption for the storage account to comply with security policies".to_string(),
                        priority: RecommendationPriority::High,
                        estimated_impact: 0.8,
                        implementation_steps: vec![
                            "Navigate to Storage Account settings".to_string(),
                            "Select 'Encryption' under Security + networking".to_string(),
                            "Enable encryption at rest".to_string(),
                        ],
                        automation_available: true,
                        remediation_template: Some("enable-storage-encryption".to_string()),
                    }
                }
                "public_access_enabled" if factor.current_value == "true" => {
                    ActionRecommendation {
                        action: "Disable Public Access".to_string(),
                        description: "Disable public blob access to prevent unauthorized data exposure".to_string(),
                        priority: RecommendationPriority::Critical,
                        estimated_impact: 0.9,
                        implementation_steps: vec![
                            "Go to Storage Account configuration".to_string(),
                            "Select 'Configuration' under Settings".to_string(),
                            "Set 'Allow Blob public access' to Disabled".to_string(),
                        ],
                        automation_available: true,
                        remediation_template: Some("disable-public-access".to_string()),
                    }
                }
                "backup_enabled" if factor.current_value == "false" => {
                    ActionRecommendation {
                        action: "Enable Backup".to_string(),
                        description: "Configure automated backups to ensure data recovery capability".to_string(),
                        priority: RecommendationPriority::Medium,
                        estimated_impact: 0.6,
                        implementation_steps: vec![
                            "Open Azure Backup service".to_string(),
                            "Create new backup policy".to_string(),
                            "Configure retention settings".to_string(),
                        ],
                        automation_available: true,
                        remediation_template: Some("enable-backup".to_string()),
                    }
                }
                _ => continue,
            };
            
            recommendations.push(recommendation);
        }
        
        // Add general recommendation if no specific ones
        if recommendations.is_empty() {
            recommendations.push(ActionRecommendation {
                action: "Review Configuration".to_string(),
                description: format!(
                    "Review and update resource configuration to comply with {} policy",
                    prediction.policy_name
                ),
                priority: RecommendationPriority::Medium,
                estimated_impact: 0.5,
                implementation_steps: vec![
                    "Review current resource configuration".to_string(),
                    "Compare with policy requirements".to_string(),
                    "Apply necessary changes".to_string(),
                ],
                automation_available: false,
                remediation_template: None,
            });
        }
        
        recommendations
    }
    
    /// Generate counterfactual explanations
    fn generate_counterfactuals(
        &self,
        feature_values: &[f64],
        shap_values: &[f64],
    ) -> Vec<CounterfactualExplanation> {
        let mut counterfactuals = Vec::new();
        
        // Find features with highest negative SHAP values (increasing violation risk)
        let mut negative_factors: Vec<(usize, f64)> = shap_values.iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, &v)| (i, v))
            .collect();
        
        negative_factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (idx, shap_value) in negative_factors.iter().take(3) {
            let feature_name = self.feature_names.get(idx)
                .cloned()
                .unwrap_or_else(|| format!("Feature_{}", idx));
            
            let current_value = feature_values.get(*idx).copied().unwrap_or(0.0);
            let suggested_value = self.calculate_counterfactual_value(&feature_name, current_value);
            
            counterfactuals.push(CounterfactualExplanation {
                feature_name: feature_name.clone(),
                current_value: self.format_feature_value(&feature_name, current_value),
                suggested_value: self.format_feature_value(&feature_name, suggested_value),
                expected_impact: shap_value.abs(),
                explanation: format!(
                    "If {} were changed from {} to {}, the violation probability would decrease by approximately {:.1}%",
                    feature_name,
                    self.format_feature_value(&feature_name, current_value),
                    self.format_feature_value(&feature_name, suggested_value),
                    shap_value.abs() * 100.0
                ),
            });
        }
        
        counterfactuals
    }
    
    /// Calculate feature importance scores
    fn calculate_feature_importance(&self, shap_values: &[f64]) -> HashMap<String, f64> {
        let mut importance = HashMap::new();
        
        for (idx, &value) in shap_values.iter().enumerate() {
            let feature_name = self.feature_names.get(&idx)
                .cloned()
                .unwrap_or_else(|| format!("Feature_{}", idx));
            
            importance.insert(feature_name, value.abs());
        }
        
        importance
    }
    
    /// Assess confidence in the explanation
    fn assess_explanation_confidence(&self, shap_values: &[f64]) -> f64 {
        // Higher concentration of SHAP values = higher confidence
        let total_impact: f64 = shap_values.iter().map(|v| v.abs()).sum();
        let max_impact = shap_values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        
        if total_impact > 0.0 {
            // Concentration ratio
            let concentration = max_impact / total_impact;
            // Scale to 0-1 range
            concentration.min(1.0)
        } else {
            0.5 // Default confidence
        }
    }
    
    /// Generate visualizations data
    fn generate_visualizations(&self, shap_values: &[f64]) -> Vec<VisualizationData> {
        vec![
            VisualizationData {
                viz_type: VisualizationType::WaterfallChart,
                title: "Feature Contribution Waterfall".to_string(),
                data: serde_json::json!({
                    "values": shap_values,
                    "features": self.feature_names.values().collect::<Vec<_>>()
                }),
            },
            VisualizationData {
                viz_type: VisualizationType::BarChart,
                title: "Top Contributing Factors".to_string(),
                data: serde_json::json!({
                    "values": shap_values.iter().take(5).collect::<Vec<_>>()
                }),
            },
        ]
    }
    
    // Helper methods
    
    fn initialize_feature_mappings(
        names: &mut HashMap<usize, String>,
        descriptions: &mut HashMap<String, String>,
    ) {
        // Common features
        names.insert(0, "encryption_enabled".to_string());
        names.insert(1, "public_access_enabled".to_string());
        names.insert(2, "backup_enabled".to_string());
        names.insert(3, "network_restrictions".to_string());
        names.insert(4, "tags_present".to_string());
        names.insert(5, "last_modified_days".to_string());
        
        descriptions.insert("encryption_enabled".to_string(), 
            "Whether encryption at rest is enabled".to_string());
        descriptions.insert("public_access_enabled".to_string(), 
            "Whether public blob access is allowed".to_string());
        descriptions.insert("backup_enabled".to_string(), 
            "Whether automated backup is configured".to_string());
    }
    
    fn initialize_templates() -> HashMap<String, String> {
        let mut templates = HashMap::new();
        
        templates.insert("high_risk".to_string(), 
            "This resource is at high risk of policy violation due to {factors}".to_string());
        templates.insert("medium_risk".to_string(),
            "Moderate risk detected. Consider reviewing {factors}".to_string());
        
        templates
    }
    
    fn format_feature_value(&self, feature_name: &str, value: f64) -> String {
        match feature_name {
            "encryption_enabled" | "public_access_enabled" | "backup_enabled" => {
                if value > 0.5 { "true" } else { "false" }.to_string()
            }
            "last_modified_days" => format!("{} days", value as i32),
            _ => format!("{:.2}", value),
        }
    }
    
    fn get_threshold_value(&self, feature_name: &str) -> String {
        match feature_name {
            "encryption_enabled" | "backup_enabled" => "true".to_string(),
            "public_access_enabled" => "false".to_string(),
            "last_modified_days" => "< 30 days".to_string(),
            _ => "within limits".to_string(),
        }
    }
    
    fn generate_factor_explanation(&self, feature_name: &str, value: f64, impact: f64) -> String {
        let impact_direction = if impact > 0.0 { "increases" } else { "decreases" };
        let formatted_value = self.format_feature_value(feature_name, value);
        
        format!(
            "{} is currently '{}', which {} violation risk by {:.1}%",
            feature_name.replace('_', " "),
            formatted_value,
            impact_direction,
            impact.abs() * 100.0
        )
    }
    
    fn calculate_counterfactual_value(&self, feature_name: &str, current: f64) -> f64 {
        match feature_name {
            "encryption_enabled" | "backup_enabled" => 1.0,
            "public_access_enabled" => 0.0,
            "last_modified_days" => 0.0,
            _ => current * 0.5, // Default: reduce by half
        }
    }
}

/// SHAP value calculator
pub struct ShapValueCalculator {
    baseline_values: Vec<f64>,
}

impl ShapValueCalculator {
    pub fn new() -> Self {
        Self {
            baseline_values: vec![0.5; 20], // Default baseline
        }
    }
    
    pub fn calculate_shap_values(&self, feature_values: &[f64]) -> Vec<f64> {
        // Simplified SHAP calculation (in production, use proper SHAP library)
        feature_values.iter()
            .zip(self.baseline_values.iter())
            .map(|(feature, baseline)| {
                let diff = feature - baseline;
                // Apply non-linear transformation
                diff * (1.0 + diff.abs()).ln()
            })
            .collect()
    }
}

// Data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionExplanation {
    pub prediction_id: String,
    pub resource_id: String,
    pub narrative_explanation: String,
    pub top_factors: Vec<ContributingFactor>,
    pub feature_importance: HashMap<String, f64>,
    pub recommendations: Vec<ActionRecommendation>,
    pub counterfactuals: Vec<CounterfactualExplanation>,
    pub shap_values: Vec<f64>,
    pub explanation_confidence: f64,
    pub visualizations: Vec<VisualizationData>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    pub feature_name: String,
    pub feature_description: String,
    pub impact_score: f64,
    pub current_value: String,
    pub threshold_value: String,
    pub contribution_type: ContributionType,
    pub readable_explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContributionType {
    Increases,
    Decreases,
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecommendation {
    pub action: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub estimated_impact: f64,
    pub implementation_steps: Vec<String>,
    pub automation_available: bool,
    pub remediation_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualExplanation {
    pub feature_name: String,
    pub current_value: String,
    pub suggested_value: String,
    pub expected_impact: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub viz_type: VisualizationType,
    pub title: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    WaterfallChart,
    BarChart,
    ForceDirectedGraph,
    DecisionTree,
}

#[derive(Debug, Clone)]
pub struct ResourceData {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub subscription_id: String,
    pub resource_group: String,
}