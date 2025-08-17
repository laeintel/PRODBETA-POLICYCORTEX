// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Cost Prediction Model
// Predicts future cloud costs and identifies optimization opportunities

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration, Datelike};
use std::collections::HashMap;

/// Cost prediction model for forecasting cloud expenses
pub struct CostPredictionModel {
    historical_data: Vec<CostDataPoint>,
    trend_model: TrendModel,
    seasonality_model: SeasonalityModel,
    resource_models: HashMap<String, ResourceCostModel>,
}

impl CostPredictionModel {
    pub fn new() -> Self {
        Self {
            historical_data: Vec::new(),
            trend_model: TrendModel::new(),
            seasonality_model: SeasonalityModel::new(),
            resource_models: HashMap::new(),
        }
    }
    
    /// Predict monthly cost for resources
    pub fn predict_monthly_cost(&self, resources: &[ResourceUsage]) -> CostPrediction {
        let mut total_cost = 0.0;
        let mut cost_breakdown = HashMap::new();
        let mut optimization_opportunities = Vec::new();
        
        // Calculate base cost for each resource
        for resource in resources {
            let resource_cost = self.calculate_resource_cost(resource);
            total_cost += resource_cost;
            cost_breakdown.insert(resource.resource_id.clone(), resource_cost);
            
            // Check for optimization opportunities
            if let Some(opportunity) = self.identify_optimization(resource) {
                optimization_opportunities.push(opportunity);
            }
        }
        
        // Apply trend adjustment
        let trend_factor = self.trend_model.get_trend_factor();
        total_cost *= trend_factor;
        
        // Apply seasonality
        let seasonal_factor = self.seasonality_model.get_seasonal_factor(Utc::now());
        total_cost *= seasonal_factor;
        
        CostPrediction {
            predicted_amount: total_cost,
            confidence: self.calculate_confidence(),
            trend: if trend_factor > 1.0 { "increasing" } else { "decreasing" }.to_string(),
            breakdown: cost_breakdown,
            optimization_opportunities,
            forecast_date: Utc::now() + Duration::days(30),
        }
    }
    
    fn calculate_resource_cost(&self, resource: &ResourceUsage) -> f64 {
        // Simplified cost calculation based on resource type and usage
        let base_rate = match resource.resource_type.as_str() {
            "Microsoft.Compute/virtualMachines" => 0.10, // per hour
            "Microsoft.Storage/storageAccounts" => 0.02, // per GB
            "Microsoft.Sql/servers/databases" => 0.15,   // per DTU hour
            "Microsoft.Network/publicIPAddresses" => 0.005, // per hour
            _ => 0.01,
        };
        
        base_rate * resource.usage_hours * resource.quantity as f64
    }
    
    fn identify_optimization(&self, resource: &ResourceUsage) -> Option<OptimizationOpportunity> {
        // Check for underutilized resources
        if resource.utilization_percent < 20.0 {
            return Some(OptimizationOpportunity {
                resource_id: resource.resource_id.clone(),
                opportunity_type: "Underutilized".to_string(),
                potential_savings: resource.usage_hours * 0.05,
                recommendation: "Consider downsizing or removing this resource".to_string(),
                confidence: 0.85,
            });
        }
        
        // Check for idle resources
        if resource.utilization_percent == 0.0 {
            return Some(OptimizationOpportunity {
                resource_id: resource.resource_id.clone(),
                opportunity_type: "Idle Resource".to_string(),
                potential_savings: resource.usage_hours * 0.10,
                recommendation: "Resource appears to be idle, consider decommissioning".to_string(),
                confidence: 0.95,
            });
        }
        
        None
    }
    
    fn calculate_confidence(&self) -> f64 {
        // Base confidence on amount of historical data
        let data_points = self.historical_data.len();
        if data_points >= 90 {
            0.95
        } else if data_points >= 30 {
            0.85
        } else if data_points >= 7 {
            0.70
        } else {
            0.50
        }
    }
    
    /// Update model with new cost data
    pub fn update_with_actual(&mut self, actual: CostDataPoint) {
        self.historical_data.push(actual.clone());
        
        // Keep only last 365 days
        if self.historical_data.len() > 365 {
            self.historical_data.remove(0);
        }
        
        // Update trend model
        self.trend_model.update(&self.historical_data);
        
        // Update seasonality model
        self.seasonality_model.update(&self.historical_data);
    }
}

/// Trend analysis model
pub struct TrendModel {
    slope: f64,
    intercept: f64,
    last_update: DateTime<Utc>,
}

impl TrendModel {
    pub fn new() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            last_update: Utc::now(),
        }
    }
    
    pub fn update(&mut self, data: &[CostDataPoint]) {
        if data.len() < 2 {
            return;
        }
        
        // Simple linear regression
        let n = data.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        
        for (i, point) in data.iter().enumerate() {
            let x = i as f64;
            let y = point.amount;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        self.intercept = (sum_y - self.slope * sum_x) / n;
        self.last_update = Utc::now();
    }
    
    pub fn get_trend_factor(&self) -> f64 {
        // Return trend factor for next period
        1.0 + (self.slope * 0.01) // Conservative trend adjustment
    }
}

/// Seasonality analysis model
pub struct SeasonalityModel {
    monthly_factors: [f64; 12],
    last_update: DateTime<Utc>,
}

impl SeasonalityModel {
    pub fn new() -> Self {
        Self {
            monthly_factors: [1.0; 12], // Default: no seasonality
            last_update: Utc::now(),
        }
    }
    
    pub fn update(&mut self, data: &[CostDataPoint]) {
        if data.len() < 30 {
            return; // Need at least a month of data
        }
        
        // Calculate average cost per month
        let mut monthly_totals: HashMap<u32, Vec<f64>> = HashMap::new();
        
        for point in data {
            let month = point.date.month();
            monthly_totals.entry(month).or_default().push(point.amount);
        }
        
        // Calculate seasonal factors
        let overall_avg: f64 = data.iter().map(|p| p.amount).sum::<f64>() / data.len() as f64;
        
        for month in 1..=12 {
            if let Some(amounts) = monthly_totals.get(&month) {
                let month_avg = amounts.iter().sum::<f64>() / amounts.len() as f64;
                self.monthly_factors[(month - 1) as usize] = month_avg / overall_avg;
            }
        }
        
        self.last_update = Utc::now();
    }
    
    pub fn get_seasonal_factor(&self, date: DateTime<Utc>) -> f64 {
        let month = date.month() as usize;
        self.monthly_factors[month - 1]
    }
}

/// Resource cost model for specific resource types
pub struct ResourceCostModel {
    resource_type: String,
    pricing_model: PricingModel,
    usage_patterns: UsagePattern,
}

/// Resource usage data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub resource_id: String,
    pub resource_type: String,
    pub usage_hours: f64,
    pub quantity: u32,
    pub utilization_percent: f64,
    pub tags: HashMap<String, String>,
}

/// Cost data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostDataPoint {
    pub date: DateTime<Utc>,
    pub amount: f64,
    pub resource_group: String,
    pub subscription_id: String,
}

/// Cost prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPrediction {
    pub predicted_amount: f64,
    pub confidence: f64,
    pub trend: String,
    pub breakdown: HashMap<String, f64>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub forecast_date: DateTime<Utc>,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub resource_id: String,
    pub opportunity_type: String,
    pub potential_savings: f64,
    pub recommendation: String,
    pub confidence: f64,
}

/// Pricing model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PricingModel {
    PayAsYouGo,
    Reserved,
    Spot,
    Hybrid,
}

/// Usage pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    pub peak_hours: Vec<u32>,
    pub average_daily_usage: f64,
    pub weekend_factor: f64,
    pub growth_rate: f64,
}

/// Cost anomaly detector
pub struct CostAnomalyDetector {
    baseline: f64,
    threshold: f64,
}

impl CostAnomalyDetector {
    pub fn new() -> Self {
        Self {
            baseline: 0.0,
            threshold: 0.3, // 30% deviation threshold
        }
    }
    
    pub fn detect_anomaly(&self, current_cost: f64) -> Option<CostAnomaly> {
        if self.baseline == 0.0 {
            return None;
        }
        
        let deviation = (current_cost - self.baseline).abs() / self.baseline;
        
        if deviation > self.threshold {
            Some(CostAnomaly {
                detected_at: Utc::now(),
                expected_cost: self.baseline,
                actual_cost: current_cost,
                deviation_percent: deviation * 100.0,
                severity: if deviation > 0.5 { "High" } else { "Medium" }.to_string(),
            })
        } else {
            None
        }
    }
    
    pub fn update_baseline(&mut self, costs: &[f64]) {
        if !costs.is_empty() {
            self.baseline = costs.iter().sum::<f64>() / costs.len() as f64;
        }
    }
}

/// Cost anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnomaly {
    pub detected_at: DateTime<Utc>,
    pub expected_cost: f64,
    pub actual_cost: f64,
    pub deviation_percent: f64,
    pub severity: String,
}