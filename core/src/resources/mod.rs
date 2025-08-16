// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod categories;
pub mod manager;
pub mod discovery;
pub mod correlations;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResourceCategory {
    Policy,
    CostManagement,
    SecurityControls,
    ComputeStorage,
    NetworksFirewalls,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureResource {
    pub id: String,
    pub name: String,
    pub display_name: String,
    pub resource_type: String,
    pub category: ResourceCategory,
    pub location: Option<String>,
    pub tags: HashMap<String, String>,
    pub status: ResourceStatus,
    pub health: ResourceHealth,
    pub cost_data: Option<CostData>,
    pub compliance_status: ComplianceStatus,
    pub quick_actions: Vec<QuickAction>,
    pub insights: Vec<ResourceInsight>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatus {
    pub state: String,
    pub provisioning_state: Option<String>,
    pub availability: f32,
    pub performance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceHealth {
    pub status: HealthStatus,
    pub issues: Vec<HealthIssue>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    pub severity: IssueSeverity,
    pub title: String,
    pub description: String,
    pub affected_components: Vec<String>,
    pub mitigation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostData {
    pub daily_cost: f64,
    pub monthly_cost: f64,
    pub yearly_projection: f64,
    pub cost_trend: CostTrend,
    pub optimization_potential: f64,
    pub currency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostTrend {
    Increasing(f32),
    Decreasing(f32),
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub is_compliant: bool,
    pub compliance_score: f32,
    pub violations: Vec<ComplianceViolation>,
    pub last_assessment: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub policy_id: String,
    pub policy_name: String,
    pub severity: IssueSeverity,
    pub description: String,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickAction {
    pub id: String,
    pub label: String,
    pub icon: String,
    pub action_type: ActionType,
    pub confirmation_required: bool,
    pub estimated_impact: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Start,
    Stop,
    Restart,
    Scale,
    Configure,
    Optimize,
    Backup,
    Delete,
    ViewDetails,
    RunDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInsight {
    pub insight_type: InsightType,
    pub title: String,
    pub description: String,
    pub impact: String,
    pub recommendation: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    CostOptimization,
    PerformanceImprovement,
    SecurityRisk,
    ComplianceGap,
    AvailabilityIssue,
    ConfigurationDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFilter {
    pub categories: Option<Vec<ResourceCategory>>,
    pub resource_types: Option<Vec<String>>,
    pub locations: Option<Vec<String>>,
    pub tags: Option<HashMap<String, String>>,
    pub health_status: Option<Vec<HealthStatus>>,
    pub compliance_filter: Option<ComplianceFilter>,
    pub cost_range: Option<CostRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFilter {
    pub only_violations: bool,
    pub min_score: Option<f32>,
    pub severity_levels: Option<Vec<IssueSeverity>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRange {
    pub min_daily: Option<f64>,
    pub max_daily: Option<f64>,
    pub currency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub total_resources: usize,
    pub by_category: HashMap<ResourceCategory, usize>,
    pub by_health: HashMap<HealthStatus, usize>,
    pub total_daily_cost: f64,
    pub compliance_score: f32,
    pub critical_issues: usize,
    pub optimization_opportunities: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceGroup {
    pub id: String,
    pub name: String,
    pub category: ResourceCategory,
    pub resources: Vec<AzureResource>,
    pub aggregated_health: HealthStatus,
    pub total_cost: CostData,
    pub group_insights: Vec<ResourceInsight>,
}