// Azure Advisor Integration for Optimization Recommendations
// Comprehensive optimization engine with cost, performance, security, and reliability recommendations
// Patent 1: Cross-Domain Governance Correlation Engine integration

use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use crate::azure_client::AzureClient;
use crate::governance::{GovernanceError, GovernanceResult, ComponentHealth, HealthStatus};

/// Azure Advisor optimization engine
pub struct OptimizationEngine {
    azure_client: Arc<AzureClient>,
    recommendation_cache: Arc<dashmap::DashMap<String, CachedRecommendationData>>,
    cost_analyzer: CostOptimizationAnalyzer,
    performance_analyzer: PerformanceAnalyzer,
    security_optimizer: SecurityOptimizer,
    reliability_monitor: ReliabilityMonitor,
    operational_excellence: OperationalExcellenceAnalyzer,
}

/// Cached recommendation data with TTL
#[derive(Debug, Clone)]
pub struct CachedRecommendationData {
    pub data: OptimizationData,
    pub cached_at: DateTime<Utc>,
    pub ttl: Duration,
}

impl CachedRecommendationData {
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.cached_at + self.ttl
    }
}

/// Comprehensive optimization data from Azure Advisor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationData {
    pub scope: String,
    pub cost_recommendations: Vec<CostRecommendation>,
    pub performance_recommendations: Vec<PerformanceRecommendation>,
    pub security_recommendations: Vec<SecurityRecommendation>,
    pub reliability_recommendations: Vec<ReliabilityRecommendation>,
    pub operational_excellence_recommendations: Vec<OperationalRecommendation>,
    pub resource_optimization_summary: ResourceOptimizationSummary,
    pub advisor_score: AdvisorScore,
    pub last_assessment: DateTime<Utc>,
}

/// Cost optimization recommendation from Azure Advisor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    pub recommendation_id: String,
    pub resource_id: String,
    pub resource_name: String,
    pub resource_type: String,
    pub category: CostCategory,
    pub impact: ImpactLevel,
    pub annual_savings_usd: f64,
    pub description: String,
    pub problem_description: String,
    pub solution_description: String,
    pub implementation_effort: ImplementationEffort,
    pub risk_level: RiskLevel,
    pub affected_resource_properties: HashMap<String, String>,
    pub recommended_actions: Vec<String>,
    pub automation_available: bool,
    pub last_updated: DateTime<Utc>,
    pub suppression_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostCategory {
    RightSize,
    Shutdown,
    ReservedInstances,
    Hybrid,
    Configuration,
    Other,
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub recommendation_id: String,
    pub resource_id: String,
    pub resource_name: String,
    pub resource_type: String,
    pub category: PerformanceCategory,
    pub impact: ImpactLevel,
    pub description: String,
    pub problem_description: String,
    pub solution_description: String,
    pub performance_improvement: PerformanceMetrics,
    pub implementation_effort: ImplementationEffort,
    pub recommended_actions: Vec<String>,
    pub monitoring_metrics: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceCategory {
    VirtualMachine,
    SqlDatabase,
    Storage,
    Network,
    AppService,
    CosmosDB,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_improvement_percentage: Option<f64>,
    pub memory_improvement_percentage: Option<f64>,
    pub throughput_improvement_percentage: Option<f64>,
    pub latency_reduction_percentage: Option<f64>,
    pub iops_improvement_percentage: Option<f64>,
}

/// Security optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRecommendation {
    pub recommendation_id: String,
    pub resource_id: String,
    pub resource_name: String,
    pub resource_type: String,
    pub category: SecurityCategory,
    pub severity: SecuritySeverity,
    pub description: String,
    pub problem_description: String,
    pub solution_description: String,
    pub security_impact: SecurityImpact,
    pub compliance_frameworks: Vec<String>,
    pub implementation_effort: ImplementationEffort,
    pub recommended_actions: Vec<String>,
    pub automation_available: bool,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityCategory {
    IdentityAccess,
    DataProtection,
    NetworkSecurity,
    VulnerabilityManagement,
    IncidentResponse,
    Configuration,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImpact {
    pub confidentiality_impact: ImpactLevel,
    pub integrity_impact: ImpactLevel,
    pub availability_impact: ImpactLevel,
    pub compliance_impact: f64,
}

/// Reliability optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityRecommendation {
    pub recommendation_id: String,
    pub resource_id: String,
    pub resource_name: String,
    pub resource_type: String,
    pub category: ReliabilityCategory,
    pub impact: ImpactLevel,
    pub description: String,
    pub problem_description: String,
    pub solution_description: String,
    pub reliability_improvement: ReliabilityMetrics,
    pub implementation_effort: ImplementationEffort,
    pub recommended_actions: Vec<String>,
    pub disaster_recovery_impact: bool,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReliabilityCategory {
    HighAvailability,
    DisasterRecovery,
    Backup,
    Monitoring,
    Redundancy,
    Configuration,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    pub availability_improvement_percentage: Option<f64>,
    pub mttr_reduction_percentage: Option<f64>,
    pub rto_improvement_percentage: Option<f64>,
    pub rpo_improvement_percentage: Option<f64>,
}

/// Operational excellence recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalRecommendation {
    pub recommendation_id: String,
    pub resource_id: String,
    pub resource_name: String,
    pub resource_type: String,
    pub category: OperationalCategory,
    pub impact: ImpactLevel,
    pub description: String,
    pub problem_description: String,
    pub solution_description: String,
    pub operational_improvement: OperationalMetrics,
    pub implementation_effort: ImplementationEffort,
    pub recommended_actions: Vec<String>,
    pub automation_potential: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationalCategory {
    Automation,
    Monitoring,
    Configuration,
    Documentation,
    ProcessImprovement,
    ResourceManagement,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalMetrics {
    pub automation_percentage: Option<f64>,
    pub monitoring_coverage_percentage: Option<f64>,
    pub incident_reduction_percentage: Option<f64>,
    pub deployment_efficiency_percentage: Option<f64>,
}

/// Common enums and types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    RequiresPlanning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Resource optimization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationSummary {
    pub total_resources_analyzed: u32,
    pub resources_with_recommendations: u32,
    pub total_potential_annual_savings: f64,
    pub cost_optimization_opportunities: u32,
    pub performance_optimization_opportunities: u32,
    pub security_optimization_opportunities: u32,
    pub reliability_optimization_opportunities: u32,
    pub operational_optimization_opportunities: u32,
    pub optimization_score: f64,
    pub top_savings_opportunities: Vec<TopSavingsOpportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopSavingsOpportunity {
    pub resource_type: String,
    pub potential_savings: f64,
    pub recommendation_count: u32,
    pub avg_implementation_effort: ImplementationEffort,
}

/// Azure Advisor Well-Architected score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisorScore {
    pub overall_score: f64,
    pub cost_score: f64,
    pub performance_score: f64,
    pub security_score: f64,
    pub reliability_score: f64,
    pub operational_excellence_score: f64,
    pub score_trend: ScoreTrend,
    pub benchmark_comparison: BenchmarkComparison,
    pub last_calculated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreTrend {
    pub overall_trend: TrendDirection,
    pub cost_trend: TrendDirection,
    pub performance_trend: TrendDirection,
    pub security_trend: TrendDirection,
    pub reliability_trend: TrendDirection,
    pub operational_trend: TrendDirection,
    pub trend_period_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub industry_percentile: f64,
    pub similar_size_percentile: f64,
    pub region_percentile: f64,
    pub peer_group_average: f64,
}

/// Cost optimization analyzer
pub struct CostOptimizationAnalyzer {
    pricing_data: HashMap<String, PricingInfo>,
    usage_patterns: HashMap<String, UsagePattern>,
    reservation_opportunities: HashMap<String, ReservationOpportunity>,
}

#[derive(Debug, Clone)]
pub struct PricingInfo {
    pub resource_type: String,
    pub current_sku: String,
    pub current_hourly_cost: f64,
    pub recommended_sku: Option<String>,
    pub recommended_hourly_cost: Option<f64>,
    pub savings_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct UsagePattern {
    pub resource_id: String,
    pub avg_cpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub peak_usage_hours: Vec<u8>, // Hours of day (0-23)
    pub idle_percentage: f64,
    pub utilization_trend: TrendDirection,
}

#[derive(Debug, Clone)]
pub struct ReservationOpportunity {
    pub resource_type: String,
    pub current_payg_cost: f64,
    pub reservation_cost: f64,
    pub potential_savings: f64,
    pub recommended_term: ReservationTerm,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReservationTerm {
    OneYear,
    ThreeYear,
}

/// Performance analyzer
pub struct PerformanceAnalyzer {
    performance_baselines: HashMap<String, PerformanceBaseline>,
    bottleneck_detector: BottleneckDetector,
}

#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub resource_id: String,
    pub cpu_baseline: f64,
    pub memory_baseline: f64,
    pub storage_baseline: f64,
    pub network_baseline: f64,
    pub response_time_baseline: f64,
}

pub struct BottleneckDetector {
    cpu_threshold: f64,
    memory_threshold: f64,
    storage_threshold: f64,
    network_threshold: f64,
}

/// Security optimizer
pub struct SecurityOptimizer {
    security_policies: HashMap<String, SecurityPolicy>,
    vulnerability_database: HashMap<String, VulnerabilityInfo>,
}

#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub compliance_frameworks: Vec<String>,
    pub required_configurations: HashMap<String, String>,
    pub severity: SecuritySeverity,
}

#[derive(Debug, Clone)]
pub struct VulnerabilityInfo {
    pub cve_id: String,
    pub severity: SecuritySeverity,
    pub affected_resources: Vec<String>,
    pub remediation_guidance: Vec<String>,
    pub patch_available: bool,
}

/// Reliability monitor
pub struct ReliabilityMonitor {
    sla_targets: HashMap<String, SlaTarget>,
    disaster_recovery_plans: HashMap<String, DrPlan>,
}

#[derive(Debug, Clone)]
pub struct SlaTarget {
    pub resource_type: String,
    pub target_availability: f64,
    pub current_availability: f64,
    pub mttr_target: Duration,
    pub current_mttr: Duration,
}

#[derive(Debug, Clone)]
pub struct DrPlan {
    pub resource_id: String,
    pub rto_target: Duration,
    pub rpo_target: Duration,
    pub backup_frequency: Duration,
    pub cross_region_replication: bool,
}

/// Operational excellence analyzer
pub struct OperationalExcellenceAnalyzer {
    automation_opportunities: HashMap<String, AutomationOpportunity>,
    monitoring_gaps: HashMap<String, MonitoringGap>,
}

#[derive(Debug, Clone)]
pub struct AutomationOpportunity {
    pub process_name: String,
    pub manual_effort_hours: f64,
    pub automation_potential: f64,
    pub estimated_roi: f64,
    pub tools_required: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MonitoringGap {
    pub resource_id: String,
    pub missing_metrics: Vec<String>,
    pub alerting_gaps: Vec<String>,
    pub dashboard_recommendations: Vec<String>,
}

impl OptimizationEngine {
    pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
        Ok(Self {
            azure_client,
            recommendation_cache: Arc::new(dashmap::DashMap::new()),
            cost_analyzer: CostOptimizationAnalyzer::new(),
            performance_analyzer: PerformanceAnalyzer::new(),
            security_optimizer: SecurityOptimizer::new(),
            reliability_monitor: ReliabilityMonitor::new(),
            operational_excellence: OperationalExcellenceAnalyzer::new(),
        })
    }

    /// Get comprehensive optimization recommendations from Azure Advisor
    pub async fn get_optimization_recommendations(&self, scope: &str) -> GovernanceResult<OptimizationData> {
        let cache_key = format!("optimization_{}", scope);

        // Check cache first
        if let Some(cached) = self.recommendation_cache.get(&cache_key) {
            if !cached.is_expired() {
                return Ok(cached.data.clone());
            }
        }

        // Fetch optimization data from Azure Advisor APIs
        let optimization_data = self.fetch_advisor_recommendations(scope).await?;

        // Cache the result
        self.recommendation_cache.insert(cache_key, CachedRecommendationData {
            data: optimization_data.clone(),
            cached_at: Utc::now(),
            ttl: Duration::hours(6), // Advisor recommendations change less frequently
        });

        Ok(optimization_data)
    }

    /// Analyze cost optimization opportunities
    pub async fn analyze_cost_optimization(&self, scope: &str) -> GovernanceResult<Vec<CostRecommendation>> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        let mut enhanced_recommendations = optimization_data.cost_recommendations;

        // Enhance recommendations with additional analysis
        for recommendation in &mut enhanced_recommendations {
            recommendation.risk_level = self.cost_analyzer.assess_risk_level(&recommendation);
            
            // Add custom insights based on usage patterns
            if let Some(usage_pattern) = self.cost_analyzer.usage_patterns.get(&recommendation.resource_id) {
                if usage_pattern.idle_percentage > 80.0 {
                    recommendation.recommended_actions.push(
                        "Consider scheduling automatic shutdown during idle periods".to_string()
                    );
                }
            }
        }

        // Sort by potential savings
        enhanced_recommendations.sort_by(|a, b| b.annual_savings_usd.partial_cmp(&a.annual_savings_usd).unwrap());

        Ok(enhanced_recommendations)
    }

    /// Analyze performance optimization opportunities
    pub async fn analyze_performance_optimization(&self, scope: &str) -> GovernanceResult<Vec<PerformanceRecommendation>> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        let mut enhanced_recommendations = optimization_data.performance_recommendations;

        // Enhance with performance baseline analysis
        for recommendation in &mut enhanced_recommendations {
            if let Some(baseline) = self.performance_analyzer.performance_baselines.get(&recommendation.resource_id) {
                // Add custom performance insights
                if baseline.cpu_baseline > 80.0 {
                    recommendation.recommended_actions.push(
                        "CPU utilization consistently high - consider scaling up or out".to_string()
                    );
                }
                
                if baseline.memory_baseline > 85.0 {
                    recommendation.recommended_actions.push(
                        "Memory pressure detected - increase memory allocation".to_string()
                    );
                }
            }
        }

        Ok(enhanced_recommendations)
    }

    /// Generate security optimization recommendations
    pub async fn analyze_security_optimization(&self, scope: &str) -> GovernanceResult<Vec<SecurityRecommendation>> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        let mut enhanced_recommendations = optimization_data.security_recommendations;

        // Enhance with security policy compliance
        for recommendation in &mut enhanced_recommendations {
            // Check against security policies
            for (_, policy) in &self.security_optimizer.security_policies {
                if policy.compliance_frameworks.contains(&"CIS".to_string()) {
                    recommendation.compliance_frameworks.push("CIS Controls".to_string());
                }
            }

            // Add automation suggestions
            if matches!(recommendation.category, SecurityCategory::Configuration) {
                recommendation.automation_available = true;
                recommendation.recommended_actions.push(
                    "Consider implementing Azure Policy for automated compliance".to_string()
                );
            }
        }

        // Sort by security severity
        enhanced_recommendations.sort_by(|a, b| {
            let severity_order = |s: &SecuritySeverity| match s {
                SecuritySeverity::Critical => 0,
                SecuritySeverity::High => 1,
                SecuritySeverity::Medium => 2,
                SecuritySeverity::Low => 3,
                SecuritySeverity::Informational => 4,
            };
            severity_order(&a.severity).cmp(&severity_order(&b.severity))
        });

        Ok(enhanced_recommendations)
    }

    /// Analyze reliability optimization opportunities
    pub async fn analyze_reliability_optimization(&self, scope: &str) -> GovernanceResult<Vec<ReliabilityRecommendation>> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        let mut enhanced_recommendations = optimization_data.reliability_recommendations;

        // Enhance with SLA analysis
        for recommendation in &mut enhanced_recommendations {
            if let Some(sla_target) = self.reliability_monitor.sla_targets.get(&recommendation.resource_type) {
                if sla_target.current_availability < sla_target.target_availability {
                    recommendation.recommended_actions.push(
                        format!("Current availability {:.2}% below target {:.2}%", 
                            sla_target.current_availability * 100.0,
                            sla_target.target_availability * 100.0)
                    );
                }
            }

            // Check disaster recovery readiness
            if let Some(_dr_plan) = self.reliability_monitor.disaster_recovery_plans.get(&recommendation.resource_id) {
                recommendation.disaster_recovery_impact = true;
            }
        }

        Ok(enhanced_recommendations)
    }

    /// Generate operational excellence recommendations
    pub async fn analyze_operational_excellence(&self, scope: &str) -> GovernanceResult<Vec<OperationalRecommendation>> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        let mut enhanced_recommendations = optimization_data.operational_excellence_recommendations;

        // Enhance with automation opportunities
        for recommendation in &mut enhanced_recommendations {
            if let Some(automation) = self.operational_excellence.automation_opportunities.get(&recommendation.resource_id) {
                recommendation.automation_potential = automation.automation_potential;
                recommendation.recommended_actions.push(
                    format!("Automation could save {:.1} hours per week", automation.manual_effort_hours)
                );
            }

            // Add monitoring recommendations
            if let Some(monitoring_gap) = self.operational_excellence.monitoring_gaps.get(&recommendation.resource_id) {
                for gap in &monitoring_gap.missing_metrics {
                    recommendation.recommended_actions.push(
                        format!("Implement monitoring for: {}", gap)
                    );
                }
            }
        }

        Ok(enhanced_recommendations)
    }

    /// Generate prioritized optimization roadmap
    pub async fn generate_optimization_roadmap(&self, scope: &str) -> GovernanceResult<OptimizationRoadmap> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        
        // Analyze all recommendations and create a prioritized roadmap
        let mut roadmap_items = Vec::new();

        // Add high-impact cost recommendations
        for cost_rec in &optimization_data.cost_recommendations {
            if cost_rec.annual_savings_usd > 10000.0 && matches!(cost_rec.impact, ImpactLevel::High | ImpactLevel::Critical) {
                roadmap_items.push(RoadmapItem {
                    item_id: uuid::Uuid::new_v4().to_string(),
                    title: format!("Cost: {}", cost_rec.description),
                    category: OptimizationCategory::Cost,
                    priority: self.calculate_priority(&cost_rec.impact, &cost_rec.implementation_effort),
                    estimated_savings: Some(cost_rec.annual_savings_usd),
                    implementation_timeline: self.estimate_timeline(&cost_rec.implementation_effort),
                    dependencies: vec![],
                    business_impact: format!("Annual savings: ${:.0}", cost_rec.annual_savings_usd),
                    technical_complexity: cost_rec.implementation_effort.clone(),
                    risk_assessment: cost_rec.risk_level.clone(),
                    success_metrics: vec![
                        "Cost reduction percentage".to_string(),
                        "Resource utilization improvement".to_string(),
                    ],
                });
            }
        }

        // Add critical security recommendations
        for sec_rec in &optimization_data.security_recommendations {
            if matches!(sec_rec.severity, SecuritySeverity::Critical | SecuritySeverity::High) {
                roadmap_items.push(RoadmapItem {
                    item_id: uuid::Uuid::new_v4().to_string(),
                    title: format!("Security: {}", sec_rec.description),
                    category: OptimizationCategory::Security,
                    priority: RoadmapPriority::Critical,
                    estimated_savings: None,
                    implementation_timeline: self.estimate_timeline(&sec_rec.implementation_effort),
                    dependencies: vec![],
                    business_impact: "Risk reduction and compliance improvement".to_string(),
                    technical_complexity: sec_rec.implementation_effort.clone(),
                    risk_assessment: match sec_rec.severity {
                        SecuritySeverity::Critical => RiskLevel::Critical,
                        SecuritySeverity::High => RiskLevel::High,
                        _ => RiskLevel::Medium,
                    },
                    success_metrics: vec![
                        "Security posture score improvement".to_string(),
                        "Compliance framework adherence".to_string(),
                    ],
                });
            }
        }

        // Sort by priority and potential impact
        roadmap_items.sort_by(|a, b| {
            let priority_order = |p: &RoadmapPriority| match p {
                RoadmapPriority::Critical => 0,
                RoadmapPriority::High => 1,
                RoadmapPriority::Medium => 2,
                RoadmapPriority::Low => 3,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });

        Ok(OptimizationRoadmap {
            roadmap_id: uuid::Uuid::new_v4().to_string(),
            scope: scope.to_string(),
            items: roadmap_items,
            total_estimated_savings: optimization_data.resource_optimization_summary.total_potential_annual_savings,
            implementation_phases: self.create_implementation_phases(&optimization_data),
            success_criteria: vec![
                "25% reduction in cloud spend".to_string(),
                "99.9% availability target achievement".to_string(),
                "Zero critical security findings".to_string(),
                "80% automation coverage".to_string(),
            ],
            created_at: Utc::now(),
            last_updated: Utc::now(),
        })
    }

    /// Get optimization metrics and KPIs
    pub async fn get_optimization_metrics(&self, scope: &str) -> GovernanceResult<OptimizationMetrics> {
        let optimization_data = self.get_optimization_recommendations(scope).await?;

        Ok(OptimizationMetrics {
            total_recommendations: (optimization_data.cost_recommendations.len() +
                optimization_data.performance_recommendations.len() +
                optimization_data.security_recommendations.len() +
                optimization_data.reliability_recommendations.len() +
                optimization_data.operational_excellence_recommendations.len()) as u32,
            high_impact_recommendations: self.count_high_impact_recommendations(&optimization_data),
            potential_annual_savings: optimization_data.resource_optimization_summary.total_potential_annual_savings,
            cost_optimization_score: optimization_data.advisor_score.cost_score,
            performance_optimization_score: optimization_data.advisor_score.performance_score,
            security_optimization_score: optimization_data.advisor_score.security_score,
            reliability_optimization_score: optimization_data.advisor_score.reliability_score,
            operational_excellence_score: optimization_data.advisor_score.operational_excellence_score,
            overall_advisor_score: optimization_data.advisor_score.overall_score,
            optimization_trend: optimization_data.advisor_score.score_trend.overall_trend.clone(),
            industry_percentile: optimization_data.advisor_score.benchmark_comparison.industry_percentile,
        })
    }

    /// Health check for optimization engine components
    pub async fn health_check(&self) -> ComponentHealth {
        let mut metrics = HashMap::new();
        metrics.insert("cache_size".to_string(), self.recommendation_cache.len() as f64);
        metrics.insert("pricing_data_entries".to_string(), self.cost_analyzer.pricing_data.len() as f64);
        metrics.insert("performance_baselines".to_string(), self.performance_analyzer.performance_baselines.len() as f64);
        metrics.insert("security_policies".to_string(), self.security_optimizer.security_policies.len() as f64);

        ComponentHealth {
            component: "OptimizationEngine".to_string(),
            status: HealthStatus::Healthy,
            message: "Optimization engine operational with Azure Advisor integration".to_string(),
            last_check: Utc::now(),
            metrics,
        }
    }

    // Private helper methods

    async fn fetch_advisor_recommendations(&self, scope: &str) -> GovernanceResult<OptimizationData> {
        // In production, would call Azure Advisor APIs:
        // GET https://management.azure.com/{scope}/providers/Microsoft.Advisor/recommendations
        // GET https://management.azure.com/{scope}/providers/Microsoft.Advisor/advisorScore

        Ok(OptimizationData {
            scope: scope.to_string(),
            cost_recommendations: vec![
                CostRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    resource_id: format!("{}/providers/Microsoft.Compute/virtualMachines/vm-web-001", scope),
                    resource_name: "vm-web-001".to_string(),
                    resource_type: "Microsoft.Compute/virtualMachines".to_string(),
                    category: CostCategory::RightSize,
                    impact: ImpactLevel::High,
                    annual_savings_usd: 2400.0,
                    description: "Right-size underutilized virtual machine".to_string(),
                    problem_description: "Virtual machine is consistently underutilized with average CPU usage below 20%".to_string(),
                    solution_description: "Resize to a smaller SKU (Standard_B2s) to reduce costs while maintaining performance".to_string(),
                    implementation_effort: ImplementationEffort::Medium,
                    risk_level: RiskLevel::Low,
                    affected_resource_properties: {
                        let mut props = HashMap::new();
                        props.insert("currentSku".to_string(), "Standard_D4s_v3".to_string());
                        props.insert("recommendedSku".to_string(), "Standard_B2s".to_string());
                        props.insert("avgCpuUsage".to_string(), "18%".to_string());
                        props
                    },
                    recommended_actions: vec![
                        "Schedule maintenance window for VM resize".to_string(),
                        "Backup VM state before resizing".to_string(),
                        "Monitor performance after resize".to_string(),
                        "Update capacity planning documentation".to_string(),
                    ],
                    automation_available: true,
                    last_updated: Utc::now() - Duration::hours(2),
                    suppression_ids: vec![],
                },
                CostRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    resource_id: format!("{}/providers/Microsoft.Compute/virtualMachines/vm-dev-002", scope),
                    resource_name: "vm-dev-002".to_string(),
                    resource_type: "Microsoft.Compute/virtualMachines".to_string(),
                    category: CostCategory::Shutdown,
                    impact: ImpactLevel::Medium,
                    annual_savings_usd: 3600.0,
                    description: "Shutdown unused development virtual machine".to_string(),
                    problem_description: "Development VM has been idle for 30+ days with no user activity".to_string(),
                    solution_description: "Deallocate or delete the VM to eliminate ongoing costs".to_string(),
                    implementation_effort: ImplementationEffort::Low,
                    risk_level: RiskLevel::Low,
                    affected_resource_properties: {
                        let mut props = HashMap::new();
                        props.insert("lastActivity".to_string(), "35 days ago".to_string());
                        props.insert("environment".to_string(), "development".to_string());
                        props
                    },
                    recommended_actions: vec![
                        "Confirm VM is no longer needed with development team".to_string(),
                        "Export any required data or configurations".to_string(),
                        "Deallocate VM to stop billing".to_string(),
                        "Schedule periodic cleanup of unused dev resources".to_string(),
                    ],
                    automation_available: true,
                    last_updated: Utc::now() - Duration::hours(1),
                    suppression_ids: vec![],
                }
            ],
            performance_recommendations: vec![
                PerformanceRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    resource_id: format!("{}/providers/Microsoft.Sql/servers/sqlsrv001/databases/proddb", scope),
                    resource_name: "proddb".to_string(),
                    resource_type: "Microsoft.Sql/servers/databases".to_string(),
                    category: PerformanceCategory::SqlDatabase,
                    impact: ImpactLevel::High,
                    description: "Upgrade SQL Database service tier for better performance".to_string(),
                    problem_description: "Database experiencing high DTU utilization (>80%) causing query timeouts".to_string(),
                    solution_description: "Upgrade from S2 to S3 service tier to provide additional DTUs and improve query performance".to_string(),
                    performance_improvement: PerformanceMetrics {
                        cpu_improvement_percentage: Some(40.0),
                        memory_improvement_percentage: Some(25.0),
                        throughput_improvement_percentage: Some(50.0),
                        latency_reduction_percentage: Some(30.0),
                        iops_improvement_percentage: Some(35.0),
                    },
                    implementation_effort: ImplementationEffort::Low,
                    recommended_actions: vec![
                        "Monitor current performance metrics".to_string(),
                        "Schedule upgrade during maintenance window".to_string(),
                        "Test application performance after upgrade".to_string(),
                        "Adjust auto-scaling policies if needed".to_string(),
                    ],
                    monitoring_metrics: vec![
                        "DTU percentage".to_string(),
                        "Database size percentage".to_string(),
                        "Connection count".to_string(),
                        "Query duration".to_string(),
                    ],
                    last_updated: Utc::now() - Duration::hours(3),
                }
            ],
            security_recommendations: vec![
                SecurityRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    resource_id: format!("{}/providers/Microsoft.Storage/storageAccounts/stgprod001", scope),
                    resource_name: "stgprod001".to_string(),
                    resource_type: "Microsoft.Storage/storageAccounts".to_string(),
                    category: SecurityCategory::DataProtection,
                    severity: SecuritySeverity::High,
                    description: "Enable storage account encryption with customer-managed keys".to_string(),
                    problem_description: "Storage account is using Microsoft-managed keys instead of customer-managed keys".to_string(),
                    solution_description: "Configure customer-managed keys using Azure Key Vault for enhanced security control".to_string(),
                    security_impact: SecurityImpact {
                        confidentiality_impact: ImpactLevel::High,
                        integrity_impact: ImpactLevel::Medium,
                        availability_impact: ImpactLevel::Low,
                        compliance_impact: 15.0,
                    },
                    compliance_frameworks: vec![
                        "SOC 2".to_string(),
                        "ISO 27001".to_string(),
                        "NIST".to_string(),
                    ],
                    implementation_effort: ImplementationEffort::Medium,
                    recommended_actions: vec![
                        "Create customer-managed key in Azure Key Vault".to_string(),
                        "Configure storage account to use customer-managed key".to_string(),
                        "Test data access and encryption functionality".to_string(),
                        "Update security documentation and procedures".to_string(),
                    ],
                    automation_available: true,
                    last_updated: Utc::now() - Duration::hours(4),
                }
            ],
            reliability_recommendations: vec![
                ReliabilityRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    resource_id: format!("{}/providers/Microsoft.Compute/virtualMachines/vm-prod-001", scope),
                    resource_name: "vm-prod-001".to_string(),
                    resource_type: "Microsoft.Compute/virtualMachines".to_string(),
                    category: ReliabilityCategory::HighAvailability,
                    impact: ImpactLevel::Critical,
                    description: "Configure virtual machine for high availability".to_string(),
                    problem_description: "Single VM deployment without availability set or zones creates single point of failure".to_string(),
                    solution_description: "Deploy VM in availability zone or availability set to ensure 99.9% SLA".to_string(),
                    reliability_improvement: ReliabilityMetrics {
                        availability_improvement_percentage: Some(15.0),
                        mttr_reduction_percentage: Some(25.0),
                        rto_improvement_percentage: Some(40.0),
                        rpo_improvement_percentage: None,
                    },
                    implementation_effort: ImplementationEffort::High,
                    recommended_actions: vec![
                        "Plan VM migration to availability zone".to_string(),
                        "Create new VM in target availability zone".to_string(),
                        "Migrate data and applications".to_string(),
                        "Update load balancer configuration".to_string(),
                        "Test failover scenarios".to_string(),
                    ],
                    disaster_recovery_impact: true,
                    last_updated: Utc::now() - Duration::hours(6),
                }
            ],
            operational_excellence_recommendations: vec![
                OperationalRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    resource_id: format!("{}/providers/Microsoft.Resources/resourceGroups/rg-prod", scope),
                    resource_name: "rg-prod".to_string(),
                    resource_type: "Microsoft.Resources/resourceGroups".to_string(),
                    category: OperationalCategory::Monitoring,
                    impact: ImpactLevel::Medium,
                    description: "Implement comprehensive monitoring and alerting".to_string(),
                    problem_description: "Missing critical monitoring metrics and alerts for production resources".to_string(),
                    solution_description: "Deploy Azure Monitor workbooks and configure alerting rules for proactive monitoring".to_string(),
                    operational_improvement: OperationalMetrics {
                        automation_percentage: Some(60.0),
                        monitoring_coverage_percentage: Some(85.0),
                        incident_reduction_percentage: Some(30.0),
                        deployment_efficiency_percentage: Some(20.0),
                    },
                    implementation_effort: ImplementationEffort::Medium,
                    recommended_actions: vec![
                        "Deploy Azure Monitor agents to all VMs".to_string(),
                        "Configure custom metrics and logs".to_string(),
                        "Create monitoring workbooks and dashboards".to_string(),
                        "Set up alerting rules and action groups".to_string(),
                        "Implement automated remediation where possible".to_string(),
                    ],
                    automation_potential: 0.75,
                    last_updated: Utc::now() - Duration::hours(8),
                }
            ],
            resource_optimization_summary: ResourceOptimizationSummary {
                total_resources_analyzed: 127,
                resources_with_recommendations: 45,
                total_potential_annual_savings: 12500.0,
                cost_optimization_opportunities: 15,
                performance_optimization_opportunities: 8,
                security_optimization_opportunities: 12,
                reliability_optimization_opportunities: 6,
                operational_optimization_opportunities: 4,
                optimization_score: 72.5,
                top_savings_opportunities: vec![
                    TopSavingsOpportunity {
                        resource_type: "Virtual Machines".to_string(),
                        potential_savings: 8200.0,
                        recommendation_count: 8,
                        avg_implementation_effort: ImplementationEffort::Medium,
                    },
                    TopSavingsOpportunity {
                        resource_type: "SQL Databases".to_string(),
                        potential_savings: 2400.0,
                        recommendation_count: 3,
                        avg_implementation_effort: ImplementationEffort::Low,
                    },
                ],
            },
            advisor_score: AdvisorScore {
                overall_score: 74.2,
                cost_score: 68.5,
                performance_score: 82.1,
                security_score: 71.3,
                reliability_score: 76.8,
                operational_excellence_score: 72.4,
                score_trend: ScoreTrend {
                    overall_trend: TrendDirection::Improving,
                    cost_trend: TrendDirection::Improving,
                    performance_trend: TrendDirection::Stable,
                    security_trend: TrendDirection::Improving,
                    reliability_trend: TrendDirection::Stable,
                    operational_trend: TrendDirection::Improving,
                    trend_period_days: 30,
                },
                benchmark_comparison: BenchmarkComparison {
                    industry_percentile: 78.5,
                    similar_size_percentile: 82.1,
                    region_percentile: 75.3,
                    peer_group_average: 71.8,
                },
                last_calculated: Utc::now() - Duration::hours(12),
            },
            last_assessment: Utc::now(),
        })
    }

    fn count_high_impact_recommendations(&self, optimization_data: &OptimizationData) -> u32 {
        let cost_high_impact = optimization_data.cost_recommendations.iter()
            .filter(|r| matches!(r.impact, ImpactLevel::High | ImpactLevel::Critical))
            .count();

        let perf_high_impact = optimization_data.performance_recommendations.iter()
            .filter(|r| matches!(r.impact, ImpactLevel::High | ImpactLevel::Critical))
            .count();

        let sec_high_impact = optimization_data.security_recommendations.iter()
            .filter(|r| matches!(r.severity, SecuritySeverity::High | SecuritySeverity::Critical))
            .count();

        let rel_high_impact = optimization_data.reliability_recommendations.iter()
            .filter(|r| matches!(r.impact, ImpactLevel::High | ImpactLevel::Critical))
            .count();

        let op_high_impact = optimization_data.operational_excellence_recommendations.iter()
            .filter(|r| matches!(r.impact, ImpactLevel::High | ImpactLevel::Critical))
            .count();

        (cost_high_impact + perf_high_impact + sec_high_impact + rel_high_impact + op_high_impact) as u32
    }

    fn calculate_priority(&self, impact: &ImpactLevel, effort: &ImplementationEffort) -> RoadmapPriority {
        match (impact, effort) {
            (ImpactLevel::Critical, _) => RoadmapPriority::Critical,
            (ImpactLevel::High, ImplementationEffort::Low) => RoadmapPriority::Critical,
            (ImpactLevel::High, ImplementationEffort::Medium) => RoadmapPriority::High,
            (ImpactLevel::High, _) => RoadmapPriority::Medium,
            (ImpactLevel::Medium, ImplementationEffort::Low) => RoadmapPriority::High,
            (ImpactLevel::Medium, _) => RoadmapPriority::Medium,
            _ => RoadmapPriority::Low,
        }
    }

    fn estimate_timeline(&self, effort: &ImplementationEffort) -> String {
        match effort {
            ImplementationEffort::Low => "1-2 weeks".to_string(),
            ImplementationEffort::Medium => "1-2 months".to_string(),
            ImplementationEffort::High => "3-6 months".to_string(),
            ImplementationEffort::RequiresPlanning => "6+ months".to_string(),
        }
    }

    fn create_implementation_phases(&self, _optimization_data: &OptimizationData) -> Vec<ImplementationPhase> {
        vec![
            ImplementationPhase {
                phase_number: 1,
                phase_name: "Quick Wins".to_string(),
                description: "Low-effort, high-impact optimizations".to_string(),
                duration_weeks: 4,
                estimated_savings: 5000.0,
                focus_areas: vec!["Cost optimization".to_string(), "Security hardening".to_string()],
            },
            ImplementationPhase {
                phase_number: 2,
                phase_name: "Performance & Reliability".to_string(),
                description: "Improve system performance and reliability".to_string(),
                duration_weeks: 12,
                estimated_savings: 3000.0,
                focus_areas: vec!["Performance tuning".to_string(), "High availability".to_string()],
            },
            ImplementationPhase {
                phase_number: 3,
                phase_name: "Operational Excellence".to_string(),
                description: "Enhance monitoring, automation, and processes".to_string(),
                duration_weeks: 16,
                estimated_savings: 4500.0,
                focus_areas: vec!["Automation".to_string(), "Monitoring".to_string(), "Documentation".to_string()],
            },
        ]
    }

    // Legacy method for backward compatibility
    pub async fn get_advisor_recommendations(&self) -> GovernanceResult<Vec<OptimizationRecommendation>> {
        // Map new comprehensive data to legacy format for backward compatibility
        let scope = "/subscriptions/default";
        let optimization_data = self.get_optimization_recommendations(scope).await?;
        
        let mut legacy_recommendations = Vec::new();

        // Convert cost recommendations
        for cost_rec in optimization_data.cost_recommendations {
            legacy_recommendations.push(OptimizationRecommendation {
                recommendation_id: cost_rec.recommendation_id,
                category: "Cost".to_string(),
                resource_id: cost_rec.resource_id,
                title: cost_rec.description,
                description: cost_rec.problem_description,
                impact: format!("{:?}", cost_rec.impact),
                potential_savings: Some(cost_rec.annual_savings_usd),
            });
        }

        // Convert performance recommendations
        for perf_rec in optimization_data.performance_recommendations {
            legacy_recommendations.push(OptimizationRecommendation {
                recommendation_id: perf_rec.recommendation_id,
                category: "Performance".to_string(),
                resource_id: perf_rec.resource_id,
                title: perf_rec.description,
                description: perf_rec.problem_description,
                impact: format!("{:?}", perf_rec.impact),
                potential_savings: None,
            });
        }

        Ok(legacy_recommendations)
    }

    pub async fn get_optimization_summary(&self) -> GovernanceResult<OptimizationSummary> {
        let scope = "/subscriptions/default";
        let optimization_data = self.get_optimization_recommendations(scope).await?;

        let total_recommendations = (optimization_data.cost_recommendations.len() +
            optimization_data.performance_recommendations.len() +
            optimization_data.security_recommendations.len() +
            optimization_data.reliability_recommendations.len() +
            optimization_data.operational_excellence_recommendations.len()) as u32;

        let high_impact_count = self.count_high_impact_recommendations(&optimization_data);
        
        Ok(OptimizationSummary {
            total_recommendations,
            high_impact_count,
            medium_impact_count: total_recommendations - high_impact_count,
            low_impact_count: 0,
            potential_total_savings: optimization_data.resource_optimization_summary.total_potential_annual_savings,
        })
    }

    pub async fn apply_optimization(&self, _recommendation_id: &str) -> GovernanceResult<()> {
        // In production, would implement actual optimization application
        Ok(())
    }
}

// Implementation for helper structs
impl CostOptimizationAnalyzer {
    pub fn new() -> Self {
        Self {
            pricing_data: HashMap::new(),
            usage_patterns: HashMap::new(),
            reservation_opportunities: HashMap::new(),
        }
    }

    pub fn assess_risk_level(&self, recommendation: &CostRecommendation) -> RiskLevel {
        match recommendation.category {
            CostCategory::Shutdown => RiskLevel::Medium,
            CostCategory::RightSize => RiskLevel::Low,
            CostCategory::ReservedInstances => RiskLevel::Low,
            _ => RiskLevel::Medium,
        }
    }
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            performance_baselines: HashMap::new(),
            bottleneck_detector: BottleneckDetector {
                cpu_threshold: 80.0,
                memory_threshold: 85.0,
                storage_threshold: 90.0,
                network_threshold: 75.0,
            },
        }
    }
}

impl SecurityOptimizer {
    pub fn new() -> Self {
        Self {
            security_policies: HashMap::new(),
            vulnerability_database: HashMap::new(),
        }
    }
}

impl ReliabilityMonitor {
    pub fn new() -> Self {
        Self {
            sla_targets: HashMap::new(),
            disaster_recovery_plans: HashMap::new(),
        }
    }
}

impl OperationalExcellenceAnalyzer {
    pub fn new() -> Self {
        Self {
            automation_opportunities: HashMap::new(),
            monitoring_gaps: HashMap::new(),
        }
    }
}

// Additional types for roadmap functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRoadmap {
    pub roadmap_id: String,
    pub scope: String,
    pub items: Vec<RoadmapItem>,
    pub total_estimated_savings: f64,
    pub implementation_phases: Vec<ImplementationPhase>,
    pub success_criteria: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapItem {
    pub item_id: String,
    pub title: String,
    pub category: OptimizationCategory,
    pub priority: RoadmapPriority,
    pub estimated_savings: Option<f64>,
    pub implementation_timeline: String,
    pub dependencies: Vec<String>,
    pub business_impact: String,
    pub technical_complexity: ImplementationEffort,
    pub risk_assessment: RiskLevel,
    pub success_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Cost,
    Performance,
    Security,
    Reliability,
    Operational,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoadmapPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationPhase {
    pub phase_number: u32,
    pub phase_name: String,
    pub description: String,
    pub duration_weeks: u32,
    pub estimated_savings: f64,
    pub focus_areas: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    pub total_recommendations: u32,
    pub high_impact_recommendations: u32,
    pub potential_annual_savings: f64,
    pub cost_optimization_score: f64,
    pub performance_optimization_score: f64,
    pub security_optimization_score: f64,
    pub reliability_optimization_score: f64,
    pub operational_excellence_score: f64,
    pub overall_advisor_score: f64,
    pub optimization_trend: TrendDirection,
    pub industry_percentile: f64,
}

// Legacy types for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_id: String,
    pub category: String,
    pub resource_id: String,
    pub title: String,
    pub description: String,
    pub impact: String,
    pub potential_savings: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    pub total_recommendations: u32,
    pub high_impact_count: u32,
    pub medium_impact_count: u32,
    pub low_impact_count: u32,
    pub potential_total_savings: f64,
}