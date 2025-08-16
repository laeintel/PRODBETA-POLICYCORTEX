// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Usage Metering Service for PolicyCortex
// Tracks API usage, predictions, and implements tiered pricing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

pub mod billing;
pub mod quotas;
pub mod analytics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetric {
    pub metric_id: Uuid,
    pub tenant_id: String,
    pub user_id: String,
    pub api_endpoint: String,
    pub operation_type: OperationType,
    pub resource_count: u64,
    pub compute_units: f64,
    pub data_processed_bytes: u64,
    pub response_time_ms: u64,
    pub status_code: u16,
    pub timestamp: DateTime<Utc>,
    pub billing_tier: BillingTier,
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Query,
    Prediction,
    Remediation,
    Analysis,
    Report,
    Training,
    Explanation,
    GraphTraversal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BillingTier {
    Free,
    Basic,
    Professional,
    Enterprise,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierQuota {
    pub tier: BillingTier,
    pub max_api_calls_per_month: u64,
    pub max_predictions_per_month: u64,
    pub max_resources_monitored: u64,
    pub max_users: u64,
    pub max_data_gb_per_month: u64,
    pub compute_units_per_month: f64,
    pub features: Vec<String>,
    pub sla_uptime: f64,
    pub support_level: String,
    pub price_per_month: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub tenant_id: String,
    pub billing_period: BillingPeriod,
    pub current_tier: BillingTier,
    pub api_calls: u64,
    pub predictions_made: u64,
    pub resources_monitored: u64,
    pub data_processed_gb: f64,
    pub compute_units_used: f64,
    pub total_cost: f64,
    pub quota_usage: QuotaUsage,
    pub top_endpoints: Vec<EndpointUsage>,
    pub daily_usage: Vec<DailyUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingPeriod {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaUsage {
    pub api_calls_percentage: f64,
    pub predictions_percentage: f64,
    pub resources_percentage: f64,
    pub data_percentage: f64,
    pub compute_percentage: f64,
    pub warnings: Vec<QuotaWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaWarning {
    pub metric: String,
    pub current_usage: f64,
    pub limit: f64,
    pub percentage: f64,
    pub estimated_exhaustion: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointUsage {
    pub endpoint: String,
    pub call_count: u64,
    pub avg_response_time_ms: f64,
    pub error_rate: f64,
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyUsage {
    pub date: DateTime<Utc>,
    pub api_calls: u64,
    pub predictions: u64,
    pub data_gb: f64,
    pub compute_units: f64,
    pub cost: f64,
}

pub struct MeteringService {
    metrics: Arc<RwLock<Vec<UsageMetric>>>,
    tenant_quotas: Arc<RwLock<HashMap<String, TierQuota>>>,
    tenant_usage: Arc<RwLock<HashMap<String, TenantUsage>>>,
    pricing_engine: Arc<PricingEngine>,
    quota_enforcer: Arc<QuotaEnforcer>,
}

struct TenantUsage {
    current_month_api_calls: u64,
    current_month_predictions: u64,
    current_month_data_gb: f64,
    current_month_compute_units: f64,
    active_resources: HashSet<String>,
    active_users: HashSet<String>,
}

impl MeteringService {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(Vec::new())),
            tenant_quotas: Arc::new(RwLock::new(Self::initialize_default_quotas())),
            tenant_usage: Arc::new(RwLock::new(HashMap::new())),
            pricing_engine: Arc::new(PricingEngine::new()),
            quota_enforcer: Arc::new(QuotaEnforcer::new()),
        }
    }

    fn initialize_default_quotas() -> HashMap<String, TierQuota> {
        let mut quotas = HashMap::new();
        
        // Free Tier
        quotas.insert("free".to_string(), TierQuota {
            tier: BillingTier::Free,
            max_api_calls_per_month: 1000,
            max_predictions_per_month: 100,
            max_resources_monitored: 50,
            max_users: 3,
            max_data_gb_per_month: 1,
            compute_units_per_month: 10.0,
            features: vec![
                "basic_monitoring".to_string(),
                "compliance_checks".to_string(),
            ],
            sla_uptime: 99.0,
            support_level: "community".to_string(),
            price_per_month: 0.0,
        });

        // Basic Tier
        quotas.insert("basic".to_string(), TierQuota {
            tier: BillingTier::Basic,
            max_api_calls_per_month: 10000,
            max_predictions_per_month: 1000,
            max_resources_monitored: 500,
            max_users: 10,
            max_data_gb_per_month: 10,
            compute_units_per_month: 100.0,
            features: vec![
                "basic_monitoring".to_string(),
                "compliance_checks".to_string(),
                "basic_remediation".to_string(),
                "reporting".to_string(),
            ],
            sla_uptime: 99.5,
            support_level: "email".to_string(),
            price_per_month: 299.0,
        });

        // Professional Tier
        quotas.insert("professional".to_string(), TierQuota {
            tier: BillingTier::Professional,
            max_api_calls_per_month: 100000,
            max_predictions_per_month: 10000,
            max_resources_monitored: 5000,
            max_users: 50,
            max_data_gb_per_month: 100,
            compute_units_per_month: 1000.0,
            features: vec![
                "advanced_monitoring".to_string(),
                "compliance_automation".to_string(),
                "auto_remediation".to_string(),
                "advanced_reporting".to_string(),
                "ml_predictions".to_string(),
                "api_access".to_string(),
            ],
            sla_uptime: 99.9,
            support_level: "priority".to_string(),
            price_per_month: 1499.0,
        });

        // Enterprise Tier
        quotas.insert("enterprise".to_string(), TierQuota {
            tier: BillingTier::Enterprise,
            max_api_calls_per_month: u64::MAX,
            max_predictions_per_month: u64::MAX,
            max_resources_monitored: u64::MAX,
            max_users: u64::MAX,
            max_data_gb_per_month: u64::MAX as f64,
            compute_units_per_month: f64::MAX,
            features: vec![
                "all_features".to_string(),
                "custom_models".to_string(),
                "white_label".to_string(),
                "dedicated_support".to_string(),
                "sla_guarantee".to_string(),
            ],
            sla_uptime: 99.99,
            support_level: "dedicated".to_string(),
            price_per_month: 0.0, // Custom pricing
        });

        quotas
    }

    pub async fn record_usage(&self, metric: UsageMetric) -> Result<(), String> {
        // Check quota before recording
        self.quota_enforcer.check_quota(&metric.tenant_id, &metric.operation_type).await?;
        
        // Calculate cost
        let cost = self.pricing_engine.calculate_cost(&metric);
        let mut metric_with_cost = metric;
        metric_with_cost.cost = cost;
        
        // Record the metric
        let mut metrics = self.metrics.write().await;
        metrics.push(metric_with_cost.clone());
        
        // Update tenant usage
        let mut usage = self.tenant_usage.write().await;
        let tenant_usage = usage.entry(metric_with_cost.tenant_id.clone())
            .or_insert_with(|| TenantUsage {
                current_month_api_calls: 0,
                current_month_predictions: 0,
                current_month_data_gb: 0.0,
                current_month_compute_units: 0.0,
                active_resources: HashSet::new(),
                active_users: HashSet::new(),
            });
        
        tenant_usage.current_month_api_calls += 1;
        if matches!(metric_with_cost.operation_type, OperationType::Prediction) {
            tenant_usage.current_month_predictions += 1;
        }
        tenant_usage.current_month_data_gb += metric_with_cost.data_processed_bytes as f64 / 1_073_741_824.0;
        tenant_usage.current_month_compute_units += metric_with_cost.compute_units;
        
        Ok(())
    }

    pub async fn get_usage_summary(&self, tenant_id: &str) -> Result<UsageSummary, String> {
        let now = Utc::now();
        let start_of_month = DateTime::from_utc(
            chrono::NaiveDate::from_ymd(now.year(), now.month(), 1).and_hms(0, 0, 0),
            Utc
        );
        let end_of_month = start_of_month + Duration::days(30);
        
        let billing_period = BillingPeriod {
            start: start_of_month,
            end: end_of_month,
        };
        
        // Get tenant's tier and quota
        let quotas = self.tenant_quotas.read().await;
        let tenant_quota = quotas.get("professional") // Default to professional
            .ok_or("Quota not found")?;
        
        // Get usage statistics
        let metrics = self.metrics.read().await;
        let tenant_metrics: Vec<_> = metrics.iter()
            .filter(|m| m.tenant_id == tenant_id && m.timestamp >= start_of_month)
            .cloned()
            .collect();
        
        let api_calls = tenant_metrics.len() as u64;
        let predictions_made = tenant_metrics.iter()
            .filter(|m| matches!(m.operation_type, OperationType::Prediction))
            .count() as u64;
        
        let data_processed_gb = tenant_metrics.iter()
            .map(|m| m.data_processed_bytes as f64 / 1_073_741_824.0)
            .sum();
        
        let compute_units_used = tenant_metrics.iter()
            .map(|m| m.compute_units)
            .sum();
        
        let total_cost = tenant_metrics.iter()
            .map(|m| m.cost)
            .sum();
        
        // Calculate quota usage
        let quota_usage = QuotaUsage {
            api_calls_percentage: (api_calls as f64 / tenant_quota.max_api_calls_per_month as f64) * 100.0,
            predictions_percentage: (predictions_made as f64 / tenant_quota.max_predictions_per_month as f64) * 100.0,
            resources_percentage: 0.0, // Would calculate from active resources
            data_percentage: (data_processed_gb / tenant_quota.max_data_gb_per_month) * 100.0,
            compute_percentage: (compute_units_used / tenant_quota.compute_units_per_month) * 100.0,
            warnings: self.generate_quota_warnings(tenant_quota, &tenant_metrics),
        };
        
        // Get top endpoints
        let mut endpoint_stats: HashMap<String, (u64, f64, u64, f64)> = HashMap::new();
        for metric in &tenant_metrics {
            let entry = endpoint_stats.entry(metric.api_endpoint.clone())
                .or_insert((0, 0.0, 0, 0.0));
            entry.0 += 1; // count
            entry.1 += metric.response_time_ms as f64; // total time
            if metric.status_code >= 400 {
                entry.2 += 1; // errors
            }
            entry.3 += metric.cost; // cost
        }
        
        let mut top_endpoints: Vec<_> = endpoint_stats.into_iter()
            .map(|(endpoint, (count, total_time, errors, cost))| EndpointUsage {
                endpoint,
                call_count: count,
                avg_response_time_ms: if count > 0 { total_time / count as f64 } else { 0.0 },
                error_rate: if count > 0 { errors as f64 / count as f64 } else { 0.0 },
                cost,
            })
            .collect();
        
        top_endpoints.sort_by(|a, b| b.call_count.cmp(&a.call_count));
        top_endpoints.truncate(10);
        
        // Generate daily usage
        let daily_usage = self.generate_daily_usage(&tenant_metrics, start_of_month);
        
        Ok(UsageSummary {
            tenant_id: tenant_id.to_string(),
            billing_period,
            current_tier: tenant_quota.tier.clone(),
            api_calls,
            predictions_made,
            resources_monitored: 0, // Would get from tenant usage
            data_processed_gb,
            compute_units_used,
            total_cost,
            quota_usage,
            top_endpoints,
            daily_usage,
        })
    }

    fn generate_quota_warnings(&self, quota: &TierQuota, metrics: &[UsageMetric]) -> Vec<QuotaWarning> {
        let mut warnings = Vec::new();
        
        let api_calls = metrics.len() as f64;
        let api_percentage = (api_calls / quota.max_api_calls_per_month as f64) * 100.0;
        
        if api_percentage > 80.0 {
            warnings.push(QuotaWarning {
                metric: "API Calls".to_string(),
                current_usage: api_calls,
                limit: quota.max_api_calls_per_month as f64,
                percentage: api_percentage,
                estimated_exhaustion: self.estimate_exhaustion(api_calls, quota.max_api_calls_per_month as f64),
            });
        }
        
        warnings
    }

    fn estimate_exhaustion(&self, current: f64, limit: f64) -> Option<DateTime<Utc>> {
        if current >= limit {
            return Some(Utc::now());
        }
        
        let days_in_month = 30;
        let day_of_month = Utc::now().day();
        let daily_rate = current / day_of_month as f64;
        let days_to_exhaustion = (limit - current) / daily_rate;
        
        if days_to_exhaustion < days_in_month as f64 {
            Some(Utc::now() + Duration::days(days_to_exhaustion as i64))
        } else {
            None
        }
    }

    fn generate_daily_usage(&self, metrics: &[UsageMetric], start_date: DateTime<Utc>) -> Vec<DailyUsage> {
        let mut daily_map: HashMap<String, (u64, u64, f64, f64, f64)> = HashMap::new();
        
        for metric in metrics {
            let date_key = metric.timestamp.date().to_string();
            let entry = daily_map.entry(date_key).or_insert((0, 0, 0.0, 0.0, 0.0));
            
            entry.0 += 1; // api calls
            if matches!(metric.operation_type, OperationType::Prediction) {
                entry.1 += 1; // predictions
            }
            entry.2 += metric.data_processed_bytes as f64 / 1_073_741_824.0; // data gb
            entry.3 += metric.compute_units; // compute
            entry.4 += metric.cost; // cost
        }
        
        let mut daily_usage: Vec<_> = daily_map.into_iter()
            .map(|(date_str, (calls, predictions, data, compute, cost))| {
                let date = date_str.parse::<chrono::NaiveDate>()
                    .unwrap_or_else(|_| Utc::now().date().naive_utc());
                DailyUsage {
                    date: DateTime::from_utc(date.and_hms(0, 0, 0), Utc),
                    api_calls: calls,
                    predictions,
                    data_gb: data,
                    compute_units: compute,
                    cost,
                }
            })
            .collect();
        
        daily_usage.sort_by_key(|d| d.date);
        daily_usage
    }

    pub async fn upgrade_tier(&self, tenant_id: &str, new_tier: BillingTier) -> Result<(), String> {
        let mut quotas = self.tenant_quotas.write().await;
        
        let tier_name = match new_tier {
            BillingTier::Free => "free",
            BillingTier::Basic => "basic",
            BillingTier::Professional => "professional",
            BillingTier::Enterprise => "enterprise",
            BillingTier::Custom => return Err("Custom tier requires contacting sales".to_string()),
        };
        
        if let Some(new_quota) = Self::initialize_default_quotas().get(tier_name) {
            quotas.insert(tenant_id.to_string(), new_quota.clone());
            Ok(())
        } else {
            Err("Invalid tier".to_string())
        }
    }
}

struct PricingEngine {
    // Pricing calculation logic
}

impl PricingEngine {
    fn new() -> Self {
        Self {}
    }

    fn calculate_cost(&self, metric: &UsageMetric) -> f64 {
        let base_cost = match metric.operation_type {
            OperationType::Query => 0.001,
            OperationType::Prediction => 0.01,
            OperationType::Remediation => 0.05,
            OperationType::Analysis => 0.02,
            OperationType::Report => 0.03,
            OperationType::Training => 0.10,
            OperationType::Explanation => 0.015,
            OperationType::GraphTraversal => 0.008,
        };
        
        // Add data processing cost
        let data_cost = (metric.data_processed_bytes as f64 / 1_073_741_824.0) * 0.02; // $0.02 per GB
        
        // Add compute cost
        let compute_cost = metric.compute_units * 0.005; // $0.005 per compute unit
        
        base_cost + data_cost + compute_cost
    }
}

struct QuotaEnforcer {
    // Quota enforcement logic
}

impl QuotaEnforcer {
    fn new() -> Self {
        Self {}
    }

    async fn check_quota(&self, tenant_id: &str, operation: &OperationType) -> Result<(), String> {
        // In production, would check actual quotas
        // For now, always allow
        Ok(())
    }
}

use std::collections::HashSet;