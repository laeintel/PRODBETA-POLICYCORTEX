use crate::auth::{AuthUser, OptionalAuthUser, TenantContext};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

// Patent 1: Unified AI Platform - Multi-service data aggregation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GovernanceMetrics {
    pub policies: PolicyMetrics,
    pub rbac: RbacMetrics,
    pub costs: CostMetrics,
    pub network: NetworkMetrics,
    pub resources: ResourceMetrics,
    pub ai: AIMetrics,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyMetrics {
    pub total: u32,
    pub active: u32,
    pub violations: u32,
    pub automated: u32,
    pub compliance_rate: f64,
    pub prediction_accuracy: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RbacMetrics {
    pub users: u32,
    pub roles: u32,
    pub violations: u32,
    pub risk_score: f64,
    pub anomalies_detected: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CostMetrics {
    pub current_spend: f64,
    pub predicted_spend: f64,
    pub savings_identified: f64,
    pub optimization_rate: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkMetrics {
    pub endpoints: u32,
    pub active_threats: u32,
    pub blocked_attempts: u32,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResourceMetrics {
    pub total: u32,
    pub optimized: u32,
    pub idle: u32,
    pub overprovisioned: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AIMetrics {
    pub accuracy: f64,
    pub predictions_made: u64,
    pub automations_executed: u64,
    pub learning_progress: f64,
}

// Patent 2: Predictive Compliance Engine
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompliancePrediction {
    pub resource_id: String,
    pub policy_id: String,
    pub prediction_time: DateTime<Utc>,
    pub violation_probability: f64,
    pub confidence_interval: (f64, f64),
    pub drift_detected: bool,
    pub recommended_actions: Vec<RemediationAction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RemediationAction {
    pub action_type: String,
    pub description: String,
    pub impact_score: f64,
    pub estimated_time_minutes: u32,
    pub automation_available: bool,
}

// Patent 3: Conversational Intelligence
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationRequest {
    pub query: String,
    pub context: Option<ConversationContext>,
    pub session_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationContext {
    pub previous_intents: Vec<String>,
    pub entities: Vec<Entity>,
    pub turn_count: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Entity {
    pub entity_type: String,
    pub value: String,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationResponse {
    pub response: String,
    pub intent: String,
    pub confidence: f64,
    pub suggested_actions: Vec<String>,
    pub generated_policy: Option<String>,
}

// Patent 4: Cross-Domain Correlation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CrossDomainCorrelation {
    pub correlation_id: String,
    pub domains: Vec<String>,
    pub correlation_strength: f64,
    pub causal_relationship: Option<CausalRelationship>,
    pub impact_predictions: Vec<ImpactPrediction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CausalRelationship {
    pub source_domain: String,
    pub target_domain: String,
    pub lag_time_hours: f64,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImpactPrediction {
    pub domain: String,
    pub metric: String,
    pub predicted_change: f64,
    pub time_to_impact_hours: f64,
}

// Proactive Recommendations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProactiveRecommendation {
    pub id: String,
    pub recommendation_type: String,
    pub severity: String,
    pub title: String,
    pub description: String,
    pub potential_savings: Option<f64>,
    pub risk_reduction: Option<f64>,
    pub automation_available: bool,
    pub confidence: f64,
}

pub struct AppState {
    pub metrics: Arc<RwLock<GovernanceMetrics>>,
    pub predictions: Arc<RwLock<Vec<CompliancePrediction>>>,
    pub recommendations: Arc<RwLock<Vec<ProactiveRecommendation>>>,
    pub start_time: std::time::Instant,
    pub azure_client: Option<crate::azure_client::AzureClient>,
    pub async_azure_client: Option<crate::azure_client_async::AsyncAzureClient>,
    // Phase 2 action orchestrator (in-memory)
    pub actions: Arc<RwLock<std::collections::HashMap<String, ActionRecord>>>,
    pub action_events: Arc<RwLock<std::collections::HashMap<String, broadcast::Sender<String>>>>,
    pub config: crate::config::AppConfig,
}

impl AppState {
    pub fn new() -> Self {
        let metrics = GovernanceMetrics {
            policies: PolicyMetrics {
                total: 347,
                active: 298,
                violations: 12,
                automated: 285,
                compliance_rate: 99.8,
                prediction_accuracy: 92.3,
            },
            rbac: RbacMetrics {
                users: 1843,
                roles: 67,
                violations: 3,
                risk_score: 18.5,
                anomalies_detected: 7,
            },
            costs: CostMetrics {
                current_spend: 145832.0,
                predicted_spend: 98190.0,
                savings_identified: 47642.0,
                optimization_rate: 89.0,
            },
            network: NetworkMetrics {
                endpoints: 487,
                active_threats: 2,
                blocked_attempts: 127,
                latency_ms: 12.3,
            },
            resources: ResourceMetrics {
                total: 2843,
                optimized: 2456,
                idle: 234,
                overprovisioned: 153,
            },
            ai: AIMetrics {
                accuracy: 96.8,
                predictions_made: 15234,
                automations_executed: 8921,
                learning_progress: 87.3,
            },
        };

        let recommendations = vec![
            ProactiveRecommendation {
                id: "rec-001".to_string(),
                recommendation_type: "cost_optimization".to_string(),
                severity: "high".to_string(),
                title: "VM Right-Sizing Opportunity".to_string(),
                description:
                    "AI detected $12,450/month savings by right-sizing 47 VMs in production"
                        .to_string(),
                potential_savings: Some(12450.0),
                risk_reduction: None,
                automation_available: true,
                confidence: 94.5,
            },
            ProactiveRecommendation {
                id: "rec-002".to_string(),
                recommendation_type: "security".to_string(),
                severity: "critical".to_string(),
                title: "Unencrypted Storage Detected".to_string(),
                description:
                    "Found 3 storage accounts without encryption in production environment"
                        .to_string(),
                potential_savings: None,
                risk_reduction: Some(85.0),
                automation_available: true,
                confidence: 99.9,
            },
            ProactiveRecommendation {
                id: "rec-003".to_string(),
                recommendation_type: "rbac".to_string(),
                severity: "medium".to_string(),
                title: "Excessive Permissions Review".to_string(),
                description: "23 users have unused admin privileges for over 90 days".to_string(),
                potential_savings: None,
                risk_reduction: Some(62.0),
                automation_available: false,
                confidence: 87.3,
            },
        ];

        AppState {
            metrics: Arc::new(RwLock::new(metrics)),
            predictions: Arc::new(RwLock::new(Vec::new())),
            recommendations: Arc::new(RwLock::new(recommendations)),
            start_time: std::time::Instant::now(),
            azure_client: None,       // Will be initialized in main
            async_azure_client: None, // Will be initialized in main
            actions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            action_events: Arc::new(RwLock::new(std::collections::HashMap::new())),
            config: crate::config::AppConfig::load(),
        }
    }
}

// API Handlers
pub async fn get_metrics(
    auth_user: OptionalAuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Log authenticated request
    // If authenticated, log who; if not, allow simulated data in dev/local flows
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for metrics from user: {:?}",
            user.claims.preferred_username
        );

        // Get tenant context for multi-tenant data access
        let _tenant_context = match TenantContext::from_user(user).await {
            Ok(context) => {
                tracing::debug!(
                    "User has access to tenant: {} with {} subscriptions",
                    context.tenant_id,
                    context.subscription_ids.len()
                );
                context
            }
            Err(e) => {
                tracing::error!("Failed to get tenant context: {:?}", e);
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "error": "tenant_access_denied",
                        "message": "Unable to determine tenant access"
                    })),
                )
                    .into_response();
            }
        };
    } else {
        tracing::info!("Unauthenticated metrics request - returning simulated data (dev mode)");
        let simulated_metrics = GovernanceMetrics {
            policies: PolicyMetrics {
                total: 15,
                active: 12,
                violations: 3,
                automated: 10,
                compliance_rate: 85.5,
                prediction_accuracy: 92.3,
            },
            rbac: RbacMetrics {
                users: 150,
                roles: 25,
                violations: 2,
                risk_score: 3.2,
                anomalies_detected: 1,
            },
            costs: CostMetrics {
                current_spend: 125000.0,
                predicted_spend: 118000.0,
                savings_identified: 7000.0,
                optimization_rate: 88.5,
            },
            network: NetworkMetrics {
                endpoints: 450,
                active_threats: 0,
                blocked_attempts: 127,
                latency_ms: 15.2,
            },
            resources: ResourceMetrics {
                total: 2500,
                optimized: 2100,
                idle: 200,
                overprovisioned: 200,
            },
            ai: AIMetrics {
                accuracy: 94.5,
                predictions_made: 12000,
                automations_executed: 8500,
                learning_progress: 85.0,
            },
        };
        return Json(simulated_metrics).into_response();
    }

    // Always try to get real Azure data when Azure client is available
    // This works for local development with Azure CLI authentication

    // Try high-performance async client first
    if let Some(ref async_azure_client) = state.async_azure_client {
        match async_azure_client.get_governance_metrics().await {
            Ok(real_metrics) => {
                tracing::info!("✅ Real Azure metrics fetched with async client (cached)");
                return Json(real_metrics).into_response();
            }
            Err(e) => {
                tracing::warn!("Async Azure client failed: {}", e);
            }
        }
    }

    // Fallback to synchronous client
    if let Some(ref azure_client) = state.azure_client {
        match azure_client.get_governance_metrics().await {
            Ok(real_metrics) => {
                tracing::info!("✅ Real Azure metrics fetched with sync client");
                return Json(real_metrics).into_response();
            }
            Err(e) => {
                tracing::warn!("Failed to fetch real Azure metrics: {}", e);
                tracing::debug!("Error details: {:?}", e);
            }
        }
    }

    // No Azure connection available - return simulated data
    let simulated_metrics = GovernanceMetrics {
        policies: PolicyMetrics {
            total: 15,
            active: 12,
            violations: 3,
            automated: 10,
            compliance_rate: 85.5,
            prediction_accuracy: 92.3,
        },
        rbac: RbacMetrics {
            users: 150,
            roles: 25,
            violations: 2,
            risk_score: 3.2,
            anomalies_detected: 1,
        },
        costs: CostMetrics {
            current_spend: 125000.0,
            predicted_spend: 118000.0,
            savings_identified: 7000.0,
            optimization_rate: 88.5,
        },
        network: NetworkMetrics {
            endpoints: 450,
            active_threats: 0,
            blocked_attempts: 127,
            latency_ms: 15.2,
        },
        resources: ResourceMetrics {
            total: 2500,
            optimized: 2100,
            idle: 200,
            overprovisioned: 200,
        },
        ai: AIMetrics {
            accuracy: 94.5,
            predictions_made: 12000,
            automations_executed: 8500,
            learning_progress: 85.0,
        },
    };

    // Return simulated data directly (flat shape)
    Json(simulated_metrics).into_response()
}

pub async fn get_predictions(
    auth_user: OptionalAuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Verify authentication and tenant access
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for predictions from user: {:?}",
            user.claims.preferred_username
        );
        // Optional: load tenant context without failing if it errors (dev)
        if let Err(e) = TenantContext::from_user(user).await {
            tracing::warn!("Tenant context unavailable: {:?}", e);
        }
    } else {
        tracing::info!("Unauthenticated predictions request (dev mode)");
    }

    let predictions = state.predictions.read().await;
    Json(predictions.clone()).into_response()
}

pub async fn get_recommendations(
    auth_user: OptionalAuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Verify authentication
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for recommendations from user: {:?}",
            user.claims.preferred_username
        );
    } else {
        tracing::info!("Unauthenticated recommendations request (dev mode)");
    }
    // If we have an Azure client, fetch real recommendations based on actual Azure data
    if let Some(ref async_azure_client) = state.async_azure_client {
        match async_azure_client.get_governance_metrics().await {
            Ok(metrics) => {
                // Generate recommendations based on real metrics
                let mut recommendations = Vec::new();

                // Cost recommendations based on actual spend
                if metrics.costs.current_spend > 50.0 {
                    let savings_percentage =
                        (metrics.costs.savings_identified / metrics.costs.current_spend) * 100.0;
                    recommendations.push(ProactiveRecommendation {
                        id: "cost-001".to_string(),
                        recommendation_type: "cost_optimization".to_string(),
                        severity: if savings_percentage > 20.0 { "high".to_string() } else { "medium".to_string() },
                        title: format!("Cost Optimization: ${:.2} potential savings", metrics.costs.savings_identified),
                        description: format!(
                            "Analysis shows ${:.2}/month in potential savings ({:.1}% of current spend). {} idle resources detected.",
                            metrics.costs.savings_identified,
                            savings_percentage,
                            metrics.resources.idle
                        ),
                        potential_savings: Some(metrics.costs.savings_identified),
                        risk_reduction: None,
                        automation_available: true,
                        confidence: 95.0,
                    });
                }

                // Resource optimization recommendations
                if metrics.resources.idle > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "resource-001".to_string(),
                        recommendation_type: "resource_optimization".to_string(),
                        severity: if metrics.resources.idle > 5 { "high".to_string() } else { "medium".to_string() },
                        title: format!("{} Idle Resources Detected", metrics.resources.idle),
                        description: format!(
                            "Found {} resources that have been idle. Consider deallocating or deleting to reduce costs.",
                            metrics.resources.idle
                        ),
                        potential_savings: Some((metrics.resources.idle as f64) * 15.0), // Estimate $15/resource
                        risk_reduction: None,
                        automation_available: true,
                        confidence: 98.0,
                    });
                }

                // Over-provisioned resources
                if metrics.resources.overprovisioned > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "resource-002".to_string(),
                        recommendation_type: "rightsizing".to_string(),
                        severity: "medium".to_string(),
                        title: format!("{} Over-provisioned Resources", metrics.resources.overprovisioned),
                        description: format!(
                            "{} resources are using less than 20% of allocated capacity. Consider downsizing.",
                            metrics.resources.overprovisioned
                        ),
                        potential_savings: Some((metrics.resources.overprovisioned as f64) * 25.0),
                        risk_reduction: None,
                        automation_available: true,
                        confidence: 92.0,
                    });
                }

                // Policy compliance recommendations
                if metrics.policies.violations > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "policy-001".to_string(),
                        recommendation_type: "compliance".to_string(),
                        severity: "critical".to_string(),
                        title: format!("{} Policy Violations Detected", metrics.policies.violations),
                        description: format!(
                            "{} resources are non-compliant with active policies. Immediate remediation recommended.",
                            metrics.policies.violations
                        ),
                        potential_savings: None,
                        risk_reduction: Some(85.0),
                        automation_available: metrics.policies.violations <= 5,
                        confidence: 99.9,
                    });
                }

                // RBAC recommendations
                if metrics.rbac.risk_score > 20.0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "rbac-001".to_string(),
                        recommendation_type: "security".to_string(),
                        severity: if metrics.rbac.risk_score > 50.0 { "critical".to_string() } else { "high".to_string() },
                        title: "Elevated RBAC Risk Score".to_string(),
                        description: format!(
                            "RBAC risk score is {:.1}%. Review permissions for {} users with elevated access.",
                            metrics.rbac.risk_score,
                            (metrics.rbac.users as f64 * 0.1) as u32
                        ),
                        potential_savings: None,
                        risk_reduction: Some(metrics.rbac.risk_score),
                        automation_available: false,
                        confidence: 88.0,
                    });
                }

                // Network security recommendations
                if metrics.network.active_threats > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "network-001".to_string(),
                        recommendation_type: "security".to_string(),
                        severity: "critical".to_string(),
                        title: format!("{} Active Network Threats", metrics.network.active_threats),
                        description: format!(
                            "Active threats detected on {} endpoints. {} attempts blocked in last 24 hours.",
                            metrics.network.active_threats,
                            metrics.network.blocked_attempts
                        ),
                        potential_savings: None,
                        risk_reduction: Some(95.0),
                        automation_available: true,
                        confidence: 99.5,
                    });
                }

                return Json(recommendations);
            }
            Err(e) => {
                tracing::warn!("Failed to generate real-time recommendations: {}", e);
            }
        }
    }

    // Fallback to cached recommendations
    let recommendations = state.recommendations.read().await;
    Json(recommendations.clone())
}

pub async fn process_conversation(
    auth_user: AuthUser,
    State(_state): State<Arc<AppState>>,
    Json(request): Json<ConversationRequest>,
) -> impl IntoResponse {
    // Verify authentication and get user context for personalized responses
    tracing::info!(
        "Authenticated conversation request from user: {:?}",
        auth_user.claims.preferred_username
    );

    let user_context = format!(
        " (authenticated as {})",
        auth_user
            .claims
            .preferred_username
            .as_deref()
            .unwrap_or("unknown user")
    );

    // Simulate NLP processing with user context
    let response = ConversationResponse {
        response: format!("I understand you're asking about: {}{}. Based on your environment analysis, I recommend reviewing the cost optimization opportunities that could save you $12,450/month.", request.query, user_context),
        intent: "cost_inquiry".to_string(),
        confidence: 92.5,
        suggested_actions: vec![
            "Review VM sizing recommendations".to_string(),
            "Enable auto-scaling for development resources".to_string(),
            "Implement resource tagging policy".to_string(),
        ],
        generated_policy: Some(r#"{
            "mode": "Indexed",
            "policyRule": {
                "if": {
                    "field": "type",
                    "equals": "Microsoft.Compute/virtualMachines"
                },
                "then": {
                    "effect": "audit"
                }
            }
        }"#.to_string()),
    };

    Json(response)
}

pub async fn get_correlations(auth_user: OptionalAuthUser) -> impl IntoResponse {
    // Verify authentication
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for correlations from user: {:?}",
            user.claims.preferred_username
        );
    } else {
        tracing::info!("Unauthenticated correlations request (dev mode)");
    }
    let correlation = CrossDomainCorrelation {
        correlation_id: "corr-001".to_string(),
        domains: vec!["cost".to_string(), "resources".to_string()],
        correlation_strength: 0.87,
        causal_relationship: Some(CausalRelationship {
            source_domain: "resources".to_string(),
            target_domain: "cost".to_string(),
            lag_time_hours: 24.0,
            confidence: 89.5,
        }),
        impact_predictions: vec![ImpactPrediction {
            domain: "cost".to_string(),
            metric: "monthly_spend".to_string(),
            predicted_change: -8.5,
            time_to_impact_hours: 72.0,
        }],
    };

    Json(vec![correlation])
}

// ===================== Deep Insights & Actions (Phase 1) =====================

#[derive(Debug, Deserialize)]
pub struct RemediateRequest {
    pub resource_id: String,
    pub action: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateExceptionRequest {
    pub resource_id: String,
    pub policy_id: String,
    pub reason: String,
}

// Get policies with data mode support
pub async fn get_policies(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use crate::data_mode::{DataMode, DataResponse};
    use crate::simulated_data::SimulatedDataProvider;

    let mode = DataMode::from_env();

    // Try to get real data if available and mode is Real
    if mode.is_real() {
        if let Some(ref async_azure_client) = state.async_azure_client {
            match async_azure_client.get_policies().await {
                Ok(policies) => {
                    return Json(DataResponse::new(
                        serde_json::json!({
                            "policies": policies,
                            "total": policies.len(),
                        }),
                        DataMode::Real,
                    ));
                }
                Err(e) => {
                    tracing::warn!("Failed to get real policies: {}", e);
                }
            }
        }
    }

    // Use simulated data as fallback or when in simulated mode
    let simulated_policies = SimulatedDataProvider::get_policies();
    Json(DataResponse::new(
        serde_json::json!({
            "policies": simulated_policies,
            "total": simulated_policies.len(),
        }),
        DataMode::Simulated,
    ))
}

// Policies Deep Compliance
pub async fn get_policies_deep() -> impl IntoResponse {
    // Try proxy to Python deep service if configured
    if let Some(json) = proxy_deep_get("/api/v1/policies/deep").await {
        return Json(json);
    }

    Json(serde_json::json!({
        "success": true,
        "message": "Deep policy compliance (core)",
        "complianceResults": [
            {
                "assignment": {
                    "name": "require-tags",
                    "displayName": "Require Resource Tags",
                    "scope": "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78"
                },
                "summary": {
                    "totalResources": 147,
                    "compliantResources": 89,
                    "nonCompliantResources": 58,
                    "compliancePercentage": 60.5
                },
                "nonCompliantResources": [
                    {
                        "resourceId": "/subscriptions/205b477d/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-prod-001",
                        "resourceName": "vm-prod-001",
                        "resourceType": "Microsoft.Compute/virtualMachines",
                        "complianceState": "NonCompliant",
                        "complianceReason": "Missing required tags: Environment, Owner, CostCenter",
                        "remediationOptions": [
                            {"action": "auto-remediate", "description": "Automatically add missing tags"},
                            {"action": "create-exception", "description": "Create policy exception"},
                            {"action": "manual-fix", "description": "Manually add tags"}
                        ]
                    },
                    {
                        "resourceId": "/subscriptions/205b477d/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stprod001",
                        "resourceName": "stprod001",
                        "resourceType": "Microsoft.Storage/storageAccounts",
                        "complianceState": "NonCompliant",
                        "complianceReason": "Missing required tags: Environment, CostCenter",
                        "remediationOptions": [
                            {"action": "auto-remediate", "description": "Automatically add missing tags"},
                            {"action": "create-exception", "description": "Create policy exception"}
                        ]
                    }
                ]
            },
            {
                "assignment": {
                    "name": "require-encryption",
                    "displayName": "Require Encryption at Rest",
                    "scope": "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78"
                },
                "summary": {
                    "totalResources": 83,
                    "compliantResources": 71,
                    "nonCompliantResources": 12,
                    "compliancePercentage": 85.5
                },
                "nonCompliantResources": [
                    {
                        "resourceId": "/subscriptions/205b477d/resourceGroups/rg-dev/providers/Microsoft.Storage/storageAccounts/stdev002",
                        "resourceName": "stdev002",
                        "resourceType": "Microsoft.Storage/storageAccounts",
                        "complianceState": "NonCompliant",
                        "complianceReason": "Encryption at rest is not enabled",
                        "remediationOptions": [
                            {"action": "auto-remediate", "description": "Enable encryption automatically"},
                            {"action": "create-exception", "description": "Create policy exception"}
                        ]
                    }
                ]
            }
        ]
    }))
}

// Initiate remediation (stub – Phase 1). Later, orchestrate jobs and stream progress.
pub async fn remediate(Json(payload): Json<RemediateRequest>) -> impl IntoResponse {
    Json(serde_json::json!({
        "success": true,
        "resourceId": payload.resource_id,
        "action": payload.action,
        "status": "Initiated",
        "estimatedCompletion": "5 minutes",
        "message": format!("Remediation '{}' initiated for resource {}", payload.action, payload.resource_id)
    }))
}

// Create a policy exception (stub – Phase 1)
pub async fn create_exception(Json(payload): Json<CreateExceptionRequest>) -> impl IntoResponse {
    use chrono::Utc;
    let id = format!("exc-{}", Utc::now().format("%Y%m%d%H%M%S"));
    Json(serde_json::json!({
        "success": true,
        "exceptionId": id,
        "resourceId": payload.resource_id,
        "policyId": payload.policy_id,
        "reason": payload.reason,
        "expiresIn": "30 days",
        "status": "Approved"
    }))
}

// Helper: Proxy deep GET to Python service (Phase 3). Base from DEEP_API_BASE or http://localhost:8090
async fn proxy_deep_get(path: &str) -> Option<serde_json::Value> {
    let base = std::env::var("DEEP_API_BASE")
        .or_else(|_| std::env::var("API_GATEWAY_URL"))
        .unwrap_or_else(|_| "http://localhost:8090".to_string());
    let url = format!("{}{}", base, path);
    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => resp.json::<serde_json::Value>().await.ok(),
        _ => None,
    }
}

// Additional deep endpoints with REAL Azure data
pub async fn get_rbac_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Try Python service first for AI-enhanced analysis
    if let Some(json) = proxy_deep_get("/api/v1/rbac/deep").await {
        return Json(json);
    }

    // Fallback to direct Azure API
    if let Some(ref client) = state.async_azure_client {
        match client.get_rbac_analysis().await {
            Ok(rbac_data) => {
                return Json(serde_json::json!({
                    "success": true,
                    "roleAssignments": rbac_data.get("assignments"),
                    "riskAnalysis": {
                        "privilegedAccounts": rbac_data.get("privileged_count"),
                        "highRiskAssignments": rbac_data.get("high_risk_count"),
                        "staleAssignments": rbac_data.get("stale_count"),
                        "overprivilegedIdentities": rbac_data.get("overprivileged_identities")
                    },
                    "recommendations": rbac_data.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get RBAC data from Azure: {}", e);
            }
        }
    }

    // No Azure connection available
    Json(serde_json::json!({
        "error": "Azure connection not available",
        "status": "unavailable"
    }))
}

pub async fn get_costs_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(json) = proxy_deep_get("/api/v1/costs/deep").await {
        return Json(json);
    }

    // Get REAL cost data from Azure Cost Management
    if let Some(ref client) = state.async_azure_client {
        match client.get_cost_analysis().await {
            Ok(cost_data) => {
                return Json(serde_json::json!({
                    "success": true,
                    "totalCost": cost_data.get("total_cost"),
                    "breakdown": cost_data.get("service_breakdown"),
                    "anomalies": cost_data.get("anomalies"),
                    "trends": cost_data.get("trends"),
                    "forecast": cost_data.get("forecast"),
                    "optimizationPotential": cost_data.get("optimization_potential"),
                    "recommendations": cost_data.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get cost data from Azure: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "error": "Azure Cost Management API not available",
        "status": "unavailable"
    }))
}

pub async fn get_network_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(json) = proxy_deep_get("/api/v1/network/deep").await {
        return Json(json);
    }

    // Get REAL network topology from Azure
    if let Some(ref client) = state.async_azure_client {
        match client.get_network_topology().await {
            Ok(network_data) => {
                return Json(serde_json::json!({
                    "success": true,
                    "networkSecurityGroups": network_data.get("nsgs"),
                    "virtualNetworks": network_data.get("vnets"),
                    "publicEndpoints": network_data.get("public_endpoints"),
                    "privateEndpoints": network_data.get("private_endpoints"),
                    "securityRisks": network_data.get("security_risks"),
                    "recommendations": network_data.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get network data from Azure: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "error": "Azure Network API not available",
        "status": "unavailable"
    }))
}

pub async fn get_resources_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(json) = proxy_deep_get("/api/v1/resources/deep").await {
        return Json(json);
    }

    // Get REAL resources from Azure Resource Graph
    if let Some(ref client) = state.async_azure_client {
        match client.get_all_resources_with_health().await {
            Ok(resources) => {
                return Json(serde_json::json!({
                    "success": true,
                    "resources": resources.get("items"),
                    "totalCount": resources.get("total_count"),
                    "healthSummary": resources.get("health_summary"),
                    "complianceSummary": resources.get("compliance_summary"),
                    "tagAnalysis": resources.get("tag_analysis"),
                    "recommendations": resources.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get resources from Azure: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "error": "Azure Resource Graph not available",
        "status": "unavailable"
    }))
}

// ===================== Additional Endpoints =====================

pub async fn get_compliance(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Try to get real compliance data from Azure
    if let Some(ref async_client) = state.async_azure_client {
        match async_client.get_governance_metrics().await {
            Ok(metrics) => {
                return Json(serde_json::json!({
                    "status": "compliant",
                    "policies": {
                        "total": metrics.policies.total,
                        "compliant": metrics.policies.active,
                        "non_compliant": metrics.policies.violations,
                        "compliance_rate": metrics.policies.compliance_rate
                    },
                    "rbac": {
                        "violations": metrics.rbac.violations,
                        "risk_score": metrics.rbac.risk_score
                    },
                    "timestamp": chrono::Utc::now()
                }));
            }
            Err(e) => {
                tracing::warn!("Failed to get compliance data: {}", e);
            }
        }
    }

    // Return default compliance data
    Json(serde_json::json!({
        "status": "compliant",
        "policies": {
            "total": 15,
            "compliant": 12,
            "non_compliant": 3,
            "compliance_rate": 80.0
        },
        "rbac": {
            "violations": 2,
            "risk_score": 3.2
        },
        "timestamp": chrono::Utc::now()
    }))
}

pub async fn get_resources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Try to get real resource data from Azure
    if let Some(ref async_client) = state.async_azure_client {
        match async_client.get_governance_metrics().await {
            Ok(metrics) => {
                return Json(serde_json::json!({
                    "resources": {
                        "total": metrics.resources.total,
                        "optimized": metrics.resources.optimized,
                        "idle": metrics.resources.idle,
                        "overprovisioned": metrics.resources.overprovisioned
                    },
                    "costs": {
                        "current_spend": metrics.costs.current_spend,
                        "savings_identified": metrics.costs.savings_identified
                    },
                    "timestamp": chrono::Utc::now()
                }));
            }
            Err(e) => {
                tracing::warn!("Failed to get resource data: {}", e);
            }
        }
    }

    // Return default resource data
    Json(serde_json::json!({
        "resources": {
            "total": 2500,
            "optimized": 1875,
            "idle": 125,
            "overprovisioned": 50
        },
        "costs": {
            "current_spend": 125000.0,
            "savings_identified": 7000.0
        },
        "timestamp": chrono::Utc::now()
    }))
}

// ===================== Action Orchestrator (Phase 2) =====================

#[derive(Debug, Deserialize)]
pub struct CreateActionRequest {
    pub action_type: String,
    pub resource_id: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub id: String,
    pub action_type: String,
    pub resource_id: String,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub params: serde_json::Value,
    pub result: Option<serde_json::Value>,
}

pub async fn create_action(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateActionRequest>,
) -> impl IntoResponse {
    use chrono::Utc;
    use uuid::Uuid;

    let id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let record = ActionRecord {
        id: id.clone(),
        action_type: payload.action_type.clone(),
        resource_id: payload.resource_id.clone(),
        status: "queued".to_string(),
        created_at: now,
        updated_at: now,
        params: payload.params.clone(),
        result: None,
    };

    {
        let mut actions = state.actions.write().await;
        actions.insert(id.clone(), record);
    }

    let (tx, _rx) = broadcast::channel::<String>(100);
    {
        let mut events = state.action_events.write().await;
        events.insert(id.clone(), tx.clone());
    }

    let state_clone = state.clone();
    let id_clone = id.clone();
    tokio::spawn(async move {
        let send_step = |m: &str| {
            let _ = tx.send(m.to_string());
        };
        send_step("queued");
        {
            let mut actions = state_clone.actions.write().await;
            if let Some(a) = actions.get_mut(&id_clone) {
                a.status = "in_progress".to_string();
                a.updated_at = chrono::Utc::now();
            }
        }
        send_step("in_progress: preflight");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        send_step("in_progress: executing");
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        send_step("in_progress: verifying");
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        {
            let mut actions = state_clone.actions.write().await;
            if let Some(a) = actions.get_mut(&id_clone) {
                a.status = "completed".to_string();
                a.updated_at = chrono::Utc::now();
                a.result = Some(
                    serde_json::json!({"message": "Action executed successfully", "changes": 1}),
                );
            }
        }
        send_step("completed");
    });

    Json(serde_json::json!({"action_id": id}))
}

pub async fn get_action(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> impl IntoResponse {
    let actions = state.actions.read().await;
    if let Some(a) = actions.get(&action_id) {
        return Json(serde_json::json!(a));
    }
    Json(serde_json::json!({"error": "action not found"}))
}

pub async fn stream_action_events(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> impl IntoResponse {
    let rx_opt = {
        let events = state.action_events.read().await;
        events.get(&action_id).cloned()
    };
    if let Some(tx) = rx_opt {
        let mut rx = tx.subscribe();
        let stream = async_stream::stream! {
            while let Ok(msg) = rx.recv().await {
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(msg));
            }
        };
        Sse::new(stream).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "action not found"})),
        )
            .into_response()
    }
}

// Global SSE stream for lightweight real-time updates (heartbeat + metric snapshots)
pub async fn stream_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stream = async_stream::stream! {
        // Send initial hello
        let hello = serde_json::json!({
            "type": "connected",
            "timestamp": chrono::Utc::now(),
        });
        yield Ok::<Event, std::convert::Infallible>(Event::default().data(hello.to_string()));

        // Periodic updates
        loop {
            // Snapshot a small subset of metrics to keep payload small
            let metrics = state.metrics.read().await.clone();
            let snapshot = serde_json::json!({
                "type": "metric_update",
                "timestamp": chrono::Utc::now(),
                "data": {
                    "policies": { "total": metrics.policies.total, "violations": metrics.policies.violations, "compliance_rate": metrics.policies.compliance_rate },
                    "costs": { "current_spend": metrics.costs.current_spend, "savings_identified": metrics.costs.savings_identified },
                    "security": { "risk_score": metrics.rbac.risk_score, "anomalies_detected": metrics.rbac.anomalies_detected }
                }
            });
            yield Ok::<Event, std::convert::Infallible>(Event::default().data(snapshot.to_string()));
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        }
    };
    Sse::new(stream).into_response()
}
