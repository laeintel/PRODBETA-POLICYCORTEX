use super::AppState;
use crate::resources::{
    AzureResource, ResourceCategory, ResourceFilter, ResourceSummary,
    HealthStatus, ComplianceFilter, CostRange, ResourceInsight,
};
use crate::resources::manager::ResourceManager;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
pub struct ResourceQuery {
    pub categories: Option<String>, // Comma-separated list
    pub resource_types: Option<String>, // Comma-separated list
    pub locations: Option<String>, // Comma-separated list
    pub health_status: Option<String>, // Comma-separated list
    pub compliance_only_violations: Option<bool>,
    pub compliance_min_score: Option<f32>,
    pub cost_min_daily: Option<f64>,
    pub cost_max_daily: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ResourcesResponse {
    pub resources: Vec<AzureResource>,
    pub summary: ResourceSummary,
    pub total: usize,
    pub filtered: usize,
}

#[derive(Debug, Serialize)]
pub struct ResourceActionResponse {
    pub success: bool,
    pub message: String,
    pub resource_id: String,
    pub action_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ExecuteActionRequest {
    pub action_id: String,
    pub confirmation: bool,
}

pub async fn get_all_resources(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ResourceQuery>,
) -> impl IntoResponse {
    info!("Fetching all Azure resources with filters");

    // Initialize resource manager if not already done
    let manager = if let Some(ref azure_client) = state.azure_client {
        ResourceManager::new(Arc::new(azure_client.clone())).await
    } else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Azure client not configured").into_response();
    };

    // Build filter from query parameters
    let filter = build_filter_from_query(query);

    // Get resources with filter
    let resources = manager.get_resources(filter).await;
    let summary = manager.get_summary().await;

    let total = resources.len();

    Json(ResourcesResponse {
        resources,
        summary,
        total,
        filtered: total,
    }).into_response()
}

pub async fn get_resources_by_category(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
) -> impl IntoResponse {
    info!("Fetching resources for category: {}", category);

    let category_enum = match category.to_lowercase().as_str() {
        "policy" => ResourceCategory::Policy,
        "cost" | "costmanagement" => ResourceCategory::CostManagement,
        "security" | "securitycontrols" => ResourceCategory::SecurityControls,
        "compute" | "storage" | "computestorage" => ResourceCategory::ComputeStorage,
        "network" | "networks" | "firewalls" => ResourceCategory::NetworksFirewalls,
        _ => return (StatusCode::BAD_REQUEST, format!("Invalid category: {}", category)).into_response(),
    };

    let manager = if let Some(ref azure_client) = state.azure_client {
        ResourceManager::new(Arc::new(azure_client.clone())).await
    } else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Azure client not configured").into_response();
    };

    let filter = Some(ResourceFilter {
        categories: Some(vec![category_enum]),
        resource_types: None,
        locations: None,
        tags: None,
        health_status: None,
        compliance_filter: None,
        cost_range: None,
    });

    let resources = manager.get_resources(filter).await;
    let summary = manager.get_summary().await;

    Json(ResourcesResponse {
        total: summary.total_resources,
        filtered: resources.len(),
        resources,
        summary,
    }).into_response()
}

pub async fn get_resource_by_id(
    State(state): State<Arc<AppState>>,
    Path(resource_id): Path<String>,
) -> impl IntoResponse {
    info!("Fetching resource by ID: {}", resource_id);

    let manager = if let Some(ref azure_client) = state.azure_client {
        ResourceManager::new(Arc::new(azure_client.clone())).await
    } else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Azure client not configured").into_response();
    };

    let resources = manager.get_resources(None).await;
    
    let resource = resources
        .into_iter()
        .find(|r| r.id == resource_id);

    match resource {
        Some(r) => Json(r).into_response(),
        None => (StatusCode::NOT_FOUND, format!("Resource not found: {}", resource_id)).into_response(),
    }
}

pub async fn execute_resource_action(
    State(state): State<Arc<AppState>>,
    Path(resource_id): Path<String>,
    Json(request): Json<ExecuteActionRequest>,
) -> impl IntoResponse {
    info!("Executing action {} on resource {}", request.action_id, resource_id);

    if !request.confirmation {
        return Json(ResourceActionResponse {
            success: false,
            message: "Action requires confirmation".to_string(),
            resource_id,
            action_id: request.action_id,
        }).into_response();
    }

    let manager = if let Some(ref azure_client) = state.azure_client {
        ResourceManager::new(Arc::new(azure_client.clone())).await
    } else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Azure client not configured").into_response();
    };

    match manager.execute_action(&resource_id, &request.action_id).await {
        Ok(message) => Json(ResourceActionResponse {
            success: true,
            message,
            resource_id,
            action_id: request.action_id,
        }).into_response(),
        Err(e) => {
            warn!("Action execution failed: {}", e);
            Json(ResourceActionResponse {
                success: false,
                message: e.to_string(),
                resource_id,
                action_id: request.action_id,
            }).into_response()
        }
    }
}

pub async fn get_resource_insights(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    info!("Fetching all resource insights");

    let manager = if let Some(ref azure_client) = state.azure_client {
        ResourceManager::new(Arc::new(azure_client.clone())).await
    } else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Azure client not configured").into_response();
    };

    let resources = manager.get_resources(None).await;
    
    let mut all_insights = Vec::new();
    for resource in resources {
        all_insights.extend(resource.insights);
    }

    // Sort by confidence score
    all_insights.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Json(all_insights).into_response()
}

pub async fn get_resource_health_summary(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    info!("Fetching resource health summary");

    let manager = if let Some(ref azure_client) = state.azure_client {
        ResourceManager::new(Arc::new(azure_client.clone())).await
    } else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Azure client not configured").into_response();
    };

    let summary = manager.get_summary().await;
    
    Json(HealthSummary {
        healthy: *summary.by_health.get(&HealthStatus::Healthy).unwrap_or(&0),
        degraded: *summary.by_health.get(&HealthStatus::Degraded).unwrap_or(&0),
        unhealthy: *summary.by_health.get(&HealthStatus::Unhealthy).unwrap_or(&0),
        unknown: *summary.by_health.get(&HealthStatus::Unknown).unwrap_or(&0),
        critical_issues: summary.critical_issues,
        total_resources: summary.total_resources,
    }).into_response()
}

fn build_filter_from_query(query: ResourceQuery) -> Option<ResourceFilter> {
    let mut has_filter = false;
    let mut filter = ResourceFilter {
        categories: None,
        resource_types: None,
        locations: None,
        tags: None,
        health_status: None,
        compliance_filter: None,
        cost_range: None,
    };

    // Parse categories
    if let Some(categories_str) = query.categories {
        has_filter = true;
        let categories: Vec<ResourceCategory> = categories_str
            .split(',')
            .filter_map(|s| match s.trim().to_lowercase().as_str() {
                "policy" => Some(ResourceCategory::Policy),
                "cost" | "costmanagement" => Some(ResourceCategory::CostManagement),
                "security" => Some(ResourceCategory::SecurityControls),
                "compute" | "storage" => Some(ResourceCategory::ComputeStorage),
                "network" | "networks" => Some(ResourceCategory::NetworksFirewalls),
                _ => None,
            })
            .collect();
        if !categories.is_empty() {
            filter.categories = Some(categories);
        }
    }

    // Parse resource types
    if let Some(types_str) = query.resource_types {
        has_filter = true;
        let types: Vec<String> = types_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        filter.resource_types = Some(types);
    }

    // Parse locations
    if let Some(locations_str) = query.locations {
        has_filter = true;
        let locations: Vec<String> = locations_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        filter.locations = Some(locations);
    }

    // Parse health status
    if let Some(health_str) = query.health_status {
        has_filter = true;
        let health_statuses: Vec<HealthStatus> = health_str
            .split(',')
            .filter_map(|s| match s.trim().to_lowercase().as_str() {
                "healthy" => Some(HealthStatus::Healthy),
                "degraded" => Some(HealthStatus::Degraded),
                "unhealthy" => Some(HealthStatus::Unhealthy),
                "unknown" => Some(HealthStatus::Unknown),
                _ => None,
            })
            .collect();
        if !health_statuses.is_empty() {
            filter.health_status = Some(health_statuses);
        }
    }

    // Parse compliance filter
    if query.compliance_only_violations.is_some() || query.compliance_min_score.is_some() {
        has_filter = true;
        filter.compliance_filter = Some(ComplianceFilter {
            only_violations: query.compliance_only_violations.unwrap_or(false),
            min_score: query.compliance_min_score,
            severity_levels: None,
        });
    }

    // Parse cost range
    if query.cost_min_daily.is_some() || query.cost_max_daily.is_some() {
        has_filter = true;
        filter.cost_range = Some(CostRange {
            min_daily: query.cost_min_daily,
            max_daily: query.cost_max_daily,
            currency: "USD".to_string(),
        });
    }

    if has_filter {
        Some(filter)
    } else {
        None
    }
}

#[derive(Debug, Serialize)]
pub struct HealthSummary {
    pub healthy: usize,
    pub degraded: usize,
    pub unhealthy: usize,
    pub unknown: usize,
    pub critical_issues: usize,
    pub total_resources: usize,
}