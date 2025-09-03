// DevOps API handlers for comprehensive navigation system
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};

use crate::{
    api::AppState,
    data_mode::{DataMode, DataResponse},
};

// Pipeline info
#[derive(Debug, Serialize, Deserialize)]
pub struct Pipeline {
    pub id: String,
    pub name: String,
    pub repository: String,
    pub branch: String,
    pub status: String,
    pub last_run: Option<DateTime<Utc>>,
    pub success_rate: f64,
    pub average_duration_minutes: f64,
    pub stages: Vec<String>,
}

// Release info
#[derive(Debug, Serialize, Deserialize)]
pub struct Release {
    pub id: String,
    pub name: String,
    pub version: String,
    pub environment: String,
    pub status: String,
    pub deployed_at: Option<DateTime<Utc>>,
    pub deployed_by: String,
    pub artifacts: Vec<String>,
    pub rollback_available: bool,
}

// Artifact info
#[derive(Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub name: String,
    pub artifact_type: String,
    pub version: String,
    pub size_mb: f64,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub downloads: u32,
    pub repository: String,
}

// Deployment info
#[derive(Debug, Serialize, Deserialize)]
pub struct Deployment {
    pub id: String,
    pub application: String,
    pub environment: String,
    pub version: String,
    pub status: String,
    pub deployed_at: DateTime<Utc>,
    pub deployed_by: String,
    pub duration_minutes: f64,
    pub changes: u32,
}

// Build info
#[derive(Debug, Serialize, Deserialize)]
pub struct Build {
    pub id: String,
    pub number: String,
    pub pipeline: String,
    pub branch: String,
    pub commit: String,
    pub status: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_minutes: f64,
    pub triggered_by: String,
}

// Repository info
#[derive(Debug, Serialize, Deserialize)]
pub struct Repository {
    pub id: String,
    pub name: String,
    pub url: String,
    pub default_branch: String,
    pub language: String,
    pub size_mb: f64,
    pub commits_count: u32,
    pub contributors: u32,
    pub last_commit: DateTime<Utc>,
    pub open_prs: u32,
}

// GET /api/v1/devops/pipelines
pub async fn get_pipelines(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mode = DataMode::from_env();
    
    // In real mode, we must have real data or fail
    if mode.is_real() {
        if let Some(ref async_client) = state.async_azure_client {
            // DevOps pipelines not yet implemented in Azure client
            // For now, fail fast in real mode
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "DevOps pipelines not available in real data mode",
                "details": "Azure DevOps integration pending implementation"
            }))).into_response();
            
            #[allow(unreachable_code)]
            match async_client.get_governance_metrics().await {
                Ok(pipelines) => {
                    return Json(DataResponse::new(pipelines, mode)).into_response();
                }
                Err(e) => {
                    tracing::error!("Failed to get pipelines from Azure DevOps: {}", e);
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": "Azure DevOps service unavailable",
                            "message": format!("Failed to retrieve pipelines: {}", e),
                            "mode": "real"
                        }))
                    ).into_response();
                }
            }
        } else {
            tracing::error!("Real mode enabled but Azure client not initialized");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Azure client not initialized",
                    "message": "Real data mode requires Azure client configuration",
                    "mode": "real"
                }))
            ).into_response();
        }
    }
    
    // Only return simulated data in simulated mode
    let pipelines = vec![
        Pipeline {
            id: "pipe-001".to_string(),
            name: "Main CI/CD Pipeline (SIMULATED)".to_string(),
            repository: "policycortex/main (SIMULATED)".to_string(),
            branch: "main".to_string(),
            status: "success".to_string(),
            last_run: Some(Utc::now() - chrono::Duration::hours(1)),
            success_rate: 95.2,
            average_duration_minutes: 12.5,
            stages: vec!["Build".to_string(), "Test".to_string(), "Deploy".to_string()],
        },
        Pipeline {
            id: "pipe-002".to_string(),
            name: "Security Scanning".to_string(),
            repository: "policycortex/main".to_string(),
            branch: "main".to_string(),
            status: "running".to_string(),
            last_run: Some(Utc::now() - chrono::Duration::minutes(15)),
            success_rate: 98.7,
            average_duration_minutes: 8.3,
            stages: vec!["SAST".to_string(), "DAST".to_string(), "Dependencies".to_string()],
        },
        Pipeline {
            id: "pipe-003".to_string(),
            name: "Infrastructure Deployment".to_string(),
            repository: "policycortex/infrastructure".to_string(),
            branch: "main".to_string(),
            status: "failed".to_string(),
            last_run: Some(Utc::now() - chrono::Duration::hours(3)),
            success_rate: 87.5,
            average_duration_minutes: 25.0,
            stages: vec!["Validate".to_string(), "Plan".to_string(), "Apply".to_string()],
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(pipelines, mode)).into_response()
}

// GET /api/v1/devops/releases
pub async fn get_releases(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let releases = vec![
        Release {
            id: "rel-001".to_string(),
            name: "PolicyCortex v2.18.0".to_string(),
            version: "2.18.0".to_string(),
            environment: "Production".to_string(),
            status: "deployed".to_string(),
            deployed_at: Some(Utc::now() - chrono::Duration::days(2)),
            deployed_by: "ci-automation".to_string(),
            artifacts: vec!["backend:2.18.0".to_string(), "frontend:2.18.0".to_string()],
            rollback_available: true,
        },
        Release {
            id: "rel-002".to_string(),
            name: "PolicyCortex v2.19.0-beta".to_string(),
            version: "2.19.0-beta".to_string(),
            environment: "Staging".to_string(),
            status: "deployed".to_string(),
            deployed_at: Some(Utc::now() - chrono::Duration::hours(6)),
            deployed_by: "dev-team".to_string(),
            artifacts: vec!["backend:2.19.0-beta".to_string(), "frontend:2.19.0-beta".to_string()],
            rollback_available: true,
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(releases, mode)).into_response()
}

// GET /api/v1/devops/artifacts
pub async fn get_artifacts(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let artifacts = vec![
        Artifact {
            id: "art-001".to_string(),
            name: "policycortex-backend".to_string(),
            artifact_type: "docker".to_string(),
            version: "2.18.0".to_string(),
            size_mb: 156.3,
            created_at: Utc::now() - chrono::Duration::days(2),
            created_by: "ci-automation".to_string(),
            downloads: 45,
            repository: "crpcxdev.azurecr.io".to_string(),
        },
        Artifact {
            id: "art-002".to_string(),
            name: "policycortex-frontend".to_string(),
            artifact_type: "docker".to_string(),
            version: "2.18.0".to_string(),
            size_mb: 89.2,
            created_at: Utc::now() - chrono::Duration::days(2),
            created_by: "ci-automation".to_string(),
            downloads: 45,
            repository: "crpcxdev.azurecr.io".to_string(),
        },
        Artifact {
            id: "art-003".to_string(),
            name: "terraform-modules".to_string(),
            artifact_type: "terraform".to_string(),
            version: "1.5.2".to_string(),
            size_mb: 2.3,
            created_at: Utc::now() - chrono::Duration::days(7),
            created_by: "infra-team".to_string(),
            downloads: 128,
            repository: "terraform-registry".to_string(),
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(artifacts, mode)).into_response()
}

// GET /api/v1/devops/deployments
pub async fn get_deployments(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let deployments = vec![
        Deployment {
            id: "dep-001".to_string(),
            application: "PolicyCortex Backend".to_string(),
            environment: "Production".to_string(),
            version: "2.18.0".to_string(),
            status: "successful".to_string(),
            deployed_at: Utc::now() - chrono::Duration::days(2),
            deployed_by: "ci-automation".to_string(),
            duration_minutes: 8.5,
            changes: 23,
        },
        Deployment {
            id: "dep-002".to_string(),
            application: "PolicyCortex Frontend".to_string(),
            environment: "Production".to_string(),
            version: "2.18.0".to_string(),
            status: "successful".to_string(),
            deployed_at: Utc::now() - chrono::Duration::days(2),
            deployed_by: "ci-automation".to_string(),
            duration_minutes: 5.2,
            changes: 15,
        },
        Deployment {
            id: "dep-003".to_string(),
            application: "PolicyCortex Backend".to_string(),
            environment: "Staging".to_string(),
            version: "2.19.0-beta".to_string(),
            status: "successful".to_string(),
            deployed_at: Utc::now() - chrono::Duration::hours(6),
            deployed_by: "dev-team".to_string(),
            duration_minutes: 7.8,
            changes: 42,
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(deployments, mode)).into_response()
}

// GET /api/v1/devops/builds
pub async fn get_builds(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let builds = vec![
        Build {
            id: "build-001".to_string(),
            number: "#1234".to_string(),
            pipeline: "Main CI/CD Pipeline".to_string(),
            branch: "main".to_string(),
            commit: "a1b2c3d4".to_string(),
            status: "success".to_string(),
            started_at: Utc::now() - chrono::Duration::hours(1),
            completed_at: Some(Utc::now() - chrono::Duration::minutes(48)),
            duration_minutes: 12.0,
            triggered_by: "git-push".to_string(),
        },
        Build {
            id: "build-002".to_string(),
            number: "#1235".to_string(),
            pipeline: "Security Scanning".to_string(),
            branch: "main".to_string(),
            commit: "a1b2c3d4".to_string(),
            status: "running".to_string(),
            started_at: Utc::now() - chrono::Duration::minutes(5),
            completed_at: None,
            duration_minutes: 5.0,
            triggered_by: "schedule".to_string(),
        },
        Build {
            id: "build-003".to_string(),
            number: "#1233".to_string(),
            pipeline: "Infrastructure Deployment".to_string(),
            branch: "feature/updates".to_string(),
            commit: "e5f6g7h8".to_string(),
            status: "failed".to_string(),
            started_at: Utc::now() - chrono::Duration::hours(3),
            completed_at: Some(Utc::now() - chrono::Duration::hours(2) - chrono::Duration::minutes(35)),
            duration_minutes: 25.0,
            triggered_by: "manual".to_string(),
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(builds, mode)).into_response()
}

// GET /api/v1/devops/repos
pub async fn get_repos(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let repos = vec![
        Repository {
            id: "repo-001".to_string(),
            name: "policycortex".to_string(),
            url: "https://github.com/policycortex/main".to_string(),
            default_branch: "main".to_string(),
            language: "Rust/TypeScript".to_string(),
            size_mb: 125.4,
            commits_count: 2847,
            contributors: 12,
            last_commit: Utc::now() - chrono::Duration::hours(2),
            open_prs: 3,
        },
        Repository {
            id: "repo-002".to_string(),
            name: "infrastructure".to_string(),
            url: "https://github.com/policycortex/infrastructure".to_string(),
            default_branch: "main".to_string(),
            language: "HCL/Terraform".to_string(),
            size_mb: 45.2,
            commits_count: 892,
            contributors: 6,
            last_commit: Utc::now() - chrono::Duration::days(1),
            open_prs: 1,
        },
        Repository {
            id: "repo-003".to_string(),
            name: "ai-models".to_string(),
            url: "https://github.com/policycortex/ai-models".to_string(),
            default_branch: "main".to_string(),
            language: "Python".to_string(),
            size_mb: 89.7,
            commits_count: 456,
            contributors: 4,
            last_commit: Utc::now() - chrono::Duration::days(3),
            open_prs: 0,
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(repos, mode)).into_response()
}