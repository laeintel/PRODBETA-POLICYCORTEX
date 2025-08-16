// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantContext {
    pub tenant_id: String,
    pub azure_tenant_id: Option<String>,
    pub subscription_ids: Vec<String>,
    pub is_active: bool,
}

impl TenantContext {
    pub fn from_token_claims(claims: &crate::auth::Claims) -> Self {
        // Extract tenant from token claims
        let tenant_id = claims.tid.clone().unwrap_or_else(|| "default".to_string());

        TenantContext {
            tenant_id: tenant_id.clone(),
            azure_tenant_id: Some(tenant_id),
            subscription_ids: vec![],
            is_active: true,
        }
    }

    pub fn default() -> Self {
        TenantContext {
            tenant_id: "default".to_string(),
            azure_tenant_id: Some("9ef5b184-d371-462a-bc75-5024ce8baff7".to_string()),
            subscription_ids: vec!["205b477d-17e7-4b3b-92c1-32cf02626b78".to_string()],
            is_active: true,
        }
    }
}

// Extension trait for Request to store tenant context
#[derive(Clone)]
pub struct TenantExtension(pub TenantContext);

// Middleware to extract and propagate tenant context
pub async fn tenant_middleware(
    State(state): State<Arc<crate::api::AppState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Try to get tenant from auth token
    let tenant = if let Some(claims) = request.extensions().get::<crate::auth::Claims>() {
        TenantContext::from_token_claims(claims)
    } else {
        // Try to get from header for service-to-service calls
        if let Some(tenant_header) = request.headers().get("X-Tenant-ID") {
            if let Ok(tenant_id) = tenant_header.to_str() {
                TenantContext {
                    tenant_id: tenant_id.to_string(),
                    azure_tenant_id: None,
                    subscription_ids: vec![],
                    is_active: true,
                }
            } else {
                TenantContext::default()
            }
        } else {
            TenantContext::default()
        }
    };

    info!("Request tenant context: {:?}", tenant.tenant_id);

    // Store in request extensions for handlers
    request
        .extensions_mut()
        .insert(TenantExtension(tenant.clone()));

    // Set database session variable for RLS
    if let Some(ref pool) = state.db_pool {
        if let Ok(mut conn) = pool.acquire().await {
            let query = format!("SELECT set_tenant_context($1)");
            let _ = sqlx::query(&query)
                .bind(&tenant.tenant_id)
                .execute(&mut *conn)
                .await;
        }
    }

    let response = next.run(request).await;
    Ok(response)
}

// Helper to extract tenant from request
pub fn get_tenant(request: &Request) -> Option<TenantContext> {
    request
        .extensions()
        .get::<TenantExtension>()
        .map(|ext| ext.0.clone())
}

// Database queries with tenant isolation
pub struct TenantAwareDb {
    pool: sqlx::PgPool,
    tenant_id: String,
}

impl TenantAwareDb {
    pub fn new(pool: sqlx::PgPool, tenant_id: String) -> Self {
        TenantAwareDb { pool, tenant_id }
    }

    pub async fn set_tenant_context(&self) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT set_tenant_context($1)")
            .bind(&self.tenant_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn query_policies(&self) -> Result<Vec<Policy>, sqlx::Error> {
        self.set_tenant_context().await?;

        let policies = sqlx::query_as::<_, Policy>(
            r#"
            SELECT id, name, description, category, severity, is_active
            FROM governance.policies
            WHERE tenant_id = $1
            "#,
        )
        .bind(&self.tenant_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(policies)
    }

    pub async fn query_resources(&self) -> Result<Vec<Resource>, sqlx::Error> {
        self.set_tenant_context().await?;

        let resources = sqlx::query_as::<_, Resource>(
            r#"
            SELECT id, azure_resource_id, name, type, location, compliance_status
            FROM governance.resources
            WHERE tenant_id = $1
            "#,
        )
        .bind(&self.tenant_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(resources)
    }
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Policy {
    pub id: uuid::Uuid,
    pub name: String,
    pub description: Option<String>,
    pub category: Option<String>,
    pub severity: Option<String>,
    pub is_active: bool,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Resource {
    pub id: uuid::Uuid,
    pub azure_resource_id: String,
    pub name: String,
    pub r#type: String,
    pub location: Option<String>,
    pub compliance_status: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_context_default() {
        let ctx = TenantContext::default();
        assert_eq!(ctx.tenant_id, "default");
        assert!(ctx.is_active);
    }

    #[test]
    fn test_tenant_from_claims() {
        let claims = crate::auth::Claims {
            sub: "user123".to_string(),
            tid: Some("tenant456".to_string()),
            aud: String::new(),
            exp: 0,
            iat: 0,
            nbf: Some(0),
            roles: Vec::<String>::new().into(),
            scp: None,
        };

        let ctx = TenantContext::from_token_claims(&claims);
        assert_eq!(ctx.tenant_id, "tenant456");
    }
}
