// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use axum::{
    extract::Extension,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::sync::Arc;
use tracing::{error, info, warn};
use uuid::Uuid;

/// Tenant context for request processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantContext {
    pub tenant_id: Uuid,
    pub tenant_name: String,
    pub is_admin: bool,
    pub user_id: Option<Uuid>,
    pub subscription_id: Option<String>,
}

impl TenantContext {
    /// Create a new tenant context from JWT claims
    pub fn from_claims(claims: &serde_json::Value) -> Result<Self, String> {
        let tenant_id = claims["tid"]
            .as_str()
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| "Missing or invalid tenant ID".to_string())?;

        let user_id = claims["oid"]
            .as_str()
            .or_else(|| claims["sub"].as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let is_admin = claims["roles"]
            .as_array()
            .map(|roles| {
                roles.iter().any(|r| {
                    r.as_str()
                        .map(|s| s.to_lowercase().contains("admin"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        Ok(Self {
            tenant_id,
            tenant_name: claims["tenant_name"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string(),
            is_admin,
            user_id,
            subscription_id: claims["subscription_id"].as_str().map(String::from),
        })
    }

    /// Check if user can access a specific tenant
    pub fn can_access_tenant(&self, target_tenant_id: &Uuid) -> bool {
        self.is_admin || self.tenant_id == *target_tenant_id
    }

    /// Check if user can access a resource
    pub fn can_access_resource(&self, resource_tenant_id: &Uuid) -> bool {
        self.is_admin || self.tenant_id == *resource_tenant_id
    }
}

/// Tenant isolation middleware
pub async fn tenant_isolation_middleware(
    Extension(tenant): Extension<Option<TenantContext>>,
    request: Request,
    next: Next,
) -> Response {
    // Check if tenant context exists
    if tenant.is_none() {
        warn!("Request without tenant context");
        return (StatusCode::UNAUTHORIZED, "Tenant context required").into_response();
    }

    let response = next.run(request).await;
    response
}

/// Apply tenant filter to database queries
pub trait TenantFilter {
    fn apply_tenant_filter(self, tenant: &TenantContext) -> Self;
}

impl TenantFilter for sqlx::QueryBuilder<'_, sqlx::Postgres> {
    fn apply_tenant_filter(mut self, tenant: &TenantContext) -> Self {
        if !tenant.is_admin {
            self.push(" AND tenant_id = ").push_bind(tenant.tenant_id);
        }
        self
    }
}

/// Resource with tenant isolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantResource<T> {
    pub tenant_id: Uuid,
    pub resource: T,
}

impl<T> TenantResource<T> {
    pub fn new(tenant_id: Uuid, resource: T) -> Self {
        Self {
            tenant_id,
            resource,
        }
    }

    /// Check if resource belongs to tenant
    pub fn belongs_to(&self, tenant: &TenantContext) -> bool {
        tenant.can_access_resource(&self.tenant_id)
    }
}

/// Database operations with tenant isolation
pub struct TenantDatabase {
    pool: Arc<PgPool>,
}

impl TenantDatabase {
    pub fn new(pool: Arc<PgPool>) -> Self {
        Self { pool }
    }

    /// Get resources for a tenant
    pub async fn get_resources(
        &self,
        tenant: &TenantContext,
        resource_type: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, sqlx::Error> {
        let mut query = sqlx::QueryBuilder::new("SELECT * FROM resources WHERE 1=1");

        // Apply tenant filter
        if !tenant.is_admin {
            query.push(" AND tenant_id = ").push_bind(tenant.tenant_id);
        }

        // Apply resource type filter
        if let Some(rtype) = resource_type {
            query.push(" AND resource_type = ").push_bind(rtype);
        }

        query.push(" ORDER BY created_at DESC");

        let rows = query.build().fetch_all(&*self.pool).await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                serde_json::json!({
                    "id": row.try_get::<Uuid, _>("id").ok(),
                    "tenant_id": row.try_get::<Uuid, _>("tenant_id").ok(),
                    "resource_type": row.try_get::<String, _>("resource_type").ok(),
                    "name": row.try_get::<String, _>("name").ok(),
                    "data": row.try_get::<serde_json::Value, _>("data").ok(),
                })
            })
            .collect())
    }

    /// Create a resource with tenant isolation
    pub async fn create_resource(
        &self,
        tenant: &TenantContext,
        resource_type: &str,
        name: &str,
        data: serde_json::Value,
    ) -> Result<Uuid, sqlx::Error> {
        let id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO resources (id, tenant_id, resource_type, name, data, created_at, created_by)
            VALUES ($1, $2, $3, $4, $5, NOW(), $6)
            "#,
        )
        .bind(id)
        .bind(&tenant.tenant_id)
        .bind(resource_type)
        .bind(name)
        .bind(&data)
        .bind(&tenant.user_id)
        .execute(&*self.pool)
        .await?;

        info!("Created resource {} for tenant {}", id, tenant.tenant_id);
        Ok(id)
    }

    /// Update a resource with tenant isolation
    pub async fn update_resource(
        &self,
        tenant: &TenantContext,
        resource_id: Uuid,
        data: serde_json::Value,
    ) -> Result<bool, sqlx::Error> {
        let result = if tenant.is_admin {
            // Admin can update any resource
            sqlx::query(
                r#"
                UPDATE resources
                SET data = $1, updated_at = NOW(), updated_by = $2
                WHERE id = $3
                "#,
            )
            .bind(&data)
            .bind(&tenant.user_id)
            .bind(resource_id)
            .execute(&*self.pool)
            .await?
        } else {
            // Regular user can only update their tenant's resources
            sqlx::query(
                r#"
                UPDATE resources
                SET data = $1, updated_at = NOW(), updated_by = $2
                WHERE id = $3 AND tenant_id = $4
                "#,
            )
            .bind(&data)
            .bind(&tenant.user_id)
            .bind(resource_id)
            .bind(&tenant.tenant_id)
            .execute(&*self.pool)
            .await?
        };

        Ok(result.rows_affected() > 0)
    }

    /// Delete a resource with tenant isolation
    pub async fn delete_resource(
        &self,
        tenant: &TenantContext,
        resource_id: Uuid,
    ) -> Result<bool, sqlx::Error> {
        let result = if tenant.is_admin {
            // Admin can delete any resource
            sqlx::query("DELETE FROM resources WHERE id = $1")
                .bind(resource_id)
                .execute(&*self.pool)
                .await?
        } else {
            // Regular user can only delete their tenant's resources
            sqlx::query("DELETE FROM resources WHERE id = $1 AND tenant_id = $2")
                .bind(resource_id)
                .bind(&tenant.tenant_id)
                .execute(&*self.pool)
                .await?
        };

        Ok(result.rows_affected() > 0)
    }

    /// Get policies for a tenant
    pub async fn get_policies(
        &self,
        tenant: &TenantContext,
    ) -> Result<Vec<serde_json::Value>, sqlx::Error> {
        let rows = if tenant.is_admin {
            sqlx::query("SELECT * FROM policies ORDER BY created_at DESC")
                .fetch_all(&*self.pool)
                .await?
        } else {
            sqlx::query("SELECT * FROM policies WHERE tenant_id = $1 ORDER BY created_at DESC")
                .bind(&tenant.tenant_id)
                .fetch_all(&*self.pool)
                .await?
        };

        Ok(rows
            .into_iter()
            .map(|row| {
                serde_json::json!({
                    "id": row.try_get::<uuid::Uuid, _>("id").unwrap_or_default(),
                    "tenant_id": row.try_get::<uuid::Uuid, _>("tenant_id").unwrap_or_default(),
                    "name": row.try_get::<String, _>("name").unwrap_or_default(),
                    "description": row.try_get::<Option<String>, _>("description").unwrap_or_default(),
                    "rules": row.try_get::<Option<serde_json::Value>, _>("rules").unwrap_or_default(),
                    "enabled": row.try_get::<bool, _>("enabled").unwrap_or_default(),
                })
            })
            .collect())
    }

    /// Get compliance data for a tenant
    pub async fn get_compliance(
        &self,
        tenant: &TenantContext,
    ) -> Result<serde_json::Value, sqlx::Error> {
        let rows = if tenant.is_admin {
            sqlx::query(
                r#"
                SELECT
                    COUNT(*) as total_resources,
                    SUM(CASE WHEN compliance_status = 'compliant' THEN 1 ELSE 0 END) as compliant,
                    SUM(CASE WHEN compliance_status = 'non_compliant' THEN 1 ELSE 0 END) as non_compliant
                FROM resources
                "#
            )
            .fetch_one(&*self.pool)
            .await?
        } else {
            sqlx::query(
                r#"
                SELECT
                    COUNT(*) as total_resources,
                    SUM(CASE WHEN compliance_status = 'compliant' THEN 1 ELSE 0 END) as compliant,
                    SUM(CASE WHEN compliance_status = 'non_compliant' THEN 1 ELSE 0 END) as non_compliant
                FROM resources
                WHERE tenant_id = $1
                "#
            )
            .bind(&tenant.tenant_id)
            .fetch_one(&*self.pool)
            .await?
        };

        let total_resources: i64 = rows.try_get("total_resources")?;
        let compliant: i64 = rows.try_get("compliant")?;
        let non_compliant: i64 = rows.try_get("non_compliant")?;

        Ok(serde_json::json!({
            "total": total_resources,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "compliance_rate": if total_resources > 0 {
                (compliant as f64 / total_resources as f64) * 100.0
            } else {
                0.0
            }
        }))
    }
}

/// Audit log with tenant context
pub async fn audit_log(
    pool: &PgPool,
    tenant: &TenantContext,
    action: &str,
    resource_type: Option<&str>,
    resource_id: Option<Uuid>,
    details: Option<serde_json::Value>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO audit_logs (id, tenant_id, user_id, action, resource_type, resource_id, details, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#
    )
    .bind(Uuid::new_v4())
    .bind(&tenant.tenant_id)
    .bind(&tenant.user_id)
    .bind(action)
    .bind(resource_type)
    .bind(resource_id)
    .bind(details)
    .execute(pool)
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_context_from_claims() {
        let claims = serde_json::json!({
            "tid": "550e8400-e29b-41d4-a716-446655440000",
            "oid": "550e8400-e29b-41d4-a716-446655440001",
            "tenant_name": "Test Tenant",
            "roles": ["admin"],
            "subscription_id": "test-sub-123"
        });

        let context = TenantContext::from_claims(&claims).unwrap();
        assert_eq!(context.tenant_name, "Test Tenant");
        assert!(context.is_admin);
        assert!(context.user_id.is_some());
    }

    #[test]
    fn test_tenant_access_control() {
        let tenant = TenantContext {
            tenant_id: Uuid::new_v4(),
            tenant_name: "Test".to_string(),
            is_admin: false,
            user_id: Some(Uuid::new_v4()),
            subscription_id: None,
        };

        let same_tenant = tenant.tenant_id;
        let different_tenant = Uuid::new_v4();

        assert!(tenant.can_access_tenant(&same_tenant));
        assert!(!tenant.can_access_tenant(&different_tenant));

        // Admin can access any tenant
        let admin = TenantContext {
            is_admin: true,
            ..tenant.clone()
        };
        assert!(admin.can_access_tenant(&different_tenant));
    }
}
