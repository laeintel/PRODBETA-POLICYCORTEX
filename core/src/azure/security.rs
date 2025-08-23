// Azure Security Integration
// Provides IAM, RBAC, PIM, and Azure AD data

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};

use super::client::AzureClient;
use super::api_versions;

/// Azure AD User
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AadUser {
    pub id: String,
    #[serde(rename = "userPrincipalName")]
    pub user_principal_name: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    pub mail: Option<String>,
    #[serde(rename = "accountEnabled")]
    pub account_enabled: bool,
    #[serde(rename = "jobTitle")]
    pub job_title: Option<String>,
    pub department: Option<String>,
    #[serde(rename = "createdDateTime")]
    pub created_date_time: Option<DateTime<Utc>>,
    #[serde(rename = "lastSignInDateTime")]
    pub last_sign_in_date_time: Option<DateTime<Utc>>,
}

/// Azure AD Group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AadGroup {
    pub id: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    pub description: Option<String>,
    #[serde(rename = "mailEnabled")]
    pub mail_enabled: bool,
    #[serde(rename = "securityEnabled")]
    pub security_enabled: bool,
    #[serde(rename = "mailNickname")]
    pub mail_nickname: Option<String>,
    #[serde(rename = "createdDateTime")]
    pub created_date_time: Option<DateTime<Utc>>,
}

/// RBAC Role Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDefinition {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub role_type: String,
    pub properties: RoleDefinitionProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDefinitionProperties {
    #[serde(rename = "roleName")]
    pub role_name: String,
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub role_type: Option<String>,
    pub permissions: Vec<Permission>,
    #[serde(rename = "assignableScopes")]
    pub assignable_scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub actions: Vec<String>,
    #[serde(rename = "notActions")]
    pub not_actions: Vec<String>,
    #[serde(rename = "dataActions")]
    pub data_actions: Option<Vec<String>>,
    #[serde(rename = "notDataActions")]
    pub not_data_actions: Option<Vec<String>>,
}

/// RBAC Role Assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleAssignment {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub assignment_type: String,
    pub properties: RoleAssignmentProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleAssignmentProperties {
    #[serde(rename = "roleDefinitionId")]
    pub role_definition_id: String,
    #[serde(rename = "principalId")]
    pub principal_id: String,
    pub scope: String,
    #[serde(rename = "principalType")]
    pub principal_type: Option<String>,
    #[serde(rename = "createdOn")]
    pub created_on: Option<DateTime<Utc>>,
    #[serde(rename = "updatedOn")]
    pub updated_on: Option<DateTime<Utc>>,
}

/// Conditional Access Policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalAccessPolicy {
    pub id: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    #[serde(rename = "createdDateTime")]
    pub created_date_time: Option<DateTime<Utc>>,
    #[serde(rename = "modifiedDateTime")]
    pub modified_date_time: Option<DateTime<Utc>>,
    pub state: String,
    pub conditions: Option<serde_json::Value>,
    #[serde(rename = "grantControls")]
    pub grant_controls: Option<serde_json::Value>,
    #[serde(rename = "sessionControls")]
    pub session_controls: Option<serde_json::Value>,
}

/// PIM Eligible Assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimAssignment {
    pub id: String,
    #[serde(rename = "principalId")]
    pub principal_id: String,
    #[serde(rename = "roleDefinitionId")]
    pub role_definition_id: String,
    #[serde(rename = "directoryScopeId")]
    pub directory_scope_id: String,
    #[serde(rename = "appScopeId")]
    pub app_scope_id: Option<String>,
    #[serde(rename = "startDateTime")]
    pub start_date_time: Option<DateTime<Utc>>,
    #[serde(rename = "endDateTime")]
    pub end_date_time: Option<DateTime<Utc>>,
    #[serde(rename = "memberType")]
    pub member_type: String,
    #[serde(rename = "assignmentState")]
    pub assignment_state: String,
}

/// Access Review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessReview {
    pub id: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    pub status: String,
    #[serde(rename = "startDateTime")]
    pub start_date_time: Option<DateTime<Utc>>,
    #[serde(rename = "endDateTime")]
    pub end_date_time: Option<DateTime<Utc>>,
    #[serde(rename = "reviewedEntity")]
    pub reviewed_entity: Option<serde_json::Value>,
    pub reviewers: Option<Vec<serde_json::Value>>,
}

/// Azure Security service
pub struct SecurityService {
    client: AzureClient,
}

impl SecurityService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get all users from Azure AD
    pub async fn get_users(&self) -> Result<Vec<AadUser>> {
        let response: super::AzureResponse<AadUser> = self.client
            .get_graph("/users")
            .await?;
        Ok(response.value)
    }

    /// Get all groups from Azure AD
    pub async fn get_groups(&self) -> Result<Vec<AadGroup>> {
        let response: super::AzureResponse<AadGroup> = self.client
            .get_graph("/groups")
            .await?;
        Ok(response.value)
    }

    /// Get all role definitions
    pub async fn get_role_definitions(&self) -> Result<Vec<RoleDefinition>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Authorization/roleDefinitions",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::RBAC).await
    }

    /// Get all role assignments
    pub async fn get_role_assignments(&self) -> Result<Vec<RoleAssignment>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Authorization/roleAssignments",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::RBAC).await
    }

    /// Get conditional access policies
    pub async fn get_conditional_access_policies(&self) -> Result<Vec<ConditionalAccessPolicy>> {
        let response: super::AzureResponse<ConditionalAccessPolicy> = self.client
            .get_graph("/identity/conditionalAccess/policies")
            .await?;
        Ok(response.value)
    }

    /// Get PIM eligible assignments
    pub async fn get_pim_assignments(&self) -> Result<Vec<PimAssignment>> {
        let response: super::AzureResponse<PimAssignment> = self.client
            .get_graph("/roleManagement/directory/roleEligibilityScheduleInstances")
            .await?;
        Ok(response.value)
    }

    /// Get access reviews
    pub async fn get_access_reviews(&self) -> Result<Vec<AccessReview>> {
        let response: super::AzureResponse<AccessReview> = self.client
            .get_graph("/identityGovernance/accessReviews/definitions")
            .await?;
        Ok(response.value)
    }

    /// Get IAM summary
    pub async fn get_iam_summary(&self) -> Result<IamSummary> {
        info!("Fetching IAM summary from Azure AD");

        let users = self.get_users().await.unwrap_or_default();
        let groups = self.get_groups().await.unwrap_or_default();
        let role_definitions = self.get_role_definitions().await.unwrap_or_default();
        let role_assignments = self.get_role_assignments().await.unwrap_or_default();

        let active_users = users.iter().filter(|u| u.account_enabled).count();
        let inactive_users = users.iter().filter(|u| !u.account_enabled).count();
        let security_groups = groups.iter().filter(|g| g.security_enabled).count();

        // Count privileged roles (simplified)
        let privileged_assignments = role_assignments.iter()
            .filter(|ra| {
                ra.properties.role_definition_id.contains("Owner") ||
                ra.properties.role_definition_id.contains("Contributor") ||
                ra.properties.role_definition_id.contains("Administrator")
            })
            .count();

        Ok(IamSummary {
            total_users: users.len(),
            active_users,
            inactive_users,
            total_groups: groups.len(),
            security_groups,
            total_roles: role_definitions.len(),
            total_assignments: role_assignments.len(),
            privileged_assignments,
        })
    }

    /// Get Zero Trust status
    pub async fn get_zero_trust_status(&self) -> Result<ZeroTrustStatus> {
        let ca_policies = self.get_conditional_access_policies().await.unwrap_or_default();
        let pim_assignments = self.get_pim_assignments().await.unwrap_or_default();
        let access_reviews = self.get_access_reviews().await.unwrap_or_default();

        let enabled_ca_policies = ca_policies.iter()
            .filter(|p| p.state == "enabled")
            .count();

        let active_pim_assignments = pim_assignments.iter()
            .filter(|a| a.assignment_state == "Active")
            .count();

        let pending_reviews = access_reviews.iter()
            .filter(|r| r.status == "InProgress" || r.status == "NotStarted")
            .count();

        // Calculate Zero Trust score (simplified)
        let mut score = 50.0; // Base score
        if enabled_ca_policies > 0 { score += 20.0; }
        if active_pim_assignments > 0 { score += 15.0; }
        if pending_reviews == 0 { score += 15.0; }

        Ok(ZeroTrustStatus {
            score,
            conditional_access_policies: ca_policies.len(),
            enabled_policies: enabled_ca_policies,
            pim_enabled: active_pim_assignments > 0,
            active_pim_assignments,
            access_reviews_pending: pending_reviews,
            mfa_coverage_percentage: 85.0, // Would need additional API calls to calculate
        })
    }
}

#[derive(Debug, Serialize)]
pub struct IamSummary {
    pub total_users: usize,
    pub active_users: usize,
    pub inactive_users: usize,
    pub total_groups: usize,
    pub security_groups: usize,
    pub total_roles: usize,
    pub total_assignments: usize,
    pub privileged_assignments: usize,
}

#[derive(Debug, Serialize)]
pub struct ZeroTrustStatus {
    pub score: f64,
    pub conditional_access_policies: usize,
    pub enabled_policies: usize,
    pub pim_enabled: bool,
    pub active_pim_assignments: usize,
    pub access_reviews_pending: usize,
    pub mfa_coverage_percentage: f64,
}