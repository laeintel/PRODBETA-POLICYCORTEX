// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use crate::error::{ApiError, ApiResult};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use uuid::Uuid;

/// Comprehensive input validation utilities for PolicyCortex API
pub struct Validator;

impl Validator {
    /// Validate UUID format
    pub fn validate_uuid(input: &str, field_name: &str) -> ApiResult<Uuid> {
        Uuid::parse_str(input)
            .map_err(|_| ApiError::InvalidInput(format!("{} must be a valid UUID", field_name)))
    }

    /// Validate email format
    pub fn validate_email(email: &str) -> ApiResult<()> {
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            .map_err(|e| ApiError::InternalError(format!("Email regex compilation failed: {}", e)))?;
        
        if email_regex.is_match(email) {
            Ok(())
        } else {
            Err(ApiError::InvalidInput("Invalid email format".to_string()))
        }
    }

    /// Validate string length constraints
    pub fn validate_length(input: &str, field_name: &str, min: usize, max: usize) -> ApiResult<()> {
        let len = input.len();
        if len < min {
            return Err(ApiError::InvalidInput(format!(
                "{} must be at least {} characters long", field_name, min
            )));
        }
        if len > max {
            return Err(ApiError::InvalidInput(format!(
                "{} must be no more than {} characters long", field_name, max
            )));
        }
        Ok(())
    }

    /// Validate alphanumeric characters only
    pub fn validate_alphanumeric(input: &str, field_name: &str) -> ApiResult<()> {
        if input.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            Ok(())
        } else {
            Err(ApiError::InvalidInput(format!(
                "{} must contain only alphanumeric characters, hyphens, and underscores", field_name
            )))
        }
    }

    /// Validate Azure resource name format
    pub fn validate_azure_resource_name(name: &str) -> ApiResult<()> {
        Self::validate_length(name, "Azure resource name", 1, 80)?;
        
        let name_regex = Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9\-_.]*[a-zA-Z0-9]$")
            .map_err(|e| ApiError::InternalError(format!("Resource name regex compilation failed: {}", e)))?;
        
        if name_regex.is_match(name) {
            Ok(())
        } else {
            Err(ApiError::InvalidInput(
                "Azure resource name must start and end with alphanumeric characters and contain only letters, numbers, hyphens, periods, and underscores".to_string()
            ))
        }
    }

    /// Validate tenant ID format (GUID)
    pub fn validate_tenant_id(tenant_id: &str) -> ApiResult<()> {
        Self::validate_uuid(tenant_id, "tenant_id")?;
        Ok(())
    }

    /// Validate subscription ID format (GUID)
    pub fn validate_subscription_id(subscription_id: &str) -> ApiResult<()> {
        Self::validate_uuid(subscription_id, "subscription_id")?;
        Ok(())
    }

    /// Validate policy name
    pub fn validate_policy_name(name: &str) -> ApiResult<()> {
        Self::validate_length(name, "policy name", 3, 128)?;
        Self::validate_alphanumeric(name, "policy name")?;
        Ok(())
    }

    /// Validate exception reason
    pub fn validate_exception_reason(reason: &str) -> ApiResult<()> {
        Self::validate_length(reason, "exception reason", 10, 1000)?;
        
        // Check for suspicious content
        let suspicious_patterns = [
            "<script", "javascript:", "vbscript:", "onload=", "onerror=",
            "eval(", "setTimeout(", "setInterval(", "document.cookie",
            "window.location", "iframe", "object", "embed"
        ];
        
        let reason_lower = reason.to_lowercase();
        for pattern in &suspicious_patterns {
            if reason_lower.contains(pattern) {
                return Err(ApiError::InvalidInput(
                    "Exception reason contains potentially unsafe content".to_string()
                ));
            }
        }
        
        Ok(())
    }

    /// Validate JSON payload size and structure
    pub fn validate_json_payload<T>(payload: &T, max_size_bytes: usize) -> ApiResult<()>
    where
        T: serde::Serialize,
    {
        let serialized = serde_json::to_string(payload)
            .map_err(|e| ApiError::SerializationError(format!("Failed to serialize payload: {}", e)))?;
        
        if serialized.len() > max_size_bytes {
            return Err(ApiError::InvalidInput(format!(
                "Payload too large: {} bytes exceeds maximum of {} bytes",
                serialized.len(), max_size_bytes
            )));
        }
        
        Ok(())
    }

    /// Validate date range
    pub fn validate_date_range(start_date: &str, end_date: &str) -> ApiResult<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)> {
        let start = chrono::DateTime::parse_from_rfc3339(start_date)
            .map_err(|_| ApiError::InvalidInput("start_date must be a valid RFC3339 timestamp".to_string()))?
            .with_timezone(&chrono::Utc);
        
        let end = chrono::DateTime::parse_from_rfc3339(end_date)
            .map_err(|_| ApiError::InvalidInput("end_date must be a valid RFC3339 timestamp".to_string()))?
            .with_timezone(&chrono::Utc);
        
        if start >= end {
            return Err(ApiError::InvalidInput("start_date must be before end_date".to_string()));
        }
        
        let max_range = chrono::Duration::days(365);
        if end - start > max_range {
            return Err(ApiError::InvalidInput("Date range cannot exceed 365 days".to_string()));
        }
        
        Ok((start, end))
    }

    /// Validate pagination parameters
    pub fn validate_pagination(page: Option<u32>, limit: Option<u32>) -> ApiResult<(u32, u32)> {
        let page = page.unwrap_or(1);
        let limit = limit.unwrap_or(20);
        
        if page < 1 {
            return Err(ApiError::InvalidInput("page must be >= 1".to_string()));
        }
        
        if limit < 1 || limit > 1000 {
            return Err(ApiError::InvalidInput("limit must be between 1 and 1000".to_string()));
        }
        
        Ok((page, limit))
    }

    /// Validate action type
    pub fn validate_action_type(action_type: &str) -> ApiResult<()> {
        let valid_actions = [
            "create_resource", "update_resource", "delete_resource",
            "modify_policy", "grant_access", "revoke_access",
            "restart_service", "scale_resource", "apply_configuration",
            "execute_script", "remediate"
        ];
        
        if valid_actions.contains(&action_type) {
            Ok(())
        } else {
            Err(ApiError::InvalidInput(format!(
                "Invalid action type: {}. Valid types: {}",
                action_type,
                valid_actions.join(", ")
            )))
        }
    }

    /// Validate query parameters for search endpoints
    pub fn validate_search_query(query: &str) -> ApiResult<()> {
        Self::validate_length(query, "search query", 1, 200)?;
        
        // Prevent SQL injection patterns
        let sql_patterns = [
            "';", "--", "/*", "*/", "union", "select", "insert", "update", 
            "delete", "drop", "alter", "create", "truncate", "exec", "execute"
        ];
        
        let query_lower = query.to_lowercase();
        for pattern in &sql_patterns {
            if query_lower.contains(pattern) {
                return Err(ApiError::InvalidInput(
                    "Search query contains potentially unsafe SQL patterns".to_string()
                ));
            }
        }
        
        Ok(())
    }

    /// Validate IP address format
    pub fn validate_ip_address(ip: &str) -> ApiResult<std::net::IpAddr> {
        ip.parse::<std::net::IpAddr>()
            .map_err(|_| ApiError::InvalidInput("Invalid IP address format".to_string()))
    }

    /// Validate port number
    pub fn validate_port(port: u16) -> ApiResult<()> {
        if port == 0 {
            return Err(ApiError::InvalidInput("Port number cannot be 0".to_string()));
        }
        Ok(())
    }

    /// Validate URL format
    pub fn validate_url(url: &str) -> ApiResult<()> {
        url::Url::parse(url)
            .map_err(|_| ApiError::InvalidInput("Invalid URL format".to_string()))?;
        Ok(())
    }

    /// Validate webhook payload
    pub fn validate_webhook_payload(payload: &serde_json::Value) -> ApiResult<()> {
        // Check payload size
        let serialized = serde_json::to_string(payload)
            .map_err(|e| ApiError::SerializationError(format!("Failed to serialize webhook payload: {}", e)))?;
        
        if serialized.len() > 10_000 {
            return Err(ApiError::InvalidInput("Webhook payload too large (max 10KB)".to_string()));
        }
        
        // Check for required fields
        if !payload.is_object() {
            return Err(ApiError::InvalidInput("Webhook payload must be a JSON object".to_string()));
        }
        
        let obj = payload.as_object().unwrap();
        if !obj.contains_key("event_type") {
            return Err(ApiError::InvalidInput("Webhook payload must contain 'event_type' field".to_string()));
        }
        
        Ok(())
    }
}

/// Request validation middleware
#[derive(Debug, Deserialize)]
pub struct CreateExceptionValidated {
    pub resource_id: String,
    pub policy_id: String,
    pub reason: String,
    pub expires_days: Option<u32>,
}

impl CreateExceptionValidated {
    pub fn validate(self) -> ApiResult<Self> {
        Validator::validate_azure_resource_name(&self.resource_id)?;
        Validator::validate_policy_name(&self.policy_id)?;
        Validator::validate_exception_reason(&self.reason)?;
        
        if let Some(days) = self.expires_days {
            if days == 0 || days > 365 {
                return Err(ApiError::InvalidInput(
                    "expires_days must be between 1 and 365".to_string()
                ));
            }
        }
        
        Ok(self)
    }
}

/// Validation for action execution requests
#[derive(Debug, Deserialize)]
pub struct ExecuteActionValidated {
    pub action_type: String,
    pub resource_id: String,
    pub params: HashMap<String, serde_json::Value>,
}

impl ExecuteActionValidated {
    pub fn validate(self) -> ApiResult<Self> {
        Validator::validate_action_type(&self.action_type)?;
        Validator::validate_azure_resource_name(&self.resource_id)?;
        
        // Validate params size
        let params_size = serde_json::to_string(&self.params)
            .map_err(|e| ApiError::SerializationError(format!("Failed to serialize params: {}", e)))?
            .len();
        
        if params_size > 50_000 {
            return Err(ApiError::InvalidInput("Action parameters too large (max 50KB)".to_string()));
        }
        
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_uuid() {
        assert!(Validator::validate_uuid("550e8400-e29b-41d4-a716-446655440000", "test").is_ok());
        assert!(Validator::validate_uuid("invalid-uuid", "test").is_err());
    }

    #[test]
    fn test_validate_email() {
        assert!(Validator::validate_email("test@example.com").is_ok());
        assert!(Validator::validate_email("invalid-email").is_err());
    }

    #[test]
    fn test_validate_length() {
        assert!(Validator::validate_length("hello", "test", 3, 10).is_ok());
        assert!(Validator::validate_length("hi", "test", 3, 10).is_err());
        assert!(Validator::validate_length("this is too long", "test", 3, 10).is_err());
    }

    #[test]
    fn test_validate_search_query() {
        assert!(Validator::validate_search_query("normal search").is_ok());
        assert!(Validator::validate_search_query("'; DROP TABLE users; --").is_err());
    }

    #[test]
    fn test_validate_exception_reason() {
        assert!(Validator::validate_exception_reason("This is a valid business justification for the exception").is_ok());
        assert!(Validator::validate_exception_reason("<script>alert('xss')</script>").is_err());
        assert!(Validator::validate_exception_reason("short").is_err());
    }
}