// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::fmt;

/// Comprehensive error type for PolicyCortex API
#[derive(Debug)]
pub enum ApiError {
    // Authentication & Authorization
    Unauthorized(String),
    Forbidden(String),
    InvalidToken(String),
    
    // Input Validation
    BadRequest(String),
    InvalidInput(String),
    MissingParameter(String),
    
    // Resource Management
    NotFound(String),
    Conflict(String),
    
    // External Services
    AzureError(String),
    DatabaseError(String),
    NetworkError(String),
    
    // Internal Errors
    InternalError(String),
    ConfigurationError(String),
    SerializationError(String),
    
    // Rate Limiting & Capacity
    TooManyRequests(String),
    ServiceUnavailable(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            ApiError::Forbidden(msg) => write!(f, "Forbidden: {}", msg),
            ApiError::InvalidToken(msg) => write!(f, "Invalid token: {}", msg),
            ApiError::BadRequest(msg) => write!(f, "Bad request: {}", msg),
            ApiError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ApiError::MissingParameter(msg) => write!(f, "Missing parameter: {}", msg),
            ApiError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ApiError::Conflict(msg) => write!(f, "Conflict: {}", msg),
            ApiError::AzureError(msg) => write!(f, "Azure error: {}", msg),
            ApiError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            ApiError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ApiError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            ApiError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ApiError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            ApiError::TooManyRequests(msg) => write!(f, "Too many requests: {}", msg),
            ApiError::ServiceUnavailable(msg) => write!(f, "Service unavailable: {}", msg),
        }
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            ApiError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "unauthorized", self.to_string()),
            ApiError::Forbidden(_) => (StatusCode::FORBIDDEN, "forbidden", self.to_string()),
            ApiError::InvalidToken(_) => (StatusCode::UNAUTHORIZED, "invalid_token", self.to_string()),
            
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request", self.to_string()),
            ApiError::InvalidInput(_) => (StatusCode::BAD_REQUEST, "invalid_input", self.to_string()),
            ApiError::MissingParameter(_) => (StatusCode::BAD_REQUEST, "missing_parameter", self.to_string()),
            
            ApiError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found", self.to_string()),
            ApiError::Conflict(_) => (StatusCode::CONFLICT, "conflict", self.to_string()),
            
            ApiError::AzureError(_) => (StatusCode::BAD_GATEWAY, "azure_error", "External service error".to_string()),
            ApiError::DatabaseError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "database_error", "Database operation failed".to_string()),
            ApiError::NetworkError(_) => (StatusCode::BAD_GATEWAY, "network_error", "Network operation failed".to_string()),
            
            ApiError::InternalError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "Internal server error".to_string()),
            ApiError::ConfigurationError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "configuration_error", "Service configuration error".to_string()),
            ApiError::SerializationError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "serialization_error", "Data serialization error".to_string()),
            
            ApiError::TooManyRequests(_) => (StatusCode::TOO_MANY_REQUESTS, "too_many_requests", self.to_string()),
            ApiError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable", self.to_string()),
        };

        // Log internal errors for debugging but don't expose details to clients
        if matches!(self, ApiError::InternalError(_) | ApiError::DatabaseError(_) | ApiError::ConfigurationError(_)) {
            tracing::error!("Internal error: {}", self);
        }

        let body = json!({
            "error": error_type,
            "message": message,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });

        (status, Json(body)).into_response()
    }
}

// Convenient Result type alias
pub type ApiResult<T> = Result<T, ApiError>;

// From implementations for common error types
impl From<sqlx::Error> for ApiError {
    fn from(err: sqlx::Error) -> Self {
        tracing::error!("Database error: {}", err);
        match err {
            sqlx::Error::RowNotFound => ApiError::NotFound("Resource not found".to_string()),
            _ => ApiError::DatabaseError(format!("Database operation failed: {}", err)),
        }
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        ApiError::SerializationError(format!("JSON serialization failed: {}", err))
    }
}

impl From<reqwest::Error> for ApiError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            ApiError::NetworkError("Request timeout".to_string())
        } else if err.is_connect() {
            ApiError::NetworkError("Connection failed".to_string())
        } else {
            ApiError::NetworkError(format!("Network error: {}", err))
        }
    }
}

impl From<std::env::VarError> for ApiError {
    fn from(err: std::env::VarError) -> Self {
        ApiError::ConfigurationError(format!("Environment variable error: {}", err))
    }
}

/// Helper macro for creating API errors
#[macro_export]
macro_rules! api_error {
    (unauthorized, $msg:expr) => {
        $crate::error::ApiError::Unauthorized($msg.to_string())
    };
    (forbidden, $msg:expr) => {
        $crate::error::ApiError::Forbidden($msg.to_string())
    };
    (bad_request, $msg:expr) => {
        $crate::error::ApiError::BadRequest($msg.to_string())
    };
    (not_found, $msg:expr) => {
        $crate::error::ApiError::NotFound($msg.to_string())
    };
    (internal, $msg:expr) => {
        $crate::error::ApiError::InternalError($msg.to_string())
    };
}

/// Helper function to handle unwrap with proper error conversion
pub fn unwrap_or_internal<T>(result: Result<T, impl std::fmt::Display>, context: &str) -> ApiResult<T> {
    result.map_err(|e| ApiError::InternalError(format!("{}: {}", context, e)))
}

/// Helper function for validating required parameters
pub fn require_param<T>(param: Option<T>, name: &str) -> ApiResult<T> {
    param.ok_or_else(|| ApiError::MissingParameter(format!("Required parameter '{}' is missing", name)))
}

/// Helper function for validating input format
pub fn validate_input<T>(result: Result<T, impl std::fmt::Display>, context: &str) -> ApiResult<T> {
    result.map_err(|e| ApiError::InvalidInput(format!("{}: {}", context, e)))
}