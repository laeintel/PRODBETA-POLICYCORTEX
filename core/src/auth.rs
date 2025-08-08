use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{header, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{debug, error};

// Azure AD Configuration
#[derive(Debug)]
pub struct AzureADConfig {
    pub tenant_id: String,
    pub client_id: String,
    pub issuer: String,
    pub jwks_uri: String,
}

impl AzureADConfig {
    pub fn new() -> Self {
        let tenant_id = std::env::var("AZURE_TENANT_ID")
            .unwrap_or_else(|_| "9ef5b184-d371-462a-bc75-5024ce8baff7".to_string());
        
        let client_id = std::env::var("AZURE_CLIENT_ID")
            .unwrap_or_else(|_| "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c".to_string());
            
        Self {
            issuer: format!("https://login.microsoftonline.com/{}/v2.0", tenant_id),
            jwks_uri: format!("https://login.microsoftonline.com/{}/discovery/v2.0/keys", tenant_id),
            tenant_id,
            client_id,
        }
    }
}

// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,                    // Subject (user ID)
    pub aud: String,                    // Audience (client ID)
    pub iss: String,                    // Issuer
    pub iat: i64,                       // Issued at
    pub exp: i64,                       // Expiration
    pub nbf: Option<i64>,               // Not before
    pub name: Option<String>,           // User name
    pub preferred_username: Option<String>, // User email
    pub oid: Option<String>,            // Object ID
    pub tid: Option<String>,            // Tenant ID
    pub roles: Option<Vec<String>>,     // Application roles
    pub scp: Option<String>,            // Scopes (space-separated)
    pub groups: Option<Vec<String>>,    // Group memberships
}

// JWKS (JSON Web Key Set) structures
#[derive(Debug, Deserialize)]
pub struct JwksResponse {
    pub keys: Vec<Jwk>,
}

#[derive(Debug, Deserialize)]
pub struct Jwk {
    pub kty: String,
    pub r#use: Option<String>,
    pub kid: String,
    pub x5t: Option<String>,
    pub n: String,
    pub e: String,
    pub x5c: Option<Vec<String>>,
    pub alg: Option<String>,
}

// Authentication extractor
pub struct AuthUser {
    pub claims: Claims,
}

// Token validation service
pub struct TokenValidator {
    config: AzureADConfig,
    client: Client,
    jwks_cache: Option<JwksResponse>,
}

impl TokenValidator {
    pub fn new() -> Self {
        Self {
            config: AzureADConfig::new(),
            client: Client::new(),
            jwks_cache: None,
        }
    }

    // Fetch JWKS (JSON Web Key Set) from Azure AD
    async fn fetch_jwks(&mut self) -> Result<&JwksResponse, AuthError> {
        if self.jwks_cache.is_none() {
            debug!("Fetching JWKS from Azure AD: {}", self.config.jwks_uri);
            
            let response = self.client
                .get(&self.config.jwks_uri)
                .send()
                .await
                .map_err(|e| {
                    error!("Failed to fetch JWKS: {}", e);
                    AuthError::JwksFetchError
                })?;

            let jwks: JwksResponse = response
                .json()
                .await
                .map_err(|e| {
                    error!("Failed to parse JWKS response: {}", e);
                    AuthError::JwksParseError
                })?;

            debug!("Successfully fetched {} keys from JWKS", jwks.keys.len());
            self.jwks_cache = Some(jwks);
        }

        Ok(self.jwks_cache.as_ref().unwrap())
    }

    // Find the appropriate key for token validation
    fn find_key<'a>(&self, kid: &str, jwks: &'a JwksResponse) -> Option<&'a Jwk> {
        jwks.keys.iter().find(|key| key.kid == kid)
    }

    // Convert JWK to DecodingKey
    fn jwk_to_decoding_key(jwk: &Jwk) -> Result<DecodingKey, AuthError> {
        if jwk.kty != "RSA" {
            return Err(AuthError::UnsupportedKeyType);
        }

        DecodingKey::from_rsa_components(&jwk.n, &jwk.e)
            .map_err(|_| AuthError::InvalidKey)
    }

    // Validate JWT token
    pub async fn validate_token(&mut self, token: &str) -> Result<Claims, AuthError> {
        // Decode header to get the key ID
        let header = decode_header(token)
            .map_err(|e| {
                error!("Failed to decode JWT header: {}", e);
                AuthError::InvalidToken
            })?;

        let kid = header.kid.ok_or_else(|| {
            error!("JWT header missing 'kid' field");
            AuthError::InvalidToken
        })?;

        // Fetch JWKS
        let jwks = self.fetch_jwks().await?;
        
        // Find the appropriate key
        let jwk = jwks.keys.iter().find(|key| key.kid == kid)
            .ok_or_else(|| {
                error!("Key with ID '{}' not found in JWKS", kid);
                AuthError::KeyNotFound
            })?;

        // Convert JWK to decoding key
        let decoding_key = Self::jwk_to_decoding_key(jwk)?;

        // Setup validation parameters
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_audience(&[&self.config.client_id]);
        validation.set_issuer(&[&self.config.issuer]);
        validation.validate_exp = true;
        validation.validate_nbf = true;

        // Decode and validate the token
        let token_data = decode::<Claims>(token, &decoding_key, &validation)
            .map_err(|e| {
                error!("JWT validation failed: {}", e);
                match e.kind() {
                    jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                    jsonwebtoken::errors::ErrorKind::InvalidAudience => AuthError::InvalidAudience,
                    jsonwebtoken::errors::ErrorKind::InvalidIssuer => AuthError::InvalidIssuer,
                    _ => AuthError::InvalidToken,
                }
            })?;

        debug!("Successfully validated token for user: {:?}", token_data.claims.preferred_username);
        Ok(token_data.claims)
    }

    // Check if user has required permissions
    pub fn check_permissions(&self, claims: &Claims, required_scopes: &[&str]) -> bool {
        // Check scopes
        if let Some(scope_str) = &claims.scp {
            let user_scopes: HashSet<&str> = scope_str.split_whitespace().collect();
            let required_scopes_set: HashSet<&str> = required_scopes.iter().copied().collect();
            
            if required_scopes_set.is_subset(&user_scopes) {
                return true;
            }
        }

        // Check roles
        if let Some(user_roles) = &claims.roles {
            let user_roles_set: HashSet<String> = user_roles.iter().cloned().collect();
            
            // Define role-based permissions
            let admin_roles = ["Global Administrator", "Security Administrator", "Compliance Administrator"];
            
            for role in &admin_roles {
                if user_roles_set.contains(*role) {
                    debug!("User has admin role: {}", role);
                    return true;
                }
            }
        }

        debug!("User lacks required permissions. Required: {:?}, User scopes: {:?}, User roles: {:?}", 
               required_scopes, claims.scp, claims.roles);
        false
    }
}

// Authentication errors
#[derive(Debug)]
pub enum AuthError {
    MissingToken,
    InvalidToken,
    TokenExpired,
    InvalidAudience,
    InvalidIssuer,
    JwksFetchError,
    JwksParseError,
    KeyNotFound,
    UnsupportedKeyType,
    InvalidKey,
    InsufficientPermissions,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingToken => (StatusCode::UNAUTHORIZED, "Missing authorization token"),
            AuthError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid authorization token"),
            AuthError::TokenExpired => (StatusCode::UNAUTHORIZED, "Authorization token expired"),
            AuthError::InvalidAudience => (StatusCode::UNAUTHORIZED, "Invalid token audience"),
            AuthError::InvalidIssuer => (StatusCode::UNAUTHORIZED, "Invalid token issuer"),
            AuthError::JwksFetchError => (StatusCode::SERVICE_UNAVAILABLE, "Failed to fetch signing keys"),
            AuthError::JwksParseError => (StatusCode::SERVICE_UNAVAILABLE, "Failed to parse signing keys"),
            AuthError::KeyNotFound => (StatusCode::UNAUTHORIZED, "Signing key not found"),
            AuthError::UnsupportedKeyType => (StatusCode::UNAUTHORIZED, "Unsupported key type"),
            AuthError::InvalidKey => (StatusCode::UNAUTHORIZED, "Invalid signing key"),
            AuthError::InsufficientPermissions => (StatusCode::FORBIDDEN, "Insufficient permissions"),
        };

        let body = Json(serde_json::json!({
            "error": "authentication_error",
            "message": message
        }));

        (status, body).into_response()
    }
}

// Axum extractor for authenticated requests
#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Extract token from Authorization header
        let auth_header = parts
            .headers
            .get(header::AUTHORIZATION)
            .ok_or(AuthError::MissingToken)?
            .to_str()
            .map_err(|_| AuthError::InvalidToken)?;

        // Check for Bearer token format
        if !auth_header.starts_with("Bearer ") {
            return Err(AuthError::InvalidToken);
        }

        let token = &auth_header[7..]; // Remove "Bearer " prefix

        // Create token validator and validate
        let mut validator = TokenValidator::new();
        let claims = validator.validate_token(token).await?;

        Ok(AuthUser { claims })
    }
}

// Middleware for checking specific permissions
pub struct RequirePermissions {
    pub scopes: Vec<String>,
}

impl RequirePermissions {
    pub fn new(scopes: Vec<&str>) -> Self {
        Self {
            scopes: scopes.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn check(&self, user: &AuthUser) -> Result<(), AuthError> {
        let validator = TokenValidator::new();
        let required_scopes: Vec<&str> = self.scopes.iter().map(|s| s.as_str()).collect();
        
        if validator.check_permissions(&user.claims, &required_scopes) {
            Ok(())
        } else {
            Err(AuthError::InsufficientPermissions)
        }
    }
}

// Helper function to create an optional auth extractor
pub struct OptionalAuthUser(pub Option<AuthUser>);

#[async_trait]
impl<S> FromRequestParts<S> for OptionalAuthUser
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match AuthUser::from_request_parts(parts, state).await {
            Ok(user) => Ok(OptionalAuthUser(Some(user))),
            Err(_) => Ok(OptionalAuthUser(None)),
        }
    }
}

// Tenant context extractor for multi-tenant scenarios
#[derive(Debug)]
pub struct TenantContext {
    pub tenant_id: String,
    pub subscription_ids: Vec<String>,
}

impl TenantContext {
    pub async fn from_user(user: &AuthUser) -> Result<Self, AuthError> {
        let tenant_id = user.claims.tid
            .as_ref()
            .ok_or(AuthError::InvalidToken)?
            .clone();

        // For now, we'll fetch all subscriptions the user has access to
        // In a real implementation, you'd use Azure Management API
        let subscription_ids = vec![
            "205b477d-17e7-4b3b-92c1-32cf02626b78".to_string(), // Your subscription
        ];

        Ok(TenantContext {
            tenant_id,
            subscription_ids,
        })
    }
}