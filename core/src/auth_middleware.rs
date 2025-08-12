use crate::auth::{AuthUser, Claims, TokenValidator};
use axum::response::IntoResponse;
use axum::{http::Request, response::Response};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tower::{Layer, Service};
use tracing::warn;

#[derive(Clone)]
pub struct AuthEnforcementLayer;

impl<S> Layer<S> for AuthEnforcementLayer {
    type Service = AuthEnforcementService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthEnforcementService { inner }
    }
}

#[derive(Clone)]
pub struct AuthEnforcementService<S> {
    inner: S,
}

impl<S, B> Service<Request<B>> for AuthEnforcementService<S>
where
    S: Service<Request<B>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        // Enforce scopes/roles on write endpoints.
        let method = req.method().clone();
        let path = req.uri().path().to_string();
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let is_write = matches!(
                method,
                axum::http::Method::POST
                    | axum::http::Method::PUT
                    | axum::http::Method::DELETE
                    | axum::http::Method::PATCH
            );
            if is_write {
                // Extract bearer token manually
                let maybe_auth = req
                    .headers()
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|h| h.to_str().ok())
                    .filter(|v| v.starts_with("Bearer "))
                    .map(|v| v.trim_start_matches("Bearer "));
                if let Some(token) = maybe_auth {
                    let mut validator = TokenValidator::new();
                    match validator.validate_token(token).await {
                        Ok(claims) => {
                            // Minimal scope enforcement: require write scope
                            let has_scope = claims
                                .scp
                                .as_deref()
                                .unwrap_or("")
                                .split_whitespace()
                                .any(|s| s == "policycortex.write" || s == ".default");
                            let has_admin_role = claims
                                .roles
                                .unwrap_or_default()
                                .iter()
                                .any(|r| r.to_lowercase().contains("admin"));
                            if !has_scope && !has_admin_role {
                                let body = axum::Json(serde_json::json!({
                                    "error": "insufficient_scope",
                                    "message": "Write operation requires policycortex.write scope or admin role"
                                }));
                                let res = (axum::http::StatusCode::FORBIDDEN, body).into_response();
                                return Ok(res);
                            }
                        }
                        Err(e) => {
                            warn!("JWT validation failed on write {}: {}", path, e as i32);
                            let body = axum::Json(serde_json::json!({
                                "error": "unauthorized",
                                "message": "Valid bearer token required"
                            }));
                            let res = (axum::http::StatusCode::UNAUTHORIZED, body).into_response();
                            return Ok(res);
                        }
                    }
                } else {
                    let body = axum::Json(serde_json::json!({
                        "error": "unauthorized",
                        "message": "Bearer token missing"
                    }));
                    let res = (axum::http::StatusCode::UNAUTHORIZED, body).into_response();
                    return Ok(res);
                }
            }

            inner.call(req).await
        })
    }
}
