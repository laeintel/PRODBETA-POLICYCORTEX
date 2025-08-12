use axum::{extract::FromRequestParts, http::request::Parts};
use metrics::{counter, describe_counter, describe_histogram, histogram};
use std::time::Instant;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tower::{Layer, Service};
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct CorrelationId(pub String);

impl Default for CorrelationId {
    fn default() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

// Extractor to read or create x-correlation-id
#[async_trait::async_trait]
impl<S> FromRequestParts<S> for CorrelationId
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        const HDR: &str = "x-correlation-id";
        if let Some(value) = parts.headers.get(HDR) {
            if let Ok(s) = value.to_str() {
                return Ok(CorrelationId(s.to_string()));
            }
        }
        Ok(CorrelationId::default())
    }
}

// Tower layer to inject correlation id and record basic metrics
#[derive(Clone)]
pub struct CorrelationLayer;

impl<S> Layer<S> for CorrelationLayer {
    type Service = CorrelationService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        CorrelationService { inner }
    }
}

#[derive(Clone)]
pub struct CorrelationService<S> {
    inner: S,
}

impl<ReqBody, ResBody, S> Service<axum::http::Request<ReqBody>> for CorrelationService<S>
where
    S: Service<axum::http::Request<ReqBody>, Response = axum::http::Response<ResBody>>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn Future<Output = Result<axum::http::Response<ResBody>, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: axum::http::Request<ReqBody>) -> Self::Future {
        const HDR: &str = "x-correlation-id";
        let corr = req
            .headers()
            .get(HDR)
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        req.extensions_mut().insert(CorrelationId(corr.clone()));
        if !req.headers().contains_key(HDR) {
            if let Ok(val) = axum::http::HeaderValue::from_str(&corr) {
                req.headers_mut().insert(HDR, val);
            }
        }

        let method = req.method().to_string();
        let path = req.uri().path().to_string();
        let start = Instant::now();

        let fut = self.inner.clone().call(req);

        // Describe metrics once (idempotent)
        describe_counter!("http_requests_total", "Total number of HTTP requests.");
        describe_histogram!(
            "http_request_duration_seconds",
            "HTTP request latencies in seconds."
        );
        counter!("http_requests_total", 1, "method" => method.clone(), "path" => path.clone());

        Box::pin(async move {
            let out = fut.await;
            let elapsed = start.elapsed().as_secs_f64();
            match &out {
                Ok(resp) => {
                    let status = resp.status().as_u16().to_string();
                    histogram!("http_request_duration_seconds", elapsed, "method" => method, "path" => path, "status" => status);
                }
                Err(_) => {
                    histogram!("http_request_duration_seconds", elapsed, "method" => method, "path" => path, "status" => "error".to_string());
                }
            }
            out
        })
    }
}
