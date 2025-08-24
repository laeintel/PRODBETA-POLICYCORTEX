// WebSocket service for real-time updates
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeUpdate {
    pub update_type: String,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
}

#[derive(Clone)]
pub struct WebSocketState {
    pub tx: broadcast::Sender<RealtimeUpdate>,
}

impl WebSocketState {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self { tx }
    }
}

// WebSocket handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WebSocketState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket(socket, state))
}

async fn websocket(socket: WebSocket, state: Arc<WebSocketState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // Task to send updates to client
    let mut send_task = tokio::spawn(async move {
        while let Ok(update) = rx.recv().await {
            if let Ok(msg) = serde_json::to_string(&update) {
                if sender.send(Message::Text(msg)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Task to receive messages from client
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Handle incoming messages
                    println!("Received: {}", text);
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    }
}

// Function to broadcast updates
pub async fn broadcast_update(state: &WebSocketState, update_type: &str, data: serde_json::Value) {
    let update = RealtimeUpdate {
        update_type: update_type.to_string(),
        timestamp: Utc::now(),
        data,
    };
    let _ = state.tx.send(update);
}

// Periodic update generator for real-time data
pub async fn start_realtime_updates(state: Arc<WebSocketState>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
    
    loop {
        interval.tick().await;
        
        // Send cost anomaly updates
        if rand::random::<f64>() > 0.7 {
            broadcast_update(
                &state,
                "cost_anomaly",
                serde_json::json!({
                    "resource_id": format!("vm-{}", rand::random::<u32>() % 100),
                    "anomaly_type": "spike",
                    "cost_impact": rand::random::<f64>() * 1000.0,
                    "severity": if rand::random::<bool>() { "high" } else { "medium" }
                })
            ).await;
        }
        
        // Send security alerts
        if rand::random::<f64>() > 0.8 {
            broadcast_update(
                &state,
                "security_alert",
                serde_json::json!({
                    "alert_type": "unauthorized_access",
                    "resource": format!("storage-{}", rand::random::<u32>() % 50),
                    "risk_level": rand::random::<u32>() % 100
                })
            ).await;
        }
        
        // Send edge node status
        broadcast_update(
            &state,
            "edge_status",
            serde_json::json!({
                "node_id": format!("edge-{}", ["us-west", "eu-central", "apac"][rand::random::<usize>() % 3]),
                "latency_ms": 10.0 + rand::random::<f64>() * 20.0,
                "load_percentage": rand::random::<f64>() * 100.0
            })
        ).await;
        
        // Send quantum migration progress
        if rand::random::<f64>() > 0.9 {
            broadcast_update(
                &state,
                "quantum_migration",
                serde_json::json!({
                    "secret_id": format!("secret-{}", rand::random::<u32>() % 200),
                    "migration_status": "completed",
                    "algorithm": "Kyber-1024"
                })
            ).await;
        }
    }
}

use rand;

// Cost prediction stream
pub async fn stream_cost_predictions() -> impl futures::Stream<Item = serde_json::Value> {
    async_stream::stream! {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            yield serde_json::json!({
                "timestamp": Utc::now(),
                "predicted_cost": 50000.0 + rand::random::<f64>() * 10000.0,
                "confidence": 0.85 + rand::random::<f64>() * 0.15,
                "trend": if rand::random::<bool>() { "increasing" } else { "decreasing" }
            });
        }
    }
}

// Security event stream
pub async fn stream_security_events() -> impl futures::Stream<Item = serde_json::Value> {
    async_stream::stream! {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3));
        loop {
            interval.tick().await;
            if rand::random::<f64>() > 0.6 {
                yield serde_json::json!({
                    "event_type": ["login_attempt", "permission_change", "resource_access"][rand::random::<usize>() % 3],
                    "user": format!("user{}", rand::random::<u32>() % 100),
                    "risk_score": rand::random::<u32>() % 100,
                    "timestamp": Utc::now()
                });
            }
        }
    }
}