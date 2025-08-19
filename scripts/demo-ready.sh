#!/bin/bash
# ========================================
# PolicyCortex Demo Ready Script
# Quick setup for local demo environment
# ========================================

set -e

echo ""
echo "=========================================="
echo "PolicyCortex Demo Environment Setup"
echo "=========================================="
echo ""

# Set demo environment variables
export NODE_ENV=demo
export USE_REAL_DATA=false
export USE_MOCK_GRAPHQL=true
export USE_DEMO_CHAT=true
export REQUIRE_AUTH=false
export NEXT_PUBLIC_API_URL=http://localhost:8080
export NEXT_PUBLIC_GRAPHQL_ENDPOINT=http://localhost:4000/graphql
export RUST_LOG=info
export NEXT_PUBLIC_DEMO_MODE=true
export NEXT_PUBLIC_USE_WS=false

echo "[1/5] Setting up demo environment variables..."
echo "      NODE_ENV=$NODE_ENV"
echo "      USE_REAL_DATA=$USE_REAL_DATA"
echo "      USE_MOCK_GRAPHQL=$USE_MOCK_GRAPHQL"
echo "      USE_DEMO_CHAT=$USE_DEMO_CHAT"
echo ""

# Check Docker
echo "[2/5] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi
echo "      Docker is installed and running"
echo ""

# Start database services
echo "[3/5] Starting database services..."
docker-compose -f docker-compose.local.yml up -d postgres dragonfly eventstore &> /dev/null || {
    echo "      WARNING: Database services may already be running"
}
echo "      PostgreSQL, Redis, and EventStore started"
sleep 5
echo ""

# Seed demo data
echo "[4/5] Seeding demo tenants and data..."
if [ -f "scripts/seed-data.sh" ]; then
    ./scripts/seed-data.sh &> /dev/null || true
    echo "      Demo data seeded successfully"
else
    echo "      WARNING: seed-data.sh not found, skipping data seeding"
fi
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    pkill -f "cargo run" &> /dev/null || true
    pkill -f "npm run dev" &> /dev/null || true
    docker-compose -f docker-compose.local.yml down &> /dev/null || true
    echo "Services stopped."
}
trap cleanup EXIT

# Start services
echo "[5/5] Starting PolicyCortex services..."
echo ""
echo "Starting services (this may take a moment)..."

# Start Core API in background
(cd core && cargo run &> /dev/null) &
CORE_PID=$!
echo "      Core API starting on http://localhost:8080"

# Start GraphQL Gateway in background (optional)
(cd graphql && npm run dev &> /dev/null) &
GRAPHQL_PID=$!
echo "      GraphQL Gateway starting on http://localhost:4000"

# Start Frontend
echo "      Frontend starting on http://localhost:3000"
echo ""
(cd frontend && npm run dev) &
FRONTEND_PID=$!

# Wait for services to start
sleep 10

echo "=========================================="
echo "Demo Environment Ready!"
echo "=========================================="
echo ""
echo "Access the demo at: http://localhost:3000"
echo ""
echo "Demo Features:"
echo "  - Multi-tenant switching (3 demo tenants)"
echo "  - Conversational AI (demo mode)"
echo "  - Security posture dashboard"
echo "  - Cost optimization panels"
echo "  - Predictive compliance"
echo "  - SHAP explainability charts"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for user interrupt
wait $FRONTEND_PID