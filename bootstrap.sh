#!/bin/bash
# ========================================================================
# PolicyCortex v2 - Linux/Mac Bootstrap Script
# One-click setup for development environment
# ========================================================================

set -e

echo ""
echo "================================================================"
echo "PolicyCortex v2 - Enterprise Bootstrap"
echo "================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
echo "[1/7] Checking prerequisites..."

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}ERROR: $1 is not installed${NC}"
        echo "Please install $1 from $2"
        exit 1
    fi
}

check_optional() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${YELLOW}WARNING: $1 is not installed${NC}"
        echo "$1 is optional but recommended for $2"
        echo "Install from $3"
    fi
}

check_command node "https://nodejs.org/"
check_command npm "https://nodejs.org/"
check_command cargo "https://rustup.rs/"
check_optional docker "full functionality" "https://www.docker.com/"
check_optional az "Azure integration" "https://docs.microsoft.com/cli/azure/install-azure-cli"

echo "Prerequisites check complete."
echo ""

# Check Azure authentication
echo "[2/7] Checking Azure authentication..."
if az account show &> /dev/null; then
    echo "Azure authentication detected"
    export USE_REAL_DATA=true
else
    echo -e "${YELLOW}WARNING: Not logged into Azure${NC}"
    echo "Run 'az login' to enable Azure features"
    export USE_REAL_DATA=false
fi
echo ""

# Install frontend dependencies
echo "[3/7] Installing frontend dependencies..."
cd frontend
npm install || { echo -e "${RED}ERROR: Failed to install frontend dependencies${NC}"; exit 1; }
cd ..
echo "Frontend dependencies installed."
echo ""

# Build Rust backend
echo "[4/7] Building Rust backend..."
cd core
cargo build --release || { echo -e "${RED}ERROR: Failed to build Rust backend${NC}"; exit 1; }
cd ..
echo "Rust backend built successfully."
echo ""

# Setup environment file
echo "[5/7] Setting up environment configuration..."
if [ ! -f frontend/.env.local ]; then
    echo "Creating frontend/.env.local..."
    cat > frontend/.env.local <<EOF
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WEBSOCKET_URL=ws://localhost:8080
NEXT_PUBLIC_GRAPHQL_URL=http://localhost:4000/graphql
NEXT_PUBLIC_ENABLE_TELEMETRY=false
NEXT_PUBLIC_DATA_MODE=$USE_REAL_DATA
EOF
fi

if [ ! -f core/.env ]; then
    echo "Creating core/.env..."
    cat > core/.env <<EOF
RUST_LOG=info,policycortex_core=debug
PORT=8080
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
REDIS_URL=redis://localhost:6379
USE_REAL_DATA=$USE_REAL_DATA
AZURE_SUBSCRIPTION_ID=$AZURE_SUBSCRIPTION_ID
AZURE_TENANT_ID=$AZURE_TENANT_ID
AZURE_CLIENT_ID=$AZURE_CLIENT_ID
EOF
fi
echo "Environment configuration complete."
echo ""

# Database setup
echo "[6/7] Setting up database..."
if docker ps &> /dev/null; then
    echo "Starting PostgreSQL with Docker..."
    docker run -d --name policycortex-postgres \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=policycortex \
        -p 5432:5432 \
        postgres:15 &> /dev/null || true
    
    echo "Starting Redis with Docker..."
    docker run -d --name policycortex-redis \
        -p 6379:6379 \
        redis:7-alpine &> /dev/null || true
    
    echo "Database services started."
else
    echo -e "${YELLOW}WARNING: Docker not running, skipping database setup${NC}"
    echo "You'll need to manually set up PostgreSQL and Redis"
fi
echo ""

# Preflight checks
echo "[7/7] Running preflight checks..."
echo ""
echo "System Information:"
echo "-------------------"
node --version && echo -e "Node.js: ${GREEN}OK${NC}" || echo -e "Node.js: ${RED}ERROR${NC}"
npm --version && echo -e "npm: ${GREEN}OK${NC}" || echo -e "npm: ${RED}ERROR${NC}"
cargo --version && echo -e "Rust: ${GREEN}OK${NC}" || echo -e "Rust: ${RED}ERROR${NC}"
echo ""

echo "Service Status:"
echo "---------------"
curl -s http://localhost:8080/health &> /dev/null && echo -e "Backend API: ${GREEN}RUNNING${NC}" || echo -e "Backend API: ${YELLOW}NOT RUNNING${NC}"
curl -s http://localhost:3000 &> /dev/null && echo -e "Frontend: ${GREEN}RUNNING${NC}" || echo -e "Frontend: ${YELLOW}NOT RUNNING${NC}"
nc -z localhost 5432 &> /dev/null && echo -e "PostgreSQL: ${GREEN}RUNNING${NC}" || echo -e "PostgreSQL: ${YELLOW}NOT RUNNING${NC}"
nc -z localhost 6379 &> /dev/null && echo -e "Redis: ${GREEN}RUNNING${NC}" || echo -e "Redis: ${YELLOW}NOT RUNNING${NC}"
echo ""

echo "================================================================"
echo "Bootstrap Complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo "1. Start the backend:  cd core && cargo run"
echo "2. Start the frontend: cd frontend && npm run dev"
echo "3. Open browser:       http://localhost:3000"
echo ""
echo "Data Mode: $USE_REAL_DATA"
if [ "$USE_REAL_DATA" = "false" ]; then
    echo -e "${YELLOW}Note: Running in SIMULATED mode. Run 'az login' for real Azure data.${NC}"
fi
echo ""
echo "For production deployment, see docs/deployment.md"
echo "================================================================"