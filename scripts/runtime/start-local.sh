#!/bin/bash

# PolicyCortex v2 Local Development Startup Script

echo "🚀 Starting PolicyCortex v2 Local Development Environment"
echo "============================================="

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.local.yml down

# Build and start services
echo "🔨 Building services..."
docker-compose -f docker-compose.local.yml build

# Start services
echo "🎯 Starting services..."
docker-compose -f docker-compose.local.yml up -d postgres redis
docker-compose -f docker-compose.local.yml up -d core graphql frontend

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
echo ""
echo "Core Service:     http://localhost:8080/health"
curl -s http://localhost:8080/health || echo "⏳ Core service starting..."
echo ""
echo "GraphQL Gateway:  http://localhost:4000/graphql"
curl -s http://localhost:4000/.well-known/apollo/server-health || echo "⏳ GraphQL starting..."
echo ""
echo "Frontend:         http://localhost:3000"
curl -s http://localhost:3000 > /dev/null && echo "✅ Frontend ready" || echo "⏳ Frontend starting..."
echo ""
echo "EventStore:       http://localhost:2113"
echo "PostgreSQL:       localhost:5432"
echo "DragonflyDB:      localhost:6379"
echo ""

# Show logs
echo "📋 Service Logs (Ctrl+C to exit):"
echo "============================================="
docker-compose -f docker-compose.local.yml logs -f