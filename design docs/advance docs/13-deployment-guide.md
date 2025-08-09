# 13. Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Development Deployment](#local-development-deployment)
4. [Staging Environment Deployment](#staging-environment-deployment)
5. [Production Deployment](#production-deployment)
6. [Azure Kubernetes Service (AKS) Deployment](#azure-kubernetes-service-aks-deployment)
7. [Database Migrations](#database-migrations)
8. [SSL/TLS Configuration](#ssltls-configuration)
9. [Load Balancing & High Availability](#load-balancing--high-availability)
10. [Disaster Recovery](#disaster-recovery)
11. [Rollback Procedures](#rollback-procedures)
12. [Health Checks & Monitoring](#health-checks--monitoring)
13. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), Windows Server 2019+, or macOS 11+
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 50GB SSD, Recommended 100GB+ SSD
- **Network**: Stable internet connection with Azure access

### Software Dependencies
```bash
# Required software versions
- Docker: 24.0+
- Docker Compose: 2.20+
- Kubernetes: 1.28+
- Helm: 3.12+
- Node.js: 18.17+
- Rust: 1.75+
- Python: 3.11+
- PostgreSQL: 15+
- Azure CLI: 2.50+
```

### Azure Requirements
```bash
# Azure subscription with required permissions
- Subscription Owner or Contributor role
- Resource Groups creation permissions
- Azure Kubernetes Service permissions
- Azure Container Registry access
- Azure Key Vault permissions
- Azure Monitor access
```

## Environment Setup

### Environment Variables Configuration

#### Base Environment Variables
```bash
# .env.base
NODE_ENV=production
RUST_ENV=production
LOG_LEVEL=info

# Database Configuration
DATABASE_URL=postgresql://username:password@host:5432/policycortex
EVENTSTORE_CONNECTION=esdb://username:password@host:2113
REDIS_URL=redis://host:6379

# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

# Application Configuration
API_BASE_URL=https://api.policycortex.com
FRONTEND_URL=https://app.policycortex.com
GRAPHQL_ENDPOINT=https://api.policycortex.com/graphql

# Security Configuration
JWT_SECRET=your-jwt-secret-key
ENCRYPTION_KEY=your-encryption-key
WEBHOOK_SECRET=your-webhook-secret

# AI Configuration
AI_MODEL_ENDPOINT=https://ai.policycortex.com
OPENAI_API_KEY=your-openai-key
AZURE_AI_ENDPOINT=your-azure-ai-endpoint
```

#### Development Environment
```bash
# .env.development
NODE_ENV=development
RUST_ENV=development
LOG_LEVEL=debug

DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex_dev
EVENTSTORE_CONNECTION=esdb://admin:changeit@localhost:2113
REDIS_URL=redis://localhost:6379

API_BASE_URL=http://localhost:8080
FRONTEND_URL=http://localhost:3000
GRAPHQL_ENDPOINT=http://localhost:4000/graphql
```

#### Production Environment
```bash
# .env.production
NODE_ENV=production
RUST_ENV=production
LOG_LEVEL=warn

# Use Azure Key Vault references
DATABASE_URL=@Microsoft.KeyVault(SecretUri=https://kv-name.vault.azure.net/secrets/database-url)
AZURE_CLIENT_SECRET=@Microsoft.KeyVault(SecretUri=https://kv-name.vault.azure.net/secrets/azure-client-secret)
JWT_SECRET=@Microsoft.KeyVault(SecretUri=https://kv-name.vault.azure.net/secrets/jwt-secret)
```

## Local Development Deployment

### Quick Start with Docker Compose
```bash
# 1. Clone repository
git clone https://github.com/aeolitech/policycortex.git
cd policycortex

# 2. Set up environment
cp .env.example .env.local
# Edit .env.local with your configuration

# 3. Start all services
docker-compose -f docker-compose.dev.yml up -d

# 4. Initialize database
docker-compose exec core-api cargo run --bin migrate
docker-compose exec postgres psql -U postgres -d policycortex -f /docker-entrypoint-initdb.d/seed.sql

# 5. Verify deployment
curl http://localhost:8080/health
curl http://localhost:3000
curl http://localhost:4000/graphql
```

### Manual Development Setup
```bash
# 1. Start infrastructure services
docker-compose -f docker-compose.infra.yml up -d

# 2. Start Rust backend
cd core
cargo install cargo-watch
cargo watch -x run

# 3. Start Next.js frontend (new terminal)
cd frontend
npm install
npm run dev

# 4. Start GraphQL gateway (new terminal)
cd graphql
npm install
npm run dev

# 5. Start AI services (new terminal)
cd backend/services/ai_engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

## Staging Environment Deployment

### Azure Container Registry Setup
```bash
# 1. Create Azure Container Registry
az acr create \
  --resource-group rg-policycortex-staging \
  --name acrpolicycortexstaging \
  --sku Basic \
  --admin-enabled true

# 2. Login to registry
az acr login --name acrpolicycortexstaging

# 3. Build and push images
./scripts/build-and-push.sh staging
```

### Build and Push Script
```bash
#!/bin/bash
# scripts/build-and-push.sh

ENVIRONMENT=$1
REGISTRY="acrpolicycortex${ENVIRONMENT}.azurecr.io"
VERSION=$(git rev-parse --short HEAD)

echo "Building images for environment: $ENVIRONMENT"
echo "Registry: $REGISTRY"
echo "Version: $VERSION"

# Build Core API (Rust)
docker build -f core/Dockerfile -t $REGISTRY/core-api:$VERSION -t $REGISTRY/core-api:latest .
docker push $REGISTRY/core-api:$VERSION
docker push $REGISTRY/core-api:latest

# Build Frontend (Next.js)
docker build -f frontend/Dockerfile -t $REGISTRY/frontend:$VERSION -t $REGISTRY/frontend:latest .
docker push $REGISTRY/frontend:$VERSION
docker push $REGISTRY/frontend:latest

# Build GraphQL Gateway
docker build -f graphql/Dockerfile -t $REGISTRY/graphql-gateway:$VERSION -t $REGISTRY/graphql-gateway:latest .
docker push $REGISTRY/graphql-gateway:$VERSION
docker push $REGISTRY/graphql-gateway:latest

# Build AI Engine
docker build -f backend/services/ai_engine/Dockerfile -t $REGISTRY/ai-engine:$VERSION -t $REGISTRY/ai-engine:latest .
docker push $REGISTRY/ai-engine:$VERSION
docker push $REGISTRY/ai-engine:latest

echo "All images pushed successfully!"
```

### Staging Kubernetes Deployment
```yaml
# k8s/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../base
  - staging-config.yaml
  - staging-secrets.yaml

replicas:
  - name: core-api
    count: 2
  - name: frontend
    count: 2
  - name: graphql-gateway
    count: 2

images:
  - name: core-api
    newName: acrpolicycortexstaging.azurecr.io/core-api
    newTag: latest
  - name: frontend
    newName: acrpolicycortexstaging.azurecr.io/frontend
    newTag: latest
  - name: graphql-gateway
    newName: acrpolicycortexstaging.azurecr.io/graphql-gateway
    newTag: latest

patchesStrategicMerge:
  - staging-patches.yaml
```

```bash
# Deploy to staging
kubectl apply -k k8s/staging/

# Verify deployment
kubectl get pods -n policycortex-staging
kubectl get services -n policycortex-staging
kubectl get ingress -n policycortex-staging
```

## Production Deployment

### Production Infrastructure with Terraform
```hcl
# terraform/production/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  backend "azurerm" {
    resource_group_name  = "rg-policycortex-tfstate"
    storage_account_name = "stapolicycortextfstate"
    container_name       = "tfstate"
    key                  = "production.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-policycortex-prod"
  location = "East US"
  tags = {
    Environment = "production"
    Project     = "PolicyCortex"
  }
}

# Azure Kubernetes Service
resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-policycortex-prod"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "policycortex-prod"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v3"
    
    upgrade_settings {
      max_surge = "10%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
  }

  tags = {
    Environment = "production"
  }
}

# Application Gateway
resource "azurerm_application_gateway" "main" {
  name                = "ag-policycortex-prod"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }

  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.gateway.id
  }

  frontend_port {
    name = "https-port"
    port = 443
  }

  frontend_port {
    name = "http-port"
    port = 80
  }

  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.gateway.id
  }

  backend_address_pool {
    name = "backend-pool"
  }

  backend_http_settings {
    name                  = "backend-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 60
  }

  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name             = "https-port"
    protocol                       = "Https"
    ssl_certificate_name           = "policycortex-ssl"
  }

  request_routing_rule {
    name                       = "routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "https-listener"
    backend_address_pool_name  = "backend-pool"
    backend_http_settings_name = "backend-http-settings"
  }

  ssl_certificate {
    name     = "policycortex-ssl"
    data     = filebase64("certs/policycortex.pfx")
    password = var.ssl_cert_password
  }
}
```

### Production Deployment Script
```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

ENVIRONMENT="production"
NAMESPACE="policycortex"
REGISTRY="acrpolicycortexprod.azurecr.io"
VERSION=${1:-latest}

echo "Starting production deployment..."
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Registry: $REGISTRY"

# 1. Build and push images
echo "Building and pushing images..."
./scripts/build-and-push.sh $ENVIRONMENT

# 2. Apply infrastructure with Terraform
echo "Deploying infrastructure..."
cd terraform/production
terraform init
terraform plan -out=tfplan
terraform apply tfplan
cd ../..

# 3. Configure kubectl
echo "Configuring kubectl..."
az aks get-credentials --resource-group rg-policycortex-prod --name aks-policycortex-prod

# 4. Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 5. Apply secrets
echo "Applying secrets..."
kubectl apply -f k8s/production/secrets.yaml -n $NAMESPACE

# 6. Deploy applications
echo "Deploying applications..."
helm upgrade --install policycortex ./helm/policycortex \
  --namespace $NAMESPACE \
  --values helm/policycortex/values-production.yaml \
  --set image.tag=$VERSION \
  --set image.registry=$REGISTRY \
  --wait \
  --timeout=600s

# 7. Verify deployment
echo "Verifying deployment..."
kubectl wait --for=condition=ready pod -l app=core-api -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=frontend -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=graphql-gateway -n $NAMESPACE --timeout=300s

# 8. Run health checks
echo "Running health checks..."
./scripts/health-check.sh $ENVIRONMENT

echo "Production deployment completed successfully!"
```

## Azure Kubernetes Service (AKS) Deployment

### Helm Chart Configuration
```yaml
# helm/policycortex/values-production.yaml
global:
  environment: production
  registry: acrpolicycortexprod.azurecr.io
  imageTag: latest
  
replicaCount:
  coreApi: 3
  frontend: 3
  graphqlGateway: 2
  aiEngine: 2

resources:
  coreApi:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  
  frontend:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

ingress:
  enabled: true
  className: "azure/application-gateway"
  annotations:
    kubernetes.io/ingress.class: azure/application-gateway
    appgw.ingress.kubernetes.io/ssl-redirect: "true"
    appgw.ingress.kubernetes.io/use-private-ip: "false"
    appgw.ingress.kubernetes.io/backend-path-prefix: "/"
  hosts:
    - host: app.policycortex.com
      paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: frontend
              port: 3000
    - host: api.policycortex.com
      paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: core-api
              port: 8080
        - path: /graphql
          pathType: Prefix
          backend:
            service:
              name: graphql-gateway
              port: 4000
  tls:
    - secretName: policycortex-tls
      hosts:
        - app.policycortex.com
        - api.policycortex.com

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
  alerts:
    enabled: true

security:
  networkPolicies:
    enabled: true
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  securityContext:
    allowPrivilegeEscalation: false
    capabilities:
      drop:
        - ALL
    readOnlyRootFilesystem: true
```

### Application Gateway Ingress Controller
```yaml
# k8s/production/agic-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agic-config
  namespace: kube-system
data:
  appgw.usePrivateIP: "false"
  appgw.shared: "false"
  appgw.verbosityLevel: "3"
  kubernetes.watchNamespace: "policycortex"
  appgw.environment.name: "AZUREPUBLICCLOUD"
```

## Database Migrations

### Migration Strategy
```rust
// core/src/migrations/mod.rs
use sqlx::{migrate::MigrateDatabase, Postgres, Pool};
use std::env;

pub async fn run_migrations() -> Result<(), Box<dyn std::error::Error>> {
    let database_url = env::var("DATABASE_URL")?;
    
    // Ensure database exists
    if !Postgres::database_exists(&database_url).await? {
        println!("Creating database...");
        Postgres::create_database(&database_url).await?;
    }
    
    // Connect to database
    let pool = Pool::<Postgres>::connect(&database_url).await?;
    
    // Run migrations
    println!("Running migrations...");
    sqlx::migrate!("./migrations").run(&pool).await?;
    
    println!("Migrations completed successfully!");
    Ok(())
}

pub async fn rollback_migration(steps: u64) -> Result<(), Box<dyn std::error::Error>> {
    let database_url = env::var("DATABASE_URL")?;
    let pool = Pool::<Postgres>::connect(&database_url).await?;
    
    println!("Rolling back {} migration(s)...", steps);
    
    for _ in 0..steps {
        sqlx::migrate!("./migrations").undo(&pool, 1).await?;
    }
    
    println!("Rollback completed successfully!");
    Ok(())
}
```

### Pre-deployment Migration Script
```bash
#!/bin/bash
# scripts/migrate-database.sh

set -e

ENVIRONMENT=$1
DATABASE_URL=$2

if [ -z "$ENVIRONMENT" ] || [ -z "$DATABASE_URL" ]; then
    echo "Usage: $0 <environment> <database_url>"
    exit 1
fi

echo "Running database migrations for environment: $ENVIRONMENT"
echo "Database URL: $(echo $DATABASE_URL | sed 's/:[^:]*@/:***@/')"

# Backup database (production only)
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Creating database backup..."
    BACKUP_FILE="backup-$(date +%Y%m%d-%H%M%S).sql"
    pg_dump "$DATABASE_URL" > "backups/$BACKUP_FILE"
    echo "Backup created: backups/$BACKUP_FILE"
fi

# Run migrations
echo "Running migrations..."
cd core
DATABASE_URL="$DATABASE_URL" cargo run --bin migrate

# Verify migrations
echo "Verifying migrations..."
DATABASE_URL="$DATABASE_URL" cargo run --bin verify-schema

echo "Database migration completed successfully!"
```

## SSL/TLS Configuration

### Let's Encrypt with Cert-Manager
```yaml
# k8s/production/cert-manager.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@policycortex.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: azure/application-gateway
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: policycortex-tls
  namespace: policycortex
spec:
  secretName: policycortex-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - app.policycortex.com
  - api.policycortex.com
  - admin.policycortex.com
```

### Custom SSL Certificate Configuration
```yaml
# k8s/production/ssl-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: policycortex-tls
  namespace: policycortex
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi... # base64 encoded certificate
  tls.key: LS0tLS1CRUdJTi... # base64 encoded private key
```

## Load Balancing & High Availability

### Azure Load Balancer Configuration
```yaml
# k8s/production/load-balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: policycortex-lb
  namespace: policycortex
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-internal: "false"
    service.beta.kubernetes.io/azure-pip-name: "pip-policycortex-prod"
    service.beta.kubernetes.io/azure-dns-label-name: "policycortex-prod"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 3000
    name: http
  - port: 443
    targetPort: 3000
    name: https
  selector:
    app: frontend
```

### Pod Disruption Budget
```yaml
# k8s/production/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: core-api-pdb
  namespace: policycortex
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: core-api
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: frontend-pdb
  namespace: policycortex
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: frontend
```

## Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# scripts/backup.sh

set -e

ENVIRONMENT=$1
BACKUP_DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="backups/$ENVIRONMENT/$BACKUP_DATE"

mkdir -p "$BACKUP_DIR"

echo "Starting backup for environment: $ENVIRONMENT"
echo "Backup directory: $BACKUP_DIR"

# Database backup
echo "Backing up PostgreSQL database..."
pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/postgres.sql.gz"

# EventStore backup
echo "Backing up EventStore..."
curl -X POST "http://$EVENTSTORE_HOST:2113/admin/scavenge" \
     -H "Authorization: Basic $(echo -n admin:changeit | base64)"
tar -czf "$BACKUP_DIR/eventstore.tar.gz" /var/lib/eventstore/data

# Configuration backup
echo "Backing up Kubernetes configurations..."
kubectl get all -n policycortex -o yaml > "$BACKUP_DIR/k8s-resources.yaml"
kubectl get secrets -n policycortex -o yaml > "$BACKUP_DIR/k8s-secrets.yaml"
kubectl get configmaps -n policycortex -o yaml > "$BACKUP_DIR/k8s-configmaps.yaml"

# Upload to Azure Blob Storage
echo "Uploading backup to Azure Blob Storage..."
az storage blob upload-batch \
    --destination "backups" \
    --source "$BACKUP_DIR" \
    --account-name "stapolicycortexbackups" \
    --destination-path "$ENVIRONMENT/$BACKUP_DATE"

echo "Backup completed successfully!"
echo "Backup location: Azure Blob Storage - backups/$ENVIRONMENT/$BACKUP_DATE"
```

### Restore Procedure
```bash
#!/bin/bash
# scripts/restore.sh

set -e

ENVIRONMENT=$1
BACKUP_DATE=$2

if [ -z "$ENVIRONMENT" ] || [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <environment> <backup_date>"
    echo "Example: $0 production 20231201-143022"
    exit 1
fi

RESTORE_DIR="restore/$ENVIRONMENT/$BACKUP_DATE"

echo "Starting restore for environment: $ENVIRONMENT"
echo "Backup date: $BACKUP_DATE"

# Download backup from Azure Blob Storage
echo "Downloading backup from Azure Blob Storage..."
mkdir -p "$RESTORE_DIR"
az storage blob download-batch \
    --destination "$RESTORE_DIR" \
    --source "backups" \
    --pattern "$ENVIRONMENT/$BACKUP_DATE/*" \
    --account-name "stapolicycortexbackups"

# Scale down applications
echo "Scaling down applications..."
kubectl scale deployment --replicas=0 -n policycortex --all

# Restore database
echo "Restoring PostgreSQL database..."
gunzip -c "$RESTORE_DIR/postgres.sql.gz" | psql "$DATABASE_URL"

# Restore EventStore
echo "Restoring EventStore..."
kubectl delete statefulset eventstore -n policycortex
tar -xzf "$RESTORE_DIR/eventstore.tar.gz" -C /
kubectl apply -f k8s/production/eventstore.yaml

# Restore Kubernetes resources
echo "Restoring Kubernetes resources..."
kubectl apply -f "$RESTORE_DIR/k8s-resources.yaml"
kubectl apply -f "$RESTORE_DIR/k8s-secrets.yaml"
kubectl apply -f "$RESTORE_DIR/k8s-configmaps.yaml"

# Scale up applications
echo "Scaling up applications..."
kubectl scale deployment core-api --replicas=3 -n policycortex
kubectl scale deployment frontend --replicas=3 -n policycortex
kubectl scale deployment graphql-gateway --replicas=2 -n policycortex

echo "Restore completed successfully!"
```

## Rollback Procedures

### Blue-Green Deployment Rollback
```bash
#!/bin/bash
# scripts/rollback-bluegreen.sh

set -e

ENVIRONMENT=$1
ROLLBACK_VERSION=$2

echo "Starting blue-green rollback..."
echo "Environment: $ENVIRONMENT"
echo "Rollback version: $ROLLBACK_VERSION"

# Update ingress to point to previous version
kubectl patch ingress policycortex-ingress -n policycortex -p '{
  "spec": {
    "rules": [{
      "host": "api.policycortex.com",
      "http": {
        "paths": [{
          "path": "/",
          "pathType": "Prefix",
          "backend": {
            "service": {
              "name": "core-api-blue",
              "port": {"number": 8080}
            }
          }
        }]
      }
    }]
  }
}'

# Verify rollback
echo "Verifying rollback..."
sleep 30
curl -f "https://api.policycortex.com/health" || {
    echo "Rollback verification failed!"
    exit 1
}

echo "Blue-green rollback completed successfully!"
```

### Canary Deployment Rollback
```bash
#!/bin/bash
# scripts/rollback-canary.sh

set -e

ENVIRONMENT=$1

echo "Starting canary rollback..."
echo "Environment: $ENVIRONMENT"

# Remove canary traffic routing
kubectl patch virtualservice policycortex-vs -n policycortex -p '{
  "spec": {
    "http": [{
      "route": [{
        "destination": {
          "host": "core-api-stable",
          "subset": "v1"
        },
        "weight": 100
      }]
    }]
  }
}'

# Scale down canary deployment
kubectl scale deployment core-api-canary --replicas=0 -n policycortex

echo "Canary rollback completed successfully!"
```

## Health Checks & Monitoring

### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

set -e

ENVIRONMENT=$1
BASE_URL=${2:-"https://api.policycortex.com"}

echo "Running health checks for environment: $ENVIRONMENT"
echo "Base URL: $BASE_URL"

# Core API health check
echo "Checking Core API health..."
CORE_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$CORE_RESPONSE" != "200" ]; then
    echo "❌ Core API health check failed (HTTP $CORE_RESPONSE)"
    exit 1
fi
echo "✅ Core API is healthy"

# Frontend health check
echo "Checking Frontend health..."
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "https://app.policycortex.com")
if [ "$FRONTEND_RESPONSE" != "200" ]; then
    echo "❌ Frontend health check failed (HTTP $FRONTEND_RESPONSE)"
    exit 1
fi
echo "✅ Frontend is healthy"

# GraphQL health check
echo "Checking GraphQL health..."
GRAPHQL_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "{ __schema { types { name } } }"}' \
    "$BASE_URL/graphql" \
    -o /dev/null -w "%{http_code}")
if [ "$GRAPHQL_RESPONSE" != "200" ]; then
    echo "❌ GraphQL health check failed (HTTP $GRAPHQL_RESPONSE)"
    exit 1
fi
echo "✅ GraphQL is healthy"

# Database connectivity check
echo "Checking database connectivity..."
DB_CHECK=$(kubectl exec -n policycortex deployment/core-api -- \
    sh -c 'curl -s http://localhost:8080/health/db' | jq -r '.status')
if [ "$DB_CHECK" != "healthy" ]; then
    echo "❌ Database connectivity check failed"
    exit 1
fi
echo "✅ Database is healthy"

# AI Engine health check
echo "Checking AI Engine health..."
AI_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/ai/health")
if [ "$AI_RESPONSE" != "200" ]; then
    echo "❌ AI Engine health check failed (HTTP $AI_RESPONSE)"
    exit 1
fi
echo "✅ AI Engine is healthy"

echo ""
echo "🎉 All health checks passed successfully!"
```

### Kubernetes Liveness and Readiness Probes
```yaml
# k8s/production/core-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: core-api
  namespace: policycortex
spec:
  template:
    spec:
      containers:
      - name: core-api
        image: acrpolicycortexprod.azurecr.io/core-api:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
```

## Troubleshooting

### Common Issues and Solutions

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n policycortex

# Describe failed pod
kubectl describe pod <pod-name> -n policycortex

# Check pod logs
kubectl logs <pod-name> -n policycortex --previous

# Check events
kubectl get events -n policycortex --sort-by=.metadata.creationTimestamp
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/core-api -n policycortex -- \
    sh -c 'psql $DATABASE_URL -c "SELECT 1;"'

# Check database pod logs
kubectl logs -l app=postgres -n policycortex

# Verify secrets
kubectl get secret db-credentials -n policycortex -o yaml
```

#### SSL/TLS Certificate Issues
```bash
# Check certificate status
kubectl describe certificate policycortex-tls -n policycortex

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manually renew certificate
kubectl delete certificate policycortex-tls -n policycortex
kubectl apply -f k8s/production/certificates.yaml
```

#### Application Gateway Issues
```bash
# Check AGIC logs
kubectl logs -n kube-system deployment/ingress-azure

# Verify backend health
az network application-gateway show-backend-health \
    --name ag-policycortex-prod \
    --resource-group rg-policycortex-prod

# Check ingress configuration
kubectl describe ingress policycortex-ingress -n policycortex
```

### Performance Troubleshooting
```bash
# Check resource usage
kubectl top pods -n policycortex
kubectl top nodes

# Check HPA status
kubectl get hpa -n policycortex

# Check network policies
kubectl get networkpolicy -n policycortex

# Analyze application metrics
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Access http://localhost:9090 for Prometheus UI
```

### Emergency Procedures
```bash
#!/bin/bash
# scripts/emergency-scale-down.sh

echo "🚨 EMERGENCY: Scaling down all applications"
kubectl scale deployment --replicas=0 -n policycortex --all

echo "Applications scaled down. Checking status..."
kubectl get pods -n policycortex

echo "Emergency scale-down completed."
echo "To restore: ./scripts/emergency-scale-up.sh"
```

```bash
#!/bin/bash
# scripts/emergency-scale-up.sh

echo "🔄 EMERGENCY RECOVERY: Scaling up applications"
kubectl scale deployment core-api --replicas=3 -n policycortex
kubectl scale deployment frontend --replicas=3 -n policycortex
kubectl scale deployment graphql-gateway --replicas=2 -n policycortex
kubectl scale deployment ai-engine --replicas=2 -n policycortex

echo "Applications scaling up. Waiting for ready status..."
kubectl wait --for=condition=ready pod -l app=core-api -n policycortex --timeout=300s
kubectl wait --for=condition=ready pod -l app=frontend -n policycortex --timeout=300s

echo "Emergency recovery completed."
echo "Running health checks..."
./scripts/health-check.sh production
```

This comprehensive deployment guide provides step-by-step instructions for deploying PolicyCortex across all environments, from local development to production. It includes infrastructure setup, database migrations, SSL configuration, monitoring, and troubleshooting procedures to ensure reliable deployments.