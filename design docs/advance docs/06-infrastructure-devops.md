# Infrastructure & DevOps

## Table of Contents
1. [Docker Configuration](#docker-configuration)
2. [Kubernetes Deployment](#kubernetes-deployment)
3. [Terraform Infrastructure](#terraform-infrastructure)
4. [CI/CD Pipelines](#cicd-pipelines)
5. [Environment Management](#environment-management)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security & Compliance](#security--compliance)
8. [Disaster Recovery](#disaster-recovery)
9. [Performance & Scaling](#performance--scaling)

## Docker Configuration

### Multi-stage Builds

```dockerfile
# Rust Core API Dockerfile
FROM rust:1.70-slim as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build dependencies first for better caching
RUN cargo build --release --bin core

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/core ./core

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
USER 1000:1000

CMD ["./core"]
```

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Runtime stage
FROM node:18-alpine

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3000
ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit 1

CMD ["node", "server.js"]
```

```dockerfile
# AI Engine Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8081/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
```

### Docker Compose Environments

```yaml
# docker-compose.local.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: policycortex
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/seed-data.sql:/docker-entrypoint-initdb.d/seed-data.sql
    networks:
      - policycortex

  eventstore:
    image: eventstore/eventstore:23.10.0-buster-slim
    environment:
      - EVENTSTORE_CLUSTER_SIZE=1
      - EVENTSTORE_RUN_PROJECTIONS=All
      - EVENTSTORE_START_STANDARD_PROJECTIONS=true
      - EVENTSTORE_EXT_TCP_PORT=1113
      - EVENTSTORE_HTTP_PORT=2113
      - EVENTSTORE_INSECURE=true
      - EVENTSTORE_ENABLE_EXTERNAL_TCP=true
      - EVENTSTORE_ENABLE_ATOM_PUB_OVER_HTTP=true
    ports:
      - "1113:1113"
      - "2113:2113"
    volumes:
      - eventstore_data:/var/lib/eventstore
    networks:
      - policycortex

  dragonfly:
    image: docker.dragonflydb.io/dragonflydb/dragonfly
    ports:
      - "6379:6379"
    volumes:
      - dragonfly_data:/data
    networks:
      - policycortex

  core:
    build:
      context: ./core
      dockerfile: Dockerfile
    depends_on:
      - postgres
      - eventstore
      - dragonfly
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/policycortex
      - EVENTSTORE_URL=tcp://eventstore:1113
      - REDIS_URL=redis://dragonfly:6379
      - AZURE_SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID}
      - AZURE_TENANT_ID=${AZURE_TENANT_ID}
      - AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
      - RUST_LOG=info
    ports:
      - "8080:8080"
    networks:
      - policycortex
    restart: unless-stopped

  ai-engine:
    build:
      context: ./backend/services/ai_engine
      dockerfile: Dockerfile
    depends_on:
      - postgres
      - dragonfly
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/policycortex
      - REDIS_URL=redis://dragonfly:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
    ports:
      - "8081:8081"
    networks:
      - policycortex
    restart: unless-stopped

  api-gateway:
    build:
      context: ./backend/services/api_gateway
      dockerfile: Dockerfile
    depends_on:
      - core
      - ai-engine
    environment:
      - CORE_API_URL=http://core:8080
      - AI_ENGINE_URL=http://ai-engine:8081
      - AZURE_SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID}
      - AZURE_TENANT_ID=${AZURE_TENANT_ID}
      - AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
    ports:
      - "8082:8082"
    networks:
      - policycortex
    restart: unless-stopped

  graphql-gateway:
    build:
      context: ./graphql
      dockerfile: Dockerfile
    depends_on:
      - core
      - ai-engine
      - api-gateway
    environment:
      - CORE_URL=http://core:8080/graphql
      - AI_URL=http://ai-engine:8081/graphql
      - AZURE_URL=http://api-gateway:8082/graphql
      - REDIS_URL=redis://dragonfly:6379
    ports:
      - "4000:4000"
    networks:
      - policycortex
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    depends_on:
      - graphql-gateway
    environment:
      - NEXT_PUBLIC_GRAPHQL_URL=http://localhost:4000/graphql
      - NEXT_PUBLIC_WS_URL=ws://localhost:4000/graphql
      - NEXTAUTH_URL=http://localhost:3000
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
    ports:
      - "3000:3000"
    networks:
      - policycortex
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    depends_on:
      - frontend
      - graphql-gateway
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./infrastructure/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./infrastructure/nginx/ssl:/etc/nginx/ssl
    networks:
      - policycortex
    restart: unless-stopped

volumes:
  postgres_data:
  eventstore_data:
  dragonfly_data:

networks:
  policycortex:
    driver: bridge
```

### Nginx Configuration

```nginx
# infrastructure/nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }
    
    upstream api {
        server graphql-gateway:4000;
    }
    
    upstream core {
        server core:8080;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=graphql:10m rate=5r/s;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    server {
        listen 80;
        server_name localhost;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name localhost;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # GraphQL API
        location /graphql {
            limit_req zone=graphql burst=10 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for subscriptions
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Core API (direct access for some endpoints)
        location /api/v1/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://core;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health checks
        location /health {
            access_log off;
            proxy_pass http://core/health;
        }
    }
}
```

## Kubernetes Deployment

### Namespace and RBAC

```yaml
# infrastructure/kubernetes/base/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: policycortex
  labels:
    name: policycortex
    environment: production

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: policycortex-sa
  namespace: policycortex

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: policycortex-role
  namespace: policycortex
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: policycortex-binding
  namespace: policycortex
subjects:
- kind: ServiceAccount
  name: policycortex-sa
  namespace: policycortex
roleRef:
  kind: Role
  name: policycortex-role
  apiGroup: rbac.authorization.k8s.io
```

### ConfigMaps and Secrets

```yaml
# infrastructure/kubernetes/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: policycortex-config
  namespace: policycortex
data:
  RUST_LOG: "info"
  DATABASE_URL: "postgresql://postgres:postgres@postgres:5432/policycortex"
  EVENTSTORE_URL: "tcp://eventstore:1113"
  REDIS_URL: "redis://dragonfly:6379"
  CORE_URL: "http://core:8080/graphql"
  AI_URL: "http://ai-engine:8081/graphql"
  AZURE_URL: "http://api-gateway:8082/graphql"

---
apiVersion: v1
kind: Secret
metadata:
  name: policycortex-secrets
  namespace: policycortex
type: Opaque
data:
  # Base64 encoded values
  AZURE_CLIENT_SECRET: <base64-encoded-secret>
  OPENAI_API_KEY: <base64-encoded-key>
  AZURE_OPENAI_KEY: <base64-encoded-key>
  NEXTAUTH_SECRET: <base64-encoded-secret>
  JWT_SECRET: <base64-encoded-secret>
```

### Core Service Deployment

```yaml
# infrastructure/kubernetes/base/core-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: core
  namespace: policycortex
  labels:
    app: core
    component: api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: core
  template:
    metadata:
      labels:
        app: core
        component: api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: policycortex-sa
      containers:
      - name: core
        image: policycortex/core:latest
        ports:
        - containerPort: 8080
          name: http
        envFrom:
        - configMapRef:
            name: policycortex-config
        - secretRef:
            name: policycortex-secrets
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL

---
apiVersion: v1
kind: Service
metadata:
  name: core
  namespace: policycortex
  labels:
    app: core
    component: api
spec:
  selector:
    app: core
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  type: ClusterIP

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: core-pdb
  namespace: policycortex
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: core
```

### HPA and VPA Configuration

```yaml
# infrastructure/kubernetes/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: core-hpa
  namespace: policycortex
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: core
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60

---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: core-vpa
  namespace: policycortex
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: core
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: core
      maxAllowed:
        cpu: "2"
        memory: "4Gi"
      minAllowed:
        cpu: "100m"
        memory: "128Mi"
```

### Ingress Configuration

```yaml
# infrastructure/kubernetes/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: policycortex-ingress
  namespace: policycortex
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization"
    nginx.ingress.kubernetes.io/enable-cors: "true"
spec:
  tls:
  - hosts:
    - policycortex.example.com
    secretName: policycortex-tls
  rules:
  - host: policycortex.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 3000
      - path: /graphql
        pathType: Exact
        backend:
          service:
            name: graphql-gateway
            port:
              number: 4000
      - path: /api/v1
        pathType: Prefix
        backend:
          service:
            name: core
            port:
              number: 8080

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: policycortex-netpol
  namespace: policycortex
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nginx-ingress
  - from:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 3000
    - protocol: TCP
      port: 4000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS outbound
    - protocol: TCP
      port: 80    # HTTP outbound
    - protocol: UDP
      port: 53    # DNS
```

## Terraform Infrastructure

### Main Infrastructure

```hcl
# infrastructure/terraform/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.70"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "azurerm" {
    resource_group_name  = "policycortex-terraform"
    storage_account_name = "policycortexstate"
    container_name       = "tfstate"
    key                  = "policycortex.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "policycortex-${var.environment}"
  location = var.location
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
    ManagedBy   = "Terraform"
  }
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "policycortex-aks-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "policycortex-${var.environment}"
  
  kubernetes_version = "1.28"
  
  default_node_pool {
    name       = "default"
    node_count = var.node_count
    vm_size    = var.node_vm_size
    
    enable_auto_scaling = true
    min_count          = var.min_node_count
    max_count          = var.max_node_count
    
    os_disk_size_gb = 100
    
    node_labels = {
      Environment = var.environment
      NodePool    = "default"
    }
    
    tags = {
      Environment = var.environment
      NodePool    = "default"
    }
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  network_profile {
    network_plugin = "azure"
    network_policy = "calico"
  }
  
  api_server_authorized_ip_ranges = var.authorized_ip_ranges
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

# Additional node pool for compute-intensive workloads
resource "azurerm_kubernetes_cluster_node_pool" "compute" {
  name                  = "compute"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = "Standard_D4s_v3"
  node_count           = 2
  
  enable_auto_scaling = true
  min_count          = 1
  max_count          = 10
  
  node_labels = {
    workload = "compute-intensive"
  }
  
  node_taints = [
    "workload=compute:NoSchedule"
  ]
  
  tags = {
    Environment = var.environment
    NodePool    = "compute"
  }
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "policycortexacr${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Premium"
  admin_enabled       = false
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

# Grant AKS access to ACR
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "policycortex-postgres-${var.environment}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  administrator_login    = var.postgres_admin_username
  administrator_password = var.postgres_admin_password
  
  storage_mb   = 32768
  sku_name     = var.postgres_sku
  
  backup_retention_days = 30
  geo_redundant_backup_enabled = true
  
  high_availability {
    mode = "ZoneRedundant"
  }
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "policycortex"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  name                = "policycortex-redis-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku
  
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  redis_configuration {
    maxmemory_reserved = 10
    maxmemory_delta    = 2
    maxmemory_policy   = "allkeys-lru"
  }
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                = "policycortex-kv-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  
  sku_name = "standard"
  
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    
    key_permissions = [
      "Get", "List", "Create", "Delete", "Recover", "Purge"
    ]
    
    secret_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Purge"
    ]
    
    certificate_permissions = [
      "Get", "List", "Create", "Delete", "Recover", "Purge"
    ]
  }
  
  # AKS access
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_kubernetes_cluster.main.identity[0].principal_id
    
    secret_permissions = [
      "Get", "List"
    ]
  }
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "policycortex-appinsights-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "policycortex-logs-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
  }
}

# Data sources
data "azurerm_client_config" "current" {}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "East US 2"
}

variable "node_count" {
  description = "Number of nodes in the default node pool"
  type        = number
  default     = 3
}

variable "min_node_count" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "node_vm_size" {
  description = "VM size for nodes"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "authorized_ip_ranges" {
  description = "IP ranges authorized to access the API server"
  type        = list(string)
  default     = []
}

variable "postgres_admin_username" {
  description = "PostgreSQL admin username"
  type        = string
  sensitive   = true
}

variable "postgres_admin_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

variable "postgres_sku" {
  description = "PostgreSQL SKU"
  type        = string
  default     = "GP_Standard_D2s_v3"
}

variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "C"
}

variable "redis_sku" {
  description = "Redis cache SKU"
  type        = string
  default     = "Standard"
}

# Outputs
output "kube_config" {
  value     = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive = true
}

output "cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "postgres_fqdn" {
  value = azurerm_postgresql_flexible_server.main.fqdn
}

output "redis_hostname" {
  value = azurerm_redis_cache.main.hostname
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "application_insights_connection_string" {
  value     = azurerm_application_insights.main.connection_string
  sensitive = true
}
```

### Environment-Specific Variables

```hcl
# infrastructure/terraform/environments/dev/terraform.tfvars
environment = "dev"
location    = "East US 2"

# AKS Configuration
node_count     = 2
min_node_count = 1
max_node_count = 5
node_vm_size   = "Standard_B2s"

# Database Configuration
postgres_sku = "B_Standard_B1ms"

# Cache Configuration
redis_capacity = 0
redis_family   = "C"
redis_sku      = "Basic"

# Security
authorized_ip_ranges = [
  "YOUR_OFFICE_IP/32"
]
```

```hcl
# infrastructure/terraform/environments/prod/terraform.tfvars
environment = "prod"
location    = "East US 2"

# AKS Configuration
node_count     = 5
min_node_count = 3
max_node_count = 20
node_vm_size   = "Standard_D4s_v3"

# Database Configuration
postgres_sku = "GP_Standard_D4s_v3"

# Cache Configuration
redis_capacity = 6
redis_family   = "P"
redis_sku      = "Premium"

# Security - Restrict to specific IPs/ranges
authorized_ip_ranges = [
  "YOUR_OFFICE_IP/32",
  "YOUR_VPN_RANGE/24"
]
```

## CI/CD Pipelines

### GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: policycortexacr.azurecr.io
  IMAGE_TAG: ${{ github.sha }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: |
          frontend/package-lock.json
          graphql/package-lock.json

    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt, clippy

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    # Rust Tests
    - name: Run Rust Tests
      working-directory: ./core
      run: |
        cargo fmt --check
        cargo clippy -- -D warnings
        cargo test --all-features
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379

    # Frontend Tests
    - name: Install Frontend Dependencies
      working-directory: ./frontend
      run: npm ci

    - name: Run Frontend Tests
      working-directory: ./frontend
      run: |
        npm run lint
        npm run type-check
        npm run test
        npm run build

    # GraphQL Tests
    - name: Install GraphQL Dependencies
      working-directory: ./graphql
      run: npm ci

    - name: Run GraphQL Tests
      working-directory: ./graphql
      run: |
        npm run lint
        npm run test

    # AI Engine Tests
    - name: Install Python Dependencies
      working-directory: ./backend/services/ai_engine
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run Python Tests
      working-directory: ./backend/services/ai_engine
      run: |
        pytest --cov=. --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379

    # Integration Tests
    - name: Run Integration Tests
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to start
        npm run test:integration
        docker-compose -f docker-compose.test.yml down

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v4

    - name: Login to Azure Container Registry
      uses: azure/docker-login@v1
      with:
        login-server: ${{ env.REGISTRY }}
        username: ${{ secrets.ACR_USERNAME }}
        password: ${{ secrets.ACR_PASSWORD }}

    # Build and push Core API
    - name: Build and push Core API
      run: |
        docker build -t ${{ env.REGISTRY }}/core:${{ env.IMAGE_TAG }} ./core
        docker push ${{ env.REGISTRY }}/core:${{ env.IMAGE_TAG }}

    # Build and push Frontend
    - name: Build and push Frontend
      run: |
        docker build -t ${{ env.REGISTRY }}/frontend:${{ env.IMAGE_TAG }} ./frontend
        docker push ${{ env.REGISTRY }}/frontend:${{ env.IMAGE_TAG }}

    # Build and push AI Engine
    - name: Build and push AI Engine
      run: |
        docker build -t ${{ env.REGISTRY }}/ai-engine:${{ env.IMAGE_TAG }} ./backend/services/ai_engine
        docker push ${{ env.REGISTRY }}/ai-engine:${{ env.IMAGE_TAG }}

    # Build and push GraphQL Gateway
    - name: Build and push GraphQL Gateway
      run: |
        docker build -t ${{ env.REGISTRY }}/graphql-gateway:${{ env.IMAGE_TAG }} ./graphql
        docker push ${{ env.REGISTRY }}/graphql-gateway:${{ env.IMAGE_TAG }}

  deploy-dev:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: development
    steps:
    - uses: actions/checkout@v4

    - name: Azure CLI Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: Get AKS credentials
      run: |
        az aks get-credentials --resource-group policycortex-dev --name policycortex-aks-dev

    - name: Deploy to Development
      run: |
        helm upgrade --install policycortex ./infrastructure/helm/policycortex \
          --namespace policycortex \
          --create-namespace \
          --set image.tag=${{ env.IMAGE_TAG }} \
          --set environment=dev \
          --values ./infrastructure/helm/values-dev.yaml

    - name: Run Smoke Tests
      run: |
        kubectl wait --for=condition=ready pod -l app=core --timeout=300s -n policycortex
        kubectl wait --for=condition=ready pod -l app=frontend --timeout=300s -n policycortex
        npm run test:smoke

  deploy-staging:
    runs-on: ubuntu-latest
    needs: [build, deploy-dev]
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
    - uses: actions/checkout@v4

    - name: Azure CLI Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: Get AKS credentials
      run: |
        az aks get-credentials --resource-group policycortex-staging --name policycortex-aks-staging

    - name: Deploy to Staging
      run: |
        helm upgrade --install policycortex ./infrastructure/helm/policycortex \
          --namespace policycortex \
          --create-namespace \
          --set image.tag=${{ env.IMAGE_TAG }} \
          --set environment=staging \
          --values ./infrastructure/helm/values-staging.yaml

    - name: Run E2E Tests
      run: |
        kubectl wait --for=condition=ready pod -l app=core --timeout=300s -n policycortex
        npm run test:e2e

  deploy-production:
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v4

    - name: Azure CLI Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: Get AKS credentials
      run: |
        az aks get-credentials --resource-group policycortex-prod --name policycortex-aks-prod

    - name: Blue-Green Deployment
      run: |
        # Deploy to green environment
        helm upgrade --install policycortex-green ./infrastructure/helm/policycortex \
          --namespace policycortex-green \
          --create-namespace \
          --set image.tag=${{ env.IMAGE_TAG }} \
          --set environment=prod \
          --set deployment.suffix=green \
          --values ./infrastructure/helm/values-prod.yaml

        # Wait for green deployment to be ready
        kubectl wait --for=condition=ready pod -l app=core,deployment=green --timeout=600s -n policycortex-green

        # Run health checks
        npm run test:health-check -- --environment=green

        # Switch traffic to green
        kubectl patch service core -n policycortex -p '{"spec":{"selector":{"deployment":"green"}}}'
        kubectl patch service frontend -n policycortex -p '{"spec":{"selector":{"deployment":"green"}}}'

        # Clean up old blue deployment after successful switch
        sleep 300  # Wait 5 minutes to ensure stability
        helm uninstall policycortex-blue -n policycortex-blue || true
        kubectl delete namespace policycortex-blue || true

  notify:
    runs-on: ubuntu-latest
    needs: [deploy-production]
    if: always()
    steps:
    - name: Notify Slack
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
      if: always()
```

### Helm Chart

```yaml
# infrastructure/helm/policycortex/Chart.yaml
apiVersion: v2
name: policycortex
description: PolicyCortex AI-powered Azure governance platform
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
- name: postgresql
  version: "12.8.2"
  repository: "https://charts.bitnami.com/bitnami"
  condition: postgresql.enabled
- name: redis
  version: "17.15.2"
  repository: "https://charts.bitnami.com/bitnami"
  condition: redis.enabled
```

```yaml
# infrastructure/helm/policycortex/values.yaml
# Default values for PolicyCortex
replicaCount: 3

image:
  registry: policycortexacr.azurecr.io
  tag: latest
  pullPolicy: IfNotPresent

environment: dev

# Core API configuration
core:
  enabled: true
  replicaCount: 3
  image:
    repository: core
  service:
    type: ClusterIP
    port: 8080
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Frontend configuration
frontend:
  enabled: true
  replicaCount: 2
  image:
    repository: frontend
  service:
    type: ClusterIP
    port: 3000
  resources:
    requests:
      memory: "128Mi"
      cpu: "100m"
    limits:
      memory: "256Mi"
      cpu: "200m"

# AI Engine configuration
aiEngine:
  enabled: true
  replicaCount: 2
  image:
    repository: ai-engine
  service:
    type: ClusterIP
    port: 8081
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

# GraphQL Gateway configuration
graphqlGateway:
  enabled: true
  replicaCount: 2
  image:
    repository: graphql-gateway
  service:
    type: ClusterIP
    port: 4000
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# PostgreSQL configuration
postgresql:
  enabled: false  # Use external Azure PostgreSQL
  external:
    host: ""
    port: 5432
    database: "policycortex"
    username: "postgres"
    existingSecret: "policycortex-secrets"
    existingSecretPasswordKey: "postgres-password"

# Redis configuration
redis:
  enabled: false  # Use external Azure Redis
  external:
    host: ""
    port: 6379
    existingSecret: "policycortex-secrets"
    existingSecretPasswordKey: "redis-password"

# Ingress configuration
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: policycortex.example.com
      paths:
        - path: /
          pathType: Prefix
          service:
            name: frontend
            port: 3000
        - path: /graphql
          pathType: Exact
          service:
            name: graphql-gateway
            port: 4000
        - path: /api/v1
          pathType: Prefix
          service:
            name: core
            port: 8080
  tls:
    - secretName: policycortex-tls
      hosts:
        - policycortex.example.com

# Monitoring
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true

# Security
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 2

nodeSelector: {}
tolerations: []
affinity: {}
```

## Environment Management

### GitOps with ArgoCD

```yaml
# infrastructure/argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: policycortex
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/policycortex
    targetRevision: HEAD
    path: infrastructure/helm/policycortex
    helm:
      valueFiles:
        - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: policycortex
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
  revisionHistoryLimit: 10
```

### Environment Configuration Management

```bash
#!/bin/bash
# scripts/environment-setup.sh

set -e

ENVIRONMENT=${1:-dev}
RESOURCE_GROUP="policycortex-${ENVIRONMENT}"
LOCATION=${2:-"East US 2"}

echo "Setting up environment: ${ENVIRONMENT}"

# Terraform deployment
cd infrastructure/terraform
terraform workspace select ${ENVIRONMENT} || terraform workspace new ${ENVIRONMENT}
terraform init
terraform plan -var-file="environments/${ENVIRONMENT}/terraform.tfvars"
terraform apply -var-file="environments/${ENVIRONMENT}/terraform.tfvars" -auto-approve

# Get outputs
CLUSTER_NAME=$(terraform output -raw cluster_name)
RESOURCE_GROUP_NAME=$(terraform output -raw resource_group_name)
ACR_LOGIN_SERVER=$(terraform output -raw acr_login_server)

# Configure kubectl
az aks get-credentials --resource-group ${RESOURCE_GROUP_NAME} --name ${CLUSTER_NAME}

# Install necessary operators
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Wait for operators to be ready
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=controller -n ingress-nginx --timeout=300s

# Create secrets
kubectl create namespace policycortex --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic policycortex-secrets \
  --from-literal=postgres-password="${POSTGRES_PASSWORD}" \
  --from-literal=redis-password="${REDIS_PASSWORD}" \
  --from-literal=jwt-secret="${JWT_SECRET}" \
  --from-literal=azure-client-secret="${AZURE_CLIENT_SECRET}" \
  --from-literal=openai-api-key="${OPENAI_API_KEY}" \
  --namespace=policycortex \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy application
cd ../helm
helm upgrade --install policycortex ./policycortex \
  --namespace policycortex \
  --values ./values-${ENVIRONMENT}.yaml \
  --set image.registry=${ACR_LOGIN_SERVER} \
  --set image.tag=latest

echo "Environment ${ENVIRONMENT} setup complete!"
echo "Access the application at: https://policycortex-${ENVIRONMENT}.example.com"
```

This comprehensive infrastructure and DevOps documentation covers all aspects of deploying and managing PolicyCortex, from local development to production environments. The configuration supports modern DevOps practices including infrastructure as code, GitOps, automated testing, and blue-green deployments.