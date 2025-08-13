# Production Deployment Guide for PolicyCortex

## 🚨 CRITICAL: Production Configuration Changes

### 1. Environment Variables (.env.production)

**NEVER commit production secrets to Git!**

```env
# Azure AD Configuration (REQUIRED for Production)
NEXT_PUBLIC_AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
NEXT_PUBLIC_AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
NEXT_PUBLIC_AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78

# CRITICAL: Disable Demo Mode in Production
NEXT_PUBLIC_DEMO_MODE=false  # ⚠️ MUST be false for production

# Production URLs
NEXT_PUBLIC_MSAL_REDIRECT_URI=https://ca-cortex-frontend-prod.azurecontainerapps.io
NEXT_PUBLIC_MSAL_POST_LOGOUT_REDIRECT_URI=https://ca-cortex-frontend-prod.azurecontainerapps.io

# API Configuration
NEXT_PUBLIC_API_BASE_URL=https://ca-cortex-core-prod.azurecontainerapps.io
NEXT_PUBLIC_GRAPHQL_URL=https://ca-cortex-graphql-prod.azurecontainerapps.io/graphql

# Environment
NODE_ENV=production
NEXT_PUBLIC_ENVIRONMENT=production

# Data Mode
NEXT_PUBLIC_USE_REAL_DATA=true
USE_REAL_DATA=true

# Azure Service Principal (Store in Key Vault)
AZURE_CLIENT_SECRET=<NEVER_COMMIT_THIS>
```

### 2. Azure AD App Registration Updates

1. **Add Production Redirect URIs**:
   - `https://ca-cortex-frontend-prod.azurecontainerapps.io`
   - `https://policycortex.yourdomain.com` (if using custom domain)

2. **Remove localhost URIs** from production app registration

3. **Enable ID tokens and Access tokens** for SPA

### 3. Code Changes Required

#### A. Update next.config.js
```javascript
// frontend/next.config.js
const nextConfig = {
  // ... existing config
  output: process.env.NODE_ENV === 'production' ? 'standalone' : undefined,
  
  // Add security headers for production
  async headers() {
    if (process.env.NODE_ENV !== 'production') return []
    
    const ContentSecurityPolicy = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://login.microsoftonline.com",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob: https:",
      "font-src 'self' data:",
      "connect-src 'self' https://login.microsoftonline.com https://graph.microsoft.com https://management.azure.com wss: https:",
      "frame-src https://login.microsoftonline.com",
      "frame-ancestors 'none'",
      "object-src 'none'",
      "base-uri 'self'",
    ].join('; ');

    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'X-XSS-Protection', value: '1; mode=block' },
          { key: 'Strict-Transport-Security', value: 'max-age=31536000; includeSubDomains' },
          { key: 'Content-Security-Policy', value: ContentSecurityPolicy },
        ],
      },
    ];
  },
}
```

#### B. Remove Demo Mode Code
```typescript
// frontend/contexts/AuthContext.tsx
const login = async () => {
  setLoading(true)
  setError(null)
  
  try {
    // REMOVE THIS BLOCK FOR PRODUCTION
    // if (!process.env.NEXT_PUBLIC_AZURE_CLIENT_ID || process.env.NEXT_PUBLIC_DEMO_MODE === 'true') {
    //   ...demo code...
    // }
    
    const loginResponse = await instance.loginPopup(loginRequest)
    instance.setActiveAccount(loginResponse.account)
    console.log('Login successful:', loginResponse)
  } catch (err: any) {
    console.error('Login failed:', err)
    setError(err.message || 'Login failed')
    // REMOVE demo fallback for production
  } finally {
    setLoading(false)
  }
}
```

### 4. Docker Configuration Updates

#### A. Production Dockerfile
```dockerfile
# frontend/Dockerfile.prod
FROM node:18-alpine AS builder
WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY . .

# Build with production environment
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Production image
FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
```

### 5. Azure Container Apps Configuration

```bash
# Update Container App with production settings
az containerapp update \
  -n ca-cortex-frontend-prod \
  -g rg-cortex-prod \
  --set-env-vars \
    NODE_ENV=production \
    NEXT_PUBLIC_ENVIRONMENT=production \
    NEXT_PUBLIC_DEMO_MODE=false \
    NEXT_PUBLIC_USE_REAL_DATA=true \
    NEXT_PUBLIC_AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c \
    NEXT_PUBLIC_AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 \
    NEXT_PUBLIC_MSAL_REDIRECT_URI=https://ca-cortex-frontend-prod.azurecontainerapps.io \
  --min-replicas 2 \
  --max-replicas 10 \
  --cpu 1.0 \
  --memory 2.0Gi
```

### 6. Security Checklist

- [ ] **Demo mode disabled** (`NEXT_PUBLIC_DEMO_MODE=false`)
- [ ] **No localhost URLs** in production configuration
- [ ] **HTTPS only** - enforce SSL/TLS
- [ ] **Secrets in Key Vault** - never in environment variables
- [ ] **CORS configured** - only allow your domains
- [ ] **CSP headers** - prevent XSS attacks
- [ ] **Rate limiting** - prevent abuse
- [ ] **WAF enabled** - Azure Front Door with WAF
- [ ] **Monitoring enabled** - Application Insights
- [ ] **Backup strategy** - database and configuration

### 7. CI/CD Pipeline Updates

Update `.github/workflows/application.yml`:

```yaml
- name: Deploy to Production
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  run: |
    # Deploy only after manual approval
    az containerapp update \
      -n ca-cortex-frontend-prod \
      -g rg-cortex-prod \
      --image ${{ env.REGISTRY }}/policycortex-frontend:${{ github.sha }}
```

### 8. Database Production Config

```bash
# Use Azure Database for PostgreSQL
az postgres flexible-server create \
  --name psql-policycortex-prod \
  --resource-group rg-cortex-prod \
  --location eastus \
  --sku-name Standard_D4s_v3 \
  --storage-size 128 \
  --version 15 \
  --high-availability Enabled \
  --backup-retention 30
```

### 9. Monitoring & Logging

```bash
# Enable Application Insights
az containerapp env dapr-component set \
  --name cae-cortex-prod \
  --resource-group rg-cortex-prod \
  --dapr-component-name appinsights \
  --yaml appinsights.yaml
```

### 10. Pre-Production Checklist

#### Frontend
- [ ] Remove all console.log statements
- [ ] Enable production builds
- [ ] Minify JavaScript and CSS
- [ ] Enable gzip compression
- [ ] Set up CDN for static assets
- [ ] Configure proper cache headers

#### Backend
- [ ] Use production database
- [ ] Enable connection pooling
- [ ] Set up Redis/DragonflyDB for caching
- [ ] Configure proper logging levels
- [ ] Enable health checks
- [ ] Set up auto-scaling

#### Security
- [ ] Run security audit: `npm audit`
- [ ] Update all dependencies
- [ ] Enable Azure AD Conditional Access
- [ ] Configure IP restrictions
- [ ] Enable DDoS protection
- [ ] Set up Azure Key Vault integration

#### Performance
- [ ] Enable HTTP/2
- [ ] Configure proper cache policies
- [ ] Optimize images
- [ ] Enable lazy loading
- [ ] Set up monitoring alerts

### 11. Rollback Strategy

```bash
# Tag production deployments
git tag -a v1.0.0-prod -m "Production release v1.0.0"
git push origin v1.0.0-prod

# Rollback command
az containerapp revision set-mode \
  --name ca-cortex-frontend-prod \
  --resource-group rg-cortex-prod \
  --mode single \
  --revision <previous-revision-name>
```

## 🚀 Deployment Commands

```bash
# 1. Build production image
docker build -f Dockerfile.prod -t policycortex-frontend:prod .

# 2. Tag for registry
docker tag policycortex-frontend:prod crcortexprodvb9v2h.azurecr.io/policycortex-frontend:v1.0.0

# 3. Push to registry
docker push crcortexprodvb9v2h.azurecr.io/policycortex-frontend:v1.0.0

# 4. Deploy to Container Apps
az containerapp update \
  -n ca-cortex-frontend-prod \
  -g rg-cortex-prod \
  --image crcortexprodvb9v2h.azurecr.io/policycortex-frontend:v1.0.0
```

## ⚠️ CRITICAL WARNINGS

1. **NEVER deploy with NEXT_PUBLIC_DEMO_MODE=true**
2. **NEVER commit production secrets to Git**
3. **ALWAYS test in staging environment first**
4. **ALWAYS have a rollback plan**
5. **ALWAYS monitor after deployment**

## 📞 Support Contacts

- **On-call Engineer**: [Your contact]
- **Azure Support**: [Ticket URL]
- **Monitoring Dashboard**: [URL]