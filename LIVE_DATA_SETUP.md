# PolicyCortex Live Azure Data Setup

## ✅ Complete - No Manual Configuration Required

The PolicyCortex application is now fully integrated with Azure and will automatically display live data from your Azure subscription.

## 🚀 Quick Start

```bash
# Start with live Azure data
.\scripts\start-with-live-data.bat

# Or manually:
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set USE_AZURE_DATA=true
cd core && cargo run --release
cd ../frontend && npm run dev
```

## 📊 Live Data Sources

### Dashboard/Tactical
- **Metrics**: Azure Monitor metrics (CPU, Memory, Network, Storage)
- **Alerts**: Active alerts from Azure Monitor
- **Activities**: Recent activities from Azure Activity Log
- **Cost**: Real-time cost data from Cost Management

### Governance
- **Compliance**: Policy compliance status from Azure Policy
- **Violations**: Active policy violations and non-compliant resources
- **Risk**: Security assessments from Azure Security Center
- **Cost Analysis**: Detailed cost breakdown and optimization recommendations
- **Policies**: Active policy definitions and initiatives

### Security & Access
- **IAM**: Users and groups from Azure Active Directory
- **RBAC**: Role assignments from Azure Resource Manager
- **PIM**: Privileged Identity Management requests
- **Conditional Access**: Policies from Azure AD
- **Zero Trust**: Security posture from Security Center
- **Entitlements**: Access packages from Identity Governance
- **Access Reviews**: Periodic reviews from Azure AD

### Operations
- **Resources**: Complete inventory from Azure Resource Graph
- **Monitoring**: Real-time metrics from Azure Monitor
- **Automation**: Runbooks from Automation Accounts
- **Notifications**: Action groups and alert rules
- **Alerts**: Alert instances and history

### DevOps
- **Container Registries**: Images and repositories from ACR
- **Deployments**: ARM deployment history
- **Pipelines**: GitHub Actions workflows (if connected)
- **Builds**: CI/CD status from connected systems

### AI Intelligence
- **Predictive Compliance**: ML-based predictions using Azure data patterns
- **Correlations**: Cross-domain analysis from Log Analytics
- **Chat**: Natural language queries about your Azure environment
- **Unified Metrics**: Aggregated insights across all domains

## 🔧 Architecture

```
Frontend (Next.js)
    ↓
API Client (TypeScript)
    ↓
Backend (Rust)
    ↓
Azure Service Modules
    ↓
Azure REST APIs / SDK
```

### Backend Modules
- `core/src/azure/client.rs` - Main Azure client with auth
- `core/src/azure/monitor.rs` - Metrics and monitoring
- `core/src/azure/policy.rs` - Policy and compliance
- `core/src/azure/identity.rs` - Azure AD and IAM
- `core/src/azure/resource_graph.rs` - Resource queries
- `core/src/azure/cost_management.rs` - Cost analysis
- `core/src/azure/security_center.rs` - Security posture
- `core/src/azure/activity_log.rs` - Activity tracking
- `core/src/azure/container_registry.rs` - ACR integration
- `core/src/azure/deployments.rs` - ARM deployments

## 🔐 Authentication

The system uses Azure CLI authentication automatically:
```bash
# Verify you're logged in
az account show

# Login if needed
az login
```

## 📋 Health Checks

```bash
# Check Azure connectivity
curl http://localhost:8080/api/v1/health/azure

# Test live data endpoints
.\scripts\test-live-data.bat
```

## 🛠️ Troubleshooting

### No Data Showing?
1. Verify Azure login: `az account show`
2. Check backend logs for errors
3. Ensure environment variables are set
4. Try health check endpoint

### Performance Issues?
- Data is cached for 5 minutes by default
- Adjust `CACHE_TTL_SECONDS` in `.env.azure`
- Check Azure API rate limits

### Missing Resources?
- Ensure your account has proper permissions
- Check Resource Graph queries in logs
- Verify subscription ID is correct

## 📈 Monitoring

The system provides detailed logging:
```bash
# Enable debug logging
set RUST_LOG=debug

# View Azure API calls
set LOG_AZURE_REQUESTS=true
```

## 🎯 Features

- **Automatic Fallback**: If Azure is unavailable, system uses mock data
- **Caching**: Reduces API calls and improves performance
- **Parallel Fetching**: Multiple Azure services queried simultaneously
- **Error Recovery**: Automatic retry with exponential backoff
- **Rate Limiting**: Respects Azure API limits

## 📊 Available Azure Resources

Your subscription includes:
- Resource Groups: 7 (including policycortex-gpt4o-resource)
- Storage Accounts: datalakeaeolitech
- Data Factory: adf-demo-aeolitech
- Static Websites: leonardesere
- Policy Assignments: Security Center defaults

## 🚦 Status Indicators

- **Green**: Live data from Azure
- **Yellow**: Cached data being used
- **Red**: Failed to fetch, using mock data

## 📝 Environment Variables

All configuration is in `.env.azure`:
```env
AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
USE_AZURE_DATA=true
CACHE_TTL_SECONDS=300
```

## 🎉 Ready to Use!

Your PolicyCortex instance is now displaying live data from your Azure subscription. No additional configuration is required!

---

See also: Value Guide at `docs/value/README.md` for stakeholder outcomes and ROI.