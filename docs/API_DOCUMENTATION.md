# PolicyCortex v2 - Complete API Documentation

**Base URL**: https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io
**GraphQL URL**: https://ca-cortex-graphql-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/graphql

## 🔑 Authentication
Currently running in demo mode with `REQUIRE_AUTH=false`. In production, all endpoints require Azure AD Bearer token:
```bash
Authorization: Bearer <azure-ad-token>
```

---

## 📊 Core Patent Endpoints

### 1. **Unified Governance Metrics** (Patent #1)
```bash
GET /api/v1/metrics
GET /api/v1/governance/unified  # Alias

# Example:
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/metrics

# Response:
{
  "policies": {
    "total": 15,
    "active": 12,
    "violations": 3,
    "automated": 10,
    "compliance_rate": 85.5,
    "prediction_accuracy": 92.3
  },
  "rbac": {
    "users": 150,
    "roles": 25,
    "violations": 2,
    "risk_score": 3.2,
    "anomalies_detected": 1
  },
  "costs": {
    "current_spend": 125000.0,
    "predicted_spend": 118000.0,
    "savings_identified": 7000.0,
    "optimization_rate": 12.5
  },
  "network": {
    "endpoints": 45,
    "active_threats": 2,
    "blocked_attempts": 5,
    "latency_ms": 23.5
  },
  "resources": {
    "total": 450,
    "optimized": 380,
    "idle": 25,
    "overprovisioned": 15
  },
  "ai": {
    "accuracy": 92.3,
    "predictions_made": 1250,
    "automations_executed": 450,
    "learning_progress": 87.5
  }
}
```

### 2. **Predictive Compliance** (Patent #2)
```bash
GET /api/v1/predictions
GET /api/v1/compliance/predict  # Alias

# Example:
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/predictions

# Response:
{
  "predictions": [
    {
      "resource_id": "vm-prod-01",
      "policy_id": "require-encryption",
      "prediction_time": "2025-08-12T20:00:00Z",
      "violation_probability": 0.78,
      "confidence_interval": [0.72, 0.84],
      "drift_detected": true,
      "recommended_actions": [
        {
          "action": "EnableEncryption",
          "priority": "High",
          "estimated_time": "5 minutes",
          "automation_available": true
        }
      ]
    }
  ],
  "summary": {
    "high_risk_resources": 3,
    "total_predictions": 45,
    "drift_detected_count": 8
  }
}
```

### 3. **Conversational Intelligence** (Patent #3)
```bash
POST /api/v1/conversation
POST /api/v1/nlp/query  # Alias

# Example:
curl -X POST https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/conversation \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my top compliance risks?"}'

# Response:
{
  "query": "What are my top compliance risks?",
  "intent": "ComplianceRiskQuery",
  "confidence": 0.95,
  "response": "Based on analysis, your top 3 compliance risks are...",
  "data": {
    "risks": [
      {"resource": "vm-prod-01", "risk": "Unencrypted disks", "severity": "High"},
      {"resource": "storage-02", "risk": "Public access enabled", "severity": "Critical"},
      {"resource": "network-03", "risk": "Open port 3389", "severity": "Medium"}
    ]
  },
  "suggested_actions": ["Enable encryption", "Disable public access", "Close RDP port"]
}
```

### 4. **Cross-Domain Correlations** (Patent #4)
```bash
GET /api/v1/correlations
GET /api/v1/analysis/cross-domain  # Alias

# Example:
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/correlations

# Response:
{
  "correlations": [
    {
      "id": "corr-001",
      "pattern": "HighCostWithLowUtilization",
      "confidence": 0.89,
      "affected_domains": ["costs", "resources"],
      "resources": ["vm-dev-cluster", "storage-backup-01"],
      "impact": {
        "cost_impact": 15000,
        "risk_score": 2.5
      },
      "recommendation": "Right-size or terminate underutilized resources"
    }
  ],
  "patterns_detected": 5,
  "cross_domain_insights": 12
}
```

---

## 🎯 Recommendations & Insights

### 5. **AI Recommendations**
```bash
GET /api/v1/recommendations
GET /api/v1/recommendations/proactive  # Proactive recommendations

# Example:
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/recommendations

# Response:
{
  "recommendations": [
    {
      "id": "rec-001",
      "type": "CostOptimization",
      "title": "Convert to Reserved Instances",
      "description": "Save $3,500/month by converting 10 VMs to reserved instances",
      "priority": "High",
      "potential_savings": 3500,
      "effort": "Low",
      "automation_available": true
    }
  ],
  "total_potential_savings": 7000,
  "implementation_effort": "2 hours"
}
```

---

## 📋 Deep Analysis Endpoints

### 6. **Policies Deep View**
```bash
GET /api/v1/policies/deep

# Example with filters:
curl "https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/policies/deep?category=Security&compliance=false"

# Response includes detailed policy analysis with violations, trends, and predictions
```

### 7. **RBAC Deep View**
```bash
GET /api/v1/rbac/deep

# Example:
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/rbac/deep

# Response includes role assignments, privileged accounts, anomalies, and recommendations
```

### 8. **Costs Deep View**
```bash
GET /api/v1/costs/deep

# Response includes departmental breakdown, trends, forecasts, and optimization opportunities
```

### 9. **Network Deep View**
```bash
GET /api/v1/network/deep

# Response includes security groups, traffic patterns, threat detection, and vulnerabilities
```

### 10. **Resources Deep View**
```bash
GET /api/v1/resources/deep

# Response includes resource inventory, utilization metrics, and optimization suggestions
```

---

## 🔧 Action & Remediation Endpoints

### 11. **Remediate Issues**
```bash
POST /api/v1/remediate

# Request:
{
  "resource_id": "vm-prod-01",
  "issue_type": "UnencryptedDisk",
  "auto_approve": false
}

# Response:
{
  "action_id": "act-123",
  "status": "PendingApproval",
  "estimated_time": "5 minutes",
  "requires_approval": true
}
```

### 12. **Create Exception**
```bash
POST /api/v1/exception

# Request:
{
  "resource_id": "vm-dev-01",
  "policy_id": "require-encryption",
  "reason": "Development environment, no sensitive data",
  "expires_at": "2025-09-01T00:00:00Z"
}
```

### 13. **Create Action**
```bash
POST /api/v1/actions

# Request:
{
  "action_type": "ScaleResource",
  "target_resource": "vmss-prod",
  "parameters": {"scale_to": 10},
  "dry_run": true
}
```

### 14. **Get Action Status**
```bash
GET /api/v1/actions/{action_id}

# Real-time events stream:
GET /api/v1/actions/{action_id}/events  # Server-Sent Events (SSE)
```

---

## ✅ Approval Workflow

### 15. **Create Approval Request**
```bash
POST /api/v1/approvals

# Request:
{
  "action_id": "act-123",
  "requested_by": "user@company.com",
  "justification": "Required for production deployment"
}
```

### 16. **List Pending Approvals**
```bash
GET /api/v1/approvals

# Response includes all pending approval requests with details
```

### 17. **Approve/Reject Request**
```bash
POST /api/v1/approvals/{approval_id}

# Request:
{
  "decision": "approve",  # or "reject"
  "comment": "Approved for production"
}
```

---

## 📜 Policy Management

### 18. **List Policies**
```bash
GET /api/v1/policies

# Query parameters:
?category=Security
?compliance_status=false
?enabled=true
```

### 19. **Generate Policy Code**
```bash
POST /api/v1/policies/generate

# Request:
{
  "description": "Ensure all VMs have backup enabled",
  "language": "terraform"  # or "bicep", "arm"
}

# Response: Generated policy code
```

### 20. **Export Policies**
```bash
GET /api/v1/policies/export

# Downloads policies as ZIP/TAR archive
```

### 21. **Policy Drift Detection**
```bash
GET /api/v1/policies/drift

# Response shows policies that have drifted from baseline
```

---

## 📊 Compliance & Frameworks

### 22. **Compliance Status**
```bash
GET /api/v1/compliance

# Response includes compliance scores by framework (ISO, SOC2, HIPAA, etc.)
```

### 23. **List Compliance Frameworks**
```bash
GET /api/v1/frameworks

# Response: Available compliance frameworks
```

### 24. **Get Framework Details**
```bash
GET /api/v1/frameworks/{framework_id}

# Example: /api/v1/frameworks/iso27001
```

### 25. **Evidence Pack Generation**
```bash
GET /api/v1/evidence

# Generates compliance evidence package for audits
```

---

## 📈 Resource Management

### 26. **List Resources**
```bash
GET /api/v1/resources

# Query parameters:
?type=VirtualMachine
?status=Running
?tag=Environment:Production
```

### 27. **List Exceptions**
```bash
GET /api/v1/exceptions

# Shows all active policy exceptions
```

### 28. **Expire Exceptions**
```bash
POST /api/v1/exceptions/expire

# Request:
{
  "exception_ids": ["exc-001", "exc-002"]
}
```

---

## 🔄 Real-Time Streams

### 29. **Global Events Stream** (Server-Sent Events)
```bash
GET /api/v1/events

# Example using curl:
curl -N -H "Accept: text/event-stream" \
  https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/events

# Streams real-time events:
data: {"event": "PolicyViolation", "resource": "vm-001", "timestamp": "2025-08-12T20:00:00Z"}
data: {"event": "CostAlert", "threshold_exceeded": true, "amount": 150000}
```

---

## 🏥 System Health & Management

### 30. **Health Check**
```bash
GET /health
GET /api/v1/health  # Alias

# Response:
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": "24h 15m",
  "mode": "simulated"
}
```

### 31. **Configuration Status**
```bash
GET /api/v1/config

# Response shows current configuration (non-sensitive)
```

### 32. **Secrets Status**
```bash
GET /api/v1/secrets/status

# Response:
{
  "configured": true,
  "provider": "AzureKeyVault",
  "last_refresh": "2025-08-12T19:00:00Z"
}
```

### 33. **Reload Secrets**
```bash
POST /api/v1/secrets/reload

# Forces refresh of secrets from Key Vault
```

### 34. **Roadmap Status**
```bash
GET /api/v1/roadmap

# Response shows implementation status of features
```

### 35. **Prometheus Metrics**
```bash
GET /metrics

# Returns Prometheus-formatted metrics for monitoring
```

---

## 🔌 GraphQL API

### GraphQL Endpoint
```bash
POST https://ca-cortex-graphql-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/graphql
```

### Example Queries:

**Get Governance Overview:**
```graphql
query GovernanceOverview {
  metrics {
    policies {
      total
      violations
      complianceRate
    }
    costs {
      currentSpend
      predictedSpend
      savingsIdentified
    }
  }
}
```

**Get Policy Details:**
```graphql
query PolicyDetails($category: String) {
  policies(category: $category) {
    id
    name
    category
    complianceStatus
    violations {
      resourceId
      severity
      detectedAt
    }
  }
}
```

**Get Recommendations:**
```graphql
query Recommendations {
  recommendations {
    id
    type
    title
    priority
    potentialSavings
    automationAvailable
  }
}
```

---

## 🧪 Testing Examples

### Quick Test All Patent Endpoints:
```bash
# Test all 4 patent endpoints
for endpoint in metrics predictions recommendations correlations; do
  echo "Testing /api/v1/$endpoint:"
  curl -s https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/$endpoint | jq '.success // true'
  echo "---"
done
```

### Test with Authentication (when enabled):
```bash
# Get token from Azure AD first
TOKEN=$(az account get-access-token --resource api://1ecc95d1-e5bb-43e2-9324-30a17cb6b01c --query accessToken -o tsv)

# Use token in requests
curl -H "Authorization: Bearer $TOKEN" \
  https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/metrics
```

### Test GraphQL:
```bash
curl -X POST https://ca-cortex-graphql-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ metrics { policies { total violations } } }"}'
```

---

## 📝 Notes

1. **Simulated Mode**: Currently running with simulated data. Set `USE_REAL_DATA=true` for production.
2. **Authentication**: Disabled for demo. Enable with `REQUIRE_AUTH=true`.
3. **Rate Limiting**: No rate limits in demo mode. Production has 1000 req/min limit.
4. **CORS**: Configured to allow all origins in demo. Restrict in production.
5. **Write Operations**: Most write operations return success in simulated mode but don't persist.

---

## 🚀 Postman Collection

Import this collection for easy testing:
```json
{
  "info": {
    "name": "PolicyCortex v2 API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "variable": [
    {
      "key": "baseUrl",
      "value": "https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io"
    }
  ]
}
```

---

**Last Updated**: 2025-08-12 20:30 UTC
**Status**: All endpoints operational in simulated mode