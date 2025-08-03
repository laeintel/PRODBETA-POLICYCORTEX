# PolicyCortex API Documentation

## Overview

PolicyCortex API provides comprehensive Azure governance, policy management, and AI-powered compliance features. This document covers all available endpoints with complete examples.

**Base URL**: `http://localhost:8000/api/v1`  
**Frontend**: `http://localhost:3000`

## Table of Contents
1. [Authentication](#authentication)
2. [Policy Management](#policy-management)
3. [Azure Integration](#azure-integration)
4. [AI Engine](#ai-engine)
5. [Conversation/Chat](#conversationchat)
6. [Data Processing](#data-processing)
7. [Notifications](#notifications)
8. [Analytics & Reports](#analytics--reports)
9. [Network Security](#network-security)
10. [System & Health](#system--health)

## Authentication

### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin@policycortex.com",
    "password": "SecurePass123!"
  }'
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_eyJhbGciOiJIUzI1NiIs..."
}
```

### Get Current User
```bash
TOKEN="your-access-token"
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### Refresh Token
```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "refresh_eyJhbGciOiJIUzI1NiIs..."
  }'
```

### Logout
```bash
curl -X POST http://localhost:8000/api/v1/auth/logout \
  -H "Authorization: Bearer $TOKEN"
```

## Policy Management

### List All Policies
```bash
curl -X GET http://localhost:8000/api/v1/policies \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "policies": [
    {
      "id": "SecurityCenterBuiltIn",
      "name": "ASC Default (subscription: 205b477d-17e7-4b3b-92c1-32cf02626b78)",
      "type": "BuiltIn",
      "category": "Security Center",
      "compliance_state": "NonCompliant",
      "resource_count": 73,
      "compliant_count": 47,
      "non_compliant_count": 26,
      "enforcement_mode": "Default",
      "metadata": {
        "assignedBy": "Azure Security Center",
        "source": "Live Azure CLI",
        "nonCompliantPolicies": 5
      }
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 20
}
```

### Get Policy Details
```bash
curl -X GET http://localhost:8000/api/v1/policies/SecurityCenterBuiltIn \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "id": "SecurityCenterBuiltIn",
  "name": "ASC Default",
  "description": "Azure Security Center default policy initiative",
  "type": "BuiltIn",
  "category": "Security Center",
  "metadata": {
    "version": "1.0.0",
    "category": "Security Center",
    "preview": false,
    "deprecated": false
  },
  "parameters": {
    "effect": {
      "type": "String",
      "defaultValue": "AuditIfNotExists",
      "allowedValues": ["AuditIfNotExists", "Disabled"]
    }
  },
  "policy_definitions": [
    {
      "id": "policy-def-1",
      "name": "Audit VMs without managed disks",
      "effect": "AuditIfNotExists"
    }
  ]
}
```

### Get Policy Compliance
```bash
curl -X GET http://localhost:8000/api/v1/policies/SecurityCenterBuiltIn/compliance \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "policy_id": "SecurityCenterBuiltIn",
  "compliance_state": "NonCompliant",
  "total_resources": 73,
  "compliant_resources": 47,
  "non_compliant_resources": 26,
  "compliance_percentage": 64.4,
  "last_evaluated": "2025-08-03T14:30:00Z",
  "resources": [
    {
      "id": "/subscriptions/205b477d.../Microsoft.Compute/virtualMachines/vm-web-01",
      "name": "vm-web-01",
      "type": "Microsoft.Compute/virtualMachines",
      "location": "eastus",
      "resource_group": "rg-production",
      "compliance_state": "NonCompliant",
      "compliance_reason": "Missing required tags: environment, owner",
      "policy_definition": "Require tags on resources",
      "last_evaluated": "2025-08-03T14:25:00Z"
    }
  ]
}
```

### Create New Policy
```bash
curl -X POST http://localhost:8000/api/v1/policies \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Require Environment Tag",
    "display_name": "Require environment tag on all resources",
    "description": "All resources must have an environment tag with values: dev, staging, or production",
    "mode": "All",
    "category": "Tagging",
    "metadata": {
      "version": "1.0.0",
      "author": "admin@policycortex.com"
    },
    "parameters": {
      "tagName": {
        "type": "String",
        "defaultValue": "environment"
      },
      "tagValues": {
        "type": "Array",
        "defaultValue": ["dev", "staging", "production"]
      }
    },
    "policy_rule": {
      "if": {
        "anyOf": [
          {
            "field": "[concat(\"tags[\", parameters(\"tagName\"), \"]\")]",
            "exists": "false"
          },
          {
            "field": "[concat(\"tags[\", parameters(\"tagName\"), \"]\")]",
            "notIn": "[parameters(\"tagValues\")]"
          }
        ]
      },
      "then": {
        "effect": "deny"
      }
    }
  }'
```

### Update Policy
```bash
curl -X PUT http://localhost:8000/api/v1/policies/custom-policy-id \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "Updated: Require environment tag",
    "description": "Updated description with more details",
    "policy_rule": {
      "if": {
        "field": "tags[\"environment\"]",
        "exists": "false"
      },
      "then": {
        "effect": "audit"
      }
    }
  }'
```

### Delete Policy
```bash
curl -X DELETE http://localhost:8000/api/v1/policies/custom-policy-id \
  -H "Authorization: Bearer $TOKEN"
```

### Evaluate Policy
```bash
curl -X POST http://localhost:8000/api/v1/policies/SecurityCenterBuiltIn/evaluate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "resource_ids": [
      "/subscriptions/205b477d.../virtualMachines/vm-001",
      "/subscriptions/205b477d.../storageAccounts/storage001"
    ],
    "dry_run": true
  }'
```

## Azure Integration

### List Subscriptions
```bash
curl -X GET http://localhost:8000/api/v1/azure/subscriptions \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "subscriptions": [
    {
      "id": "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78",
      "subscription_id": "205b477d-17e7-4b3b-92c1-32cf02626b78",
      "display_name": "Policy Cortex",
      "state": "Enabled",
      "tenant_id": "9ef5b184-d371-462a-bc75-5024ce8baff7"
    }
  ]
}
```

### List Resource Groups
```bash
curl -X GET http://localhost:8000/api/v1/azure/resource-groups \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "resource_groups": [
    {
      "id": "/subscriptions/205b477d.../resourceGroups/rg-policortex001-app-dev",
      "name": "rg-policortex001-app-dev",
      "location": "eastus",
      "tags": {
        "environment": "dev",
        "project": "policycortex",
        "owner": "admin@aeoliTech.com"
      },
      "provisioning_state": "Succeeded"
    }
  ]
}
```

### List All Resources
```bash
# List all resources
curl -X GET http://localhost:8000/api/v1/azure/resources \
  -H "Authorization: Bearer $TOKEN"

# Filter by resource type
curl -X GET "http://localhost:8000/api/v1/azure/resources?type=Microsoft.Compute/virtualMachines" \
  -H "Authorization: Bearer $TOKEN"

# Filter by resource group
curl -X GET "http://localhost:8000/api/v1/azure/resources?resource_group=rg-production" \
  -H "Authorization: Bearer $TOKEN"

# Filter by tags
curl -X GET "http://localhost:8000/api/v1/azure/resources?tag=environment:production" \
  -H "Authorization: Bearer $TOKEN"
```

### List Azure Policies
```bash
curl -X GET http://localhost:8000/api/v1/azure/policies \
  -H "Authorization: Bearer $TOKEN"
```

### List RBAC Roles
```bash
curl -X GET http://localhost:8000/api/v1/azure/rbac/roles \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "roles": [
    {
      "id": "/subscriptions/205b477d.../providers/Microsoft.Authorization/roleDefinitions/contributor-id",
      "name": "Contributor",
      "type": "BuiltInRole",
      "description": "Grants full access to manage all resources",
      "permissions": [
        {
          "actions": ["*"],
          "notActions": [
            "Microsoft.Authorization/*/Delete",
            "Microsoft.Authorization/*/Write"
          ]
        }
      ]
    }
  ]
}
```

### List Role Assignments
```bash
curl -X GET http://localhost:8000/api/v1/azure/rbac/assignments \
  -H "Authorization: Bearer $TOKEN"
```

### Create Role Assignment
```bash
curl -X POST http://localhost:8000/api/v1/azure/rbac/assignments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "principal_id": "user-object-id-here",
    "principal_type": "User",
    "role_definition_id": "/subscriptions/205b477d.../providers/Microsoft.Authorization/roleDefinitions/contributor",
    "scope": "/subscriptions/205b477d.../resourceGroups/rg-production",
    "description": "Grant contributor access to production RG"
  }'
```

### Delete Role Assignment
```bash
curl -X DELETE http://localhost:8000/api/v1/azure/rbac/assignments/assignment-id \
  -H "Authorization: Bearer $TOKEN"
```

## AI Engine

### Analyze Resources
```bash
curl -X POST http://localhost:8000/api/v1/ai/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "resource_ids": [
      "/subscriptions/205b477d.../virtualMachines/vm-001",
      "/subscriptions/205b477d.../storageAccounts/storage001"
    ],
    "analysis_types": ["security", "compliance", "cost"],
    "include_recommendations": true
  }'
```

**Response**:
```json
{
  "analysis_id": "ana-20250803-145623",
  "status": "completed",
  "findings": [
    {
      "resource_id": "/subscriptions/205b477d.../virtualMachines/vm-001",
      "resource_name": "vm-001",
      "findings": {
        "security": {
          "risk_level": "high",
          "issues": [
            "Public IP address exposed",
            "No antimalware extension installed",
            "Unencrypted OS disk"
          ],
          "score": 3.5
        },
        "compliance": {
          "violations": 4,
          "policies_violated": ["RequireEncryption", "RequireAntimalware"]
        },
        "cost": {
          "monthly_cost": 125.50,
          "optimization_potential": 45.00,
          "recommendations": ["Resize to B2s", "Enable auto-shutdown"]
        }
      }
    }
  ],
  "summary": {
    "total_issues": 15,
    "critical_issues": 3,
    "estimated_savings": 450.00
  }
}
```

### Get Predictions
```bash
curl -X POST http://localhost:8000/api/v1/ai/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_type": "cost_forecast",
    "resource_groups": ["rg-production"],
    "time_range": "next_30_days",
    "include_breakdown": true
  }'
```

### Get Optimization Suggestions
```bash
curl -X POST http://localhost:8000/api/v1/ai/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_goals": ["cost", "security", "performance"],
    "resource_scope": "subscription",
    "max_recommendations": 10
  }'
```

### Generate Policy from Natural Language
```bash
curl -X POST http://localhost:8000/api/v1/ai/generate-policy \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a policy that blocks any virtual machine creation that does not have antimalware extension installed and is not using managed disks",
    "target_resource_types": ["Microsoft.Compute/virtualMachines"],
    "effect": "deny",
    "additional_requirements": [
      "Must check for Microsoft Antimalware extension",
      "Must verify managed disks are used"
    ]
  }'
```

### Get AI Insights
```bash
curl -X GET http://localhost:8000/api/v1/ai/insights \
  -H "Authorization: Bearer $TOKEN"
```

## Conversation/Chat

### Create Chat Session
```bash
curl -X POST http://localhost:8000/api/v1/chat/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Policy Compliance Discussion",
    "context": {
      "focus_area": "azure_policies",
      "resource_groups": ["rg-production"]
    }
  }'
```

**Response**:
```json
{
  "session_id": "sess-20250803-145623",
  "name": "Policy Compliance Discussion",
  "created_at": "2025-08-03T14:56:23Z",
  "context": {
    "focus_area": "azure_policies",
    "resource_groups": ["rg-production"]
  }
}
```

### Send Chat Message
```bash
curl -X POST http://localhost:8000/api/v1/chat/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess-20250803-145623",
    "message": "Show me all non-compliant virtual machines and explain how to fix them",
    "include_context": true,
    "stream": false
  }'
```

**Response**:
```json
{
  "message_id": "msg-20250803-145630",
  "session_id": "sess-20250803-145623",
  "role": "assistant",
  "content": "I found 12 non-compliant virtual machines in your environment. Here's a breakdown:\n\n**Critical Issues (3 VMs):**\n\n1. **vm-web-01** (rg-production)\n   - Missing antimalware extension\n   - Public IP directly attached\n   - Fix: `az vm extension set --name IaaSAntimalware --publisher Microsoft.Azure.Security --vm-name vm-web-01 --resource-group rg-production`\n\n2. **vm-db-02** (rg-production)\n   - Unencrypted OS disk\n   - Fix: Enable Azure Disk Encryption\n\n**Moderate Issues (9 VMs):**\n- Missing required tags (environment, owner)\n- Not using managed disks\n\nWould you like me to generate a remediation script for all issues?",
  "metadata": {
    "processing_time": 1.2,
    "context_used": true,
    "data_sources": ["azure_resources", "policy_compliance"]
  },
  "suggestions": [
    {
      "type": "action",
      "label": "Generate remediation script",
      "action": "generate_script"
    },
    {
      "type": "action",
      "label": "Create work items",
      "action": "create_tickets"
    }
  ]
}
```

### Get Chat History
```bash
curl -X GET "http://localhost:8000/api/v1/chat/messages?session_id=sess-20250803-145623&limit=50" \
  -H "Authorization: Bearer $TOKEN"
```

### List Chat Sessions
```bash
curl -X GET http://localhost:8000/api/v1/chat/sessions \
  -H "Authorization: Bearer $TOKEN"
```

### Delete Chat Session
```bash
curl -X DELETE http://localhost:8000/api/v1/chat/sessions/sess-20250803-145623 \
  -H "Authorization: Bearer $TOKEN"
```

### WebSocket Connection (Real-time Chat)
```javascript
// JavaScript example for WebSocket connection
const ws = new WebSocket('ws://localhost:8000/api/v1/chat/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Send message
ws.send(JSON.stringify({
  type: 'message',
  session_id: 'sess-20250803-145623',
  content: 'What are my compliance issues?'
}));
```

## Data Processing

### Create Data Pipeline
```bash
curl -X POST http://localhost:8000/api/v1/data/pipelines \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Compliance Report",
    "description": "Aggregate compliance data and generate daily report",
    "schedule": {
      "type": "cron",
      "expression": "0 2 * * *",
      "timezone": "UTC"
    },
    "steps": [
      {
        "name": "Extract Resources",
        "type": "extract",
        "config": {
          "source": "azure_resources",
          "filters": {
            "resource_types": ["Microsoft.Compute/virtualMachines"]
          }
        }
      },
      {
        "name": "Check Compliance",
        "type": "transform",
        "config": {
          "operation": "check_compliance",
          "policies": ["SecurityCenterBuiltIn"]
        }
      },
      {
        "name": "Generate Report",
        "type": "load",
        "config": {
          "destination": "blob_storage",
          "format": "csv",
          "path": "reports/compliance/{date}.csv"
        }
      }
    ]
  }'
```

### List Pipelines
```bash
curl -X GET http://localhost:8000/api/v1/data/pipelines \
  -H "Authorization: Bearer $TOKEN"
```

### Get Pipeline Details
```bash
curl -X GET http://localhost:8000/api/v1/data/pipelines/pipe-123456 \
  -H "Authorization: Bearer $TOKEN"
```

### Run Pipeline
```bash
curl -X POST http://localhost:8000/api/v1/data/pipelines/pipe-123456/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "date_override": "2025-08-03"
    }
  }'
```

### Get Pipeline Status
```bash
curl -X GET http://localhost:8000/api/v1/data/pipelines/pipe-123456/status \
  -H "Authorization: Bearer $TOKEN"
```

### Export Data
```bash
curl -X POST http://localhost:8000/api/v1/data/export \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "export_type": "compliance_report",
    "format": "csv",
    "filters": {
      "compliance_state": "NonCompliant",
      "resource_types": ["Microsoft.Compute/virtualMachines"],
      "date_range": {
        "start": "2025-08-01",
        "end": "2025-08-03"
      }
    },
    "columns": [
      "resource_id",
      "resource_name",
      "resource_type",
      "compliance_state",
      "policy_name",
      "last_evaluated"
    ]
  }'
```

**Response**:
```json
{
  "export_id": "exp-20250803-150000",
  "status": "completed",
  "download_url": "http://localhost:8000/api/v1/data/export/exp-20250803-150000/download",
  "expires_at": "2025-08-04T15:00:00Z",
  "file_size": 45678,
  "row_count": 156
}
```

### Import Data
```bash
curl -X POST http://localhost:8000/api/v1/data/import \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@compliance_data.csv" \
  -F "import_type=policy_definitions" \
  -F "mode=merge"
```

## Notifications

### List Notifications
```bash
curl -X GET http://localhost:8000/api/v1/notifications \
  -H "Authorization: Bearer $TOKEN"
```

### Create Notification Rule
```bash
curl -X POST http://localhost:8000/api/v1/notifications \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Critical Policy Violations",
    "description": "Alert on critical policy violations",
    "enabled": true,
    "conditions": {
      "event_type": "policy_violation",
      "severity": ["critical", "high"],
      "resource_types": ["Microsoft.Compute/virtualMachines"]
    },
    "actions": [
      {
        "type": "email",
        "recipients": ["admin@company.com", "security@company.com"],
        "template": "critical_alert"
      },
      {
        "type": "teams",
        "webhook_url": "https://outlook.office.com/webhook/..."
      }
    ],
    "cooldown_minutes": 30
  }'
```

### Update Notification
```bash
curl -X PUT http://localhost:8000/api/v1/notifications/notif-123456 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": false,
    "cooldown_minutes": 60
  }'
```

### Delete Notification
```bash
curl -X DELETE http://localhost:8000/api/v1/notifications/notif-123456 \
  -H "Authorization: Bearer $TOKEN"
```

### Subscribe to Alerts
```bash
curl -X POST http://localhost:8000/api/v1/notifications/subscribe \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notification_types": ["policy_violation", "cost_anomaly", "security_alert"],
    "channels": [
      {
        "type": "email",
        "address": "admin@company.com"
      },
      {
        "type": "sms",
        "phone": "+1234567890"
      },
      {
        "type": "webhook",
        "url": "https://myapp.com/webhooks/alerts"
      }
    ],
    "filters": {
      "severity": ["critical", "high"],
      "resource_groups": ["rg-production"],
      "tags": {
        "environment": "production"
      }
    }
  }'
```

### Unsubscribe from Alerts
```bash
curl -X POST http://localhost:8000/api/v1/notifications/unsubscribe \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "subscription_id": "sub-123456"
  }'
```

## Analytics & Reports

### Get Dashboard Data
```bash
curl -X GET http://localhost:8000/api/v1/analytics/dashboard \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "summary": {
    "total_resources": 287,
    "compliant_resources": 234,
    "non_compliant_resources": 53,
    "compliance_percentage": 81.5,
    "critical_issues": 8,
    "cost": {
      "monthly_total": 12500.00,
      "daily_average": 416.67,
      "trend": "increasing"
    }
  },
  "resource_breakdown": {
    "Microsoft.Compute/virtualMachines": {
      "total": 45,
      "compliant": 32,
      "percentage": 71.1
    },
    "Microsoft.Storage/storageAccounts": {
      "total": 23,
      "compliant": 21,
      "percentage": 91.3
    }
  },
  "trends": {
    "compliance_7d": [78.2, 79.1, 80.5, 79.8, 81.2, 80.9, 81.5],
    "cost_7d": [1250, 1180, 1200, 1350, 1290, 1310, 1275],
    "issues_7d": [65, 62, 58, 60, 55, 54, 53]
  },
  "top_violations": [
    {
      "policy": "Require encryption at rest",
      "count": 23,
      "trend": "decreasing"
    },
    {
      "policy": "Require managed disks",
      "count": 18,
      "trend": "stable"
    }
  ],
  "recent_changes": [
    {
      "timestamp": "2025-08-03T14:30:00Z",
      "type": "resource_created",
      "resource": "vm-new-01",
      "impact": "compliance_decreased"
    }
  ]
}
```

### Get Compliance Analytics
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/compliance?date_range=last_30_days&group_by=policy" \
  -H "Authorization: Bearer $TOKEN"
```

### Get Cost Analytics
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/costs?date_range=current_month&breakdown=service" \
  -H "Authorization: Bearer $TOKEN"
```

### Get Security Analytics
```bash
curl -X GET http://localhost:8000/api/v1/analytics/security \
  -H "Authorization: Bearer $TOKEN"
```

### Generate Report
```bash
curl -X POST http://localhost:8000/api/v1/reports/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "executive_summary",
    "format": "pdf",
    "period": {
      "start": "2025-08-01",
      "end": "2025-08-31"
    },
    "sections": [
      "compliance_overview",
      "cost_analysis",
      "security_posture",
      "recommendations"
    ],
    "filters": {
      "resource_groups": ["rg-production"],
      "include_trends": true,
      "include_details": false
    }
  }'
```

### Get Report
```bash
curl -X GET http://localhost:8000/api/v1/reports/report-123456 \
  -H "Authorization: Bearer $TOKEN"
```

## Network Security

### List Network Security Groups
```bash
curl -X GET http://localhost:8000/api/v1/network/nsgs \
  -H "Authorization: Bearer $TOKEN"
```

### Get NSG Rules
```bash
curl -X GET http://localhost:8000/api/v1/network/nsgs/nsg-web-prod/rules \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "nsg_id": "nsg-web-prod",
  "rules": [
    {
      "name": "AllowHTTPS",
      "priority": 100,
      "direction": "Inbound",
      "access": "Allow",
      "protocol": "Tcp",
      "source_address_prefix": "Internet",
      "source_port_range": "*",
      "destination_address_prefix": "VirtualNetwork",
      "destination_port_range": "443",
      "description": "Allow HTTPS traffic"
    },
    {
      "name": "DenyAll",
      "priority": 4096,
      "direction": "Inbound",
      "access": "Deny",
      "protocol": "*",
      "source_address_prefix": "*",
      "source_port_range": "*",
      "destination_address_prefix": "*",
      "destination_port_range": "*"
    }
  ]
}
```

### Add NSG Rule
```bash
curl -X POST http://localhost:8000/api/v1/network/nsgs/nsg-web-prod/rules \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AllowSSH",
    "priority": 150,
    "direction": "Inbound",
    "access": "Allow",
    "protocol": "Tcp",
    "source_address_prefixes": ["10.0.1.0/24", "10.0.2.0/24"],
    "source_port_range": "*",
    "destination_address_prefix": "VirtualNetwork",
    "destination_port_range": "22",
    "description": "Allow SSH from internal networks"
  }'
```

### Update NSG Rule
```bash
curl -X PUT http://localhost:8000/api/v1/network/nsgs/nsg-web-prod/rules/AllowHTTPS \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "priority": 110,
    "source_address_prefixes": ["CloudFlare", "10.0.0.0/8"],
    "description": "Allow HTTPS from CloudFlare and internal"
  }'
```

### Delete NSG Rule
```bash
curl -X DELETE http://localhost:8000/api/v1/network/nsgs/nsg-web-prod/rules/AllowSSH \
  -H "Authorization: Bearer $TOKEN"
```

## System & Health

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-03T15:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "azure_connection": "healthy"
  }
}
```

### Readiness Check
```bash
curl -X GET http://localhost:8000/ready
```

### Metrics (Prometheus format)
```bash
curl -X GET http://localhost:8000/metrics
```

**Response**:
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/api/v1/policies",status="200"} 1234

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 1000
http_request_duration_seconds_bucket{le="0.5"} 1200
```

### System Information
```bash
curl -X GET http://localhost:8000/api/v1/system/info \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "version": "1.0.0",
  "environment": "development",
  "uptime_seconds": 3600,
  "start_time": "2025-08-03T14:00:00Z",
  "git_commit": "abc123def",
  "features": {
    "ai_engine": true,
    "real_time_sync": true,
    "multi_tenant": false
  },
  "limits": {
    "max_resources_per_query": 1000,
    "max_export_rows": 50000,
    "rate_limit_per_hour": 1000
  }
}
```

## Development/Testing Endpoints

### Simple Policy List (Testing)
```bash
curl -X GET http://localhost:8000/policies/list
```

### Policy Details (Testing)
```bash
curl -X GET http://localhost:8000/policies/SecurityCenterBuiltIn/details
```

### Compliance Summary (Testing)
```bash
curl -X GET http://localhost:8000/policies/compliance/summary
```

### Simple Chat (Testing)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are my compliance issues?"
  }'
```

## Error Handling

All API errors follow this format:

```json
{
  "error": {
    "code": "ResourceNotFound",
    "message": "The specified resource was not found",
    "details": {
      "resource_type": "Policy",
      "resource_id": "invalid-policy-id"
    },
    "timestamp": "2025-08-03T15:00:00Z",
    "trace_id": "abc-123-def-456",
    "documentation_url": "https://docs.policycortex.com/errors/ResourceNotFound"
  }
}
```

Common error codes:
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing/invalid token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

## Rate Limiting

API requests are rate limited:
- **1000 requests per hour** per user
- **100 requests per minute** per user
- **10 concurrent requests** per user

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1627847900
X-RateLimit-Window: 3600
```

## Pagination

List endpoints support pagination:

```bash
# Page 2, 50 items per page
curl -X GET "http://localhost:8000/api/v1/policies?page=2&limit=50" \
  -H "Authorization: Bearer $TOKEN"
```

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 50,
    "total": 234,
    "pages": 5,
    "has_next": true,
    "has_prev": true,
    "next_page": 3,
    "prev_page": 1
  }
}
```

## Filtering and Sorting

Most list endpoints support filtering and sorting:

```bash
# Filter and sort policies
curl -X GET "http://localhost:8000/api/v1/policies?compliance_state=NonCompliant&sort=name:asc&category=Security" \
  -H "Authorization: Bearer $TOKEN"

# Multiple sort fields
curl -X GET "http://localhost:8000/api/v1/resources?sort=compliance_state:desc,name:asc" \
  -H "Authorization: Bearer $TOKEN"
```

## Webhooks

Configure webhooks for real-time events:

```bash
curl -X POST http://localhost:8000/api/v1/webhooks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://myapp.com/webhooks/policycortex",
    "events": ["policy.violation", "resource.created", "cost.anomaly"],
    "secret": "webhook-secret-key",
    "active": true
  }'
```

## Testing Tips

1. **Set up environment variables:**
```bash
export API_URL="http://localhost:8000"
export TOKEN="your-jwt-token"
```

2. **Use curl with saved token:**
```bash
# Save token after login
TOKEN=$(curl -X POST $API_URL/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"pass"}' \
  | jq -r '.access_token')

# Use in subsequent requests
curl -H "Authorization: Bearer $TOKEN" $API_URL/api/v1/policies
```

3. **Pretty print JSON responses:**
```bash
curl -H "Authorization: Bearer $TOKEN" $API_URL/api/v1/policies | jq '.'
```

4. **Save responses for debugging:**
```bash
curl -H "Authorization: Bearer $TOKEN" $API_URL/api/v1/policies > policies.json
```

## Postman Collection

Import this collection URL in Postman:
```
http://localhost:8000/api/v1/postman-collection
```

## Support

- **API Status**: http://localhost:8000/health
- **Documentation**: http://localhost:3000/docs
- **Support Email**: support@policycortex.com