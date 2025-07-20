"""
Local development API Gateway for PolicyCortex.
Simplified version without complex dependencies for faster local testing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="PolicyCortex API Gateway (Local)",
    description="AI-powered Azure governance platform - Local Development",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "PolicyCortex API Gateway (Local)", "status": "running", "environment": "development"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "api_gateway",
        "environment": "local",
        "timestamp": "2025-07-20T00:00:00Z"
    }

@app.get("/api/v1/health")
async def api_health():
    return {
        "status": "healthy",
        "services": {
            "api_gateway": "running",
            "azure_integration": "mock",
            "ai_engine": "mock",
            "data_processing": "mock",
            "conversation": "mock",
            "notification": "mock"
        }
    }

@app.get("/api/v1/auth/me")
async def get_current_user():
    return {
        "id": "local-user-123",
        "email": "l.esere@AeoliTech.com",
        "name": "Leonard Esere",
        "roles": ["admin", "user"],
        "tenant": "AeoliTech"
    }

@app.get("/api/v1/azure/subscriptions")
async def get_subscriptions():
    return {
        "subscriptions": [
            {
                "id": "sub-12345678-1234-1234-1234-123456789012",
                "name": "AeoliTech Development Subscription",
                "state": "Enabled",
                "tenantId": "9ef5b184-d371-462a-bc75-5024ce8baff7"
            },
            {
                "id": "sub-87654321-4321-4321-4321-210987654321",
                "name": "AeoliTech Production Subscription", 
                "state": "Enabled",
                "tenantId": "9ef5b184-d371-462a-bc75-5024ce8baff7"
            }
        ]
    }

@app.get("/api/v1/policies")
async def get_policies():
    return {
        "policies": [
            {
                "id": "policy-001",
                "name": "Storage Account Security",
                "description": "Ensures storage accounts have proper security settings",
                "status": "active",
                "category": "security",
                "severity": "high",
                "resourceType": "Microsoft.Storage/storageAccounts"
            },
            {
                "id": "policy-002", 
                "name": "VM Compliance Check",
                "description": "Validates virtual machine compliance with organizational standards",
                "status": "active",
                "category": "compliance",
                "severity": "medium",
                "resourceType": "Microsoft.Compute/virtualMachines"
            },
            {
                "id": "policy-003",
                "name": "Network Security Groups",
                "description": "Monitors NSG rules for security vulnerabilities",
                "status": "draft",
                "category": "networking",
                "severity": "high",
                "resourceType": "Microsoft.Network/networkSecurityGroups"
            }
        ]
    }

@app.get("/api/v1/resources")
async def get_resources():
    return {
        "resources": [
            {
                "id": "/subscriptions/sub-123/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/proddata001",
                "name": "proddata001",
                "type": "Microsoft.Storage/storageAccounts",
                "resourceGroup": "rg-prod",
                "location": "eastus",
                "status": "compliant",
                "lastScanned": "2025-07-20T00:00:00Z"
            },
            {
                "id": "/subscriptions/sub-123/resourceGroups/rg-dev/providers/Microsoft.Compute/virtualMachines/dev-vm-01",
                "name": "dev-vm-01",
                "type": "Microsoft.Compute/virtualMachines",
                "resourceGroup": "rg-dev", 
                "location": "eastus",
                "status": "non-compliant",
                "lastScanned": "2025-07-20T00:00:00Z"
            }
        ]
    }

@app.get("/api/v1/conversations")
async def get_conversations():
    return {
        "conversations": [
            {
                "id": "conv-001",
                "title": "Storage Account Configuration",
                "status": "active",
                "lastMessage": "How can I improve the security of my storage accounts?",
                "createdAt": "2025-07-19T10:30:00Z",
                "updatedAt": "2025-07-19T15:45:00Z"
            },
            {
                "id": "conv-002", 
                "title": "Cost Optimization Query",
                "status": "completed",
                "lastMessage": "Thanks for the recommendations on reducing compute costs!",
                "createdAt": "2025-07-18T09:15:00Z",
                "updatedAt": "2025-07-18T16:22:00Z"
            }
        ]
    }

@app.post("/api/v1/conversations")
async def create_conversation(conversation_data: dict = None):
    return {
        "id": "conv-new-123",
        "title": "New Conversation",
        "status": "active",
        "createdAt": "2025-07-20T00:00:00Z",
        "message": "Conversation created successfully"
    }

@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_analytics():
    return {
        "summary": {
            "totalResources": 147,
            "compliantResources": 132,
            "nonCompliantResources": 15,
            "complianceScore": 89.8
        },
        "trends": {
            "complianceOverTime": [
                {"date": "2025-07-13", "score": 85.2},
                {"date": "2025-07-14", "score": 87.1},
                {"date": "2025-07-15", "score": 88.5},
                {"date": "2025-07-16", "score": 89.8}
            ]
        },
        "topIssues": [
            {"category": "Security", "count": 8},
            {"category": "Cost Optimization", "count": 4},
            {"category": "Performance", "count": 3}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)