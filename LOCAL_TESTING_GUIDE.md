# PolicyCortex Local Testing Guide

## 🎯 Objective
Test the complete PolicyCortex system locally to verify all 4 patent implementations are working before production deployment.

## 📋 Prerequisites

### Required Software
- **Docker Desktop** (with Docker Compose support)
- **Python 3.11+** 
- **Node.js 18+**
- **PowerShell** (for Windows scripts)

### System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB free space
- **CPU**: 4+ cores recommended

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```powershell
# Run the automated setup script
.\test-local-setup.ps1
```

### Option 2: Manual Setup
```bash
# 1. Start all services
docker-compose -f docker-compose.local.yml up -d --build

# 2. Wait for services to initialize (2-3 minutes)
docker-compose -f docker-compose.local.yml logs -f

# 3. Test the APIs
python test_patent_apis.py
```

## 🔧 What Gets Started

### Infrastructure Services
- **Redis** (localhost:6379) - Caching and session storage
- **Cosmos DB Emulator** (localhost:8081) - NoSQL database

### Backend Microservices
- **API Gateway** (localhost:8000) - Main API entry point
- **Azure Integration** (localhost:8001) - Azure API integration
- **AI Engine** (localhost:8002) - **Patent implementations**
- **Data Processing** (localhost:8003) - Data pipelines
- **Conversation** (localhost:8004) - Conversation management
- **Notification** (localhost:8005) - Alerts and notifications

### Frontend
- **React App** (localhost:5173) - User interface

## 🧪 Testing the Patent Implementations

### Patent 1: Predictive Policy Compliance Engine
```bash
# Test compliance prediction
curl -X POST http://localhost:8002/api/v1/compliance/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "policy_data": {
      "policies": [{"id": "pol_001", "name": "VM Security"}],
      "resources": [{"id": "vm_001", "type": "virtual_machine"}]
    },
    "time_horizon": "7_days"
  }'
```

### Patent 2: Unified AI-Driven Platform
```bash
# Test unified AI analysis
curl -X POST http://localhost:8002/api/v1/unified-ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_002",
    "governance_data": {
      "resource_data": [[[0.5, 0.3, 0.7]]],
      "service_data": [[0.6, 0.4, 0.8]],
      "domain_data": [[[0.7, 0.5, 0.9]]]
    },
    "analysis_scope": ["security", "compliance", "cost"]
  }'

# Test governance optimization
curl -X POST http://localhost:8002/api/v1/unified-ai/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_003",
    "governance_data": {"budget_limit": 10000},
    "preferences": {
      "security_weight": 0.3,
      "compliance_weight": 0.3,
      "cost_weight": 0.2
    }
  }'
```

### Patent 3: Conversational Governance Intelligence
```bash
# Test conversational AI
curl -X POST http://localhost:8002/api/v1/conversation/governance \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What are the current security policies for VMs?",
    "session_id": "test_session_001",
    "user_id": "test_user"
  }'

# Test policy synthesis
curl -X POST http://localhost:8002/api/v1/conversation/policy-synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_004",
    "description": "Create a network security policy",
    "domain": "security",
    "policy_type": "network"
  }'
```

### Patent 4: Cross-Domain Correlation Engine
```bash
# Test correlation analysis
curl -X POST http://localhost:8002/api/v1/correlation/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_005",
    "events": [
      {
        "event_id": "evt_001",
        "domain": "security",
        "timestamp": "2024-01-15T10:30:00Z",
        "event_type": "policy_violation",
        "severity": "high"
      }
    ]
  }'
```

## 🌐 Frontend Testing

### Access the Application
1. Open http://localhost:5173 in your browser
2. Navigate to **AI Assistant** page
3. Test the conversational interface

### Key Features to Test
- **Dashboard**: Overview of governance metrics
- **AI Assistant**: Conversational interface (Patent 3)
- **Policies**: Policy management and analysis
- **Security**: Security assessment and compliance
- **Analytics**: Cross-domain insights (Patent 4)

## 📊 Expected Results

### Successful Test Indicators
- ✅ All services show "healthy" status
- ✅ Patent APIs return structured responses
- ✅ Frontend loads without errors
- ✅ Conversational AI responds intelligently
- ✅ Mock data demonstrates patent capabilities

### Mock vs Production Models
**During local testing, the system uses mock models that:**
- Simulate AI processing with realistic delays
- Return structured, believable responses
- Demonstrate the API contracts and data flows
- Show the patent implementations work end-to-end

**In production, these would be replaced with:**
- Real PyTorch neural networks
- Azure ML model endpoints
- Live training pipelines
- Real-time data processing

## 🐛 Troubleshooting

### Services Won't Start
```bash
# Check logs
docker-compose -f docker-compose.local.yml logs [service-name]

# Restart specific service
docker-compose -f docker-compose.local.yml restart [service-name]

# Full restart
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.local.yml up -d --build
```

### API Errors
```bash
# Check AI Engine logs specifically
docker-compose -f docker-compose.local.yml logs ai-engine

# Test service health
curl http://localhost:8002/health
```

### Frontend Issues
```bash
# Check frontend logs
docker-compose -f docker-compose.local.yml logs frontend

# Access directly
curl http://localhost:5173
```

### Common Issues
1. **Port conflicts**: Ensure ports 8000-8005, 5173, 6379, 8081 are free
2. **Memory issues**: Close unnecessary applications
3. **Docker space**: Run `docker system prune` if low on disk space
4. **Network issues**: Restart Docker Desktop

## 🎯 Success Criteria

### ✅ System is "Functionally Complete" when:
1. **All 6 microservices** start and show healthy status
2. **All 4 patent APIs** respond with structured data
3. **Frontend** loads and connects to backend
4. **Conversational AI** processes queries and generates responses
5. **Cross-domain analysis** shows correlations and insights
6. **Policy synthesis** generates governance policies
7. **Optimization engine** provides recommendations

## 🚀 Next Steps After Successful Testing

1. **Development**: Continue adding features and refining models
2. **Production Setup**: Configure real Azure ML pipelines
3. **Security**: Implement full authentication and encryption
4. **Monitoring**: Add comprehensive observability
5. **CI/CD**: Set up automated deployment pipelines

## 📞 Support

If you encounter issues:
1. Check the logs using the troubleshooting commands above
2. Verify all prerequisites are installed
3. Ensure adequate system resources
4. Review the error messages for specific issues

The local testing setup demonstrates that PolicyCortex is architecturally sound and ready for production enhancement!