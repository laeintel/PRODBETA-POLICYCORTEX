# 🚀 PolicyCortex Local Access Guide

## 🌐 Access Your Platform

### Main Dashboard
**URL:** http://localhost:3000
- Modern UI with real-time updates
- AI-powered insights dashboard
- Voice interface enabled

### Available Pages
- **Dashboard:** http://localhost:3000/dashboard
- **AI Expert:** http://localhost:3000/ai-expert
- **Policies:** http://localhost:3000/policies
- **Resources:** http://localhost:3000/resources
- **Chat Interface:** http://localhost:3000/chat
- **Settings:** http://localhost:3000/settings

## 📊 Live Data Examples

### Current Metrics (Real-time)
```json
{
  "policies": {
    "total": 347,
    "active": 298,
    "compliance_rate": 99.8%
  },
  "costs": {
    "current_spend": $145,832,
    "savings_identified": $47,642
  },
  "security": {
    "active_threats": 2,
    "blocked_attempts": 127
  }
}
```

## 🤖 AI Features to Try

### 1. Conversational AI
Visit http://localhost:3000/chat and try:
- "How can I reduce my Azure costs?"
- "Show me RBAC violations"
- "Create a SOC 2 compliance checklist"
- "Analyze my security posture"

### 2. Voice Commands
Click the microphone icon and say:
- "Assess security posture"
- "Show cost optimization"
- "Check compliance status"

### 3. AI Predictions
Visit http://localhost:3000/ai-expert to see:
- Policy drift predictions
- Cost forecasting
- Security threat analysis
- Compliance risk scoring

## 🔧 API Endpoints to Test

### Core APIs (Direct Access)
- **Health Check:** http://localhost:8080/health
- **Metrics:** http://localhost:8080/api/v1/metrics
- **Predictions:** http://localhost:8080/api/v1/predictions
- **Correlations:** http://localhost:8080/api/v1/correlations
- **Recommendations:** http://localhost:8080/api/v1/recommendations

### GraphQL Playground
- **URL:** http://localhost:4000/graphql
- Try this query:
```graphql
query {
  governanceMetrics {
    policies {
      total
      violations
      complianceRate
    }
    costs {
      currentSpend
      savingsIdentified
    }
  }
}
```

## 🎯 Quick Test Commands

### Test Patent Features
```bash
# Patent 1: Unified Platform
curl http://localhost:8080/api/v1/metrics

# Patent 2: Predictive Compliance
curl http://localhost:8080/api/v1/predictions

# Patent 3: Conversational AI
curl -X POST http://localhost:8080/api/v1/conversation \
  -H "Content-Type: application/json" \
  -d '{"query": "How to improve compliance?", "session_id": "test"}'

# Patent 4: Cross-Domain Correlation
curl http://localhost:8080/api/v1/correlations
```

## 🎨 What You'll See

### Dashboard Features:
1. **Real-time Metrics Cards** - Live governance data
2. **AI Insights Panel** - Predictive analytics
3. **Cost Optimization Graph** - Savings opportunities
4. **Security Heat Map** - Threat visualization
5. **Compliance Score** - Real-time compliance status
6. **Voice Assistant** - Floating AI interface

### AI Expert Page:
- Domain-specific recommendations
- Automated policy generation
- Compliance automation
- Multi-cloud insights

## 🔍 Troubleshooting

If pages don't load:
1. Check services are running:
   ```bash
   curl http://localhost:8080/health
   ```

2. Restart frontend if needed:
   ```bash
   cd frontend && npm run dev
   ```

3. Check browser console for errors (F12)

## 📱 Mobile View
The platform is responsive! Try:
- Resize your browser window
- Open DevTools (F12) > Toggle device toolbar
- Test on your phone: http://[your-ip]:3000

## 💡 Pro Tips
1. Use Chrome/Edge for best voice recognition
2. Allow microphone permissions for voice features
3. The AI learns from your interactions
4. All data is processed locally for privacy