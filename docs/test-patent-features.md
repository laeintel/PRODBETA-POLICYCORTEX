# Testing Patent #1 and #2 Frontend Integration

## Patent #1 - Cross-Domain Correlation Engine

### Access Points:
1. **Navigation Menu**: Look for "AI & Intelligence" section in the sidebar
2. **Direct URL**: http://localhost:3000/correlations
3. **Badge**: Should show "Patent #1" badge with "NEW" indicator

### Features to Test:
- **Correlation Graph Tab**: Visual network graph showing resource correlations
- **Risk Propagation Tab**: Shows domain amplification matrix (50%, 80%, 60% increases)
- **What-If Analysis Tab**: Simulate governance changes with 6 change types
- **ML Insights Tab**: SHAP values, attention weights, and GNN performance metrics

### API Endpoints:
- GET `/api/v1/correlations` - Fetch correlations with filtering
- POST `/api/v1/correlations` with action: 'analyze' - Deep analysis
- POST `/api/v1/correlations` with action: 'what-if' - Simulation
- POST `/api/v1/correlations` with action: 'risk-propagation' - Blast radius

## Patent #2 - Conversational Governance Intelligence

### Access Points:
1. **Navigation Menu**: "Conversational AI" in "AI & Intelligence" section
2. **Direct URL**: http://localhost:3000/chat
3. **Badge**: Should show "Patent #2" badge with "NEW" indicator

### Features to Test:
- **Intent Classification**: 13 governance-specific intents
- **Entity Extraction**: 10 entity types (resources, frameworks, time ranges, etc.)
- **Suggested Actions**: Context-aware action recommendations
- **Policy Generation**: Natural language to JSON policy conversion

### Test Queries:
1. "Check compliance status for all Azure VMs"
2. "Generate a policy to enforce encryption"
3. "What would happen if I remove the admin role?"
4. "Show me high risk resources in the last 7 days"
5. "How much are we spending on compute resources?"

### API Endpoint:
- POST `/api/v1/conversation` - Process natural language queries

## Visual Indicators of Success:
- ✅ New menu items appear with Patent badges
- ✅ Correlation page loads with 4 tabs
- ✅ Mock data displays (3 sample correlations)
- ✅ Chat page processes queries with intent/entity display
- ✅ Risk amplification factors match patent specs (1.5x, 1.8x, 1.6x)

## Browser Testing:
1. Open http://localhost:3000
2. Navigate to AI & Intelligence section
3. Click on "Cross-Domain Correlations" (Patent #1)
4. Test all 4 tabs
5. Click on "Conversational AI" (Patent #2)
6. Try the test queries listed above

## Expected Results:
- Correlation Graph shows interactive network visualization
- Risk Propagation displays amplification matrix
- What-If Simulator allows selecting changes and running simulations
- ML Insights shows SHAP importance and attention weights
- Chat processes queries and shows intent classification