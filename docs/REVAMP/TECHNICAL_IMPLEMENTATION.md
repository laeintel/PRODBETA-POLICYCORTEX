# PolicyCortex PCG - Technical Implementation Guide

## Overview
This document details the technical implementation of the PolicyCortex Predictive Cloud Governance (PCG) platform revamp, including architecture decisions, code structure, and implementation details.

---

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│  │ PREVENT │    │  PROVE  │    │ PAYBACK │            │
│  └────┬────┘    └────┬────┘    └────┬────┘            │
│       └──────────────┼──────────────┘                  │
│                      ▼                                  │
│              ┌──────────────┐                          │
│              │  API Client  │                          │
│              └──────┬───────┘                          │
└──────────────────────┼──────────────────────────────────┘
                       ▼
        ┌──────────────────────────┐
        │   Mock Server (8081)     │
        │  ┌──────────────────┐   │
        │  │ /api/v1/predictions│  │
        │  │ /api/v1/evidence   │  │
        │  │ /api/v1/roi/metrics│  │
        │  └──────────────────┘   │
        └──────────────────────────┘
```

---

## 📁 Project Structure

### Root Directory
```
policycortex/
├── frontend/               # Next.js application
├── backend/               # Python ML services
├── core/                  # Rust CQRS backend
├── docs/REVAMP/          # Revamp documentation
├── mock-server.js        # General mock API (port 8080)
├── mock-server-pcg.js    # PCG-specific API (port 8081)
├── package.json          # Root dependencies
├── README.md             # Main documentation
└── CLAUDE.md             # AI assistant instructions
```

### Frontend Structure
```
frontend/
├── app/                   # Next.js App Router
│   ├── layout.tsx        # Root layout with providers
│   ├── page.tsx          # Dashboard/home page
│   ├── prevent/          # PREVENT pillar
│   │   └── page.tsx     # Predictions interface
│   ├── prove/            # PROVE pillar
│   │   └── page.tsx     # Evidence chain interface
│   └── payback/          # PAYBACK pillar
│       └── page.tsx     # ROI metrics interface
├── components/           # Reusable components
│   ├── Navigation.tsx   # Main navigation
│   └── ui/              # UI components
├── lib/                  # Utilities
│   └── api-client.ts    # API communication layer
├── stores/              # State management
│   └── resourceStore.ts # PCG Zustand store
└── types/               # TypeScript definitions
    └── api.ts          # API types
```

---

## 💻 Implementation Details

### 1. Frontend Implementation

#### State Management (Zustand)
```typescript
// stores/resourceStore.ts
interface PCGStore {
  predictions: PredictionData[];
  evidence: EvidenceItem[];
  roiMetrics: ROIMetrics | null;
  isLoading: boolean;
  error: string | null;
  
  fetchPredictions: () => Promise<void>;
  fetchEvidence: () => Promise<void>;
  fetchROIMetrics: () => Promise<void>;
}
```

#### API Client Pattern
```typescript
// lib/api-client.ts
class PCGApiClient {
  async getPredictions(): Promise<PredictionData[]> {
    const response = await fetch('/api/v1/predictions');
    const data = await response.json();
    // Transform PCG format to frontend format
    return this.transformPredictions(data);
  }
}
```

#### Component Structure
```typescript
// app/prevent/page.tsx
export default function PreventPage() {
  const { predictions, fetchPredictions } = usePCGStore();
  
  useEffect(() => {
    fetchPredictions();
  }, []);
  
  return <PredictionsList predictions={predictions} />;
}
```

### 2. Mock Server Implementation

#### PCG Mock Server Structure
```javascript
// mock-server-pcg.js
const endpoints = {
  '/api/v1/predictions': getPredictions,
  '/api/v1/evidence': getEvidence,
  '/api/v1/roi/metrics': getROIMetrics,
  '/api/v1/predict/mttp': getMTTP,
  '/api/v1/evidence/chain': getEvidenceChain,
  '/api/v1/roi/simulate': simulateROI
};
```

#### Data Generation Pattern
```javascript
function getPredictions() {
  return {
    predictions: [
      {
        id: 'pred-001',
        resource_name: 'prodstorage',
        violation_type: 'data_encryption',
        prediction_date: getFutureDate(7),
        probability: 0.89,
        severity: 'HIGH',
        estimated_impact: '$45,000',
        recommended_action: 'Enable encryption'
      }
    ],
    summary: {
      total: 3,
      critical: 1,
      mttp_hours: 18
    }
  };
}
```

### 3. API Response Transformation

#### Prediction Transformation
```typescript
function transformPredictions(pcgData: any): PredictionData[] {
  return pcgData.predictions.map(p => ({
    id: p.id,
    type: mapViolationType(p.violation_type),
    score: Math.round(p.probability * 100),
    confidence: Math.round(p.probability * 100),
    prediction: `${p.resource_name}: ${p.recommended_action}`,
    impact: p.severity.toLowerCase(),
    timestamp: p.prediction_date
  }));
}
```

---

## 🔧 Configuration

### Environment Variables
```env
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8081
NEXT_PUBLIC_DEMO_MODE=true
NEXT_PUBLIC_AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb
NEXT_PUBLIC_AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f
```

### Port Configuration
- **Frontend**: 3000 (or 3001 if busy)
- **Mock Server (General)**: 8080
- **Mock Server (PCG)**: 8081
- **Rust Backend**: 8000 (when running)
- **Python ML Service**: 8082 (when running)

---

## 🚀 Development Workflow

### Starting Development Environment
```bash
# Terminal 1: Start PCG mock server
node mock-server-pcg.js

# Terminal 2: Start frontend
cd frontend && npm run dev

# Access at http://localhost:3000
```

### Building for Production
```bash
# Frontend build
cd frontend
npm run build
npm start

# Rust backend build
cd core
cargo build --release

# Python services
cd backend/services/api_gateway
uvicorn main:app --host 0.0.0.0 --port 8082
```

---

## 🔄 Data Flow

### Prediction Flow (PREVENT)
1. User navigates to `/prevent`
2. Component calls `fetchPredictions()`
3. Store calls `apiClient.getPredictions()`
4. API Client fetches from `http://localhost:8081/api/v1/predictions`
5. Mock server returns PCG-formatted data
6. API Client transforms to frontend format
7. Store updates state
8. Component re-renders with predictions

### Evidence Flow (PROVE)
1. User navigates to `/prove`
2. Component calls `fetchEvidence()`
3. Store calls `apiClient.getEvidence()`
4. API Client fetches from `http://localhost:8081/api/v1/evidence`
5. Mock server returns evidence with hashes
6. API Client transforms to frontend format
7. Store updates state
8. Component displays evidence chain

### ROI Flow (PAYBACK)
1. User navigates to `/payback`
2. Component calls `fetchROIMetrics()`
3. Store calls `apiClient.getROIMetrics()`
4. API Client fetches from `http://localhost:8081/api/v1/roi/metrics`
5. Mock server returns financial metrics
6. API Client transforms to frontend format
7. Store updates state
8. Component displays ROI dashboard

---

## 🔌 Integration Points

### Frontend → Backend
- **Authentication**: MSAL integration ready (currently bypassed)
- **API Calls**: Unified through api-client.ts
- **State Management**: Centralized in Zustand store
- **Error Handling**: Consistent error state management

### Backend Services
- **Rust Core**: CQRS pattern with command/query separation
- **Python ML**: FastAPI endpoints for model serving
- **Mock Servers**: Drop-in replaceable with real services

---

## 📊 Performance Optimizations

### Frontend Optimizations
- **Code Splitting**: Automatic with Next.js App Router
- **Image Optimization**: Next/Image component used
- **Bundle Size**: Minimal dependencies
- **Caching**: API responses cached in store

### API Optimizations
- **Response Caching**: Hot/warm/cold data strategy
- **Batch Requests**: Promise.all for parallel fetches
- **Compression**: Gzip enabled on responses
- **Connection Pooling**: Reuse HTTP connections

---

## 🔒 Security Considerations

### Authentication
- Azure AD integration prepared
- Demo mode for development
- Role-based access control (RBAC) ready

### Data Protection
- HTTPS in production
- Environment variables for secrets
- No hardcoded credentials
- CSP headers configured

---

## 🐛 Debugging

### Common Issues and Solutions

#### API Connection Refused
```bash
# Check if mock server is running
curl http://localhost:8081/health

# Restart mock server
node mock-server-pcg.js
```

#### Frontend Build Errors
```bash
# Clear cache and rebuild
rm -rf .next
npm run build
```

#### State Not Updating
```javascript
// Check store subscription
const { predictions } = usePCGStore();
console.log('Predictions:', predictions);
```

---

## 📈 Monitoring

### Key Metrics to Track
- **API Response Time**: Target <500ms
- **Page Load Time**: Target <2s
- **Error Rate**: Target <1%
- **User Engagement**: Time on pillar pages

### Logging Strategy
```javascript
// API Client logging
console.log('[API] Fetching predictions...');
console.time('predictions-fetch');
const data = await fetch(url);
console.timeEnd('predictions-fetch');
```

---

## 🔮 Future Enhancements

### Planned Improvements
1. **Real Azure Integration**: Replace mock with Azure APIs
2. **ML Model Deployment**: Deploy trained models
3. **WebSocket Support**: Real-time predictions
4. **Advanced Analytics**: Detailed drill-downs
5. **Mobile Optimization**: Responsive improvements

### Scalability Considerations
- **Microservices**: Each pillar as separate service
- **Event-Driven**: CQRS with event sourcing
- **Caching Layer**: Redis for hot data
- **CDN**: Static assets distribution

---

## 📝 Development Guidelines

### Code Style
- **TypeScript**: Strict mode enabled
- **React**: Functional components with hooks
- **Formatting**: Prettier configuration
- **Linting**: ESLint rules enforced

### Git Workflow
```bash
# Feature development
git checkout -b feature/new-feature
# Make changes
git add .
git commit -m "feat: Add new feature"
git push origin feature/new-feature
# Create PR to main
```

### Testing Strategy
- **Unit Tests**: Jest for components
- **Integration Tests**: API endpoint testing
- **E2E Tests**: Playwright for user flows
- **Performance Tests**: Lighthouse CI

---

*Last Updated: December 5, 2024*
*Version: 1.0.0*