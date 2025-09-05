# PolicyCortex PCG - Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Prerequisites
- Node.js 18+ installed
- Git installed
- Port 3000, 8080, 8081 available

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/laeintel/policycortex.git
cd policycortex

# Install dependencies
npm install
cd frontend && npm install && cd ..

# Start the platform
node mock-server-pcg.js &
cd frontend && npm run dev

# Open browser
# Navigate to http://localhost:3000
```

---

## 📱 What You'll See

### 1. Dashboard
- Overview of all three pillars
- Key metrics at a glance
- Quick navigation cards

### 2. PREVENT Page (`/prevent`)
- **7-day violation predictions**
- Risk levels (HIGH/MEDIUM/LOW)
- Recommended actions
- Confidence scores

### 3. PROVE Page (`/prove`)
- **Immutable evidence chain**
- Compliance verification status
- Cryptographic hashes
- Audit trail records

### 4. PAYBACK Page (`/payback`)
- **$485,000 total savings**
- 350% ROI achievement
- Monthly trend analysis
- Cost breakdown by category

---

## 🔧 Common Commands

### Development
```bash
# Start mock server (Terminal 1)
node mock-server-pcg.js

# Start frontend (Terminal 2)
cd frontend
npm run dev

# Access at http://localhost:3000
```

### Building
```bash
# Build frontend
cd frontend
npm run build
npm start  # Production server

# Build Rust backend (optional)
cd core
cargo build --release
```

### Testing
```bash
# Test API endpoints
curl http://localhost:8081/api/v1/predictions
curl http://localhost:8081/api/v1/evidence
curl http://localhost:8081/api/v1/roi/metrics
```

---

## 🎯 Key Features to Explore

### PREVENT - Predictive Compliance
1. Click "Prevent" in navigation
2. View upcoming violations (7-day window)
3. Check risk scores and impacts
4. Review recommended actions

### PROVE - Evidence Chain
1. Click "Prove" in navigation
2. Browse compliance evidence
3. Verify cryptographic hashes
4. Track audit trail integrity

### PAYBACK - ROI Dashboard
1. Click "Payback" in navigation
2. Review total savings ($485k)
3. Analyze cost breakdown
4. View monthly trends

---

## 🔍 Understanding the Data

### Prediction Structure
```json
{
  "id": "pred-001",
  "resource_name": "prodstorage",
  "violation_type": "data_encryption",
  "probability": 0.89,
  "severity": "HIGH",
  "estimated_impact": "$45,000",
  "recommended_action": "Enable encryption"
}
```

### Evidence Structure
```json
{
  "id": "ev-001",
  "control": "NIST-800-53-AC-2",
  "status": "compliant",
  "hash": "0x3f4a8b9c2d1e5f6a7b8c9d0e1f2a3b4c5d6e7f8a",
  "verified": true
}
```

### ROI Structure
```json
{
  "total_savings": 485000,
  "roi_percentage": 350,
  "prevented_incidents": {
    "count": 12,
    "value": 285000
  }
}
```

---

## 🛠️ Configuration

### Change API Port
Edit `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8081
```

### Enable Real Azure Data
Edit `frontend/.env.local`:
```env
NEXT_PUBLIC_USE_REAL_DATA=true
NEXT_PUBLIC_DEMO_MODE=false
```

### Change Frontend Port
```bash
cd frontend
npm run dev -- -p 3001
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill existing processes
taskkill /F /IM node.exe

# Or use different ports
node mock-server-pcg.js --port 8082
cd frontend && npm run dev -- -p 3001
```

### API Connection Refused
```bash
# Verify mock server is running
curl http://localhost:8081/health

# Restart if needed
node mock-server-pcg.js
```

### Frontend Build Errors
```bash
# Clear cache and reinstall
cd frontend
rm -rf .next node_modules
npm install
npm run dev
```

---

## 📊 Mock Data Endpoints

### Available Endpoints
- `GET /api/v1/predictions` - 7-day predictions
- `GET /api/v1/evidence` - Audit trail
- `GET /api/v1/roi/metrics` - Financial metrics
- `GET /api/v1/predict/mttp` - Mean time to prevention
- `GET /api/v1/evidence/chain` - Blockchain status
- `POST /api/v1/roi/simulate` - ROI simulation

### Test with cURL
```bash
# Get predictions
curl http://localhost:8081/api/v1/predictions | json_pp

# Get ROI metrics
curl http://localhost:8081/api/v1/roi/metrics | json_pp

# Simulate ROI
curl -X POST http://localhost:8081/api/v1/roi/simulate \
  -H "Content-Type: application/json" \
  -d '{"prevention_rate": 0.4, "automation_level": 0.7}'
```

---

## 🚢 Next Steps

### For Development
1. Explore the three pillar pages
2. Modify mock data in `mock-server-pcg.js`
3. Customize UI in `frontend/app/`
4. Add new features to stores

### For Production
1. Replace mock server with real APIs
2. Configure Azure AD authentication
3. Deploy ML models
4. Set up PostgreSQL database
5. Deploy to Azure/AWS/GCP

---

## 📚 Learn More

- [Accomplishment Summary](./ACCOMPLISHMENT_SUMMARY.md)
- [Technical Implementation](./TECHNICAL_IMPLEMENTATION.md)
- [Main README](../../README.md)
- [CLAUDE.md](../../CLAUDE.md)

---

## 💡 Tips

1. **Demo Mode**: Currently bypasses authentication
2. **Mock Data**: All data is simulated, perfect for demos
3. **Hot Reload**: Frontend updates automatically on save
4. **API Testing**: Use Postman or cURL to test endpoints
5. **State Inspection**: Use React DevTools to inspect Zustand store

---

*Get started in 2 minutes, deliver value in 7 days!*

**"Prevent. Prove. Pays for itself."**