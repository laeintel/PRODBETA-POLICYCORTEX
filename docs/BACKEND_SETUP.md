# Backend Setup Instructions

## Current Status
The application is now running with:
- **Frontend**: http://localhost:3000 (Next.js)
- **Mock Backend**: http://localhost:8080 (Express.js)

## Mock Backend Features
The mock backend provides all necessary endpoints for the tactical operations center:
- `/health` - Health check endpoint
- `/api/v1/health` - API health status
- `/api/v1/compliance` - Compliance data
- `/api/v1/security/threats` - Security threats
- `/api/v1/resources` - Resource management
- `/api/v1/cost/analysis` - Cost analytics
- `/api/v1/correlations` - Correlation data
- `/api/v1/predictions` - AI predictions
- `/api/v1/metrics` - Unified metrics
- `/api/v1/recommendations` - AI recommendations
- `/api/v1/actions` - Action orchestration

## Starting the Services

### 1. Start Mock Backend
```bash
node mock-backend.js
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

### 3. Access the Application
Open http://localhost:3000 in your browser

## Authentication Setup

### For Development (without Azure AD)
The application currently bypasses authentication in development mode.

### For Production (with Azure AD)
1. Set environment variables:
```powershell
$env:ALLOW_AZURE_MGMT_SCOPE = "1"
$env:AZURE_TENANT_ID = "9ef5b184-d371-462a-bc75-5024ce8baff7"
$env:AZURE_CLIENT_ID = "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c"
```

2. Configure Azure AD app registration:
   - Set up as Single Page Application (SPA)
   - Add redirect URI: http://localhost:3000
   - Grant necessary API permissions

3. Update `.env.local`:
```env
NEXT_PUBLIC_DEMO_MODE=false
```

4. Sign in via the application and verify token:
```javascript
localStorage.getItem('pcx_token')
```

## Testing the Backend

### Health Check
```bash
curl http://localhost:8080/health
```

### API Endpoints
```bash
curl http://localhost:8080/api/v1/metrics
curl http://localhost:8080/api/v1/compliance
curl http://localhost:8080/api/v1/resources
```

## Production Backend (Rust)

When ready to use the actual Rust backend:

### Build
```bash
cd core
cargo build --release
```

### Run with Environment Variables
```bash
set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
set ALLOW_AZURE_MGMT_SCOPE=1
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379

cd core
cargo run --release
```

### Using Docker Compose
```bash
docker-compose -f docker-compose.local.yml up -d
```

## Troubleshooting

### Backend not accessible
- Check if port 8080 is available: `netstat -an | findstr :8080`
- Kill process using port: `taskkill /F /PID <PID>`

### Frontend not updating
- Restart Next.js server
- Clear Next.js cache: `rm -rf frontend/.next`

### Authentication issues
- Check localStorage for token: `localStorage.getItem('pcx_token')`
- Verify Azure AD configuration in `.env.local`
- Check browser console for MSAL errors

## Features Working
✅ Tactical Operations Center with 200+ clickable elements
✅ All navigation links functional (no 404 errors)
✅ Mock data for all API endpoints
✅ Real-time action monitoring
✅ WebSocket support for live updates
✅ CORS properly configured
✅ Authentication token support