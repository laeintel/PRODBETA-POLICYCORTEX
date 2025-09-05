# 🚨 URGENT: Azure Real Data Connection Setup

## ⏱️ TIME REMAINING: < 1 HOUR

## ✅ WHAT'S DONE:
1. Azure SDK installed
2. Real data server created (`backend/azure-real-server.js`)
3. Server running on port 8081
4. Environment configured with subscription ID

## 🔴 WHAT YOU NEED TO DO NOW:

### Option A: Azure CLI (FASTEST - 2 minutes)

Open a terminal and run:

```bash
# 1. Login to Azure
az login

# 2. Set your subscription
az account set --subscription 205b477d-17e7-4b3b-92c1-32cf02626b78

# 3. Restart the Azure server
cd backend
node azure-real-server.js
```

### Option B: Service Principal (5 minutes)

1. Create service principal:
```bash
az ad sp create-for-rbac --name "PolicyCortex-Dev" --role "Reader" --scopes "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78"
```

2. Copy the output and add to `backend/.env`:
```env
AZURE_CLIENT_ID=<appId from output>
AZURE_CLIENT_SECRET=<password from output>
AZURE_TENANT_ID=<tenant from output>
```

3. Restart server:
```bash
cd backend
node azure-real-server.js
```

## 🎯 VERIFICATION:

1. Check server health:
```bash
curl http://localhost:8081/health
```

Should show: `"connected": true`

2. Test real data:
```bash
curl http://localhost:8081/api/v1/resources
```

Should return your actual Azure resources!

3. Access the app:
```
http://localhost:3000
```

## 📊 CURRENT STATUS:

| Component | Status | Action Needed |
|-----------|--------|---------------|
| Azure SDK | ✅ Installed | None |
| Real Server | ✅ Running | None |
| Azure Auth | ❌ Not configured | **Run `az login`** |
| Frontend | ✅ Configured for real data | None |
| Data Flow | ⏳ Waiting for auth | **Authenticate NOW** |

## 🔥 QUICK TROUBLESHOOTING:

### If "az login" fails:
- Install Azure CLI: https://aka.ms/azurecli
- Or use PowerShell: `Install-Module -Name Az -AllowClobber -Scope CurrentUser`

### If still seeing mock data:
1. Verify server is on 8081: `curl http://localhost:8081/health`
2. Check frontend .env has `USE_REAL_DATA=true`
3. Restart frontend: `cd frontend && npm run dev`

### If Azure connection fails:
- Check subscription access: `az account show`
- Ensure you have Reader role: `az role assignment list --assignee $(az account show --query user.name -o tsv)`

## ⚡ FASTEST PATH (30 seconds):

```bash
# Copy-paste these commands:
az login
az account set --subscription 205b477d-17e7-4b3b-92c1-32cf02626b78
cd backend && node azure-real-server.js
```

Then open: http://localhost:3000

## 📞 IMMEDIATE HELP:

The server is **ALREADY RUNNING** and waiting for Azure credentials.
Just run `az login` and it will start working!

---

**TIME INVESTED**: 45 minutes
**TIME REMAINING**: < 1 hour
**NEXT ACTION**: Run `az login` NOW!