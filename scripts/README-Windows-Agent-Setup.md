# Windows Self-Hosted Agent Setup Guide

Your Windows self-hosted agent (`aeolitech-runner1`) is missing required tools. Here's how to fix it:

## 🚨 Current Issues
- ❌ `git` command not found
- ❌ `az` (Azure CLI) command not found  
- ❌ Likely missing Node.js, Python, Docker

## ✅ Quick Fix

**Run this PowerShell script as Administrator on your agent machine:**

```powershell
# Navigate to the project directory
cd "C:\Users\leonardesere\actions-runner\_work\policycortex\policycortex"

# Run the setup script
powershell -ExecutionPolicy Bypass -File "scripts\setup-windows-agent.ps1"
```

## 🛠️ What the Script Installs

1. **Chocolatey** - Windows package manager
2. **Git** - Version control system
3. **Azure CLI** - For Azure operations
4. **Node.js LTS** - For frontend builds
5. **Python** - For backend services
6. **Docker Desktop** - For container operations
7. **Visual Studio Code** - Useful for debugging
8. **PowerShell Core** - Latest PowerShell version

## 📋 Manual Installation (Alternative)

If the script fails, install manually:

### 1. Install Git
```powershell
# Download and install Git from https://git-scm.com/download/win
# OR use winget:
winget install --id Git.Git -e --source winget
```

### 2. Install Azure CLI  
```powershell
# Download from https://aka.ms/installazurecliwindows
# OR use winget:
winget install -e --id Microsoft.AzureCLI
```

### 3. Install Node.js
```powershell
# Download from https://nodejs.org/
# OR use winget:
winget install OpenJS.NodeJS
```

### 4. Install Python
```powershell
# Download from https://python.org/
# OR use winget:
winget install Python.Python.3.11
```

## 🔄 After Installation

1. **Restart your computer** (important for PATH updates)
2. **Start Docker Desktop** manually if installed
3. **Test the tools** by opening a new PowerShell window and running:
   ```powershell
   git --version
   az --version
   node --version  
   python --version
   docker --version
   ```
4. **Authenticate Azure CLI**:
   ```powershell
   az login
   ```
5. **Restart the GitHub Actions Runner Service**:
   - Open Services (services.msc)
   - Find "GitHub Actions Runner (policycortex.aeolitech-runner1)"
   - Right-click → Restart

## 🎯 Verify Agent is Ready

Run the Quick Agent Test workflow again to verify all tools are working:

1. Go to GitHub Actions → "Quick Agent Test" → "Run workflow"
2. Should now show successful Git, Azure CLI, Node.js, and Python detection

## 🚀 Expected Result

After setup, your CI/CD pipeline should successfully:
- ✅ Detect file changes with Git
- ✅ Authenticate with Azure CLI  
- ✅ Build frontend with Node.js/npm
- ✅ Run backend tests with Python/pip
- ✅ Build and deploy Docker containers

Your Windows self-hosted agent will then be fully operational! 🏗️✨