# Windows Self-Hosted Agent Setup Script
# Run this on your Windows agent machine to install required tools

param(
    [switch]$SkipRestart,
    [switch]$InstallAll = $true
)

Write-Host "🔧 Setting up Windows Self-Hosted GitHub Agent..." -ForegroundColor Green
Write-Host "This script will install: Git, Azure CLI, Node.js, Python, Docker Desktop" -ForegroundColor Yellow

# Set execution policy for this session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Install Chocolatey if not present
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing Chocolatey package manager..." -ForegroundColor Cyan
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    refreshenv
}

# Function to install package with Chocolatey
function Install-ChocoPackage {
    param($PackageName, $DisplayName)
    
    Write-Host "📦 Installing $DisplayName..." -ForegroundColor Cyan
    try {
        choco install $PackageName -y --no-progress
        Write-Host "✅ $DisplayName installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install $DisplayName : $_" -ForegroundColor Red
    }
}

if ($InstallAll) {
    # Install Git
    Write-Host "`n🔨 Installing Development Tools..." -ForegroundColor Yellow
    Install-ChocoPackage "git" "Git"
    
    # Install Azure CLI
    Install-ChocoPackage "azure-cli" "Azure CLI"
    
    # Install Node.js (LTS)
    Install-ChocoPackage "nodejs-lts" "Node.js LTS"
    
    # Install Python
    Install-ChocoPackage "python" "Python"
    
    # Install Docker Desktop
    Write-Host "📦 Installing Docker Desktop..." -ForegroundColor Cyan
    Install-ChocoPackage "docker-desktop" "Docker Desktop"
    
    # Install additional useful tools
    Write-Host "`n🛠️ Installing Additional Tools..." -ForegroundColor Yellow
    Install-ChocoPackage "vscode" "Visual Studio Code"
    Install-ChocoPackage "powershell-core" "PowerShell Core"
    Install-ChocoPackage "7zip" "7-Zip"
}

# Refresh environment variables
Write-Host "`n🔄 Refreshing environment variables..." -ForegroundColor Cyan
refreshenv

# Update PATH for current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Verify installations
Write-Host "`n✅ Verifying installations..." -ForegroundColor Green

# Check Git
try {
    $gitVersion = git --version 2>$null
    Write-Host "✅ Git: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git: Not found or not in PATH" -ForegroundColor Red
}

# Check Azure CLI
try {
    $azVersion = az version --output table 2>$null | Select-String "azure-cli"
    Write-Host "✅ Azure CLI: $azVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI: Not found or not in PATH" -ForegroundColor Red
}

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js: Not found or not in PATH" -ForegroundColor Red
}

# Check Python
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python: Not found or not in PATH" -ForegroundColor Red
}

# Check Docker
try {
    $dockerVersion = docker --version 2>$null
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Docker: Not found (may need manual start or restart)" -ForegroundColor Yellow
}

Write-Host "`n🎉 Setup completed!" -ForegroundColor Green
Write-Host "📝 Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your computer if Docker was installed" -ForegroundColor White
Write-Host "2. Start Docker Desktop manually if needed" -ForegroundColor White
Write-Host "3. Run 'az login' to authenticate Azure CLI" -ForegroundColor White
Write-Host "4. Restart the GitHub Actions runner service" -ForegroundColor White

if (!$SkipRestart) {
    Write-Host "`n⚠️  A restart is recommended for all PATH changes to take effect." -ForegroundColor Yellow
    $restart = Read-Host "Do you want to restart now? (y/N)"
    if ($restart -eq 'y' -or $restart -eq 'Y') {
        Write-Host "🔄 Restarting computer..." -ForegroundColor Cyan
        Restart-Computer -Force
    }
}

Write-Host "`n🚀 Your Windows GitHub Actions agent should now have all required tools!" -ForegroundColor Green