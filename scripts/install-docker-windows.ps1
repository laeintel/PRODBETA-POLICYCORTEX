# PowerShell script to install Docker on Windows
# Run as Administrator

Write-Host "Docker Installation for Windows" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

# Check if Docker is already installed
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "✅ Docker is already installed:" -ForegroundColor Green
    docker --version
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        Write-Host "✅ Docker daemon is running" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️ Docker is installed but not running" -ForegroundColor Yellow
        Write-Host "Please start Docker Desktop" -ForegroundColor Yellow
    }
    exit 0
}

Write-Host "📦 Installing Docker..." -ForegroundColor Yellow

# Check Windows version
$os = Get-WmiObject -Class Win32_OperatingSystem
$version = [System.Version]$os.Version

if ($version.Major -lt 10) {
    Write-Host "❌ Docker Desktop requires Windows 10 or higher" -ForegroundColor Red
    exit 1
}

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Enable required Windows features
Write-Host "Enabling required Windows features..." -ForegroundColor Yellow

# Enable WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Download and install Docker Desktop
Write-Host "Downloading Docker Desktop..." -ForegroundColor Yellow
$dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
$installerPath = "$env:TEMP\DockerDesktopInstaller.exe"

try {
    Invoke-WebRequest -Uri $dockerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "✅ Download complete" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to download Docker Desktop" -ForegroundColor Red
    Write-Host "Please download manually from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Install Docker Desktop
Write-Host "Installing Docker Desktop..." -ForegroundColor Yellow
Start-Process -FilePath $installerPath -ArgumentList "install", "--quiet" -Wait

# Clean up installer
Remove-Item $installerPath -Force

Write-Host "" 
Write-Host "✅ Docker Desktop installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️ IMPORTANT:" -ForegroundColor Yellow
Write-Host "1. Restart your computer to complete the installation" -ForegroundColor Yellow
Write-Host "2. After restart, start Docker Desktop from the Start Menu" -ForegroundColor Yellow
Write-Host "3. Complete the Docker Desktop setup wizard" -ForegroundColor Yellow
Write-Host ""
Write-Host "To verify after restart: docker run hello-world" -ForegroundColor Cyan