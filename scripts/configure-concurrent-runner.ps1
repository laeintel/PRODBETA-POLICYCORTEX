# Configure GitHub Self-Hosted Runner for Concurrent Jobs
# Run this script with administrator privileges

param(
    [Parameter(Mandatory=$true)]
    [string]$RunnerDirectory,
    
    [Parameter(Mandatory=$false)]
    [int]$MaxConcurrentJobs = 3,
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "actions.runner.aeolitech-policycortex.aeolitech-runner1"
)

Write-Host "Configuring GitHub Runner for Concurrent Jobs..." -ForegroundColor Green

# Clean up the runner directory path (remove quotes if present)
$RunnerDirectory = $RunnerDirectory.Trim('"').Trim("'")

# Verify runner directory exists
if (-not (Test-Path $RunnerDirectory)) {
    Write-Host "Error: Runner directory not found at: $RunnerDirectory" -ForegroundColor Red
    Write-Host "Please provide a valid path to your GitHub Actions runner directory" -ForegroundColor Yellow
    exit 1
}

# Verify svc.cmd exists in the runner directory
$svcPath = Join-Path $RunnerDirectory "svc.cmd"
if (-not (Test-Path $svcPath)) {
    Write-Host "Error: svc.cmd not found in: $RunnerDirectory" -ForegroundColor Red
    Write-Host "This doesn't appear to be a valid GitHub Actions runner directory" -ForegroundColor Yellow
    exit 1
}

# Stop the existing runner service
Write-Host "Stopping runner service..." -ForegroundColor Yellow
Stop-Service -Name $ServiceName -ErrorAction SilentlyContinue

# Navigate to runner directory
Write-Host "Navigating to: $RunnerDirectory" -ForegroundColor Cyan
Set-Location $RunnerDirectory

# Remove existing service
Write-Host "Removing existing service..." -ForegroundColor Yellow
.\svc.cmd stop
.\svc.cmd uninstall

# Configure runner for concurrent jobs
Write-Host "Configuring runner for $MaxConcurrentJobs concurrent jobs..." -ForegroundColor Yellow

# Update runner configuration
$configPath = Join-Path $RunnerDirectory ".runner"
if (Test-Path $configPath) {
    $config = Get-Content $configPath | ConvertFrom-Json
    $config | Add-Member -Name "concurrency" -Value $MaxConcurrentJobs -MemberType NoteProperty -Force
    $config | ConvertTo-Json -Depth 10 | Set-Content $configPath
}

# Install service with concurrency support
Write-Host "Installing runner service with concurrency support..." -ForegroundColor Yellow
.\svc.cmd install --concurrency $MaxConcurrentJobs

# Start the service
Write-Host "Starting runner service..." -ForegroundColor Yellow
.\svc.cmd start

# Verify service status
Start-Sleep -Seconds 5
$serviceStatus = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue

if ($serviceStatus -and $serviceStatus.Status -eq 'Running') {
    Write-Host "✅ Runner service configured successfully for $MaxConcurrentJobs concurrent jobs" -ForegroundColor Green
    Write-Host "Service Status: $($serviceStatus.Status)" -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to start runner service" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔧 Runner Configuration:" -ForegroundColor Cyan
Write-Host "- Max Concurrent Jobs: $MaxConcurrentJobs" -ForegroundColor White
Write-Host "- Service Name: $ServiceName" -ForegroundColor White
Write-Host "- Runner Directory: $RunnerDirectory" -ForegroundColor White

Write-Host "`n📋 To verify concurrent execution:" -ForegroundColor Yellow
Write-Host "1. Trigger multiple pipeline runs simultaneously" -ForegroundColor White
Write-Host "2. Check GitHub Actions tab to see multiple jobs running" -ForegroundColor White
Write-Host "3. Monitor Windows Task Manager for multiple runner processes" -ForegroundColor White