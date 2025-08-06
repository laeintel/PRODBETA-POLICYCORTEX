# Simple script to enable concurrent jobs on GitHub self-hosted runner
# Run as Administrator

Write-Host "GitHub Self-Hosted Runner Concurrent Jobs Enabler" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Common runner locations - try to auto-detect
$possiblePaths = @(
    "C:\actions-runner",
    "C:\Users\$env:USERNAME\actions-runner",
    "C:\Users\leonardesere\actions-runner",
    "D:\actions-runner",
    "$env:USERPROFILE\actions-runner"
)

$runnerPath = $null
foreach ($path in $possiblePaths) {
    if (Test-Path "$path\svc.cmd") {
        $runnerPath = $path
        Write-Host "Found runner at: $runnerPath" -ForegroundColor Cyan
        break
    }
}

if (-not $runnerPath) {
    Write-Host "Could not auto-detect runner location." -ForegroundColor Yellow
    $runnerPath = Read-Host "Enter the full path to your runner directory (without quotes)"
    $runnerPath = $runnerPath.Trim('"').Trim("'")
    
    if (-not (Test-Path "$runnerPath\svc.cmd")) {
        Write-Host "Error: Invalid runner directory. svc.cmd not found." -ForegroundColor Red
        exit 1
    }
}

# Ask for number of concurrent jobs
Write-Host ""
Write-Host "How many concurrent jobs should the runner handle?" -ForegroundColor Yellow
Write-Host "  Recommended based on your system: 3-4 jobs" -ForegroundColor White
Write-Host "  (Press Enter for default of 3)" -ForegroundColor Gray
$jobCount = Read-Host "Number of concurrent jobs"
if ([string]::IsNullOrWhiteSpace($jobCount)) {
    $jobCount = 3
}

# Find the service name
Write-Host ""
Write-Host "Looking for runner service..." -ForegroundColor Yellow
$runnerServices = Get-Service -Name "actions.runner.*" -ErrorAction SilentlyContinue

if ($runnerServices) {
    $serviceName = $runnerServices[0].Name
    Write-Host "Found service: $serviceName" -ForegroundColor Green
} else {
    Write-Host "Warning: Could not find runner service automatically" -ForegroundColor Yellow
    $serviceName = "actions.runner.aeolitech-policycortex.aeolitech-runner1"
}

# Confirmation
Write-Host ""
Write-Host "Configuration Summary:" -ForegroundColor Cyan
Write-Host "  Runner Path: $runnerPath" -ForegroundColor White
Write-Host "  Service Name: $serviceName" -ForegroundColor White
Write-Host "  Concurrent Jobs: $jobCount" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "Continue with configuration? (Y/N)"
if ($confirm -ne 'Y' -and $confirm -ne 'y') {
    Write-Host "Configuration cancelled." -ForegroundColor Yellow
    exit 0
}

# Execute configuration
try {
    Write-Host ""
    Write-Host "Configuring runner for concurrent jobs..." -ForegroundColor Green
    
    # Stop service
    Write-Host "Stopping service..." -ForegroundColor Yellow
    Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    
    # Navigate to runner directory
    Set-Location $runnerPath
    
    # Reconfigure service
    Write-Host "Removing old service configuration..." -ForegroundColor Yellow
    & cmd /c "svc.cmd stop" 2>&1 | Out-Null
    & cmd /c "svc.cmd uninstall" 2>&1 | Out-Null
    
    Write-Host "Installing service with concurrent job support..." -ForegroundColor Yellow
    
    # Create config file for concurrent jobs
    $configContent = @"
{
  "maxParallel": $jobCount
}
"@
    $configContent | Out-File -FilePath ".service" -Encoding UTF8
    
    # Install service
    & cmd /c "svc.cmd install" 2>&1 | Out-Null
    
    # Start service
    Write-Host "Starting service..." -ForegroundColor Yellow
    & cmd /c "svc.cmd start" 2>&1 | Out-Null
    Start-Sleep -Seconds 3
    
    # Verify
    $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($service -and $service.Status -eq 'Running') {
        Write-Host ""
        Write-Host "SUCCESS! Runner configured for $jobCount concurrent jobs" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Cyan
        Write-Host "1. Trigger multiple workflow runs in your GitHub repository" -ForegroundColor White
        Write-Host "2. Check the Actions tab to see concurrent job execution" -ForegroundColor White
        Write-Host "3. Monitor system performance in Task Manager" -ForegroundColor White
    } else {
        Write-Host "Warning: Service may not have started properly" -ForegroundColor Yellow
        Write-Host "Try starting it manually with: svc.cmd start" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "Error during configuration: $_" -ForegroundColor Red
    Write-Host "You may need to configure manually" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Configuration complete!" -ForegroundColor Green