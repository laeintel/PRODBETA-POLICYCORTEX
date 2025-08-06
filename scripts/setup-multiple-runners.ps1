# Setup Multiple GitHub Self-Hosted Runners
# Run this script with administrator privileges

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubToken,
    
    [Parameter(Mandatory=$true)]
    [string]$Repository = "AeoliTech/policycortex",
    
    [Parameter(Mandatory=$false)]
    [int]$RunnerCount = 3,
    
    [Parameter(Mandatory=$false)]
    [string]$BaseDirectory = "C:\actions-runners"
)

Write-Host "Setting up $RunnerCount GitHub Self-Hosted Runners..." -ForegroundColor Green

# Create base directory
if (-not (Test-Path $BaseDirectory)) {
    New-Item -ItemType Directory -Path $BaseDirectory -Force | Out-Null
}

# Download GitHub Actions Runner if not exists
$runnerVersion = "2.311.0"  # Use latest stable version
$runnerArchive = "actions-runner-win-x64-$runnerVersion.zip"
$runnerUrl = "https://github.com/actions/runner/releases/download/v$runnerVersion/$runnerArchive"
$archivePath = Join-Path $BaseDirectory $runnerArchive

if (-not (Test-Path $archivePath)) {
    Write-Host "Downloading GitHub Actions Runner..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $runnerUrl -OutFile $archivePath
}

# Setup each runner instance
for ($i = 1; $i -le $RunnerCount; $i++) {
    Write-Host "`n🚀 Setting up Runner $i..." -ForegroundColor Cyan
    
    $runnerDir = Join-Path $BaseDirectory "runner$i"
    $runnerName = "aeolitech-runner$i"
    $runnerLabels = "self-hosted,Windows,X64,aeolitech-runner$i"
    
    # Create runner directory
    if (Test-Path $runnerDir) {
        Remove-Item $runnerDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $runnerDir -Force | Out-Null
    
    # Extract runner
    Write-Host "Extracting runner to $runnerDir..." -ForegroundColor Yellow
    Expand-Archive -Path $archivePath -DestinationPath $runnerDir -Force
    
    # Configure runner
    Set-Location $runnerDir
    
    Write-Host "Configuring runner '$runnerName'..." -ForegroundColor Yellow
    
    # Generate registration token (you'll need to implement token generation)
    $registrationToken = $GitHubToken  # In practice, you'd call GitHub API to get registration token
    
    # Configure the runner
    $configArgs = @(
        "--url", "https://github.com/$Repository",
        "--token", $registrationToken,
        "--name", $runnerName,
        "--labels", $runnerLabels,
        "--work", "_work",
        "--unattended"
    )
    
    Start-Process -FilePath ".\config.cmd" -ArgumentList $configArgs -Wait -NoNewWindow
    
    # Install as service
    Write-Host "Installing runner '$runnerName' as service..." -ForegroundColor Yellow
    .\svc.cmd install "actions.runner.$($Repository -replace '/','.').$runnerName"
    .\svc.cmd start
    
    Write-Host "✅ Runner '$runnerName' configured and started" -ForegroundColor Green
}

Write-Host "`n🎉 All $RunnerCount runners configured successfully!" -ForegroundColor Green
Write-Host "`n📋 Runner Summary:" -ForegroundColor Cyan

for ($i = 1; $i -le $RunnerCount; $i++) {
    $serviceName = "actions.runner.$($Repository -replace '/','.')." + "aeolitech-runner$i"
    $serviceStatus = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    
    Write-Host "- Runner $i (aeolitech-runner$i): $($serviceStatus.Status)" -ForegroundColor White
}

Write-Host "`n⚠️  Important Notes:" -ForegroundColor Yellow
Write-Host "1. Each runner can process jobs independently" -ForegroundColor White
Write-Host "2. Use specific runner labels in workflow files to target runners" -ForegroundColor White
Write-Host "3. Monitor system resources to ensure adequate performance" -ForegroundColor White
Write-Host "4. Consider runner maintenance and updates" -ForegroundColor White