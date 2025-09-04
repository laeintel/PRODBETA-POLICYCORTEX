# PowerShell script to track DORA deployment metrics
param(
    [Parameter(Mandatory=$true)]
    [string]$Environment,
    
    [Parameter(Mandatory=$true)]
    [string]$Version,
    
    [Parameter(Mandatory=$false)]
    [string]$CommitHash = (git rev-parse HEAD),
    
    [Parameter(Mandatory=$false)]
    [string]$Status = "success",
    
    [Parameter(Mandatory=$false)]
    [string]$PrometheusGateway = "http://localhost:9091"
)

Write-Host "Tracking deployment metrics for PolicyCortex" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Green
Write-Host "Version: $Version" -ForegroundColor Green
Write-Host "Commit: $CommitHash" -ForegroundColor Green
Write-Host "Status: $Status" -ForegroundColor Green

# Calculate lead time (time from commit to deployment)
$CommitTime = git show -s --format=%ct $CommitHash
$CurrentTime = [int](Get-Date -UFormat %s)
$LeadTimeSeconds = $CurrentTime - $CommitTime

# Prepare metrics for Prometheus pushgateway
$Metrics = @"
# TYPE deployments_total counter
# HELP deployments_total Total number of deployments
deployments_total{environment="$Environment",version="$Version",status="$Status"} 1

# TYPE lead_time_seconds gauge
# HELP lead_time_seconds Lead time from commit to deployment in seconds
lead_time_seconds{environment="$Environment",version="$Version"} $LeadTimeSeconds

# TYPE deployment_info info
# HELP deployment_info Information about the deployment
deployment_info{environment="$Environment",version="$Version",commit="$CommitHash",timestamp="$CurrentTime"} 1
"@

if ($Status -eq "failed") {
    $Metrics += @"

# TYPE deployments_failed_total counter
# HELP deployments_failed_total Total number of failed deployments
deployments_failed_total{environment="$Environment",version="$Version"} 1
"@
}

# Push metrics to Prometheus Pushgateway
try {
    $Headers = @{
        "Content-Type" = "text/plain; version=0.0.4"
    }
    
    $Response = Invoke-RestMethod -Uri "$PrometheusGateway/metrics/job/deployments/instance/$Environment" -Method POST -Body $Metrics -Headers $Headers
    Write-Host "Metrics pushed successfully to Prometheus" -ForegroundColor Green
} catch {
    Write-Host "Failed to push metrics to Prometheus: $_" -ForegroundColor Red
}

# Store deployment event in JSON for backup
$DeploymentEvent = @{
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    environment = $Environment
    version = $Version
    commit = $CommitHash
    status = $Status
    lead_time_seconds = $LeadTimeSeconds
    deployed_by = $env:USERNAME
    pipeline_id = $env:GITHUB_RUN_ID
}

$EventsFile = "deployment-events.json"
$Events = @()

if (Test-Path $EventsFile) {
    $Events = Get-Content $EventsFile | ConvertFrom-Json
}

$Events += $DeploymentEvent
$Events | ConvertTo-Json -Depth 10 | Set-Content $EventsFile

Write-Host "Deployment event logged to $EventsFile" -ForegroundColor Green

# Calculate and display DORA metrics
Write-Host "`nDORA Metrics Summary:" -ForegroundColor Cyan
Write-Host "- Lead Time: $([math]::Round($LeadTimeSeconds / 3600, 2)) hours" -ForegroundColor Yellow

# Calculate deployment frequency (last 7 days)
$RecentDeployments = $Events | Where-Object { 
    [DateTime]$_.timestamp -gt (Get-Date).AddDays(-7) 
} | Measure-Object
Write-Host "- Deployment Frequency: $($RecentDeployments.Count) deployments in last 7 days" -ForegroundColor Yellow

# Calculate change failure rate (last 30 days)
$RecentDeploymentsMonth = $Events | Where-Object { 
    [DateTime]$_.timestamp -gt (Get-Date).AddDays(-30) 
}
$FailedDeployments = $RecentDeploymentsMonth | Where-Object { $_.status -eq "failed" } | Measure-Object
$TotalDeployments = $RecentDeploymentsMonth | Measure-Object
if ($TotalDeployments.Count -gt 0) {
    $FailureRate = [math]::Round(($FailedDeployments.Count / $TotalDeployments.Count) * 100, 2)
    Write-Host "- Change Failure Rate: $FailureRate%" -ForegroundColor Yellow
}

exit 0