# PowerShell script to calculate and report DORA metrics
param(
    [Parameter(Mandatory=$false)]
    [int]$Days = 30,
    
    [Parameter(Mandatory=$false)]
    [switch]$ExportJson,
    
    [Parameter(Mandatory=$false)]
    [switch]$ExportCsv,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath = "."
)

Write-Host "PolicyCortex DORA Metrics Report" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Period: Last $Days days" -ForegroundColor Yellow
Write-Host ""

$StartDate = (Get-Date).AddDays(-$Days)
$Report = @{
    period_start = $StartDate.ToString("yyyy-MM-dd")
    period_end = (Get-Date).ToString("yyyy-MM-dd")
    days = $Days
}

# Load deployment data
$DeploymentEvents = @()
if (Test-Path "deployment-events.json") {
    $DeploymentEvents = Get-Content "deployment-events.json" | ConvertFrom-Json
}

# Load incident data
$Incidents = @()
if (Test-Path "incidents.json") {
    $Incidents = Get-Content "incidents.json" | ConvertFrom-Json
}

# 1. DEPLOYMENT FREQUENCY
Write-Host "1. Deployment Frequency" -ForegroundColor Green
Write-Host "-----------------------" -ForegroundColor Green

$RecentDeployments = $DeploymentEvents | Where-Object { 
    [DateTime]$_.timestamp -gt $StartDate 
}

$DeploymentsPerDay = @{}
foreach ($deployment in $RecentDeployments) {
    $day = [DateTime]$deployment.timestamp | Get-Date -Format "yyyy-MM-dd"
    if (-not $DeploymentsPerDay.ContainsKey($day)) {
        $DeploymentsPerDay[$day] = 0
    }
    $DeploymentsPerDay[$day]++
}

$TotalDeployments = $RecentDeployments.Count
$AvgDeploymentsPerDay = [math]::Round($TotalDeployments / $Days, 2)
$DeploymentFrequency = switch ($AvgDeploymentsPerDay) {
    {$_ -ge 1} { "Daily" }
    {$_ -ge 0.25} { "Weekly" }
    {$_ -ge 0.033} { "Monthly" }
    default { "Less than monthly" }
}

Write-Host "  Total Deployments: $TotalDeployments" -ForegroundColor White
Write-Host "  Average per Day: $AvgDeploymentsPerDay" -ForegroundColor White
Write-Host "  Frequency Class: $DeploymentFrequency" -ForegroundColor Yellow

$Report.deployment_frequency = @{
    total = $TotalDeployments
    average_per_day = $AvgDeploymentsPerDay
    frequency_class = $DeploymentFrequency
    target_met = $AvgDeploymentsPerDay -ge 1
}

# 2. LEAD TIME FOR CHANGES
Write-Host "`n2. Lead Time for Changes" -ForegroundColor Green
Write-Host "------------------------" -ForegroundColor Green

$LeadTimes = $RecentDeployments | Where-Object { $_.lead_time_seconds } | ForEach-Object { $_.lead_time_seconds }

if ($LeadTimes.Count -gt 0) {
    $AvgLeadTime = ($LeadTimes | Measure-Object -Average).Average
    $MedianLeadTime = $LeadTimes | Sort-Object | Select-Object -Index ([Math]::Floor($LeadTimes.Count / 2))
    $P95LeadTime = $LeadTimes | Sort-Object | Select-Object -Index ([Math]::Floor($LeadTimes.Count * 0.95))
    
    Write-Host "  Average: $([math]::Round($AvgLeadTime / 3600, 2)) hours" -ForegroundColor White
    Write-Host "  Median: $([math]::Round($MedianLeadTime / 3600, 2)) hours" -ForegroundColor White
    Write-Host "  95th Percentile: $([math]::Round($P95LeadTime / 3600, 2)) hours" -ForegroundColor White
    
    $LeadTimeClass = switch ($AvgLeadTime) {
        {$_ -le 3600} { "Elite (< 1 hour)" }
        {$_ -le 86400} { "High (< 1 day)" }
        {$_ -le 604800} { "Medium (< 1 week)" }
        default { "Low (> 1 week)" }
    }
    Write-Host "  Performance Class: $LeadTimeClass" -ForegroundColor Yellow
    
    $Report.lead_time = @{
        average_hours = [math]::Round($AvgLeadTime / 3600, 2)
        median_hours = [math]::Round($MedianLeadTime / 3600, 2)
        p95_hours = [math]::Round($P95LeadTime / 3600, 2)
        performance_class = $LeadTimeClass
        target_met = $AvgLeadTime -le 86400
    }
} else {
    Write-Host "  No lead time data available" -ForegroundColor Yellow
}

# 3. CHANGE FAILURE RATE
Write-Host "`n3. Change Failure Rate" -ForegroundColor Green
Write-Host "----------------------" -ForegroundColor Green

$FailedDeployments = $RecentDeployments | Where-Object { $_.status -eq "failed" }
$ChangeFailureRate = if ($TotalDeployments -gt 0) { 
    [math]::Round(($FailedDeployments.Count / $TotalDeployments) * 100, 2) 
} else { 0 }

Write-Host "  Failed Deployments: $($FailedDeployments.Count)" -ForegroundColor White
Write-Host "  Total Deployments: $TotalDeployments" -ForegroundColor White
Write-Host "  Failure Rate: $ChangeFailureRate%" -ForegroundColor White

$FailureRateClass = switch ($ChangeFailureRate) {
    {$_ -le 5} { "Elite (0-5%)" }
    {$_ -le 10} { "High (5-10%)" }
    {$_ -le 15} { "Medium (10-15%)" }
    default { "Low (>15%)" }
}
Write-Host "  Performance Class: $FailureRateClass" -ForegroundColor Yellow

$Report.change_failure_rate = @{
    failed_deployments = $FailedDeployments.Count
    total_deployments = $TotalDeployments
    failure_rate_percent = $ChangeFailureRate
    performance_class = $FailureRateClass
    target_met = $ChangeFailureRate -le 10
}

# 4. MEAN TIME TO RECOVERY (MTTR)
Write-Host "`n4. Mean Time to Recovery (MTTR)" -ForegroundColor Green
Write-Host "--------------------------------" -ForegroundColor Green

$ResolvedIncidents = $Incidents | Where-Object { 
    $_.status -eq "resolved" -and 
    [DateTime]$_.resolved_at -gt $StartDate -and
    $_.resolution_time_seconds
}

if ($ResolvedIncidents.Count -gt 0) {
    $ResolutionTimes = $ResolvedIncidents | ForEach-Object { $_.resolution_time_seconds }
    $AvgMTTR = ($ResolutionTimes | Measure-Object -Average).Average
    $MedianMTTR = $ResolutionTimes | Sort-Object | Select-Object -Index ([Math]::Floor($ResolutionTimes.Count / 2))
    
    Write-Host "  Incidents Resolved: $($ResolvedIncidents.Count)" -ForegroundColor White
    Write-Host "  Average MTTR: $([math]::Round($AvgMTTR / 3600, 2)) hours" -ForegroundColor White
    Write-Host "  Median MTTR: $([math]::Round($MedianMTTR / 3600, 2)) hours" -ForegroundColor White
    
    $MTTRClass = switch ($AvgMTTR) {
        {$_ -le 3600} { "Elite (< 1 hour)" }
        {$_ -le 86400} { "High (< 1 day)" }
        {$_ -le 604800} { "Medium (< 1 week)" }
        default { "Low (> 1 week)" }
    }
    Write-Host "  Performance Class: $MTTRClass" -ForegroundColor Yellow
    
    $Report.mttr = @{
        incidents_resolved = $ResolvedIncidents.Count
        average_hours = [math]::Round($AvgMTTR / 3600, 2)
        median_hours = [math]::Round($MedianMTTR / 3600, 2)
        performance_class = $MTTRClass
        target_met = $AvgMTTR -le 14400
    }
} else {
    Write-Host "  No incident data available" -ForegroundColor Yellow
}

# Overall DORA Performance
Write-Host "`nOverall DORA Performance" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

$TargetsMet = 0
$TotalTargets = 0

foreach ($metric in @("deployment_frequency", "lead_time", "change_failure_rate", "mttr")) {
    if ($Report.ContainsKey($metric)) {
        $TotalTargets++
        if ($Report[$metric].target_met) {
            $TargetsMet++
        }
    }
}

$OverallScore = if ($TotalTargets -gt 0) { 
    [math]::Round(($TargetsMet / $TotalTargets) * 100, 0) 
} else { 0 }

Write-Host "Targets Met: $TargetsMet / $TotalTargets" -ForegroundColor Yellow
Write-Host "Overall Score: $OverallScore%" -ForegroundColor Yellow

$Report.overall_performance = @{
    targets_met = $TargetsMet
    total_targets = $TotalTargets
    score_percent = $OverallScore
    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
}

# Export results if requested
if ($ExportJson) {
    $JsonPath = Join-Path $OutputPath "dora-metrics-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $Report | ConvertTo-Json -Depth 10 | Set-Content $JsonPath
    Write-Host "`nReport exported to: $JsonPath" -ForegroundColor Green
}

if ($ExportCsv) {
    $CsvPath = Join-Path $OutputPath "dora-metrics-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"
    $CsvData = @()
    
    foreach ($metric in $Report.Keys) {
        if ($metric -ne "overall_performance" -and $Report[$metric] -is [hashtable]) {
            $row = [PSCustomObject]@{
                Metric = $metric
                Value = $Report[$metric].average_per_day ?? $Report[$metric].average_hours ?? $Report[$metric].failure_rate_percent ?? "N/A"
                PerformanceClass = $Report[$metric].performance_class ?? "N/A"
                TargetMet = $Report[$metric].target_met ?? "N/A"
            }
            $CsvData += $row
        }
    }
    
    $CsvData | Export-Csv -Path $CsvPath -NoTypeInformation
    Write-Host "CSV exported to: $CsvPath" -ForegroundColor Green
}

exit 0