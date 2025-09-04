# PowerShell script to calculate and monitor error budgets
param(
    [Parameter(Mandatory=$false)]
    [string]$ConfigFile = ".\slo-config.yaml",
    
    [Parameter(Mandatory=$false)]
    [string]$PrometheusUrl = "http://localhost:9090",
    
    [Parameter(Mandatory=$false)]
    [int]$Window = 30,  # Days
    
    [Parameter(Mandatory=$false)]
    [switch]$AlertOnBreach,
    
    [Parameter(Mandatory=$false)]
    [switch]$ExportReport
)

Write-Host "PolicyCortex Error Budget Calculator" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Calculation Window: $Window days" -ForegroundColor Yellow

# Install required module for YAML parsing if not present
if (-not (Get-Module -ListAvailable -Name powershell-yaml)) {
    Write-Host "Installing YAML module..." -ForegroundColor Yellow
    Install-Module -Name powershell-yaml -Force -Scope CurrentUser
}

Import-Module powershell-yaml

# Load SLO configuration
$Config = Get-Content $ConfigFile -Raw | ConvertFrom-Yaml

# Function to query Prometheus
function Query-Prometheus {
    param(
        [string]$Query,
        [string]$Time = "now"
    )
    
    $EncodedQuery = [System.Web.HttpUtility]::UrlEncode($Query)
    $Url = "$PrometheusUrl/api/v1/query?query=$EncodedQuery&time=$Time"
    
    try {
        $Response = Invoke-RestMethod -Uri $Url -Method Get
        if ($Response.status -eq "success" -and $Response.data.result.Count -gt 0) {
            return $Response.data.result[0].value[1]
        }
    } catch {
        Write-Host "  Failed to query Prometheus: $_" -ForegroundColor Red
        return $null
    }
    
    return 0
}

# Function to calculate error budget
function Calculate-ErrorBudget {
    param(
        [hashtable]$SLO,
        [string]$Service
    )
    
    $Result = @{
        service = $Service
        slo_name = $SLO.name
        target = $SLO.target
        window = $SLO.window
        current_performance = 0
        error_budget_total = 0
        error_budget_consumed = 0
        error_budget_remaining = 0
        error_budget_remaining_pct = 0
        burn_rate = 0
        time_until_exhausted = "N/A"
        status = "healthy"
    }
    
    # Calculate total error budget (in minutes for a 30-day window)
    $WindowMinutes = $Window * 24 * 60
    $Result.error_budget_total = $WindowMinutes * (100 - $SLO.target) / 100
    
    # Query current SLI performance
    $Query = switch ($SLO.sli.type) {
        "ratio" {
            "($($SLO.sli.good_events)) / ($($SLO.sli.total_events))"
        }
        "threshold" {
            "$($SLO.sli.metric) < $($SLO.sli.threshold)"
        }
        "probe" {
            "avg_over_time($($SLO.sli.metric)[$($Window)d])"
        }
        "custom" {
            $SLO.sli.metric
        }
    }
    
    if ($Query) {
        $Performance = Query-Prometheus -Query $Query
        if ($null -ne $Performance) {
            $Result.current_performance = [math]::Round([double]$Performance * 100, 3)
        }
    }
    
    # Calculate error budget consumption
    if ($Result.current_performance -gt 0) {
        $ErrorRate = 100 - $Result.current_performance
        $Result.error_budget_consumed = $WindowMinutes * $ErrorRate / 100
        $Result.error_budget_remaining = $Result.error_budget_total - $Result.error_budget_consumed
        $Result.error_budget_remaining_pct = [math]::Round(($Result.error_budget_remaining / $Result.error_budget_total) * 100, 2)
        
        # Calculate burn rate (how fast we're consuming the budget)
        # Burn rate = (actual error rate) / (allowed error rate)
        $AllowedErrorRate = 100 - $SLO.target
        if ($AllowedErrorRate -gt 0) {
            $Result.burn_rate = [math]::Round($ErrorRate / $AllowedErrorRate, 2)
        }
        
        # Calculate time until exhausted
        if ($Result.burn_rate -gt 1 -and $Result.error_budget_remaining -gt 0) {
            $DaysUntilExhausted = $Result.error_budget_remaining / ($Result.error_budget_consumed / $Window)
            if ($DaysUntilExhausted -lt 1) {
                $Result.time_until_exhausted = "$([math]::Round($DaysUntilExhausted * 24, 1)) hours"
            } else {
                $Result.time_until_exhausted = "$([math]::Round($DaysUntilExhausted, 1)) days"
            }
        } elseif ($Result.burn_rate -le 1) {
            $Result.time_until_exhausted = "Budget sustainable"
        } else {
            $Result.time_until_exhausted = "Already exhausted"
        }
        
        # Determine status
        $Result.status = switch ($Result.error_budget_remaining_pct) {
            {$_ -le 0} { "exhausted" }
            {$_ -le 10} { "critical" }
            {$_ -le 25} { "warning" }
            {$_ -le 50} { "attention" }
            default { "healthy" }
        }
    }
    
    return $Result
}

# Calculate error budgets for all SLOs
$Results = @()
$CriticalAlerts = @()

foreach ($Service in $Config.slos) {
    Write-Host "`nService: $($Service.service)" -ForegroundColor Green
    Write-Host "========================" -ForegroundColor Green
    
    foreach ($Objective in $Service.objectives) {
        Write-Host "`n  SLO: $($Objective.name)" -ForegroundColor Yellow
        Write-Host "  Description: $($Objective.description)" -ForegroundColor White
        
        $Budget = Calculate-ErrorBudget -SLO $Objective -Service $Service.service
        $Results += $Budget
        
        # Display results
        Write-Host "  Target: $($Budget.target)%" -ForegroundColor White
        Write-Host "  Current Performance: $($Budget.current_performance)%" -ForegroundColor White
        
        $BudgetColor = switch ($Budget.status) {
            "exhausted" { "Red" }
            "critical" { "Magenta" }
            "warning" { "Yellow" }
            "attention" { "DarkYellow" }
            default { "Green" }
        }
        
        Write-Host "  Error Budget Remaining: $($Budget.error_budget_remaining_pct)% ($($Budget.status))" -ForegroundColor $BudgetColor
        Write-Host "  Burn Rate: $($Budget.burn_rate)x" -ForegroundColor White
        Write-Host "  Time Until Exhausted: $($Budget.time_until_exhausted)" -ForegroundColor White
        
        # Check for critical alerts
        if ($Budget.status -in @("exhausted", "critical")) {
            $CriticalAlerts += $Budget
        }
    }
}

# Calculate composite SLOs
Write-Host "`nComposite SLOs" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan

foreach ($Composite in $Config.composite_slos) {
    Write-Host "`n$($Composite.name): $($Composite.description)" -ForegroundColor Green
    
    $WeightedSum = 0
    $TotalWeight = 0
    
    foreach ($Component in $Composite.components) {
        $ComponentResult = $Results | Where-Object { 
            $_.service -eq $Component.service -and 
            $_.slo_name -eq $Component.objective 
        } | Select-Object -First 1
        
        if ($ComponentResult) {
            $WeightedSum += $ComponentResult.current_performance * $Component.weight
            $TotalWeight += $Component.weight
        }
    }
    
    if ($TotalWeight -gt 0) {
        $CompositePerformance = [math]::Round($WeightedSum / $TotalWeight, 2)
        $CompositeMet = $CompositePerformance -ge $Composite.target
        
        Write-Host "  Target: $($Composite.target)%" -ForegroundColor White
        Write-Host "  Current: $CompositePerformance%" -ForegroundColor $(if ($CompositeMet) { "Green" } else { "Red" })
        Write-Host "  Status: $(if ($CompositeMet) { 'MEETING SLO' } else { 'BREACHING SLO' })" -ForegroundColor $(if ($CompositeMet) { "Green" } else { "Red" })
    }
}

# Check error budget policies
Write-Host "`nError Budget Policy Evaluation" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

foreach ($Policy in $Config.error_budget_policies) {
    Write-Host "`nPolicy: $($Policy.name)" -ForegroundColor Yellow
    Write-Host "Description: $($Policy.description)" -ForegroundColor White
    
    $TriggeredServices = $Results | Where-Object { 
        $_.error_budget_remaining_pct -lt $Policy.threshold 
    }
    
    if ($TriggeredServices.Count -gt 0) {
        Write-Host "  STATUS: TRIGGERED" -ForegroundColor Red
        Write-Host "  Affected Services:" -ForegroundColor White
        
        foreach ($Service in $TriggeredServices) {
            Write-Host "    - $($Service.service)/$($Service.slo_name): $($Service.error_budget_remaining_pct)% remaining" -ForegroundColor Red
        }
        
        Write-Host "  Actions Required:" -ForegroundColor Yellow
        foreach ($Action in $Policy.actions.PSObject.Properties) {
            if ($Action.Value -eq $true) {
                Write-Host "    - $($Action.Name)" -ForegroundColor White
            }
        }
    } else {
        Write-Host "  STATUS: Not Triggered" -ForegroundColor Green
    }
}

# Send alerts if requested
if ($AlertOnBreach -and $CriticalAlerts.Count -gt 0) {
    Write-Host "`nSending Critical Alerts..." -ForegroundColor Red
    
    foreach ($Alert in $CriticalAlerts) {
        $AlertMessage = @{
            severity = "critical"
            service = $Alert.service
            slo = $Alert.slo_name
            error_budget_remaining = $Alert.error_budget_remaining_pct
            burn_rate = $Alert.burn_rate
            time_until_exhausted = $Alert.time_until_exhausted
            timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        }
        
        # Here you would send to your alerting system (Slack, PagerDuty, etc.)
        Write-Host "  Alert sent for $($Alert.service)/$($Alert.slo_name)" -ForegroundColor Red
    }
}

# Export report if requested
if ($ExportReport) {
    $Report = @{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        window_days = $Window
        slos = $Results
        critical_alerts = $CriticalAlerts
        summary = @{
            total_slos = $Results.Count
            healthy = ($Results | Where-Object { $_.status -eq "healthy" }).Count
            attention = ($Results | Where-Object { $_.status -eq "attention" }).Count
            warning = ($Results | Where-Object { $_.status -eq "warning" }).Count
            critical = ($Results | Where-Object { $_.status -eq "critical" }).Count
            exhausted = ($Results | Where-Object { $_.status -eq "exhausted" }).Count
        }
    }
    
    $ReportFile = "error-budget-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $Report | ConvertTo-Json -Depth 10 | Set-Content $ReportFile
    
    Write-Host "`nReport exported to: $ReportFile" -ForegroundColor Green
}

# Display summary
Write-Host "`nError Budget Summary" -ForegroundColor Cyan
Write-Host "====================" -ForegroundColor Cyan

$Summary = @{
    Healthy = ($Results | Where-Object { $_.status -eq "healthy" }).Count
    Attention = ($Results | Where-Object { $_.status -eq "attention" }).Count
    Warning = ($Results | Where-Object { $_.status -eq "warning" }).Count
    Critical = ($Results | Where-Object { $_.status -eq "critical" }).Count
    Exhausted = ($Results | Where-Object { $_.status -eq "exhausted" }).Count
}

foreach ($Status in $Summary.GetEnumerator()) {
    $Color = switch ($Status.Key) {
        "Exhausted" { "Red" }
        "Critical" { "Magenta" }
        "Warning" { "Yellow" }
        "Attention" { "DarkYellow" }
        default { "Green" }
    }
    
    if ($Status.Value -gt 0) {
        Write-Host "$($Status.Key): $($Status.Value)" -ForegroundColor $Color
    }
}

$OverallHealth = if ($Summary.Exhausted -gt 0) { "CRITICAL" }
                 elseif ($Summary.Critical -gt 0) { "SEVERE" }
                 elseif ($Summary.Warning -gt 0) { "DEGRADED" }
                 else { "HEALTHY" }

Write-Host "`nOverall System Health: $OverallHealth" -ForegroundColor $(
    if ($OverallHealth -eq "CRITICAL") { "Red" }
    elseif ($OverallHealth -eq "SEVERE") { "Magenta" }
    elseif ($OverallHealth -eq "DEGRADED") { "Yellow" }
    else { "Green" }
)

exit 0