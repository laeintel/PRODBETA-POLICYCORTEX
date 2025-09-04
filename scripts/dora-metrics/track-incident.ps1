# PowerShell script to track incident and MTTR metrics
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("open", "resolved")]
    [string]$Action,
    
    [Parameter(Mandatory=$true)]
    [string]$IncidentId,
    
    [Parameter(Mandatory=$false)]
    [string]$Service = "policycortex",
    
    [Parameter(Mandatory=$false)]
    [string]$Severity = "medium",
    
    [Parameter(Mandatory=$false)]
    [string]$PrometheusGateway = "http://localhost:9091"
)

$IncidentsFile = "incidents.json"
$Incidents = @()

if (Test-Path $IncidentsFile) {
    $Incidents = Get-Content $IncidentsFile | ConvertFrom-Json
}

$CurrentTime = Get-Date

if ($Action -eq "open") {
    Write-Host "Opening incident $IncidentId" -ForegroundColor Red
    
    # Create new incident record
    $Incident = @{
        id = $IncidentId
        service = $Service
        severity = $Severity
        opened_at = $CurrentTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
        opened_timestamp = [int](Get-Date -UFormat %s)
        status = "open"
    }
    
    $Incidents += $Incident
    
    # Push metric to Prometheus
    $Metrics = @"
# TYPE incidents_opened_total counter
# HELP incidents_opened_total Total number of incidents opened
incidents_opened_total{service="$Service",severity="$Severity"} 1

# TYPE incidents_open gauge
# HELP incidents_open Number of currently open incidents
incidents_open{service="$Service",severity="$Severity"} 1
"@

} elseif ($Action -eq "resolved") {
    Write-Host "Resolving incident $IncidentId" -ForegroundColor Green
    
    # Find and update the incident
    $Incident = $Incidents | Where-Object { $_.id -eq $IncidentId -and $_.status -eq "open" } | Select-Object -First 1
    
    if ($null -eq $Incident) {
        Write-Host "Incident $IncidentId not found or already resolved" -ForegroundColor Yellow
        exit 1
    }
    
    $Incident.status = "resolved"
    $Incident.resolved_at = $CurrentTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
    $Incident.resolved_timestamp = [int](Get-Date -UFormat %s)
    $Incident.resolution_time_seconds = $Incident.resolved_timestamp - $Incident.opened_timestamp
    
    # Calculate MTTR from recent incidents
    $ResolvedIncidents = $Incidents | Where-Object { 
        $_.status -eq "resolved" -and 
        $_.resolution_time_seconds -gt 0 
    }
    
    if ($ResolvedIncidents.Count -gt 0) {
        $AvgMTTR = ($ResolvedIncidents | Measure-Object -Property resolution_time_seconds -Average).Average
    } else {
        $AvgMTTR = 0
    }
    
    # Push metrics to Prometheus
    $Metrics = @"
# TYPE incidents_resolved_total counter
# HELP incidents_resolved_total Total number of incidents resolved
incidents_resolved_total{service="$Service",severity="$Severity"} 1

# TYPE incident_resolution_time_seconds gauge
# HELP incident_resolution_time_seconds Time to resolve incident in seconds
incident_resolution_time_seconds{service="$Service",severity="$Severity",incident_id="$IncidentId"} $($Incident.resolution_time_seconds)

# TYPE mttr_seconds gauge
# HELP mttr_seconds Mean time to recovery in seconds
mttr_seconds{service="$Service"} $AvgMTTR
"@
    
    Write-Host "Incident resolved in $([math]::Round($Incident.resolution_time_seconds / 3600, 2)) hours" -ForegroundColor Green
    Write-Host "Current MTTR: $([math]::Round($AvgMTTR / 3600, 2)) hours" -ForegroundColor Yellow
}

# Save incidents to file
$Incidents | ConvertTo-Json -Depth 10 | Set-Content $IncidentsFile

# Push metrics to Prometheus
try {
    $Headers = @{
        "Content-Type" = "text/plain; version=0.0.4"
    }
    
    $Response = Invoke-RestMethod -Uri "$PrometheusGateway/metrics/job/incidents/instance/$Service" -Method POST -Body $Metrics -Headers $Headers
    Write-Host "Metrics pushed successfully to Prometheus" -ForegroundColor Green
} catch {
    Write-Host "Failed to push metrics to Prometheus: $_" -ForegroundColor Red
}

# Display current incident statistics
Write-Host "`nIncident Statistics:" -ForegroundColor Cyan
$OpenIncidents = $Incidents | Where-Object { $_.status -eq "open" } | Measure-Object
$ResolvedLast30Days = $Incidents | Where-Object { 
    $_.status -eq "resolved" -and 
    [DateTime]$_.resolved_at -gt (Get-Date).AddDays(-30) 
} | Measure-Object

Write-Host "- Open Incidents: $($OpenIncidents.Count)" -ForegroundColor Yellow
Write-Host "- Resolved (30d): $($ResolvedLast30Days.Count)" -ForegroundColor Yellow
if ($AvgMTTR -gt 0) {
    Write-Host "- Average MTTR: $([math]::Round($AvgMTTR / 3600, 2)) hours" -ForegroundColor Yellow
}

exit 0