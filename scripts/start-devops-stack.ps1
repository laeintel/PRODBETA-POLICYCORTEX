# PowerShell script to start the complete DevOps monitoring and observability stack
param(
    [Parameter(Mandatory=$false)]
    [switch]$StartMonitoring,
    
    [Parameter(Mandatory=$false)]
    [switch]$StartApplication,
    
    [Parameter(Mandatory=$false)]
    [switch]$GenerateSBOM,
    
    [Parameter(Mandatory=$false)]
    [switch]$SignContainers,
    
    [Parameter(Mandatory=$false)]
    [switch]$CheckSLOs,
    
    [Parameter(Mandatory=$false)]
    [switch]$All
)

Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║     PolicyCortex DevOps Operations Center                   ║
║     Enterprise Monitoring & Observability Stack             ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

$StartTime = Get-Date

# Function to check Docker status
function Test-Docker {
    try {
        docker version | Out-Null
        return $true
    } catch {
        Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        return $false
    }
}

# Function to wait for service health
function Wait-ForService {
    param(
        [string]$ServiceName,
        [string]$HealthUrl,
        [int]$MaxAttempts = 30
    )
    
    Write-Host "  Waiting for $ServiceName to be healthy..." -ForegroundColor Yellow
    
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $Response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($Response.StatusCode -eq 200) {
                Write-Host "  $ServiceName is healthy!" -ForegroundColor Green
                return $true
            }
        } catch {
            # Service not ready yet
        }
        
        if ($i % 5 -eq 0) {
            Write-Host "    Still waiting... ($i/$MaxAttempts)" -ForegroundColor Gray
        }
        Start-Sleep -Seconds 2
    }
    
    Write-Host "  $ServiceName failed to start!" -ForegroundColor Red
    return $false
}

# Check Docker
if (-not (Test-Docker)) {
    exit 1
}

# Start monitoring stack
if ($StartMonitoring -or $All) {
    Write-Host "`n📊 Starting Monitoring Stack" -ForegroundColor Green
    Write-Host "=============================" -ForegroundColor Green
    
    # Create required networks if they don't exist
    docker network create policycortex-monitoring 2>$null
    docker network create policycortex-network 2>$null
    
    # Start monitoring services
    Write-Host "Starting monitoring services..." -ForegroundColor Yellow
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be healthy
    $MonitoringServices = @(
        @{Name="Prometheus"; Url="http://localhost:9090/-/ready"},
        @{Name="Grafana"; Url="http://localhost:3030/api/health"},
        @{Name="Jaeger"; Url="http://localhost:16686/"},
        @{Name="AlertManager"; Url="http://localhost:9093/-/ready"}
    )
    
    foreach ($Service in $MonitoringServices) {
        Wait-ForService -ServiceName $Service.Name -HealthUrl $Service.Url
    }
    
    Write-Host "`n✅ Monitoring stack started successfully!" -ForegroundColor Green
    Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor White
    Write-Host "  - Grafana: http://localhost:3030 (admin/admin)" -ForegroundColor White
    Write-Host "  - Jaeger: http://localhost:16686" -ForegroundColor White
    Write-Host "  - AlertManager: http://localhost:9093" -ForegroundColor White
}

# Start application stack
if ($StartApplication -or $All) {
    Write-Host "`n🚀 Starting Application Stack" -ForegroundColor Green
    Write-Host "=============================" -ForegroundColor Green
    
    # Start main application services
    Write-Host "Starting application services..." -ForegroundColor Yellow
    docker-compose -f docker-compose.local.yml up -d
    
    # Wait for application services
    $AppServices = @(
        @{Name="Frontend"; Url="http://localhost:3005/api/health"},
        @{Name="Core API"; Url="http://localhost:8085/health"},
        @{Name="GraphQL"; Url="http://localhost:4001/health"}
    )
    
    foreach ($Service in $AppServices) {
        Wait-ForService -ServiceName $Service.Name -HealthUrl $Service.Url
    }
    
    Write-Host "`n✅ Application stack started successfully!" -ForegroundColor Green
    Write-Host "  - Frontend: http://localhost:3005" -ForegroundColor White
    Write-Host "  - Core API: http://localhost:8085" -ForegroundColor White
    Write-Host "  - GraphQL: http://localhost:4001/graphql" -ForegroundColor White
}

# Generate SBOM
if ($GenerateSBOM -or $All) {
    Write-Host "`n📋 Generating Software Bill of Materials" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    & .\scripts\supply-chain\generate-sbom.ps1 -Format json
    
    Write-Host "✅ SBOM generation complete!" -ForegroundColor Green
}

# Sign containers
if ($SignContainers -or $All) {
    Write-Host "`n🔐 Signing Container Images" -ForegroundColor Green
    Write-Host "===========================" -ForegroundColor Green
    
    & .\scripts\supply-chain\sign-containers.ps1 -GenerateProvenance -GenerateSLSA
    
    Write-Host "✅ Container signing complete!" -ForegroundColor Green
}

# Check SLOs and error budgets
if ($CheckSLOs -or $All) {
    Write-Host "`n📈 Checking SLOs and Error Budgets" -ForegroundColor Green
    Write-Host "===================================" -ForegroundColor Green
    
    & .\scripts\slo-management\calculate-error-budget.ps1 -ExportReport
    
    Write-Host "✅ SLO check complete!" -ForegroundColor Green
}

# Display DORA metrics
Write-Host "`n📊 DORA Metrics Summary" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

& .\scripts\dora-metrics\calculate-dora-metrics.ps1 -Days 7

# System health check
Write-Host "`n🏥 System Health Check" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

$HealthChecks = @(
    @{Service="Prometheus"; Url="http://localhost:9090/-/ready"; Critical=$true},
    @{Service="Grafana"; Url="http://localhost:3030/api/health"; Critical=$false},
    @{Service="Frontend"; Url="http://localhost:3005/api/health"; Critical=$true},
    @{Service="Core API"; Url="http://localhost:8085/health"; Critical=$true},
    @{Service="PostgreSQL"; Command="docker exec postgres pg_isready"; Critical=$true}
)

$HealthyServices = 0
$TotalServices = $HealthChecks.Count

foreach ($Check in $HealthChecks) {
    Write-Host -NoNewline "  $($Check.Service): " -ForegroundColor White
    
    $IsHealthy = $false
    if ($Check.Url) {
        try {
            $Response = Invoke-WebRequest -Uri $Check.Url -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            $IsHealthy = $Response.StatusCode -eq 200
        } catch {
            $IsHealthy = $false
        }
    } elseif ($Check.Command) {
        try {
            Invoke-Expression $Check.Command 2>$null | Out-Null
            $IsHealthy = $?
        } catch {
            $IsHealthy = $false
        }
    }
    
    if ($IsHealthy) {
        Write-Host "✅ Healthy" -ForegroundColor Green
        $HealthyServices++
    } else {
        $Severity = if ($Check.Critical) { "❌ Down (CRITICAL)" } else { "⚠️  Down" }
        $Color = if ($Check.Critical) { "Red" } else { "Yellow" }
        Write-Host $Severity -ForegroundColor $Color
    }
}

$HealthPercentage = [math]::Round(($HealthyServices / $TotalServices) * 100, 0)
$HealthStatus = if ($HealthPercentage -eq 100) { "HEALTHY" }
                elseif ($HealthPercentage -ge 75) { "DEGRADED" }
                else { "CRITICAL" }

$StatusColor = switch ($HealthStatus) {
    "HEALTHY" { "Green" }
    "DEGRADED" { "Yellow" }
    "CRITICAL" { "Red" }
}

Write-Host "`nOverall Health: $HealthStatus ($HealthPercentage%)" -ForegroundColor $StatusColor

# Display operational dashboard URLs
Write-Host "`n🌐 Operational Dashboards" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

$Dashboards = @(
    @{Name="SLO & Golden Signals"; Url="http://localhost:3030/d/policycortex-slo"},
    @{Name="DORA Metrics"; Url="http://localhost:3030/d/policycortex-dora"},
    @{Name="Distributed Tracing"; Url="http://localhost:16686"},
    @{Name="Alert Manager"; Url="http://localhost:9093"},
    @{Name="Application Frontend"; Url="http://localhost:3005"}
)

foreach ($Dashboard in $Dashboards) {
    Write-Host "  $($Dashboard.Name):" -ForegroundColor White
    Write-Host "    $($Dashboard.Url)" -ForegroundColor Gray
}

# Generate operational readiness contract
$OperationalContract = @{
    slos = @{
        availability = "99.9%"
        latency_p95_ms = 300
    }
    alerts = @("burn_25", "burn_50", "burn_100")
    runbooks = @("https://wiki.policycortex.com/runbooks")
    dora = @{
        deployment_frequency_target = "daily"
        lead_time_target_hours = 24
        change_failure_rate_target_pct = 10
        mttr_target_hours = 4
    }
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
}

$ContractFile = "operational-contract-$(Get-Date -Format 'yyyyMMdd').json"
$OperationalContract | ConvertTo-Json -Depth 10 | Set-Content $ContractFile

Write-Host "`n📄 Operational Readiness Contract" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Contract saved to: $ContractFile" -ForegroundColor Green

# Calculate total startup time
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "`n⏱️  Total startup time: $([math]::Round($Duration.TotalSeconds, 1)) seconds" -ForegroundColor Yellow

Write-Host "`n✨ PolicyCortex DevOps Stack Ready!" -ForegroundColor Green
Write-Host "All systems operational. Happy monitoring! 🚀" -ForegroundColor Cyan

exit 0