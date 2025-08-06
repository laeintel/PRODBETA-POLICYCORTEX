# Check System Resources for Concurrent Job Execution
# Ensures your system can handle multiple GitHub Actions jobs

Write-Host "Checking System Resources for Concurrent Job Execution" -ForegroundColor Green

# Get system information
$cpu = Get-CimInstance -ClassName Win32_Processor
$memory = Get-CimInstance -ClassName Win32_ComputerSystem
$disk = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "C:" }

Write-Host "`nSystem Specifications:" -ForegroundColor Cyan
Write-Host "CPU: $($cpu.Name)" -ForegroundColor White
Write-Host "Cores: $($cpu.NumberOfCores)" -ForegroundColor White
Write-Host "Logical Processors: $($cpu.NumberOfLogicalProcessors)" -ForegroundColor White
Write-Host "Total RAM: $([math]::Round($memory.TotalPhysicalMemory / 1GB, 2)) GB" -ForegroundColor White
Write-Host "Available Disk Space: $([math]::Round($disk.FreeSpace / 1GB, 2)) GB" -ForegroundColor White

# Calculate recommended concurrent jobs
$recommendedJobs = [math]::Floor($cpu.NumberOfLogicalProcessors / 2)
$maxJobs = [math]::Min($recommendedJobs, 6)  # Cap at 6 for stability

Write-Host "`nConcurrency Recommendations:" -ForegroundColor Yellow
Write-Host "Recommended Concurrent Jobs: $maxJobs" -ForegroundColor White
Write-Host "Max Safe Concurrent Jobs: $([math]::Min($cpu.NumberOfLogicalProcessors, 8))" -ForegroundColor White

# Check memory per job
$memoryPerJob = [math]::Round(($memory.TotalPhysicalMemory / 1GB) / $maxJobs, 2)
Write-Host "Memory per Job: $memoryPerJob GB" -ForegroundColor White

# Resource warnings
Write-Host "`nResource Considerations:" -ForegroundColor Yellow

if ($cpu.NumberOfLogicalProcessors -lt 4) {
    Write-Host "- Consider limiting to 1-2 concurrent jobs due to limited CPU cores" -ForegroundColor Red
} elseif ($cpu.NumberOfLogicalProcessors -ge 8) {
    Write-Host "- System has excellent CPU resources for concurrent execution" -ForegroundColor Green
}

if (($memory.TotalPhysicalMemory / 1GB) -lt 8) {
    Write-Host "- Consider limiting concurrent jobs due to limited RAM" -ForegroundColor Red
} elseif (($memory.TotalPhysicalMemory / 1GB) -ge 16) {
    Write-Host "- System has excellent memory resources for concurrent execution" -ForegroundColor Green
}

if (($disk.FreeSpace / 1GB) -lt 20) {
    Write-Host "- Low disk space may impact job execution" -ForegroundColor Red
} else {
    Write-Host "- Adequate disk space available" -ForegroundColor Green
}

# Active runner services
Write-Host "`nCurrent GitHub Runner Services:" -ForegroundColor Cyan
$runnerServices = Get-Service -Name "actions.runner.*" -ErrorAction SilentlyContinue

if ($runnerServices) {
    foreach ($service in $runnerServices) {
        $status = if ($service.Status -eq 'Running') { "[RUNNING]" } else { "[" + $service.Status + "]" }
        Write-Host "- $($service.Name): $status" -ForegroundColor White
    }
} else {
    Write-Host "- No GitHub Runner services found" -ForegroundColor Yellow
}

# Performance monitoring suggestion
Write-Host "`nPerformance Monitoring:" -ForegroundColor Cyan
Write-Host "- Monitor CPU usage during concurrent job execution" -ForegroundColor White
Write-Host "- Watch memory consumption to prevent system instability" -ForegroundColor White
Write-Host "- Consider using Performance Monitor (perfmon) for detailed analysis" -ForegroundColor White

Write-Host "`nRecommended Configuration:" -ForegroundColor Green
Write-Host "Configure your runner for $maxJobs concurrent jobs using:" -ForegroundColor White
Write-Host ".\configure-concurrent-runner.ps1 -MaxConcurrentJobs $maxJobs" -ForegroundColor Cyan