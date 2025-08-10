param(
  [switch]$KillPorts = $false,
  [int[]]$Ports = @(3000,8080),
  [switch]$VerboseMode = $false
)

$ErrorActionPreference = 'SilentlyContinue'

function Write-Info($msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg){ Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Ok($msg){ Write-Host "[OK] $msg" -ForegroundColor Green }

function Test-Cmd($name){
  $cmd = Get-Command $name -ErrorAction SilentlyContinue
  return $null -ne $cmd
}

function Show-Version($name,$args){
  try { & $name $args | Select-Object -First 1 | ForEach-Object { Write-Info "$name $_" } } catch { Write-Warn "$name not available" }
}

function Get-PortProcess($port){
  try {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if($conns){
      $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
      return $pids
    }
    return @()
  } catch { return @() }
}

function Ensure-Ports(){
  foreach($p in $Ports){
    $pids = Get-PortProcess $p
    if($pids.Count -gt 0){
      Write-Warn "Port $p in use by PIDs: $($pids -join ', ')"
      if($KillPorts){
        foreach($pid in $pids){
          try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue; Write-Ok "Killed PID $pid on port $p" } catch { Write-Err "Failed to kill PID $pid on port $p" }
        }
      } else {
        Write-Info "Run with -KillPorts to free port $p"
      }
    } else { Write-Ok "Port $p is free" }
  }
}

function Test-Env($name){
  $val = [Environment]::GetEnvironmentVariable($name,'Process');
  if(-not $val){ $val = [Environment]::GetEnvironmentVariable($name,'User') }
  if(-not $val){ $val = [Environment]::GetEnvironmentVariable($name,'Machine') }
  if([string]::IsNullOrWhiteSpace($val)){
    Write-Warn "$name is not set"
  } else {
    if($VerboseMode){ Write-Info "$name=$val" } else { Write-Ok "$name present" }
  }
}

Write-Host "=== PolicyCortex Windows Preflight ===" -ForegroundColor Magenta

# Tooling
if(Test-Cmd node){ Show-Version node '-v' } else { Write-Warn 'Node.js not found in PATH' }
if(Test-Cmd npm){ Show-Version npm '-v' } else { Write-Warn 'npm not found in PATH' }
if(Test-Cmd cargo){ Show-Version cargo '--version' } else { Write-Warn 'cargo not found in PATH' }
if(Test-Cmd python){ Show-Version python '--version' } else { Write-Warn 'python not found in PATH' }

# Ports
Ensure-Ports

# Frontend env (MSAL + API)
Test-Env 'NEXT_PUBLIC_API_URL'
Test-Env 'NEXT_PUBLIC_MSAL_CLIENT_ID'
Test-Env 'NEXT_PUBLIC_MSAL_AUTHORITY'
Test-Env 'NEXT_PUBLIC_MSAL_REDIRECT_URI'

# Backend env (Azure AD / JWT)
Test-Env 'AZURE_TENANT_ID'
Test-Env 'AZURE_CLIENT_ID'
Test-Env 'AZURE_CLIENT_SECRET'
Test-Env 'JWT_AUDIENCE'
Test-Env 'JWT_ISSUER'

# Optional feature flags
Test-Env 'NEXT_PUBLIC_USE_MOCK_DATA'
Test-Env 'NEXT_PUBLIC_DISABLE_DEEP'

Write-Host "Preflight completed." -ForegroundColor Green
