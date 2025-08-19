Param(
    [switch]$InstallTools = $false,
    [switch]$SkipNode = $false,
    [switch]$SkipRust = $false,
    [switch]$SkipPython = $false,
    [string]$OutputSubdir = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

function Write-Info([string]$msg) { Write-Host "[codescan] $msg" -ForegroundColor Cyan }
function Write-Warn([string]$msg) { Write-Host "[codescan] $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg) { Write-Host "[codescan] $msg" -ForegroundColor Red }

function Should-SkipPath([string]$fullPath) {
    if ([string]::IsNullOrWhiteSpace($fullPath)) { return $false }
    $pattern = '(?i)(?:\\|/)(node_modules|dist|build|out|coverage|target|\.venv|\.next)(?:\\|/|$)'
    return ($fullPath -match $pattern)
}

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Push-Location $root
try {
    $timestamp = if ([string]::IsNullOrWhiteSpace($OutputSubdir)) { (Get-Date).ToString('yyyyMMdd-HHmmss') } else { $OutputSubdir }
    $scanRoot = Join-Path $root "codescan"
    $scanDir = Join-Path $scanRoot $timestamp

    New-Item -ItemType Directory -Force -Path $scanRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $scanDir | Out-Null

    $dirs = @(
        (Join-Path $scanDir 'node'),
        (Join-Path $scanDir 'rust'),
        (Join-Path $scanDir 'python'),
        (Join-Path $scanDir 'duplication'),
        (Join-Path $scanDir 'logs')
    )
    foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

    Write-Info "Output directory: $scanDir"

    function Ensure-Command([string]$cmd, [string]$installHint) {
        $exists = Get-Command $cmd -ErrorAction SilentlyContinue
        if (-not $exists) { Write-Warn "$cmd not found. $installHint" }
        return $null -ne $exists
    }

    function Try-Install-CargoTool([string]$tool) {
        $found = Get-Command $tool -ErrorAction SilentlyContinue
        if ($found) { return }
        Write-Info "Installing $tool (cargo install $tool)"
        try { & cargo install $tool | Tee-Object -FilePath (Join-Path $scanDir "logs/$tool-install.log") } catch { Write-Warn "Failed to install ${tool}: $_" }
    }

    function Ensure-PythonVenv() {
        $venvDir = Join-Path $scanDir '.venv'
        if (-not (Test-Path $venvDir)) {
            Write-Info 'Creating Python virtual environment for scanners'
            & python -m venv $venvDir 2>&1 | Tee-Object -FilePath (Join-Path $scanDir 'logs/venv-create.log') | Out-Null
        }
        $pip = Join-Path $venvDir 'Scripts/pip.exe'
        $pkgs = @('pip-audit','bandit','ruff','vulture')
        Write-Info "Installing Python scanner packages: $($pkgs -join ', ')"
        & $pip install --upgrade pip 2>&1 | Tee-Object -FilePath (Join-Path $scanDir 'logs/pip-upgrade.log') | Out-Null
        & $pip install $pkgs 2>&1 | Tee-Object -FilePath (Join-Path $scanDir 'logs/pip-install.log') | Out-Null
        return $venvDir
    }

    function Save-Json($obj, $path) {
        try { ($obj | ConvertTo-Json -Depth 10) | Out-File -Encoding utf8 -FilePath $path } catch { $_ | Out-File -Encoding utf8 -FilePath $path }
    }

    # --------------------------------
    # Node/JavaScript/TypeScript scans
    # --------------------------------
    if (-not $SkipNode) {
        $hasNode = Ensure-Command 'node' 'Install Node.js from https://nodejs.org and ensure it is on PATH.'
        if ($hasNode) {
            $nodeOutDir = Join-Path $scanDir 'node'
            $pkgFiles = Get-ChildItem -Recurse -Filter package.json
            foreach ($pkg in $pkgFiles) {
                if (Should-SkipPath $pkg.FullName) { continue }
                $pkgDir = Split-Path -Parent $pkg.FullName
                $relDir = Resolve-Path $pkgDir | ForEach-Object { $_.ToString().Substring($root.Path.Length).TrimStart('/','\') }
                $safeName = ($relDir -replace '[\\/]', '_'); if ([string]::IsNullOrWhiteSpace($safeName)) { $safeName = 'root' }
                Write-Info "Node scans for $relDir"
                Push-Location $pkgDir
                try {
                    # Vulnerabilities
                    try { npm audit --json 2>&1 | Out-File -Encoding utf8 (Join-Path $nodeOutDir "$safeName-npm-audit.json") } catch { Write-Warn "npm audit failed in ${relDir}: $_" }

                    # Dependency updates
                    try { npx --yes npm-check-updates --dep prod,dev,peer,optional --target latest --jsonUpgraded 2>&1 | Out-File -Encoding utf8 (Join-Path $nodeOutDir "$safeName-ncu.json") } catch { Write-Warn "npm-check-updates failed in ${relDir}: $_" }

                    # Unused dependencies
                    try { npx --yes depcheck --json 2>&1 | Out-File -Encoding utf8 (Join-Path $nodeOutDir "$safeName-depcheck.json") } catch { Write-Warn "depcheck failed in ${relDir}: $_" }

                    # Unused exports (TypeScript)
                    $tsconfig = Join-Path $pkgDir 'tsconfig.json'
                    if (Test-Path $tsconfig) {
                        try { npx --yes ts-unused-exports $tsconfig --showLineNumber 2>&1 | Out-File -Encoding utf8 (Join-Path $nodeOutDir "$safeName-ts-unused-exports.txt") } catch { Write-Warn "ts-unused-exports failed in ${relDir}: $_" }
                    }

                    # Orphaned/unused files
                    try { npx --yes unimported -f json 2>&1 | Out-File -Encoding utf8 (Join-Path $nodeOutDir "$safeName-unimported.json") } catch { Write-Warn "unimported failed in ${relDir}: $_" }

                    # Circular dependency detection
                    try { npx --yes madge --circular --extensions ts,tsx,js,jsx . --json 2>&1 | Out-File -Encoding utf8 (Join-Path $nodeOutDir "$safeName-madge-circular.json") } catch { Write-Warn "madge failed in ${relDir}: $_" }
                } finally { Pop-Location }
            }

            # Cross-language duplication detection
            try {
                Write-Info 'Running jscpd for duplicate code detection'
                npx --yes jscpd --reporters html,json,markdown --output (Join-Path $scanDir 'duplication') --ignore "**/node_modules/**" --ignore "**/target/**" --ignore "**/.venv/**" --ignore "**/dist/**" --ignore "**/build/**" --ignore "**/.next/**" --ignore "**/out/**" --ignore "**/coverage/**" --silent . 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir 'logs/jscpd.log')
            } catch { Write-Warn "jscpd failed: $_" }
        }
    } else { Write-Info 'Skipping Node scans by request.' }

    # --------
    # Rust scans
    # --------
    if (-not $SkipRust) {
        $hasCargo = Ensure-Command 'cargo' 'Install Rust (rustup) from https://rustup.rs and ensure cargo is on PATH.'
        if ($hasCargo) {
            Write-Info 'Preparing Rust tooling (clippy, cargo-audit, cargo-outdated, cargo-udeps)'
            try { & rustup component add clippy 2>&1 | Tee-Object -FilePath (Join-Path $scanDir 'logs/clippy-install.log') | Out-Null } catch { Write-Warn "clippy component add failed: $_" }
            if ($InstallTools) { Try-Install-CargoTool 'cargo-audit'; Try-Install-CargoTool 'cargo-outdated'; Try-Install-CargoTool 'cargo-udeps' }

            Push-Location $root
            try {
                # cargo audit
                try { cargo audit --json 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir 'rust/cargo-audit.json') } catch { Write-Warn "cargo audit failed: $_" }

                # cargo outdated (text output for compatibility)
                try { cargo outdated --workspace --depth 1 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir 'rust/cargo-outdated.txt') } catch { Write-Warn "cargo outdated failed: $_" }

                # cargo udeps (text output)
                try { cargo udeps --workspace --all-targets 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir 'rust/cargo-udeps.txt') } catch { Write-Warn "cargo udeps failed: $_" }

                # clippy diagnostics (json)
                try { cargo clippy --workspace --all-targets --message-format=json 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir 'rust/clippy.json') } catch { Write-Warn "cargo clippy failed: $_" }
            } finally { Pop-Location }
        }
    } else { Write-Info 'Skipping Rust scans by request.' }

    # ----------
    # Python scans
    # ----------
    if (-not $SkipPython) {
        $hasPython = Ensure-Command 'python' 'Install Python 3 and ensure it is on PATH.'
        if ($hasPython) {
            $venv = Ensure-PythonVenv
            $pipAudit = Join-Path $venv 'Scripts/pip-audit.exe'
            $bandit = Join-Path $venv 'Scripts/bandit.exe'
            $ruff = Join-Path $venv 'Scripts/ruff.exe'
            $vulture = Join-Path $venv 'Scripts/vulture.exe'

            $reqFiles = Get-ChildItem -Recurse -Filter requirements.txt | Where-Object { $_.FullName -notmatch "\\.venv" }
            foreach ($req in $reqFiles) {
                if (Should-SkipPath $req.FullName) { continue }
                $svcDir = Split-Path -Parent $req.FullName
                $relDir = Resolve-Path $svcDir | ForEach-Object { $_.ToString().Substring($root.Path.Length).TrimStart('/','\') }
                $safeName = ($relDir -replace '[\\/]', '_')
                Write-Info "Python scans for $relDir"
                try {
                    # Vulnerabilities from requirements
                    try { & $pipAudit -r $req.FullName -f json -o (Join-Path $scanDir "python/$safeName-pip-audit.json") 2>&1 | Out-Null } catch { Write-Warn "pip-audit failed in ${relDir}: $_" }

                    # Static security analysis
                    try { & $bandit -q -r $svcDir -f json -o (Join-Path $scanDir "python/$safeName-bandit.json") 2>&1 | Out-Null } catch { Write-Warn "bandit failed in ${relDir}: $_" }

                    # Lint
                    try { & $ruff $svcDir --output-format json 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir "python/$safeName-ruff.json") } catch { Write-Warn "ruff failed in ${relDir}: $_" }

                    # Unused code signals
                    try { & $vulture $svcDir 2>&1 | Out-File -Encoding utf8 (Join-Path $scanDir "python/$safeName-vulture.txt") } catch { Write-Warn "vulture failed in ${relDir}: $_" }
                } catch { Write-Warn "Python scanning failed in ${relDir}: $_" }
            }
        }
    } else { Write-Info 'Skipping Python scans by request.' }

    # ---------------
    # Summary builder
    # ---------------
    Write-Info 'Building summary'
    $summaryPath = Join-Path $scanDir 'summary.md'
    $lines = @()
    $lines += "# Code Scan Summary ($timestamp)"
    $lines += ""

    # Node summary
    if (-not $SkipNode) {
        $lines += "## Node/JS/TS"
        $auditFiles = Get-ChildItem (Join-Path $scanDir 'node') -Filter '*-npm-audit.json' -ErrorAction SilentlyContinue
        foreach ($f in $auditFiles) {
            try {
                $j = Get-Content -Raw -Path $f.FullName | ConvertFrom-Json
                $low = $j.vulnerabilities.low; $mod = $j.vulnerabilities.moderate; $high = $j.vulnerabilities.high; $crit = $j.vulnerabilities.critical
                if ($null -eq $low -and $null -eq $mod -and $null -eq $high -and $null -eq $crit) {
                    $advisoriesCount = if ($j.advisories) { ($j.advisories | Get-Member -MemberType NoteProperty | Measure-Object).Count } else { 0 }
                    $lines += "- $($f.BaseName.Replace('-npm-audit','')): advisories=$advisoriesCount ([report]($(Resolve-Path $f.FullName)))"
                } else {
                    $total = ($low + $mod + $high + $crit)
                    $lines += "- $($f.BaseName.Replace('-npm-audit','')): low=$low, moderate=$mod, high=$high, critical=$crit, total=$total ([report]($(Resolve-Path $f.FullName)))"
                }
            } catch { $lines += "- $($f.BaseName.Replace('-npm-audit','')): could not parse npm audit JSON" }
        }
        $ncuFiles = Get-ChildItem (Join-Path $scanDir 'node') -Filter '*-ncu.json' -ErrorAction SilentlyContinue
        foreach ($f in $ncuFiles) {
            try { $j = Get-Content -Raw -Path $f.FullName | ConvertFrom-Json; $count = ($j | Get-Member -MemberType NoteProperty | Measure-Object).Count; $lines += "- updates available ($($f.BaseName.Replace('-ncu',''))): $count ([report]($(Resolve-Path $f.FullName)))" } catch { $lines += "- updates available ($($f.BaseName.Replace('-ncu',''))): parse error" }
        }
        $depcheckFiles = Get-ChildItem (Join-Path $scanDir 'node') -Filter '*-depcheck.json' -ErrorAction SilentlyContinue
        foreach ($f in $depcheckFiles) {
            try {
                $j = Get-Content -Raw -Path $f.FullName | ConvertFrom-Json
                $unused = ($j.unusedDependencies | Measure-Object).Count + ($j.unusedDevDependencies | Measure-Object).Count
                $missing = ($j.missing | Get-Member -MemberType NoteProperty -ErrorAction SilentlyContinue | Measure-Object).Count
                $lines += "- unused deps ($($f.BaseName.Replace('-depcheck',''))): unused=$unused, missing=$missing ([report]($(Resolve-Path $f.FullName)))"
            } catch { $lines += "- unused deps ($($f.BaseName.Replace('-depcheck',''))): parse error" }
        }
        $tse = Get-ChildItem (Join-Path $scanDir 'node') -Filter '*-ts-unused-exports.txt' -ErrorAction SilentlyContinue
        foreach ($f in $tse) { $lines += "- ts-unused-exports ($($f.BaseName.Replace('-ts-unused-exports',''))): see [report]($(Resolve-Path $f.FullName))" }
        $lines += ""
    }

    # Rust summary
    if (-not $SkipRust) {
        $lines += "## Rust"
        $audit = Join-Path $scanDir 'rust/cargo-audit.json'
        if (Test-Path $audit) {
            try { $j = Get-Content -Raw -Path $audit | ConvertFrom-Json; $count = if ($j.vulnerabilities -and $j.vulnerabilities.found) { ($j.vulnerabilities.list | Measure-Object).Count } else { 0 }; $lines += "- cargo-audit vulnerabilities: $count ([report]($(Resolve-Path $audit)))" } catch { $lines += "- cargo-audit: parse error" }
        }
        foreach ($fname in @('cargo-outdated.txt','cargo-udeps.txt','clippy.json')) {
            $p = Join-Path $scanDir "rust/$fname"; if (Test-Path $p) { $lines += "- ${fname}: see [report]($(Resolve-Path $p))" }
        }
        $lines += ""
    }

    # Python summary
    if (-not $SkipPython) {
        $lines += "## Python"
        $pa = Get-ChildItem (Join-Path $scanDir 'python') -Filter '*-pip-audit.json' -ErrorAction SilentlyContinue
        foreach ($f in $pa) {
            try { $j = Get-Content -Raw -Path $f.FullName | ConvertFrom-Json; $count = ($j.vulnerabilities | Measure-Object).Count; $lines += "- pip-audit ($($f.BaseName.Replace('-pip-audit',''))): vulnerabilities=$count ([report]($(Resolve-Path $f.FullName)))" } catch { $lines += "- pip-audit ($($f.BaseName.Replace('-pip-audit',''))): parse error" }
        }
        $band = Get-ChildItem (Join-Path $scanDir 'python') -Filter '*-bandit.json' -ErrorAction SilentlyContinue
        foreach ($f in $band) {
            try { $j = Get-Content -Raw -Path $f.FullName | ConvertFrom-Json; $count = ($j.results | Measure-Object).Count; $lines += "- bandit ($($f.BaseName.Replace('-bandit',''))): findings=$count ([report]($(Resolve-Path $f.FullName)))" } catch { $lines += "- bandit ($($f.BaseName.Replace('-bandit',''))): parse error" }
        }
        $ruffFiles = Get-ChildItem (Join-Path $scanDir 'python') -Filter '*-ruff.json' -ErrorAction SilentlyContinue
        foreach ($f in $ruffFiles) {
            try { $j = Get-Content -Raw -Path $f.FullName | ConvertFrom-Json; $count = ($j | Measure-Object).Count; $lines += "- ruff ($($f.BaseName.Replace('-ruff',''))): issues=$count ([report]($(Resolve-Path $f.FullName)))" } catch { $lines += "- ruff ($($f.BaseName.Replace('-ruff',''))): parse error" }
        }
        $vulTxt = Get-ChildItem (Join-Path $scanDir 'python') -Filter '*-vulture.txt' -ErrorAction SilentlyContinue
        foreach ($f in $vulTxt) { $lines += "- vulture ($($f.BaseName.Replace('-vulture',''))): see [report]($(Resolve-Path $f.FullName))" }
        $lines += ""
    }

    # Duplication summary
    $jscpdJson = Join-Path $scanDir 'duplication/jscpd-report.json'
    if (Test-Path $jscpdJson) {
        try {
            $j = Get-Content -Raw -Path $jscpdJson | ConvertFrom-Json
            $dupCount = ($j.duplicates | Measure-Object).Count
            $pct = if ($j.statistics -and $j.statistics.total) { [math]::Round([double]$j.statistics.total.percentage,2) } else { 0 }
            $lines += "## Duplication"
            $lines += "- clones: $dupCount, percentage: $pct% ([html]($(Resolve-Path (Join-Path $scanDir 'duplication/index.html'))), [json]($(Resolve-Path $jscpdJson)))"
            $lines += ""
        } catch { $lines += "## Duplication`n- jscpd: parse error" }
    }

    $lines | Out-File -Encoding utf8 -FilePath $summaryPath
    # Also write a latest pointer for convenience
    Copy-Item -Force $summaryPath (Join-Path $scanRoot 'latest.md')

    Write-Info "Done. See summary: $summaryPath"
} finally {
    Pop-Location
}


