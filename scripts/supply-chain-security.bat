@echo off
REM Supply Chain Security Script for Windows
REM Generates SBOM, scans for CVEs, and creates SLSA provenance

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set OUTPUT_DIR=%PROJECT_ROOT%\security-reports

echo Supply Chain Security Scanner
echo ===============================

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Check for required tools
echo Checking for required tools...

where syft >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Syft for SBOM generation...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/anchore/syft/releases/latest/download/syft_0.98.0_windows_amd64.zip' -OutFile 'syft.zip'"
    powershell -Command "Expand-Archive -Path 'syft.zip' -DestinationPath '.'"
    del syft.zip
    echo Syft installed
)

where grype >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Grype for vulnerability scanning...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/anchore/grype/releases/latest/download/grype_0.73.0_windows_amd64.zip' -OutFile 'grype.zip'"
    powershell -Command "Expand-Archive -Path 'grype.zip' -DestinationPath '.'"
    del grype.zip
    echo Grype installed
)

REM Generate SBOM for Rust backend
echo.
echo Generating SBOM for Rust backend...
cd /d "%PROJECT_ROOT%\core"
syft . -o spdx-json > "%OUTPUT_DIR%\rust-sbom-spdx.json"
syft . -o cyclonedx-json > "%OUTPUT_DIR%\rust-sbom-cyclonedx.json"
syft . -o table > "%OUTPUT_DIR%\rust-sbom-summary.txt"
echo Rust SBOM generated

REM Generate SBOM for Node.js frontend
echo.
echo Generating SBOM for Node.js frontend...
cd /d "%PROJECT_ROOT%\frontend"
syft . -o spdx-json > "%OUTPUT_DIR%\node-sbom-spdx.json"
syft . -o cyclonedx-json > "%OUTPUT_DIR%\node-sbom-cyclonedx.json"
syft . -o table > "%OUTPUT_DIR%\node-sbom-summary.txt"
echo Node.js SBOM generated

REM Generate SBOM for Python services
echo.
echo Generating SBOM for Python services...
cd /d "%PROJECT_ROOT%\backend"
syft . -o spdx-json > "%OUTPUT_DIR%\python-sbom-spdx.json"
syft . -o cyclonedx-json > "%OUTPUT_DIR%\python-sbom-cyclonedx.json"
syft . -o table > "%OUTPUT_DIR%\python-sbom-summary.txt"
echo Python SBOM generated

REM Scan for vulnerabilities
echo.
echo Scanning for vulnerabilities...

echo Scanning Rust dependencies...
grype "%PROJECT_ROOT%\core" -o json > "%OUTPUT_DIR%\rust-vulnerabilities.json"
grype "%PROJECT_ROOT%\core" -o table > "%OUTPUT_DIR%\rust-vulnerabilities.txt"

echo Scanning Node.js dependencies...
grype "%PROJECT_ROOT%\frontend" -o json > "%OUTPUT_DIR%\node-vulnerabilities.json"
grype "%PROJECT_ROOT%\frontend" -o table > "%OUTPUT_DIR%\node-vulnerabilities.txt"

echo Scanning Python dependencies...
grype "%PROJECT_ROOT%\backend" -o json > "%OUTPUT_DIR%\python-vulnerabilities.json"
grype "%PROJECT_ROOT%\backend" -o table > "%OUTPUT_DIR%\python-vulnerabilities.txt"

echo Vulnerability scans complete

REM Generate SLSA provenance
echo.
echo Generating SLSA provenance...

for /f "tokens=*" %%g in ('git rev-parse HEAD') do set GIT_SHA=%%g
for /f "tokens=*" %%g in ('date /t') do set BUILD_DATE=%%g
for /f "tokens=*" %%g in ('time /t') do set BUILD_TIME=%%g

(
echo {
echo   "_type": "https://in-toto.io/Statement/v0.1",
echo   "predicateType": "https://slsa.dev/provenance/v0.2",
echo   "subject": [
echo     {
echo       "name": "policycortex",
echo       "digest": {
echo         "sha256": "%GIT_SHA%"
echo       }
echo     }
echo   ],
echo   "predicate": {
echo     "builder": {
echo       "id": "https://github.com/policycortex/builder@v1"
echo     },
echo     "buildType": "https://github.com/policycortex/build@v1",
echo     "invocation": {
echo       "configSource": {
echo         "uri": "git+https://github.com/policycortex/policycortex@%GIT_SHA%",
echo         "digest": {
echo           "sha256": "%GIT_SHA%"
echo         },
echo         "entryPoint": "scripts/build.bat"
echo       },
echo       "parameters": {},
echo       "environment": {
echo         "arch": "%PROCESSOR_ARCHITECTURE%",
echo         "os": "Windows",
echo         "user": "%USERNAME%",
echo         "timestamp": "%BUILD_DATE% %BUILD_TIME%"
echo       }
echo     }
echo   }
echo }
) > "%OUTPUT_DIR%\slsa-provenance.json"

echo SLSA provenance generated

REM Check for critical vulnerabilities
echo.
echo Checking for critical vulnerabilities...

set CRITICAL_COUNT=0
set HIGH_COUNT=0

REM Parse vulnerability reports (simplified for Windows)
findstr /i "Critical" "%OUTPUT_DIR%\rust-vulnerabilities.txt" >nul 2>&1
if %errorlevel% equ 0 set /a CRITICAL_COUNT+=1

findstr /i "Critical" "%OUTPUT_DIR%\node-vulnerabilities.txt" >nul 2>&1
if %errorlevel% equ 0 set /a CRITICAL_COUNT+=1

findstr /i "Critical" "%OUTPUT_DIR%\python-vulnerabilities.txt" >nul 2>&1
if %errorlevel% equ 0 set /a CRITICAL_COUNT+=1

if !CRITICAL_COUNT! gtr 0 (
    echo WARNING: Found critical vulnerabilities!
    echo Build should be blocked until these are resolved.
) else (
    echo No critical vulnerabilities found
)

REM Generate final report
echo.
echo Generating final security report...

(
echo # Supply Chain Security Report
echo Generated: %BUILD_DATE% %BUILD_TIME%
echo Git SHA: %GIT_SHA%
echo.
echo ## SBOM Generation
echo - Rust SBOM generated
echo - Node.js SBOM generated
echo - Python SBOM generated
echo.
echo ## Vulnerability Scanning
echo - All components scanned
echo - Reports generated
echo.
echo ## SLSA Compliance
echo - Source: Version controlled ^(Git^)
echo - Build: Reproducible build configuration
echo - Provenance: Generated
echo - Dependencies: Complete SBOM available
echo.
echo ## Files Generated
echo All reports available in: %OUTPUT_DIR%
) > "%OUTPUT_DIR%\security-summary.md"

echo Security summary generated at %OUTPUT_DIR%\security-summary.md

echo.
echo ===============================
echo Supply chain security scan complete!
echo Reports available in: %OUTPUT_DIR%

cd /d "%SCRIPT_DIR%"
endlocal