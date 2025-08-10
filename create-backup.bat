@echo off
echo Creating PolicyCortex Enterprise Backup...

:: Set timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%-%dt:~8,6%"

:: Create backup directory if not exists
if not exist "C:\CODEBACKUP-V1" mkdir "C:\CODEBACKUP-V1"

:: Create subdirectory for this backup
set "backup_dir=C:\CODEBACKUP-V1\policycortex-enterprise-%timestamp%"
mkdir "%backup_dir%"

:: Copy all source files (excluding build artifacts)
echo Copying source files...
xcopy /E /I /H /Y "backend" "%backup_dir%\backend" /EXCLUDE:backup-exclude.txt
xcopy /E /I /H /Y "core" "%backup_dir%\core" /EXCLUDE:backup-exclude.txt
xcopy /E /I /H /Y "frontend" "%backup_dir%\frontend" /EXCLUDE:backup-exclude.txt
xcopy /E /I /H /Y "graphql" "%backup_dir%\graphql" /EXCLUDE:backup-exclude.txt
xcopy /E /I /H /Y "edge" "%backup_dir%\edge" /EXCLUDE:backup-exclude.txt
xcopy /E /I /H /Y "training" "%backup_dir%\training" /EXCLUDE:backup-exclude.txt
xcopy /E /I /H /Y "scripts" "%backup_dir%\scripts"

:: Copy important root files
copy "*.md" "%backup_dir%\"
copy "*.json" "%backup_dir%\"
copy "*.toml" "%backup_dir%\"
copy "*.yaml" "%backup_dir%\"
copy "*.yml" "%backup_dir%\"
copy "*.bat" "%backup_dir%\"
copy "*.sh" "%backup_dir%\"
copy "docker-compose*.yml" "%backup_dir%\"
copy "Dockerfile*" "%backup_dir%\"

:: Create backup info file
echo PolicyCortex Enterprise Backup > "%backup_dir%\BACKUP_INFO.txt"
echo Created: %date% %time% >> "%backup_dir%\BACKUP_INFO.txt"
echo Version: 3.0.0 (80 Fixes Applied) >> "%backup_dir%\BACKUP_INFO.txt"
echo Git Commit: 807290f >> "%backup_dir%\BACKUP_INFO.txt"
echo Features: >> "%backup_dir%\BACKUP_INFO.txt"
echo - Enterprise Security (JWT, RBAC, Tenant Isolation) >> "%backup_dir%\BACKUP_INFO.txt"
echo - Multi-Cloud Support (AWS, GCP, Azure) >> "%backup_dir%\BACKUP_INFO.txt"
echo - Event Sourcing and Audit Trail >> "%backup_dir%\BACKUP_INFO.txt"
echo - WebSocket Real-time Updates >> "%backup_dir%\BACKUP_INFO.txt"
echo - Observability (OpenTelemetry, Prometheus) >> "%backup_dir%\BACKUP_INFO.txt"
echo - Multi-tier Caching >> "%backup_dir%\BACKUP_INFO.txt"
echo - Rate Limiting and Circuit Breakers >> "%backup_dir%\BACKUP_INFO.txt"

:: Create a zip archive as well
echo Creating ZIP archive...
powershell Compress-Archive -Path "%backup_dir%" -DestinationPath "C:\CODEBACKUP-V1\policycortex-enterprise-%timestamp%.zip" -CompressionLevel Optimal

echo.
echo Backup completed successfully!
echo Location: %backup_dir%
echo Archive: C:\CODEBACKUP-V1\policycortex-enterprise-%timestamp%.zip
echo.