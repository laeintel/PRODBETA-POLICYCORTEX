@echo off
REM ==================================================================
REM ML Docker Infrastructure Test Script for Windows
REM Tests CPU-only ML services on Windows without GPU support
REM ==================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

echo.
echo ========================================
echo ML Docker Infrastructure Test
echo ========================================
echo.

REM Color codes for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "RESET=[0m"

REM Test configuration
set "COMPOSE_FILE=docker-compose.ml-windows.yml"
set "COMPOSE_PROJECT_NAME=policycortex-ml"
set "MAX_WAIT_TIME=180"

REM Function to print colored messages
goto :main

:print_success
echo %GREEN%[SUCCESS]%RESET% %~1
exit /b

:print_error
echo %RED%[ERROR]%RESET% %~1
exit /b

:print_warning
echo %YELLOW%[WARNING]%RESET% %~1
exit /b

:print_info
echo [INFO] %~1
exit /b

:main
REM ==================================================================
REM 1. Pre-flight checks
REM ==================================================================
call :print_info "Running pre-flight checks..."

REM Check Docker
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "Docker is not installed or not in PATH"
    exit /b 1
)
call :print_success "Docker is installed"

REM Check Docker Compose
docker-compose --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "Docker Compose is not installed"
    exit /b 1
)
call :print_success "Docker Compose is installed"

REM Check if Docker daemon is running
docker ps >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "Docker daemon is not running. Please start Docker Desktop."
    exit /b 1
)
call :print_success "Docker daemon is running"

REM ==================================================================
REM 2. Clean up any existing containers
REM ==================================================================
call :print_info "Cleaning up existing containers..."
docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% down -v --remove-orphans >nul 2>&1
call :print_success "Cleanup completed"

REM ==================================================================
REM 3. Build ML Docker image
REM ==================================================================
call :print_info "Building ML Docker image (CPU-only)..."
docker build -f Dockerfile.ml-cpu -t policycortex-ml-cpu:latest .
if %ERRORLEVEL% neq 0 (
    call :print_error "Failed to build ML Docker image"
    exit /b 1
)
call :print_success "ML Docker image built successfully"

REM ==================================================================
REM 4. Start services
REM ==================================================================
call :print_info "Starting ML services..."
docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% up -d
if %ERRORLEVEL% neq 0 (
    call :print_error "Failed to start services"
    docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% logs
    exit /b 1
)
call :print_success "Services started"

REM ==================================================================
REM 5. Wait for services to be healthy
REM ==================================================================
call :print_info "Waiting for services to be healthy (max %MAX_WAIT_TIME% seconds)..."

set "services=postgres-ml redis-ml mlflow ml-prediction-server ml-websocket-server"
set /a "elapsed=0"

:health_check_loop
set "all_healthy=1"

for %%s in (%services%) do (
    docker inspect %%s --format="{{.State.Health.Status}}" 2>nul | findstr /i "healthy" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        docker inspect %%s --format="{{.State.Status}}" 2>nul | findstr /i "running" >nul 2>&1
        if !ERRORLEVEL! neq 0 (
            set "all_healthy=0"
        )
    )
)

if !all_healthy! equ 1 (
    call :print_success "All services are healthy"
    goto :test_services
)

timeout /t 5 /nobreak >nul
set /a "elapsed+=5"

if !elapsed! geq %MAX_WAIT_TIME% (
    call :print_error "Services failed to become healthy within %MAX_WAIT_TIME% seconds"
    docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% ps
    docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% logs --tail=50
    exit /b 1
)

goto :health_check_loop

:test_services
REM ==================================================================
REM 6. Test individual services
REM ==================================================================
call :print_info "Testing individual services..."

REM Test PostgreSQL
call :print_info "Testing PostgreSQL..."
docker exec postgres-ml pg_isready -U postgres >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "PostgreSQL is not ready"
    set "test_failed=1"
) else (
    call :print_success "PostgreSQL is ready"
)

REM Test Redis
call :print_info "Testing Redis..."
docker exec redis-ml redis-cli ping | findstr /i "PONG" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "Redis is not responding"
    set "test_failed=1"
) else (
    call :print_success "Redis is responding"
)

REM Test MLflow
call :print_info "Testing MLflow..."
curl -f -s http://localhost:5000/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_warning "MLflow health endpoint not accessible (this may be normal)"
) else (
    call :print_success "MLflow is accessible"
)

REM Test ML Prediction Server
call :print_info "Testing ML Prediction Server..."
timeout /t 10 /nobreak >nul
curl -f -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_warning "ML Prediction Server health check failed (may still be starting)"
    REM Try a simple connection test
    curl -s http://localhost:8080/ >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        call :print_error "ML Prediction Server is not accessible"
        set "test_failed=1"
    ) else (
        call :print_warning "ML Prediction Server is accessible but health check not implemented"
    )
) else (
    call :print_success "ML Prediction Server is healthy"
)

REM Test WebSocket Server
call :print_info "Testing WebSocket Server..."
powershell -Command "try { $tcp = New-Object System.Net.Sockets.TcpClient; $tcp.Connect('localhost', 8765); $tcp.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "WebSocket Server is not accessible on port 8765"
    set "test_failed=1"
) else (
    call :print_success "WebSocket Server is accessible on port 8765"
)

REM Test Prometheus metrics endpoint
call :print_info "Testing Prometheus metrics..."
curl -f -s http://localhost:9090/metrics >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_warning "Metrics endpoint not accessible (may not be implemented yet)"
) else (
    call :print_success "Metrics endpoint is accessible"
)

REM ==================================================================
REM 7. Display service status
REM ==================================================================
echo.
call :print_info "Service Status:"
docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% ps

REM ==================================================================
REM 8. Show service URLs
REM ==================================================================
echo.
echo ========================================
echo Service URLs:
echo ========================================
echo ML Prediction API:  http://localhost:8080
echo WebSocket Server:   ws://localhost:8765
echo MLflow UI:          http://localhost:5000
echo Metrics:            http://localhost:9090/metrics
echo PostgreSQL:         localhost:5432
echo Redis:              localhost:6379
echo.

REM ==================================================================
REM 9. Check for errors
REM ==================================================================
if defined test_failed (
    echo.
    call :print_warning "Some services failed tests. Showing recent logs..."
    echo.
    docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% logs --tail=20
    echo.
    call :print_info "To view full logs, run: docker-compose -f %COMPOSE_FILE% logs"
    call :print_info "To stop services, run: docker-compose -f %COMPOSE_FILE% down"
    exit /b 1
)

REM ==================================================================
REM 10. Success
REM ==================================================================
echo.
call :print_success "All ML services are running successfully!"
echo.
call :print_info "To view logs: docker-compose -f %COMPOSE_FILE% logs -f"
call :print_info "To stop services: docker-compose -f %COMPOSE_FILE% down"
call :print_info "To stop and remove volumes: docker-compose -f %COMPOSE_FILE% down -v"
echo.

REM Optional: Keep services running or stop them
choice /C YN /M "Do you want to keep the services running"
if %ERRORLEVEL% equ 2 (
    call :print_info "Stopping services..."
    docker-compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% down
    call :print_success "Services stopped"
)

exit /b 0