@echo off
REM PolicyCortex Master Setup Script for Windows
REM This script sets up the complete PolicyCortex environment
REM Run as Administrator for best results

setlocal enabledelayedexpansion

echo =============================================================
echo PolicyCortex Master Setup Script v2.0
echo Setting up complete PolicyCortex environment...
echo =============================================================
echo.

REM Set error handling
set SETUP_DIR=%~dp0
set PROJECT_ROOT=%SETUP_DIR%\..\..
set ERRORS_OCCURRED=0

REM Jump to main execution
goto :main

REM Function to log messages
:log_info
echo [INFO] %~1
goto :eof

:log_error
echo [ERROR] %~1
set ERRORS_OCCURRED=1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:check_prerequisites
echo.
call :log_info "Checking prerequisites..."

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "This script should be run as Administrator for best results"
)

REM Check Docker
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Docker is not installed or not in PATH"
    call :log_info "Please install Docker Desktop from https://docker.com/get-started"
    goto :error_exit
)

REM Check Docker Compose
docker compose version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Docker Compose is not available"
    call :log_info "Please ensure Docker Compose is installed"
    goto :error_exit
)

REM Check Azure CLI
az version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Azure CLI is not installed"
    call :log_info "Please install Azure CLI from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    goto :error_exit
)

REM Check GitHub CLI
gh version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "GitHub CLI is not installed"
    call :log_info "Please install GitHub CLI from https://cli.github.com/"
    goto :error_exit
)

REM Check Node.js
node --version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Node.js is not installed"
    call :log_info "Please install Node.js from https://nodejs.org/"
    goto :error_exit
)

REM Check Rust
cargo --version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Rust is not installed"
    call :log_info "Please install Rust from https://rustup.rs/"
    goto :error_exit
)

REM Check Python
python --version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Python is not installed"
    call :log_info "Please install Python from https://python.org/"
    goto :error_exit
)

call :log_success "All prerequisites are installed"
goto :eof

:setup_environment_files
echo.
call :log_info "Setting up environment configuration files..."

if not exist "%PROJECT_ROOT%\.env.development" (
    call :log_info "Creating development environment file..."
    copy "%SETUP_DIR%\.env.development.template" "%PROJECT_ROOT%\.env.development" >nul 2>&1
    if %errorLevel% equ 0 (
        call :log_success "Development environment file created"
    ) else (
        call :log_error "Failed to create development environment file"
    )
) else (
    call :log_info "Development environment file already exists"
)

if not exist "%PROJECT_ROOT%\.env.production" (
    call :log_info "Creating production environment file..."
    copy "%SETUP_DIR%\.env.production.template" "%PROJECT_ROOT%\.env.production" >nul 2>&1
    if %errorLevel% equ 0 (
        call :log_success "Production environment file created"
    ) else (
        call :log_error "Failed to create production environment file"
    )
) else (
    call :log_info "Production environment file already exists"
)

goto :eof

:setup_docker_services
echo.
call :log_info "Setting up Docker services..."

cd /d "%PROJECT_ROOT%"

REM Stop any existing services
call :log_info "Stopping existing Docker services..."
docker compose -f scripts\setup\docker-services.yml down >nul 2>&1

REM Pull latest images
call :log_info "Pulling latest Docker images..."
docker compose -f scripts\setup\docker-services.yml pull
if %errorLevel% neq 0 (
    call :log_error "Failed to pull Docker images"
    goto :eof
)

REM Start infrastructure services
call :log_info "Starting infrastructure services..."
docker compose -f scripts\setup\docker-services.yml up -d postgres redis eventstore nats
if %errorLevel% neq 0 (
    call :log_error "Failed to start infrastructure services"
    goto :eof
)

REM Wait for services to be ready
call :log_info "Waiting for services to be ready..."
timeout /t 30 /nobreak >nul

REM Check service health
call :log_info "Checking service health..."
docker compose -f scripts\setup\docker-services.yml ps

call :log_success "Docker services are running"
goto :eof

:setup_database
echo.
call :log_info "Setting up database..."

REM Run database initialization
call :log_info "Running database initialization..."
docker exec policycortex-postgres psql -U postgres -d policycortex -f /docker-entrypoint-initdb.d/init-db.sql
if %errorLevel% neq 0 (
    call :log_error "Failed to initialize database"
    goto :eof
)

REM Run migrations
call :log_info "Running database migrations..."
cd /d "%PROJECT_ROOT%\scripts\migrations"
for %%f in (*.sql) do (
    call :log_info "Running migration: %%f"
    docker exec policycortex-postgres psql -U postgres -d policycortex -f /scripts/migrations/%%f
)

call :log_success "Database setup complete"
goto :eof

:setup_azure_resources
echo.
call :log_info "Setting up Azure resources..."

REM Check Azure login
az account show >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Please log in to Azure..."
    az login
    if %errorLevel% neq 0 (
        call :log_error "Azure login failed"
        goto :eof
    )
)

REM Run Azure setup script
call :log_info "Running Azure resource setup..."
call "%SETUP_DIR%\azure-setup.bat"
if %errorLevel% neq 0 (
    call :log_error "Azure setup failed"
    goto :eof
)

call :log_success "Azure resources configured"
goto :eof

:setup_github_secrets
echo.
call :log_info "Setting up GitHub repository secrets..."

REM Check GitHub authentication
gh auth status >nul 2>&1
if %errorLevel% neq 0 (
    call :log_info "Please authenticate with GitHub..."
    gh auth login
    if %errorLevel% neq 0 (
        call :log_error "GitHub authentication failed"
        goto :eof
    )
)

REM Run GitHub setup script
call :log_info "Configuring GitHub secrets and variables..."
call "%SETUP_DIR%\github-setup.bat"
if %errorLevel% neq 0 (
    call :log_error "GitHub setup failed"
    goto :eof
)

call :log_success "GitHub secrets configured"
goto :eof

:setup_application_dependencies
echo.
call :log_info "Setting up application dependencies..."

cd /d "%PROJECT_ROOT%"

REM Install frontend dependencies
call :log_info "Installing frontend dependencies..."
cd frontend
call npm install
if %errorLevel% neq 0 (
    call :log_error "Failed to install frontend dependencies"
    cd /d "%PROJECT_ROOT%"
    goto :eof
)
cd ..

REM Install GraphQL dependencies
call :log_info "Installing GraphQL dependencies..."
cd graphql
call npm install
if %errorLevel% neq 0 (
    call :log_error "Failed to install GraphQL dependencies"
    cd /d "%PROJECT_ROOT%"
    goto :eof
)
cd ..

REM Build Rust backend
call :log_info "Building Rust backend..."
cd core
call cargo build --release
if %errorLevel% neq 0 (
    call :log_error "Failed to build Rust backend"
    cd /d "%PROJECT_ROOT%"
    goto :eof
)
cd ..

call :log_success "Application dependencies installed"
goto :eof

:setup_ml_models
echo.
call :log_info "Setting up ML models and AI services..."

REM Install Python dependencies
call :log_info "Installing Python ML dependencies..."
pip install -r requirements-ml.txt
if %errorLevel% neq 0 (
    call :log_error "Failed to install Python ML dependencies"
    goto :eof
)

REM Initialize ML models
call :log_info "Initializing ML models..."
python -c "import backend.services.ml_models.train_models; print('ML models initialized')"
if %errorLevel% neq 0 (
    call :log_error "Failed to initialize ML models"
    goto :eof
)

call :log_success "ML models configured"
goto :eof

:setup_monitoring
echo.
call :log_info "Setting up monitoring stack..."

REM Run monitoring setup script
call "%SETUP_DIR%\setup-monitoring.bat"
if %errorLevel% neq 0 (
    call :log_error "Monitoring setup failed"
    goto :eof
)

call :log_success "Monitoring stack configured"
goto :eof

:run_smoke_tests
echo.
call :log_info "Running smoke tests..."

REM Test Docker services
call :log_info "Testing Docker services..."
docker compose -f scripts\setup\docker-services.yml ps | findstr "Up"
if %errorLevel% neq 0 (
    call :log_error "Some Docker services are not running"
    goto :eof
)

REM Test database connection
call :log_info "Testing database connection..."
docker exec policycortex-postgres pg_isready -U postgres
if %errorLevel% neq 0 (
    call :log_error "Database connection test failed"
    goto :eof
)

REM Test Redis connection
call :log_info "Testing Redis connection..."
docker exec policycortex-redis redis-cli ping | findstr "PONG"
if %errorLevel% neq 0 (
    call :log_error "Redis connection test failed"
    goto :eof
)

call :log_success "All smoke tests passed"
goto :eof

:cleanup_on_error
echo.
call :log_info "Cleaning up due to errors..."
docker compose -f scripts\setup\docker-services.yml down >nul 2>&1
goto :eof

:error_exit
echo.
call :log_error "Setup failed. Please check the errors above and try again."
call :cleanup_on_error
exit /b 1

:main
REM Main execution flow
call :check_prerequisites
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_environment_files
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_docker_services
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_database
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_azure_resources
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_github_secrets
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_application_dependencies
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_ml_models
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :setup_monitoring
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :run_smoke_tests
if !ERRORS_OCCURRED! equ 1 goto :error_exit

echo.
echo =============================================================
call :log_success "PolicyCortex setup completed successfully!"
echo =============================================================
echo.
echo Next steps:
echo 1. Review and update environment variables in .env files
echo 2. Configure Azure AD application permissions
echo 3. Update GitHub secrets with your specific values
echo 4. Run: docker compose up -d to start all services
echo 5. Access the application at: http://localhost:3005
echo.
echo Documentation:
echo - Frontend: http://localhost:3005
echo - Backend API: http://localhost:8085
echo - GraphQL: http://localhost:4001/graphql
echo - EventStore UI: http://localhost:2113 (admin/changeit)
echo - Database UI: http://localhost:8081 (user: postgres, pass: postgres)
echo.

REM End of main execution
exit /b 0