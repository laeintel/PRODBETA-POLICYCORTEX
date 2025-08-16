# PolicyCortex Scripts

## Directory Structure

### `/runtime/`
Core runtime scripts for starting and managing the application:
- `start-dev.bat` - Start development environment (Windows)
- `start-local.bat` - Start with Docker Compose (Windows)
- `start-local.sh` - Start with Docker Compose (Linux/Mac)
- `start-production.bat` - Start production environment

### `/testing/`
Testing and validation scripts:
- `test-all-windows.bat` - Complete test suite for Windows
- `test-all-linux.sh` - Complete test suite for Linux/Mac
- `test-workflow.sh` - Test API endpoints and workflows
- `performance-tests.js` - K6 performance testing

### `/cli-tools/`
Command-line utilities for specific features:
- `azure-sync-cli.py` - Azure policy synchronization
- `soc2-compliance-cli.py` - SOC2 compliance checks
- `usage-meter-cli.py` - Usage metering and billing
- `what-if-cli.py` - What-if analysis for policies

### `/migrations/`
Database migration scripts (executed automatically):
- SQL migration files for schema updates

### `/azure-policy-sync/`
Azure policy synchronization components:
- `auto_refresh.py` - Automatic policy refresh

### `/disaster-recovery/`
Backup and restore utilities:
- `backup.sh` - Create system backups
- `restore.sh` - Restore from backups

### Root Scripts
- `run-autonomous.ps1` - Autonomous execution script for AI-driven operations
- `seed-data.bat/sh/sql` - Database initialization scripts
- `init.sql` - Initial database setup

## Usage

### Starting the Application
```bash
# Windows Development
.\scripts\runtime\start-dev.bat

# Windows Docker
.\scripts\runtime\start-local.bat

# Linux/Mac Docker
./scripts/runtime/start-local.sh

# Production
.\scripts\runtime\start-production.bat
```

### Running Tests
```bash
# Windows
.\scripts\testing\test-all-windows.bat

# Linux/Mac
./scripts/testing/test-all-linux.sh

# Performance Testing
k6 run scripts/testing/performance-tests.js
```

### Using CLI Tools
```bash
# Azure Sync
python scripts/cli-tools/azure-sync-cli.py

# SOC2 Compliance Check
python scripts/cli-tools/soc2-compliance-cli.py

# Usage Metering
python scripts/cli-tools/usage-meter-cli.py

# What-If Analysis
python scripts/cli-tools/what-if-cli.py
```

## Notes
- All non-essential setup, installation, and utility scripts have been removed
- Scripts are organized by function for better maintainability
- Use environment variables or .env files for configuration instead of set-env scripts