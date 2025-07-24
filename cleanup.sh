#!/bin/bash

# Clean up script for PolicyCortex repository
echo "Cleaning up unnecessary files..."

# Remove Python cache files and directories
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name ".coverage" -delete
find . -type f -name "coverage.xml" -delete

# Remove temporary and backup files
find . -type f -name "*.bak" -delete
find . -type f -name "*.tmp" -delete
find . -type f -name "*.swp" -delete
find . -type f -name "*~" -delete
find . -type f -name ".DS_Store" -delete
find . -type f -name "Thumbs.db" -delete

# Remove log files (except important ones)
find . -type f -name "*.log" -not -path "./infrastructure/*" -delete

# Remove build directories (except node_modules and venv)
find . -type d -name "dist" -not -path "./frontend/dist" -not -path "./node_modules/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "build" -not -path "./node_modules/*" -exec rm -rf {} + 2>/dev/null || true

# Remove development test files
find . -type f -name "main_simple.py" -delete
find . -type f -name "main_local.py" -delete
find . -type f -name "test_simple.py" -delete

# Remove empty directories
find . -type d -empty -delete 2>/dev/null || true

echo "Cleanup completed!"