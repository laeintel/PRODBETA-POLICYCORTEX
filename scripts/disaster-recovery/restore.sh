#!/bin/bash

# PolicyCortex Disaster Recovery - Restore Script
# Restores system from backup with verification and rollback capabilities

set -e

# Configuration
BACKUP_DIR=${BACKUP_DIR:-"/backup/policycortex"}
RESTORE_POINT=${1:-"latest"}

# Azure Storage Configuration
AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT:-"policycortexbackup"}
AZURE_CONTAINER_NAME=${AZURE_CONTAINER_NAME:-"backups"}

# Database Configuration
DB_HOST=${DB_HOST:-"localhost"}
DB_PORT=${DB_PORT:-"5432"}
DB_NAME=${DB_NAME:-"policycortex"}
DB_USER=${DB_USER:-"postgres"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Pre-restore checks
pre_restore_checks() {
    log "Performing pre-restore checks..."
    
    # Check if services are stopped
    if systemctl is-active --quiet policycortex; then
        error "PolicyCortex services are still running. Please stop them before restore."
    fi
    
    # Check disk space
    REQUIRED_SPACE=10240  # 10GB in MB
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    
    if [ "${AVAILABLE_SPACE}" -lt "${REQUIRED_SPACE}" ]; then
        error "Insufficient disk space. Required: ${REQUIRED_SPACE}MB, Available: ${AVAILABLE_SPACE}MB"
    fi
    
    log "Pre-restore checks passed"
}

# Find backup to restore
find_backup() {
    if [ "${RESTORE_POINT}" == "latest" ]; then
        # Find the latest backup
        BACKUP_FILE=$(ls -t ${BACKUP_DIR}/policycortex_backup_*.tar.gz 2>/dev/null | head -1)
        
        if [ -z "${BACKUP_FILE}" ]; then
            # Try to download from Azure
            log "No local backup found. Downloading from Azure..."
            download_from_azure
        fi
    else
        # Use specified backup
        BACKUP_FILE="${BACKUP_DIR}/${RESTORE_POINT}"
        
        if [ ! -f "${BACKUP_FILE}" ]; then
            error "Backup file not found: ${BACKUP_FILE}"
        fi
    fi
    
    log "Using backup: ${BACKUP_FILE}"
}

# Download backup from Azure
download_from_azure() {
    if ! command -v az &> /dev/null; then
        error "Azure CLI not found. Cannot download backup from Azure."
    fi
    
    # Get latest backup from Azure
    LATEST_BLOB=$(az storage blob list \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --container-name "${AZURE_CONTAINER_NAME}" \
        --query "sort_by([?starts_with(name, 'policycortex_backup_')], &properties.lastModified)[-1].name" \
        -o tsv)
    
    if [ -z "${LATEST_BLOB}" ]; then
        error "No backup found in Azure Storage"
    fi
    
    BACKUP_FILE="${BACKUP_DIR}/${LATEST_BLOB}"
    
    log "Downloading ${LATEST_BLOB} from Azure..."
    az storage blob download \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --container-name "${AZURE_CONTAINER_NAME}" \
        --name "${LATEST_BLOB}" \
        --file "${BACKUP_FILE}"
}

# Extract backup
extract_backup() {
    log "Extracting backup..."
    
    TEMP_DIR=$(mktemp -d)
    tar -xzf "${BACKUP_FILE}" -C "${TEMP_DIR}"
    
    # Find the extracted directory
    BACKUP_NAME=$(basename "${BACKUP_FILE}" .tar.gz)
    EXTRACT_PATH="${TEMP_DIR}/${BACKUP_NAME}"
    
    if [ ! -d "${EXTRACT_PATH}" ]; then
        error "Backup extraction failed"
    fi
    
    # Verify checksums
    if [ -f "${EXTRACT_PATH}/checksums.sha256" ]; then
        log "Verifying backup integrity..."
        cd "${EXTRACT_PATH}"
        sha256sum -c checksums.sha256 || error "Backup integrity check failed"
        cd - > /dev/null
    fi
    
    echo "${EXTRACT_PATH}"
}

# Create restore point before making changes
create_restore_point() {
    log "Creating restore point..."
    
    RESTORE_POINT_DIR="/backup/restore_points/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${RESTORE_POINT_DIR}"
    
    # Backup current database
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -F custom \
        -f "${RESTORE_POINT_DIR}/database_current.dump" 2>/dev/null || true
    
    # Backup current Redis
    redis-cli --rdb "${RESTORE_POINT_DIR}/redis_current.rdb" 2>/dev/null || true
    
    log "Restore point created at ${RESTORE_POINT_DIR}"
}

# Restore database
restore_database() {
    local EXTRACT_PATH=$1
    
    log "Restoring PostgreSQL database..."
    
    # Drop existing database and recreate
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "CREATE DATABASE ${DB_NAME};"
    
    # Restore from backup
    PGPASSWORD="${DB_PASSWORD}" pg_restore \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -v \
        "${EXTRACT_PATH}/database.dump" || error "Database restore failed"
    
    log "Database restored successfully"
}

# Restore EventStore
restore_eventstore() {
    local EXTRACT_PATH=$1
    
    if [ -d "${EXTRACT_PATH}/eventstore" ]; then
        log "Restoring EventStore..."
        
        # Stop EventStore
        systemctl stop eventstore 2>/dev/null || true
        
        # Restore data
        rm -rf /var/lib/eventstore/*
        cp -r "${EXTRACT_PATH}/eventstore/"* /var/lib/eventstore/
        
        # Start EventStore
        systemctl start eventstore
        
        log "EventStore restored successfully"
    else
        warning "EventStore backup not found in backup"
    fi
}

# Restore Redis
restore_redis() {
    local EXTRACT_PATH=$1
    
    if [ -f "${EXTRACT_PATH}/redis.rdb" ]; then
        log "Restoring Redis/DragonflyDB..."
        
        # Stop Redis
        systemctl stop redis 2>/dev/null || true
        
        # Restore dump file
        cp "${EXTRACT_PATH}/redis.rdb" /var/lib/redis/dump.rdb
        chown redis:redis /var/lib/redis/dump.rdb
        
        # Start Redis
        systemctl start redis
        
        log "Redis restored successfully"
    else
        warning "Redis backup not found"
    fi
}

# Restore configuration
restore_configuration() {
    local EXTRACT_PATH=$1
    
    log "Restoring configuration files..."
    
    # Restore environment files
    if [ -d "${EXTRACT_PATH}/config" ]; then
        cp "${EXTRACT_PATH}/config/.env"* . 2>/dev/null || true
        cp "${EXTRACT_PATH}/config/docker-compose"*.yml . 2>/dev/null || true
    fi
    
    # Restore Kubernetes manifests
    if [ -d "${EXTRACT_PATH}/config/kubernetes" ]; then
        cp -r "${EXTRACT_PATH}/config/kubernetes" infrastructure/
    fi
    
    log "Configuration restored"
}

# Restore security artifacts
restore_security() {
    local EXTRACT_PATH=$1
    
    log "Restoring security artifacts..."
    
    # Restore SSL certificates
    if [ -d "${EXTRACT_PATH}/security/ssl" ]; then
        cp -r "${EXTRACT_PATH}/security/ssl" .
    fi
    
    # Restore Key Vault secrets
    if command -v az &> /dev/null && [ -d "${EXTRACT_PATH}/security" ]; then
        KV_NAME=${KEY_VAULT_NAME:-"policycortex-kv"}
        
        for secret_file in "${EXTRACT_PATH}/security/"*.json; do
            if [ -f "${secret_file}" ]; then
                SECRET_NAME=$(jq -r '.name' "${secret_file}")
                SECRET_VALUE=$(jq -r '.value' "${secret_file}")
                
                az keyvault secret set \
                    --vault-name "${KV_NAME}" \
                    --name "${SECRET_NAME}" \
                    --value "${SECRET_VALUE}" || warning "Failed to restore secret: ${SECRET_NAME}"
            fi
        done
    fi
    
    log "Security artifacts restored"
}

# Verify restoration
verify_restoration() {
    log "Verifying restoration..."
    
    # Check database connectivity
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -c "SELECT COUNT(*) FROM events;" > /dev/null || error "Database verification failed"
    
    # Check Redis
    redis-cli ping > /dev/null || warning "Redis verification failed"
    
    log "Restoration verification completed"
}

# Main restore process
main() {
    info "PolicyCortex Disaster Recovery - Restore Process"
    info "================================================"
    
    # Perform pre-restore checks
    pre_restore_checks
    
    # Find backup to restore
    find_backup
    
    # Extract backup
    EXTRACT_PATH=$(extract_backup)
    
    # Show backup metadata
    if [ -f "${EXTRACT_PATH}/metadata.json" ]; then
        info "Backup Information:"
        cat "${EXTRACT_PATH}/metadata.json" | jq '.'
    fi
    
    # Confirm restoration
    read -p "Do you want to proceed with restoration? (yes/no): " CONFIRM
    if [ "${CONFIRM}" != "yes" ]; then
        error "Restoration cancelled by user"
    fi
    
    # Create restore point
    create_restore_point
    
    # Restore components
    restore_database "${EXTRACT_PATH}"
    restore_eventstore "${EXTRACT_PATH}"
    restore_redis "${EXTRACT_PATH}"
    restore_configuration "${EXTRACT_PATH}"
    restore_security "${EXTRACT_PATH}"
    
    # Verify restoration
    verify_restoration
    
    # Clean up
    rm -rf "${EXTRACT_PATH}"
    
    log "Restoration completed successfully!"
    info "Please restart PolicyCortex services: systemctl start policycortex"
}

# Run main function
main