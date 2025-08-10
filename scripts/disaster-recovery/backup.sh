#!/bin/bash

# PolicyCortex Disaster Recovery - Backup Script
# Performs comprehensive backup of all critical system components

set -e

# Configuration
BACKUP_DIR=${BACKUP_DIR:-"/backup/policycortex"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="policycortex_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

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

# Create backup directory
mkdir -p "${BACKUP_PATH}"

log "Starting PolicyCortex backup to ${BACKUP_PATH}"

# 1. Backup PostgreSQL Database
log "Backing up PostgreSQL database..."
PGPASSWORD="${DB_PASSWORD}" pg_dump \
    -h "${DB_HOST}" \
    -p "${DB_PORT}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    -F custom \
    -b \
    -v \
    -f "${BACKUP_PATH}/database.dump" || error "Database backup failed"

# Also create SQL script for manual recovery
PGPASSWORD="${DB_PASSWORD}" pg_dump \
    -h "${DB_HOST}" \
    -p "${DB_PORT}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    --schema-only \
    -f "${BACKUP_PATH}/schema.sql"

log "Database backup completed"

# 2. Backup EventStore
log "Backing up EventStore..."
if command -v eventstoredb-admin &> /dev/null; then
    eventstoredb-admin backup \
        --source="esdb://localhost:2113?tls=false" \
        --destination="${BACKUP_PATH}/eventstore" || warning "EventStore backup failed"
else
    warning "EventStore admin tool not found, skipping EventStore backup"
fi

# 3. Backup Redis/DragonflyDB
log "Backing up Redis/DragonflyDB..."
redis-cli --rdb "${BACKUP_PATH}/redis.rdb" || warning "Redis backup failed"

# 4. Backup configuration files
log "Backing up configuration files..."
mkdir -p "${BACKUP_PATH}/config"

# Copy environment files
cp .env* "${BACKUP_PATH}/config/" 2>/dev/null || true
cp docker-compose*.yml "${BACKUP_PATH}/config/" 2>/dev/null || true

# Copy Kubernetes manifests if they exist
if [ -d "infrastructure/kubernetes" ]; then
    cp -r infrastructure/kubernetes "${BACKUP_PATH}/config/"
fi

# Copy Terraform state if it exists
if [ -d "infrastructure/terraform" ]; then
    mkdir -p "${BACKUP_PATH}/terraform"
    find infrastructure/terraform -name "*.tfstate*" -exec cp {} "${BACKUP_PATH}/terraform/" \;
fi

# 5. Backup encryption keys and certificates
log "Backing up security artifacts..."
mkdir -p "${BACKUP_PATH}/security"

# Backup SSL certificates
if [ -d "ssl" ]; then
    cp -r ssl "${BACKUP_PATH}/security/"
fi

# Export Key Vault secrets (if Azure CLI is available)
if command -v az &> /dev/null; then
    log "Exporting Azure Key Vault secrets..."
    KV_NAME=${KEY_VAULT_NAME:-"policycortex-kv"}
    
    az keyvault secret list --vault-name "${KV_NAME}" --query "[].name" -o tsv | while read -r secret; do
        az keyvault secret show --vault-name "${KV_NAME}" --name "${secret}" \
            --query "{name:name, value:value}" -o json > "${BACKUP_PATH}/security/${secret}.json"
    done
fi

# 6. Create metadata file
log "Creating backup metadata..."
cat > "${BACKUP_PATH}/metadata.json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "2.0.0",
    "hostname": "$(hostname)",
    "components": {
        "database": true,
        "eventstore": $([ -d "${BACKUP_PATH}/eventstore" ] && echo "true" || echo "false"),
        "redis": $([ -f "${BACKUP_PATH}/redis.rdb" ] && echo "true" || echo "false"),
        "configuration": true,
        "security": true
    },
    "environment": "${ENVIRONMENT:-production}",
    "backup_size": "$(du -sh ${BACKUP_PATH} | cut -f1)"
}
EOF

# 7. Create checksum for verification
log "Creating checksums..."
find "${BACKUP_PATH}" -type f -exec sha256sum {} \; > "${BACKUP_PATH}/checksums.sha256"

# 8. Compress backup
log "Compressing backup..."
tar -czf "${BACKUP_PATH}.tar.gz" -C "${BACKUP_DIR}" "${BACKUP_NAME}"

# 9. Upload to Azure Blob Storage
if command -v az &> /dev/null; then
    log "Uploading backup to Azure Blob Storage..."
    
    # Upload with encryption
    az storage blob upload \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --container-name "${AZURE_CONTAINER_NAME}" \
        --name "${BACKUP_NAME}.tar.gz" \
        --file "${BACKUP_PATH}.tar.gz" \
        --encryption-scope "policycortex-encryption" \
        --metadata "timestamp=${TIMESTAMP}" "environment=${ENVIRONMENT:-production}" || warning "Azure upload failed"
    
    # Also upload to geo-redundant location
    AZURE_STORAGE_ACCOUNT_DR=${AZURE_STORAGE_ACCOUNT_DR:-"policycortexbackupdr"}
    az storage blob upload \
        --account-name "${AZURE_STORAGE_ACCOUNT_DR}" \
        --container-name "${AZURE_CONTAINER_NAME}" \
        --name "${BACKUP_NAME}.tar.gz" \
        --file "${BACKUP_PATH}.tar.gz" \
        --encryption-scope "policycortex-encryption" \
        --metadata "timestamp=${TIMESTAMP}" "environment=${ENVIRONMENT:-production}" || warning "DR Azure upload failed"
fi

# 10. Clean up old backups (keep last 30 days)
log "Cleaning up old backups..."
find "${BACKUP_DIR}" -name "policycortex_backup_*.tar.gz" -mtime +30 -delete

# 11. Verify backup
log "Verifying backup integrity..."
tar -tzf "${BACKUP_PATH}.tar.gz" > /dev/null || error "Backup verification failed"

# 12. Send notification
if [ -n "${WEBHOOK_URL}" ]; then
    curl -X POST "${WEBHOOK_URL}" \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": \"PolicyCortex backup completed successfully\",
            \"backup_name\": \"${BACKUP_NAME}\",
            \"size\": \"$(du -sh ${BACKUP_PATH}.tar.gz | cut -f1)\",
            \"timestamp\": \"${TIMESTAMP}\"
        }"
fi

log "Backup completed successfully: ${BACKUP_PATH}.tar.gz"

# Output backup information
echo "----------------------------------------"
echo "Backup Summary:"
echo "  Name: ${BACKUP_NAME}"
echo "  Path: ${BACKUP_PATH}.tar.gz"
echo "  Size: $(du -sh ${BACKUP_PATH}.tar.gz | cut -f1)"
echo "  Checksum: $(sha256sum ${BACKUP_PATH}.tar.gz | cut -d' ' -f1)"
echo "----------------------------------------"