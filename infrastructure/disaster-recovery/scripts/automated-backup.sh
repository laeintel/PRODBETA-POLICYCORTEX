#!/bin/bash

# PolicyCortex Automated Backup System
# This script handles automated backups for all critical components

set -euo pipefail

# Configuration
BACKUP_ROOT="/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"
LOG_FILE="${BACKUP_ROOT}/backup_${TIMESTAMP}.log"
RETENTION_DAYS=30
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"

# Azure Storage Configuration
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-policycortexbackups}"
STORAGE_CONTAINER="${AZURE_STORAGE_CONTAINER:-backups}"
STORAGE_KEY="${AZURE_STORAGE_KEY:-}"

# Database Configuration
DB_HOST="${DB_HOST:-postgresql.policycortex.svc.cluster.local}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-policycortex}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Error handling
error_handler() {
    log "${RED}ERROR: Backup failed at line $1${NC}"
    send_alert "ERROR" "Backup failed: $2"
    cleanup_failed_backup
    exit 1
}

trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

# Send alerts
send_alert() {
    local severity=$1
    local message=$2
    
    if [ -n "${ALERT_WEBHOOK}" ]; then
        curl -X POST "${ALERT_WEBHOOK}" \
            -H "Content-Type: application/json" \
            -d "{\"severity\":\"${severity}\",\"message\":\"${message}\",\"timestamp\":\"$(date -Iseconds)\"}" \
            2>/dev/null || log "${YELLOW}Warning: Failed to send alert${NC}"
    fi
}

# Initialize backup
init_backup() {
    log "${GREEN}Starting PolicyCortex backup at ${TIMESTAMP}${NC}"
    mkdir -p "${BACKUP_DIR}"
    
    # Create backup metadata
    cat > "${BACKUP_DIR}/metadata.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "version": "$(kubectl get deployment core-api -n policycortex -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)",
    "type": "automated",
    "components": []
}
EOF
}

# Backup PostgreSQL database
backup_postgresql() {
    log "Backing up PostgreSQL database..."
    local backup_file="${BACKUP_DIR}/postgresql_${TIMESTAMP}.sql.gz"
    
    # Create backup with compression
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --verbose \
        --no-owner \
        --no-privileges \
        --format=custom \
        --compress=9 \
        --file="${backup_file}" 2>&1 | tee -a "${LOG_FILE}"
    
    # Verify backup
    if [ -f "${backup_file}" ]; then
        local size=$(du -h "${backup_file}" | cut -f1)
        log "${GREEN}PostgreSQL backup completed: ${size}${NC}"
        
        # Update metadata
        jq '.components += ["postgresql"]' "${BACKUP_DIR}/metadata.json" > tmp.json && mv tmp.json "${BACKUP_DIR}/metadata.json"
    else
        error_handler $LINENO "PostgreSQL backup file not created"
    fi
}

# Backup EventStore
backup_eventstore() {
    log "Backing up EventStore..."
    local backup_file="${BACKUP_DIR}/eventstore_${TIMESTAMP}.backup"
    
    # EventStore backup using Admin API
    curl -X POST "http://eventstore.policycortex.svc.cluster.local:2113/admin/backup" \
        -u admin:changeit \
        -H "Content-Type: application/json" \
        -d "{\"path\":\"${backup_file}\"}" \
        --fail \
        2>&1 | tee -a "${LOG_FILE}"
    
    # Compress the backup
    if [ -f "${backup_file}" ]; then
        gzip -9 "${backup_file}"
        local size=$(du -h "${backup_file}.gz" | cut -f1)
        log "${GREEN}EventStore backup completed: ${size}${NC}"
        
        # Update metadata
        jq '.components += ["eventstore"]' "${BACKUP_DIR}/metadata.json" > tmp.json && mv tmp.json "${BACKUP_DIR}/metadata.json"
    fi
}

# Backup DragonflyDB (Redis-compatible)
backup_dragonfly() {
    log "Backing up DragonflyDB..."
    local backup_file="${BACKUP_DIR}/dragonfly_${TIMESTAMP}.rdb"
    
    # Trigger BGSAVE
    redis-cli -h dragonfly.policycortex.svc.cluster.local BGSAVE
    
    # Wait for backup to complete
    while [ $(redis-cli -h dragonfly.policycortex.svc.cluster.local LASTSAVE) -lt $(date +%s) ]; do
        sleep 1
    done
    
    # Copy the dump file
    kubectl cp policycortex/$(kubectl get pod -n policycortex -l app=dragonfly -o jsonpath='{.items[0].metadata.name}'):/data/dump.rdb "${backup_file}"
    
    # Compress the backup
    gzip -9 "${backup_file}"
    local size=$(du -h "${backup_file}.gz" | cut -f1)
    log "${GREEN}DragonflyDB backup completed: ${size}${NC}"
    
    # Update metadata
    jq '.components += ["dragonfly"]' "${BACKUP_DIR}/metadata.json" > tmp.json && mv tmp.json "${BACKUP_DIR}/metadata.json"
}

# Backup Kubernetes configurations
backup_kubernetes() {
    log "Backing up Kubernetes configurations..."
    local k8s_dir="${BACKUP_DIR}/kubernetes"
    mkdir -p "${k8s_dir}"
    
    # Backup namespaces
    kubectl get namespaces -o yaml > "${k8s_dir}/namespaces.yaml"
    
    # Backup all resources in policycortex namespace
    for resource in $(kubectl api-resources --verbs=list --namespaced -o name | grep -v "events.events.k8s.io" | grep -v "events" | sort | uniq); do
        kubectl get -n policycortex "${resource}" -o yaml > "${k8s_dir}/policycortex_${resource}.yaml" 2>/dev/null || true
    done
    
    # Backup secrets (encrypted)
    kubectl get secrets -n policycortex -o yaml | \
        kubectl-encrypt - | \
        gzip -9 > "${k8s_dir}/secrets_encrypted.yaml.gz"
    
    # Backup persistent volume claims
    kubectl get pvc -n policycortex -o yaml > "${k8s_dir}/pvcs.yaml"
    
    # Create archive
    tar -czf "${BACKUP_DIR}/kubernetes_${TIMESTAMP}.tar.gz" -C "${k8s_dir}" .
    rm -rf "${k8s_dir}"
    
    local size=$(du -h "${BACKUP_DIR}/kubernetes_${TIMESTAMP}.tar.gz" | cut -f1)
    log "${GREEN}Kubernetes backup completed: ${size}${NC}"
    
    # Update metadata
    jq '.components += ["kubernetes"]' "${BACKUP_DIR}/metadata.json" > tmp.json && mv tmp.json "${BACKUP_DIR}/metadata.json"
}

# Backup ML models
backup_ml_models() {
    log "Backing up ML models..."
    local models_dir="${BACKUP_DIR}/ml_models"
    mkdir -p "${models_dir}"
    
    # Copy models from persistent volume
    kubectl cp policycortex/$(kubectl get pod -n policycortex -l app=ai-engine -o jsonpath='{.items[0].metadata.name}'):/app/models "${models_dir}/"
    
    # Create archive with compression
    tar -czf "${BACKUP_DIR}/ml_models_${TIMESTAMP}.tar.gz" -C "${models_dir}" .
    rm -rf "${models_dir}"
    
    local size=$(du -h "${BACKUP_DIR}/ml_models_${TIMESTAMP}.tar.gz" | cut -f1)
    log "${GREEN}ML models backup completed: ${size}${NC}"
    
    # Update metadata
    jq '.components += ["ml_models"]' "${BACKUP_DIR}/metadata.json" > tmp.json && mv tmp.json "${BACKUP_DIR}/metadata.json"
}

# Upload to Azure Storage
upload_to_azure() {
    log "Uploading backups to Azure Storage..."
    
    # Create container if it doesn't exist
    az storage container create \
        --name "${STORAGE_CONTAINER}" \
        --account-name "${STORAGE_ACCOUNT}" \
        --account-key "${STORAGE_KEY}" \
        --fail-on-exist false \
        2>&1 | tee -a "${LOG_FILE}"
    
    # Upload each backup file
    for file in "${BACKUP_DIR}"/*; do
        if [ -f "${file}" ]; then
            local filename=$(basename "${file}")
            log "Uploading ${filename}..."
            
            az storage blob upload \
                --container-name "${STORAGE_CONTAINER}" \
                --file "${file}" \
                --name "${TIMESTAMP}/${filename}" \
                --account-name "${STORAGE_ACCOUNT}" \
                --account-key "${STORAGE_KEY}" \
                --tier "Cool" \
                2>&1 | tee -a "${LOG_FILE}"
        fi
    done
    
    log "${GREEN}Upload to Azure Storage completed${NC}"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    local errors=0
    
    # Check PostgreSQL backup
    if [ -f "${BACKUP_DIR}/postgresql_${TIMESTAMP}.sql.gz" ]; then
        pg_restore --list "${BACKUP_DIR}/postgresql_${TIMESTAMP}.sql.gz" > /dev/null 2>&1 || {
            log "${RED}PostgreSQL backup verification failed${NC}"
            ((errors++))
        }
    fi
    
    # Check archive integrity
    for archive in "${BACKUP_DIR}"/*.tar.gz; do
        if [ -f "${archive}" ]; then
            tar -tzf "${archive}" > /dev/null 2>&1 || {
                log "${RED}Archive verification failed: ${archive}${NC}"
                ((errors++))
            }
        fi
    done
    
    # Check gzip files
    for gzfile in "${BACKUP_DIR}"/*.gz; do
        if [ -f "${gzfile}" ] && [[ ! "${gzfile}" == *.tar.gz ]]; then
            gzip -t "${gzfile}" 2>&1 || {
                log "${RED}Gzip verification failed: ${gzfile}${NC}"
                ((errors++))
            }
        fi
    done
    
    if [ ${errors} -eq 0 ]; then
        log "${GREEN}Backup verification completed successfully${NC}"
        jq '.verified = true' "${BACKUP_DIR}/metadata.json" > tmp.json && mv tmp.json "${BACKUP_DIR}/metadata.json"
    else
        error_handler $LINENO "Backup verification failed with ${errors} errors"
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Local cleanup
    find "${BACKUP_ROOT}" -maxdepth 1 -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null || true
    
    # Azure Storage cleanup
    local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y-%m-%d)
    
    az storage blob list \
        --container-name "${STORAGE_CONTAINER}" \
        --account-name "${STORAGE_ACCOUNT}" \
        --account-key "${STORAGE_KEY}" \
        --query "[?properties.lastModified < '${cutoff_date}'].name" \
        -o tsv | while read -r blob; do
        
        log "Deleting old backup: ${blob}"
        az storage blob delete \
            --container-name "${STORAGE_CONTAINER}" \
            --name "${blob}" \
            --account-name "${STORAGE_ACCOUNT}" \
            --account-key "${STORAGE_KEY}" \
            2>&1 | tee -a "${LOG_FILE}"
    done
    
    log "${GREEN}Cleanup completed${NC}"
}

# Cleanup failed backup
cleanup_failed_backup() {
    log "${YELLOW}Cleaning up failed backup...${NC}"
    if [ -d "${BACKUP_DIR}" ]; then
        rm -rf "${BACKUP_DIR}"
    fi
}

# Generate backup report
generate_report() {
    log "Generating backup report..."
    
    local total_size=$(du -sh "${BACKUP_DIR}" | cut -f1)
    local end_time=$(date +%s)
    local start_time=$(date -d "${TIMESTAMP:0:8} ${TIMESTAMP:9:2}:${TIMESTAMP:11:2}:${TIMESTAMP:13:2}" +%s)
    local duration=$((end_time - start_time))
    
    cat > "${BACKUP_DIR}/report.txt" <<EOF
PolicyCortex Backup Report
==========================
Timestamp: ${TIMESTAMP}
Duration: ${duration} seconds
Total Size: ${total_size}
Components: $(jq -r '.components | join(", ")' "${BACKUP_DIR}/metadata.json")
Verified: $(jq -r '.verified' "${BACKUP_DIR}/metadata.json")
Status: SUCCESS
EOF
    
    log "${GREEN}Backup completed successfully!${NC}"
    log "$(cat ${BACKUP_DIR}/report.txt)"
    
    # Send success notification
    send_alert "INFO" "Backup completed successfully. Size: ${total_size}, Duration: ${duration}s"
}

# Main execution
main() {
    init_backup
    
    # Perform backups
    backup_postgresql
    backup_eventstore
    backup_dragonfly
    backup_kubernetes
    backup_ml_models
    
    # Verify and upload
    verify_backup
    upload_to_azure
    
    # Cleanup and report
    cleanup_old_backups
    generate_report
    
    # Copy log to backup directory
    cp "${LOG_FILE}" "${BACKUP_DIR}/"
}

# Run main function
main "$@"