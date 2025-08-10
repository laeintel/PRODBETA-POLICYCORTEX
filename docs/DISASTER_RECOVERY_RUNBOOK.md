# PolicyCortex Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for disaster recovery scenarios in PolicyCortex. It covers backup strategies, failure scenarios, recovery procedures, and validation steps.

## Table of Contents

1. [Backup Strategy](#backup-strategy)
2. [Failure Scenarios](#failure-scenarios)
3. [Recovery Procedures](#recovery-procedures)
4. [Validation and Testing](#validation-and-testing)
5. [Contact Information](#contact-information)

## Backup Strategy

### Automated Backups

PolicyCortex performs automated backups every 6 hours:

- **Database**: PostgreSQL full backup with WAL archiving
- **EventStore**: Event stream snapshots
- **Redis/DragonflyDB**: RDB snapshots
- **Configuration**: All config files and secrets
- **Retention**: 30 days local, 90 days in Azure Blob Storage

### Manual Backup

```bash
# Perform immediate backup
./scripts/disaster-recovery/backup.sh

# Verify backup
tar -tzf /backup/policycortex/policycortex_backup_[timestamp].tar.gz
```

### Backup Locations

| Component | Primary Location | Secondary Location | Geo-Redundant |
|-----------|-----------------|-------------------|---------------|
| Database | Local disk | Azure Blob (West US) | Azure Blob (East US) |
| EventStore | Local disk | Azure Blob (West US) | Azure Blob (East US) |
| Redis | Local disk | Azure Blob (West US) | N/A |
| Configs | Git repository | Azure Key Vault | Azure Blob |

## Failure Scenarios

### Scenario 1: Database Corruption

**Symptoms:**
- Database queries failing
- Data inconsistencies
- Application errors related to data access

**Recovery Time Objective (RTO):** 1 hour
**Recovery Point Objective (RPO):** 1 hour

**Recovery Steps:**

1. Stop all services:
```bash
systemctl stop policycortex
systemctl stop policycortex-api
systemctl stop policycortex-worker
```

2. Verify database corruption:
```bash
PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -d policycortex -c "SELECT 1;"
```

3. Restore from latest backup:
```bash
./scripts/disaster-recovery/restore.sh latest
```

4. Verify restoration:
```bash
PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -d policycortex \
    -c "SELECT COUNT(*) FROM events;"
```

5. Restart services:
```bash
systemctl start policycortex
systemctl start policycortex-api
systemctl start policycortex-worker
```

### Scenario 2: Complete System Failure

**Symptoms:**
- All services down
- Infrastructure unavailable
- No response from health endpoints

**RTO:** 4 hours
**RPO:** 6 hours

**Recovery Steps:**

1. Provision new infrastructure:
```bash
cd infrastructure/terraform
terraform init
terraform apply -auto-approve
```

2. Deploy PolicyCortex:
```bash
kubectl apply -f infrastructure/kubernetes/prod/
```

3. Restore from backup:
```bash
# Download latest backup from Azure
az storage blob download \
    --account-name policycortexbackup \
    --container-name backups \
    --name [latest-backup] \
    --file /tmp/backup.tar.gz

# Restore
./scripts/disaster-recovery/restore.sh /tmp/backup.tar.gz
```

4. Verify services:
```bash
kubectl get pods -n policycortex
curl http://[load-balancer-ip]/health
```

### Scenario 3: Data Center Outage

**Symptoms:**
- Primary region unavailable
- Network connectivity issues
- Azure region failure

**RTO:** 2 hours
**RPO:** 1 hour

**Recovery Steps:**

1. Failover to secondary region:
```bash
# Update DNS to point to secondary region
az network dns record-set a update \
    --resource-group policycortex-rg \
    --zone-name policycortex.com \
    --name @ \
    --set aRecords[0].ipv4Address=[secondary-ip]
```

2. Activate secondary database:
```bash
# Promote read replica to primary
az postgres server replica stop \
    --name policycortex-db-secondary \
    --resource-group policycortex-rg
```

3. Update application configuration:
```bash
kubectl set env deployment/policycortex-api \
    DATABASE_URL=postgresql://[secondary-db-url] \
    -n policycortex
```

4. Verify failover:
```bash
curl http://policycortex.com/health
```

### Scenario 4: Ransomware Attack

**Symptoms:**
- Encrypted files
- Unauthorized access detected
- Suspicious activity in audit logs

**RTO:** 8 hours
**RPO:** 24 hours

**Recovery Steps:**

1. Isolate affected systems:
```bash
# Disable network access
iptables -A INPUT -j DROP
iptables -A OUTPUT -j DROP
```

2. Assess damage:
```bash
# Check file integrity
find / -type f -name "*.encrypted" 2>/dev/null
sha256sum -c /backup/checksums.sha256
```

3. Restore from immutable backup:
```bash
# Use offline backup from secure location
./scripts/disaster-recovery/restore.sh /secure-backup/policycortex_backup_[date].tar.gz
```

4. Reset all credentials:
```bash
# Rotate all secrets in Key Vault
./scripts/rotate-all-secrets.sh
```

5. Audit and forensics:
```bash
# Export audit logs for investigation
kubectl logs -n policycortex --since=72h > audit_logs.txt
```

## Recovery Procedures

### Pre-Recovery Checklist

- [ ] Identify failure type and scope
- [ ] Notify stakeholders
- [ ] Assemble recovery team
- [ ] Document start time
- [ ] Create incident ticket

### Standard Recovery Process

1. **Assessment Phase** (15 minutes)
   - Identify affected components
   - Determine RPO/RTO requirements
   - Select appropriate recovery strategy

2. **Preparation Phase** (30 minutes)
   - Stop affected services
   - Create current state backup
   - Prepare recovery environment

3. **Recovery Phase** (Variable)
   - Execute recovery procedure
   - Monitor progress
   - Document actions taken

4. **Validation Phase** (30 minutes)
   - Verify data integrity
   - Test functionality
   - Check performance metrics

5. **Restoration Phase** (15 minutes)
   - Resume normal operations
   - Monitor for stability
   - Update documentation

### Rollback Procedures

If recovery fails:

1. Stop recovery attempt:
```bash
# Kill any running restore processes
pkill -f restore.sh
```

2. Restore from recovery point:
```bash
# Use the backup created before recovery
./scripts/disaster-recovery/restore.sh /backup/restore_points/[timestamp]/
```

3. Escalate to next level support

## Validation and Testing

### Post-Recovery Validation

1. **System Health Checks:**
```bash
# Check all services
curl http://localhost:8080/health
curl http://localhost:3000/api/health
curl http://localhost:4000/graphql?query={__typename}
```

2. **Data Integrity Verification:**
```bash
# Verify event count
PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -d policycortex \
    -c "SELECT COUNT(*) FROM events WHERE created_at > NOW() - INTERVAL '1 hour';"

# Check audit trail integrity
./scripts/verify-audit-chain.sh
```

3. **Functional Testing:**
```bash
# Run smoke tests
npm run test:smoke

# Run integration tests
./scripts/test-workflow.sh
```

### DR Testing Schedule

| Test Type | Frequency | Duration | Scope |
|-----------|-----------|----------|-------|
| Backup Verification | Daily | 5 min | Automated checksum validation |
| Restore Test | Weekly | 1 hour | Single component restore |
| Failover Test | Monthly | 2 hours | Regional failover |
| Full DR Test | Quarterly | 4 hours | Complete system recovery |

### Test Scenarios

1. **Backup Restoration Test:**
```bash
# Create test environment
docker-compose -f docker-compose.test.yml up -d

# Restore backup to test environment
RESTORE_ENV=test ./scripts/disaster-recovery/restore.sh

# Validate
./scripts/validate-restore.sh
```

2. **Failover Test:**
```bash
# Simulate primary failure
kubectl delete deployment policycortex-api -n policycortex

# Verify automatic failover
watch kubectl get pods -n policycortex
```

## Contact Information

### Escalation Matrix

| Level | Role | Contact | Available |
|-------|------|---------|-----------|
| L1 | On-Call Engineer | PagerDuty | 24/7 |
| L2 | DevOps Lead | +1-555-0100 | Business Hours |
| L3 | Infrastructure Manager | +1-555-0101 | Business Hours |
| L4 | CTO | +1-555-0102 | Emergency Only |

### External Contacts

- **Azure Support:** 1-800-AZURE-00
- **Database Vendor:** support@postgres.com
- **Security Team:** security@policycortex.com

### Communication Channels

- **Incident Channel:** #incidents (Slack)
- **Status Page:** https://status.policycortex.com
- **War Room:** https://meet.policycortex.com/warroom

## Appendix

### Recovery Time Estimates

| Component | Backup Size | Restore Time |
|-----------|------------|--------------|
| Database (1GB) | 200MB | 5 minutes |
| Database (10GB) | 2GB | 15 minutes |
| Database (100GB) | 20GB | 60 minutes |
| EventStore | 500MB | 10 minutes |
| Redis | 100MB | 2 minutes |
| Full System | Variable | 30-120 minutes |

### Useful Commands

```bash
# Check backup status
ls -lah /backup/policycortex/

# Monitor restore progress
tail -f /var/log/policycortex/restore.log

# Verify database connectivity
pg_isready -h localhost -p 5432

# Check service status
systemctl status policycortex

# View recent errors
journalctl -u policycortex --since "1 hour ago" | grep ERROR

# Export metrics
curl http://localhost:8080/metrics
```

### Recovery Metrics

Track these KPIs for each DR event:

- Time to Detection (TTD)
- Time to Declaration (TTDec)
- Time to Recovery (TTR)
- Data Loss (actual vs RPO)
- Downtime (actual vs RTO)
- Incidents during recovery
- Rollback required (Y/N)
- Root cause identified (Y/N)

---

**Last Updated:** January 2025
**Version:** 1.0
**Review Cycle:** Quarterly