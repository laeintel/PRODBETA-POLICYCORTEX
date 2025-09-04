# PolicyCortex PROVE Pillar - Immutable Evidence Chain

## Overview

The PROVE pillar provides tamper-proof evidence collection and cryptographic verification for all policy compliance checks in the PolicyCortex PCG platform. This system ensures auditors can verify the authenticity and integrity of compliance evidence through an immutable blockchain-based evidence chain.

## Architecture

### Core Components

1. **Hash Chain Engine** (`hash_chain/chain.rs`)
   - Merkle tree structure using SHA3-256
   - Ed25519 digital signatures
   - Block creation every 100 events or 1 hour
   - Immutable append-only storage
   - Chain verification and Merkle proof generation

2. **Evidence Collector** (`collector/collector.py`)
   - Python FastAPI service
   - Gathers evidence from all policy checks
   - PostgreSQL storage with row-level immutability
   - Redis caching for recent evidence
   - CQRS event store integration

3. **Audit Reporter** (`audit_reporter/reporter.ts`)
   - TypeScript service for report generation
   - Board-ready PDF reports with QR codes
   - SSP/POA&M format export
   - Digital signatures using RSA-4096
   - Cryptographic verification support

## API Endpoints

### Evidence Collection
- `POST /api/v1/evidence/collect` - Store new evidence
  ```json
  {
    "event_type": "PolicyCheck",
    "resource_id": "vm-001",
    "policy_id": "policy-001",
    "policy_name": "VM Compliance Policy",
    "compliance_status": "Compliant",
    "actor": "system",
    "subscription_id": "sub-001",
    "resource_group": "rg-001",
    "resource_type": "Microsoft.Compute/virtualMachines",
    "details": {},
    "metadata": {}
  }
  ```

### Chain Verification
- `GET /api/v1/evidence/verify/{hash}` - Verify evidence integrity
- `GET /api/v1/evidence/chain` - Get chain status
- `GET /api/v1/evidence/proof/{hash}` - Get Merkle proof

### Report Generation
- `POST /api/v1/evidence/report` - Generate audit report
  ```json
  {
    "subscription_id": "sub-001",
    "format": "PDF",
    "include_qr_code": true,
    "digital_signature": true,
    "include_evidence": true,
    "evidence_limit": 1000
  }
  ```

### Evidence Retrieval
- `GET /api/v1/evidence/{id}` - Get evidence by ID
- `GET /api/v1/evidence/block/{index}` - Get block by index

## Setup Instructions

### 1. Rust Hash Chain Engine

The hash chain is integrated into the core Rust application and starts automatically.

### 2. Python Evidence Collector

```bash
cd core/src/evidence/collector
pip install -r requirements.txt
python collector.py
```

The collector runs on port 8001 by default.

### 3. TypeScript Audit Reporter

```bash
cd core/src/evidence/audit_reporter
npm install
npm run build
npm start
```

## Database Schema

The evidence table includes:
- Unique hash for each evidence record
- Immutability constraints via PostgreSQL triggers
- Block index and hash once added to chain
- JSONB storage for flexible metadata

```sql
CREATE TABLE evidence (
    id UUID PRIMARY KEY,
    hash VARCHAR(64) NOT NULL UNIQUE,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    resource_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    compliance_status VARCHAR(20) NOT NULL,
    actor TEXT NOT NULL,
    details JSONB NOT NULL,
    metadata JSONB NOT NULL,
    block_index BIGINT,
    block_hash VARCHAR(64),
    immutable BOOLEAN DEFAULT FALSE
);
```

## Security Features

### Cryptographic Protection
- SHA3-256 hashing for evidence integrity
- Ed25519 signatures for block authentication
- RSA-4096 signatures for reports
- Merkle proofs for individual evidence verification

### Immutability Guarantees
- PostgreSQL triggers prevent modification of immutable records
- Append-only blockchain structure
- Cryptographic linking between blocks
- Distributed verification capability

### Audit Trail
- Complete evidence history
- Tamper detection through chain verification
- QR codes for easy verification
- Digital signatures on all reports

## Testing

### Unit Tests
```bash
# Rust tests
cd core
cargo test evidence

# Python tests
cd core/src/evidence/collector
pytest test_collector.py

# TypeScript tests
cd core/src/evidence/audit_reporter
npm test
```

### Integration Tests
```bash
# Start all services
./scripts/start-evidence-chain.sh

# Run integration tests
./scripts/test-evidence-integration.sh
```

## Compliance Standards Support

The PROVE pillar supports evidence collection for:
- SOC 2 Type II
- ISO 27001/27017/27018
- NIST 800-53
- CIS Controls
- PCI DSS
- HIPAA
- FedRAMP

## Performance Metrics

- Evidence collection: <50ms latency
- Chain verification: <100ms for 10,000 blocks
- Report generation: <5 seconds for 1,000 evidence records
- Merkle proof generation: <10ms
- Block creation: <500ms for 100 events

## Monitoring

Key metrics to monitor:
- Evidence collection rate
- Chain growth rate
- Block creation frequency
- Verification success rate
- Report generation time

## Troubleshooting

### Common Issues

1. **Chain verification fails**
   - Check block signatures
   - Verify Merkle roots
   - Ensure no database tampering

2. **Evidence not appearing in reports**
   - Check evidence timestamps
   - Verify subscription ID filters
   - Ensure evidence is in immutable state

3. **Report generation timeout**
   - Reduce evidence limit
   - Check database indexes
   - Verify network connectivity

## Future Enhancements

- [ ] Distributed ledger with multiple nodes
- [ ] Zero-knowledge proofs for privacy
- [ ] Homomorphic encryption for sensitive data
- [ ] Integration with external blockchain networks
- [ ] AI-powered anomaly detection in evidence patterns