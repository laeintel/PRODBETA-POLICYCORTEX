-- Evidence durable store

CREATE TABLE IF NOT EXISTS evidence_store (
    id UUID PRIMARY KEY,
    evidence_type TEXT NOT NULL,
    source JSONB NOT NULL,
    subject TEXT NOT NULL,
    description TEXT,
    data JSONB NOT NULL,
    hash TEXT NOT NULL,
    signature TEXT NOT NULL,
    signing_key_id TEXT,
    chain_of_custody JSONB,
    metadata JSONB,
    tenant_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    verification_status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_tenant ON evidence_store(tenant_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence_store(evidence_type);
