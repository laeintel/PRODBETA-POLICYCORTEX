-- PolicyCortex Core Schema
-- Evidence table for tamper-evident audit trail

CREATE TABLE IF NOT EXISTS evidence (
    content_hash VARCHAR(64) PRIMARY KEY,
    signer VARCHAR(255),
    merkle_root VARCHAR(64),
    ts TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_evidence_ts ON evidence(ts);
CREATE INDEX IF NOT EXISTS idx_evidence_signer ON evidence(signer);

-- Event sourcing table
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, created_at);

-- Seed test data for demo
INSERT INTO evidence(content_hash, signer) 
VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'dev')
ON CONFLICT DO NOTHING;