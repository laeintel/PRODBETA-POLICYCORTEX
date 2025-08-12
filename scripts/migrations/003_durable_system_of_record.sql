-- Durable System of Record Migration
-- Implements event sourcing for complete audit trail and transaction history

-- Event store table for all system events
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_version INTEGER NOT NULL,
    event_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    correlation_id UUID,
    causation_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_aggregate_version UNIQUE (aggregate_id, event_version)
);

-- Indexes for efficient querying
CREATE INDEX idx_events_aggregate_id ON events(aggregate_id);
CREATE INDEX idx_events_aggregate_type ON events(aggregate_type);
CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_tenant_id ON events(tenant_id);
CREATE INDEX idx_events_created_at ON events(created_at DESC);
CREATE INDEX idx_events_correlation_id ON events(correlation_id);

-- Snapshots table for performance optimization
CREATE TABLE IF NOT EXISTS event_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(255) NOT NULL,
    snapshot_version INTEGER NOT NULL,
    snapshot_data JSONB NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_aggregate_snapshot UNIQUE (aggregate_id, snapshot_version)
);

CREATE INDEX idx_snapshots_aggregate_id ON event_snapshots(aggregate_id);
CREATE INDEX idx_snapshots_tenant_id ON event_snapshots(tenant_id);

-- Command store for command sourcing and replay
CREATE TABLE IF NOT EXISTS commands (
    command_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    command_type VARCHAR(255) NOT NULL,
    command_data JSONB NOT NULL,
    aggregate_id UUID,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result JSONB,
    error_message TEXT,
    idempotency_key VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT unique_idempotency_key UNIQUE (idempotency_key)
);

CREATE INDEX idx_commands_aggregate_id ON commands(aggregate_id);
CREATE INDEX idx_commands_tenant_id ON commands(tenant_id);
CREATE INDEX idx_commands_status ON commands(status);
CREATE INDEX idx_commands_idempotency_key ON commands(idempotency_key);
CREATE INDEX idx_commands_created_at ON commands(created_at DESC);

-- Transaction log for financial and critical operations
CREATE TABLE IF NOT EXISTS transaction_log (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_type VARCHAR(255) NOT NULL,
    source_system VARCHAR(255) NOT NULL,
    target_system VARCHAR(255),
    transaction_data JSONB NOT NULL,
    amount DECIMAL(20, 4),
    currency VARCHAR(3),
    status VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    reference_id VARCHAR(255),
    external_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX idx_transaction_log_tenant_id ON transaction_log(tenant_id);
CREATE INDEX idx_transaction_log_status ON transaction_log(status);
CREATE INDEX idx_transaction_log_reference_id ON transaction_log(reference_id);
CREATE INDEX idx_transaction_log_external_id ON transaction_log(external_id);
CREATE INDEX idx_transaction_log_created_at ON transaction_log(created_at DESC);

-- State transitions table for workflow tracking
CREATE TABLE IF NOT EXISTS state_transitions (
    transition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL,
    entity_type VARCHAR(255) NOT NULL,
    from_state VARCHAR(255),
    to_state VARCHAR(255) NOT NULL,
    transition_reason TEXT,
    metadata JSONB DEFAULT '{}',
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_state_transitions_entity ON state_transitions(entity_id, entity_type);
CREATE INDEX idx_state_transitions_tenant_id ON state_transitions(tenant_id);
CREATE INDEX idx_state_transitions_created_at ON state_transitions(created_at DESC);

-- Immutable audit trail with cryptographic verification
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL,
    entity_type VARCHAR(255) NOT NULL,
    action VARCHAR(255) NOT NULL,
    before_state JSONB,
    after_state JSONB,
    changes JSONB,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    user_email VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    hash VARCHAR(64) NOT NULL, -- SHA-256 hash of the record
    previous_hash VARCHAR(64), -- Hash of the previous record for chain verification
    signature TEXT, -- Digital signature for non-repudiation
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_trail_entity ON audit_trail(entity_id, entity_type);
CREATE INDEX idx_audit_trail_tenant_id ON audit_trail(tenant_id);
CREATE INDEX idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX idx_audit_trail_action ON audit_trail(action);
CREATE INDEX idx_audit_trail_created_at ON audit_trail(created_at DESC);
CREATE INDEX idx_audit_trail_hash ON audit_trail(hash);

-- Evidence store for WORM-like persistence of artifacts
CREATE TABLE IF NOT EXISTS evidence_store (
    id UUID PRIMARY KEY,
    evidence_type VARCHAR(64) NOT NULL,
    source JSONB NOT NULL,
    subject TEXT NOT NULL,
    description TEXT,
    data JSONB NOT NULL,
    hash VARCHAR(128) NOT NULL,
    signature TEXT NOT NULL,
    signing_key_id TEXT,
    chain_of_custody JSONB NOT NULL,
    metadata JSONB NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    verification_status VARCHAR(32) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_tenant ON evidence_store(tenant_id);
CREATE INDEX IF NOT EXISTS idx_evidence_created_at ON evidence_store(created_at DESC);

-- Idempotency records for action deduplication
CREATE TABLE IF NOT EXISTS idempotency_records (
    key VARCHAR(255) PRIMARY KEY,
    action_id UUID NOT NULL,
    result JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_records(expires_at);

-- Approval requests table
CREATE TABLE IF NOT EXISTS approval_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    action_id UUID,
    action_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    requester_id TEXT,
    requester_email TEXT,
    title TEXT,
    description TEXT,
    impact_analysis JSONB,
    approval_type JSONB,
    required_approvers TEXT[],
    status TEXT NOT NULL,
    approvals JSONB,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_approval_req_tenant ON approval_requests(tenant_id);
CREATE INDEX IF NOT EXISTS idx_approval_req_status ON approval_requests(status);

-- Exceptions lifecycle table (GRC-managed exceptions)
CREATE TABLE IF NOT EXISTS exceptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    resource_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Approved',
    created_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    recertify_at TIMESTAMP WITH TIME ZONE,
    evidence JSONB,
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_exceptions_tenant ON exceptions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_exceptions_status ON exceptions(status);
CREATE INDEX IF NOT EXISTS idx_exceptions_expires ON exceptions(expires_at);

-- Function to calculate audit record hash
CREATE OR REPLACE FUNCTION calculate_audit_hash(
    p_entity_id UUID,
    p_entity_type VARCHAR,
    p_action VARCHAR,
    p_changes JSONB,
    p_user_id VARCHAR,
    p_timestamp TIMESTAMP WITH TIME ZONE,
    p_previous_hash VARCHAR
) RETURNS VARCHAR AS $$
BEGIN
    RETURN encode(
        digest(
            CONCAT(
                p_entity_id::TEXT,
                p_entity_type,
                p_action,
                p_changes::TEXT,
                p_user_id,
                p_timestamp::TEXT,
                COALESCE(p_previous_hash, '')
            ),
            'sha256'
        ),
        'hex'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger to automatically calculate hash for audit records
CREATE OR REPLACE FUNCTION audit_trail_hash_trigger()
RETURNS TRIGGER AS $$
DECLARE
    v_previous_hash VARCHAR(64);
BEGIN
    -- Get the hash of the most recent audit record
    SELECT hash INTO v_previous_hash
    FROM audit_trail
    WHERE tenant_id = NEW.tenant_id
    ORDER BY created_at DESC
    LIMIT 1;
    
    -- Calculate hash for the new record
    NEW.previous_hash := v_previous_hash;
    NEW.hash := calculate_audit_hash(
        NEW.entity_id,
        NEW.entity_type,
        NEW.action,
        NEW.changes,
        NEW.user_id,
        NEW.created_at,
        NEW.previous_hash
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_audit_trail_hash
    BEFORE INSERT ON audit_trail
    FOR EACH ROW
    EXECUTE FUNCTION audit_trail_hash_trigger();

-- Projections table for read models
CREATE TABLE IF NOT EXISTS projections (
    projection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    projection_name VARCHAR(255) NOT NULL,
    aggregate_id UUID NOT NULL,
    projection_data JSONB NOT NULL,
    version INTEGER NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_projection UNIQUE (projection_name, aggregate_id)
);

CREATE INDEX idx_projections_name ON projections(projection_name);
CREATE INDEX idx_projections_aggregate_id ON projections(aggregate_id);
CREATE INDEX idx_projections_tenant_id ON projections(tenant_id);

-- Saga state table for managing distributed transactions
CREATE TABLE IF NOT EXISTS sagas (
    saga_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    saga_type VARCHAR(255) NOT NULL,
    saga_state JSONB NOT NULL,
    current_step VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'running',
    tenant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT check_saga_status CHECK (status IN ('running', 'completed', 'failed', 'compensating', 'compensated'))
);

CREATE INDEX idx_sagas_type ON sagas(saga_type);
CREATE INDEX idx_sagas_status ON sagas(status);
CREATE INDEX idx_sagas_tenant_id ON sagas(tenant_id);

-- Function to replay events for an aggregate
CREATE OR REPLACE FUNCTION replay_events(
    p_aggregate_id UUID,
    p_from_version INTEGER DEFAULT 0,
    p_to_version INTEGER DEFAULT NULL
) RETURNS TABLE (
    event_id UUID,
    event_type VARCHAR,
    event_version INTEGER,
    event_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.event_id,
        e.event_type,
        e.event_version,
        e.event_data,
        e.created_at
    FROM events e
    WHERE e.aggregate_id = p_aggregate_id
    AND e.event_version > p_from_version
    AND (p_to_version IS NULL OR e.event_version <= p_to_version)
    ORDER BY e.event_version ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to get the current state of an aggregate
CREATE OR REPLACE FUNCTION get_aggregate_state(
    p_aggregate_id UUID
) RETURNS JSONB AS $$
DECLARE
    v_snapshot JSONB;
    v_snapshot_version INTEGER;
    v_event RECORD;
    v_state JSONB;
BEGIN
    -- Get the latest snapshot if available
    SELECT snapshot_data, snapshot_version 
    INTO v_snapshot, v_snapshot_version
    FROM event_snapshots
    WHERE aggregate_id = p_aggregate_id
    ORDER BY snapshot_version DESC
    LIMIT 1;
    
    -- Initialize state with snapshot or empty object
    v_state := COALESCE(v_snapshot, '{}'::JSONB);
    
    -- Replay events after the snapshot
    FOR v_event IN 
        SELECT event_data 
        FROM events 
        WHERE aggregate_id = p_aggregate_id
        AND event_version > COALESCE(v_snapshot_version, 0)
        ORDER BY event_version ASC
    LOOP
        -- Apply event to state (simplified - actual implementation would be event-specific)
        v_state := v_state || v_event.event_data;
    END LOOP;
    
    RETURN v_state;
END;
$$ LANGUAGE plpgsql;

-- Enable Row Level Security on new tables
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE commands ENABLE ROW LEVEL SECURITY;
ALTER TABLE transaction_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE state_transitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_trail ENABLE ROW LEVEL SECURITY;
ALTER TABLE projections ENABLE ROW LEVEL SECURITY;
ALTER TABLE sagas ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for events table
CREATE POLICY tenant_isolation_events ON events
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_snapshots ON event_snapshots
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_commands ON commands
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_transactions ON transaction_log
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_transitions ON state_transitions
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_audit ON audit_trail
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_projections ON projections
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_sagas ON sagas
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;