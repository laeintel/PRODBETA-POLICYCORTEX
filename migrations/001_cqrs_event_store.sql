-- Migration: Create CQRS Event Store tables
-- Required for T04 - Events/CQRS implementation

-- Event store table
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_version INTEGER NOT NULL,
    event_data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    correlation_id UUID,
    causation_id UUID,
    tenant_id VARCHAR(255),
    
    -- Indexes for query performance
    INDEX idx_events_aggregate (aggregate_id, event_version),
    INDEX idx_events_type (event_type),
    INDEX idx_events_created (created_at DESC),
    INDEX idx_events_tenant (tenant_id),
    INDEX idx_events_correlation (correlation_id)
);

-- Snapshots table for event sourcing optimization
CREATE TABLE IF NOT EXISTS snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(255) NOT NULL,
    snapshot_version INTEGER NOT NULL,
    snapshot_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique snapshot per aggregate and version
    UNIQUE(aggregate_id, snapshot_version),
    INDEX idx_snapshots_aggregate (aggregate_id, snapshot_version DESC)
);

-- Projections table for read models
CREATE TABLE IF NOT EXISTS projections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    projection_name VARCHAR(255) NOT NULL,
    projection_key VARCHAR(255) NOT NULL,
    projection_data JSONB NOT NULL,
    last_event_id UUID,
    last_event_version INTEGER,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique projection per name and key
    UNIQUE(projection_name, projection_key),
    INDEX idx_projections_name (projection_name),
    INDEX idx_projections_updated (updated_at DESC)
);

-- Command log table for audit trail
CREATE TABLE IF NOT EXISTS commands (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    command_type VARCHAR(255) NOT NULL,
    command_data JSONB NOT NULL,
    result_status VARCHAR(50),
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    correlation_id UUID,
    tenant_id VARCHAR(255),
    processing_time_ms INTEGER,
    
    INDEX idx_commands_type (command_type),
    INDEX idx_commands_created (created_at DESC),
    INDEX idx_commands_correlation (correlation_id),
    INDEX idx_commands_tenant (tenant_id)
);

-- Saga state table for distributed transactions
CREATE TABLE IF NOT EXISTS sagas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    saga_type VARCHAR(255) NOT NULL,
    saga_state VARCHAR(50) NOT NULL,
    saga_data JSONB NOT NULL,
    current_step VARCHAR(255),
    completed_steps TEXT[],
    failed_steps TEXT[],
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    correlation_id UUID,
    
    INDEX idx_sagas_type (saga_type),
    INDEX idx_sagas_state (saga_state),
    INDEX idx_sagas_correlation (correlation_id)
);

-- Create function to maintain projection consistency
CREATE OR REPLACE FUNCTION update_projection_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for projection updates
CREATE TRIGGER projection_update_timestamp
    BEFORE UPDATE ON projections
    FOR EACH ROW
    EXECUTE FUNCTION update_projection_timestamp();

-- Create function to validate event ordering
CREATE OR REPLACE FUNCTION validate_event_version()
RETURNS TRIGGER AS $$
DECLARE
    max_version INTEGER;
BEGIN
    SELECT COALESCE(MAX(event_version), 0) INTO max_version
    FROM events
    WHERE aggregate_id = NEW.aggregate_id;
    
    IF NEW.event_version != max_version + 1 THEN
        RAISE EXCEPTION 'Invalid event version. Expected %, got %', max_version + 1, NEW.event_version;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for event version validation
CREATE TRIGGER event_version_check
    BEFORE INSERT ON events
    FOR EACH ROW
    EXECUTE FUNCTION validate_event_version();

-- Create view for latest aggregate states
CREATE OR REPLACE VIEW aggregate_current_state AS
SELECT DISTINCT ON (aggregate_id)
    aggregate_id,
    aggregate_type,
    event_version AS current_version,
    created_at AS last_modified,
    tenant_id
FROM events
ORDER BY aggregate_id, event_version DESC;

-- Create materialized view for event statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS event_statistics AS
SELECT 
    event_type,
    COUNT(*) as event_count,
    DATE_TRUNC('day', created_at) as event_date,
    AVG(EXTRACT(EPOCH FROM (created_at - LAG(created_at) OVER (PARTITION BY aggregate_id ORDER BY event_version)))) as avg_time_between_events
FROM events
GROUP BY event_type, DATE_TRUNC('day', created_at);

-- Create index on materialized view
CREATE INDEX idx_event_stats_type ON event_statistics(event_type);
CREATE INDEX idx_event_stats_date ON event_statistics(event_date DESC);

-- Grant permissions (adjust as needed for your user/role setup)
-- GRANT SELECT, INSERT ON events TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON projections TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON commands TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON sagas TO policycortex_app;
-- GRANT SELECT ON aggregate_current_state TO policycortex_app;
-- GRANT SELECT ON event_statistics TO policycortex_app;