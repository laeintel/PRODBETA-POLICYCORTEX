-- Migration for CQRS Event Store and Read Models
-- PolicyCortex v2.26.0

-- Create event store table for event sourcing
CREATE TABLE IF NOT EXISTS event_store (
    id BIGSERIAL PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    event_version BIGINT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure events are unique per aggregate and version
    UNIQUE(aggregate_id, event_version)
);

-- Indexes for event store performance
CREATE INDEX idx_event_store_aggregate_id ON event_store(aggregate_id);
CREATE INDEX idx_event_store_event_type ON event_store(event_type);
CREATE INDEX idx_event_store_occurred_at ON event_store(occurred_at DESC);

-- Create snapshots table for aggregate state caching
CREATE TABLE IF NOT EXISTS aggregate_snapshots (
    aggregate_id UUID PRIMARY KEY,
    aggregate_type VARCHAR(255) NOT NULL,
    version BIGINT NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Policies read model
CREATE TABLE IF NOT EXISTS policies (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    rules TEXT[] NOT NULL DEFAULT '{}',
    resource_types TEXT[] NOT NULL DEFAULT '{}',
    enforcement_mode VARCHAR(50) NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ,
    deleted_by VARCHAR(255),
    deletion_reason TEXT
);

CREATE INDEX idx_policies_resource_types ON policies USING GIN(resource_types);
CREATE INDEX idx_policies_enforcement_mode ON policies(enforcement_mode);
CREATE INDEX idx_policies_created_at ON policies(created_at DESC);
CREATE INDEX idx_policies_deleted_at ON policies(deleted_at) WHERE deleted_at IS NULL;

-- Resources read model
CREATE TABLE IF NOT EXISTS resources (
    id UUID PRIMARY KEY,
    resource_type VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    tags JSONB DEFAULT '{}',
    compliance_status VARCHAR(50) DEFAULT 'unknown',
    risk_score DECIMAL(5,2) DEFAULT 0.0,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_resources_type ON resources(resource_type);
CREATE INDEX idx_resources_location ON resources(location);
CREATE INDEX idx_resources_compliance_status ON resources(compliance_status);
CREATE INDEX idx_resources_risk_score ON resources(risk_score DESC);
CREATE INDEX idx_resources_tags ON resources USING GIN(tags);
CREATE INDEX idx_resources_deleted_at ON resources(deleted_at) WHERE deleted_at IS NULL;

-- Compliance checks read model
CREATE TABLE IF NOT EXISTS compliance_checks (
    id UUID PRIMARY KEY,
    resource_id UUID NOT NULL,
    policy_id UUID NOT NULL,
    compliant BOOLEAN NOT NULL,
    violations TEXT[] DEFAULT '{}',
    checked_by VARCHAR(255) NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL,
    
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE,
    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE
);

CREATE INDEX idx_compliance_checks_resource_id ON compliance_checks(resource_id);
CREATE INDEX idx_compliance_checks_policy_id ON compliance_checks(policy_id);
CREATE INDEX idx_compliance_checks_compliant ON compliance_checks(compliant);
CREATE INDEX idx_compliance_checks_checked_at ON compliance_checks(checked_at DESC);

-- Compliance violations read model
CREATE TABLE IF NOT EXISTS compliance_violations (
    id UUID PRIMARY KEY,
    resource_id UUID NOT NULL,
    policy_id UUID NOT NULL,
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE,
    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE
);

CREATE INDEX idx_violations_resource_id ON compliance_violations(resource_id);
CREATE INDEX idx_violations_policy_id ON compliance_violations(policy_id);
CREATE INDEX idx_violations_severity ON compliance_violations(severity);
CREATE INDEX idx_violations_detected_at ON compliance_violations(detected_at DESC);
CREATE INDEX idx_violations_unresolved ON compliance_violations(resolved_at) WHERE resolved_at IS NULL;

-- Remediation actions read model
CREATE TABLE IF NOT EXISTS remediations (
    id UUID PRIMARY KEY,
    resource_id UUID NOT NULL,
    policy_id UUID NOT NULL,
    violation_id UUID,
    action_type VARCHAR(100) NOT NULL,
    parameters JSONB DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    triggered_by VARCHAR(255) NOT NULL,
    triggered_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    error_message TEXT,
    result JSONB,
    
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE,
    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE,
    FOREIGN KEY (violation_id) REFERENCES compliance_violations(id) ON DELETE SET NULL
);

CREATE INDEX idx_remediations_resource_id ON remediations(resource_id);
CREATE INDEX idx_remediations_policy_id ON remediations(policy_id);
CREATE INDEX idx_remediations_status ON remediations(status);
CREATE INDEX idx_remediations_triggered_at ON remediations(triggered_at DESC);

-- Analytics aggregation tables for performance
CREATE TABLE IF NOT EXISTS metrics_daily (
    date DATE NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (date, metric_type)
);

CREATE TABLE IF NOT EXISTS metrics_hourly (
    hour TIMESTAMPTZ NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (hour, metric_type)
);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for auto-updating timestamps
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_resources_updated_at BEFORE UPDATE ON resources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create materialized view for compliance dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS compliance_dashboard AS
SELECT 
    r.id as resource_id,
    r.name as resource_name,
    r.resource_type,
    r.compliance_status,
    r.risk_score,
    COUNT(DISTINCT cc.policy_id) as policies_checked,
    SUM(CASE WHEN cc.compliant THEN 1 ELSE 0 END)::float / 
        NULLIF(COUNT(cc.id), 0) * 100 as compliance_rate,
    ARRAY_AGG(DISTINCT cv.severity) FILTER (WHERE cv.resolved_at IS NULL) as active_violation_severities,
    COUNT(DISTINCT cv.id) FILTER (WHERE cv.resolved_at IS NULL) as active_violations_count,
    MAX(cc.checked_at) as last_checked_at
FROM resources r
LEFT JOIN compliance_checks cc ON r.id = cc.resource_id
LEFT JOIN compliance_violations cv ON r.id = cv.resource_id AND cv.resolved_at IS NULL
WHERE r.deleted_at IS NULL
GROUP BY r.id, r.name, r.resource_type, r.compliance_status, r.risk_score;

-- Create index on materialized view
CREATE UNIQUE INDEX idx_compliance_dashboard_resource ON compliance_dashboard(resource_id);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_compliance_dashboard()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY compliance_dashboard;
END;
$$ LANGUAGE plpgsql;

-- Grant appropriate permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO policycortex_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO policycortex_app;