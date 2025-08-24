-- PolicyCortex Database Schema Initialization
-- Version 2.0.0

-- Create schemas
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS audit;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    tenant_id VARCHAR(255),
    roles TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policies table
CREATE TABLE IF NOT EXISTS governance.policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    rules JSONB,
    enabled BOOLEAN DEFAULT true,
    compliance_score FLOAT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resources table
CREATE TABLE IF NOT EXISTS governance.resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) UNIQUE NOT NULL,
    resource_type VARCHAR(100),
    name VARCHAR(255),
    subscription_id VARCHAR(255),
    resource_group VARCHAR(255),
    location VARCHAR(100),
    tags JSONB,
    metadata JSONB,
    compliance_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Compliance results table
CREATE TABLE IF NOT EXISTS governance.compliance_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id UUID REFERENCES governance.resources(id),
    policy_id UUID REFERENCES governance.policies(id),
    status VARCHAR(50),
    score FLOAT,
    issues JSONB,
    recommendations JSONB,
    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML Predictions table
CREATE TABLE IF NOT EXISTS ml.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500),
    prediction_type VARCHAR(100),
    risk_score FLOAT,
    confidence FLOAT,
    features JSONB,
    recommendations JSONB,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations table (Patent #2)
CREATE TABLE IF NOT EXISTS ml.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    user_id UUID REFERENCES users(id),
    message TEXT,
    response TEXT,
    intent VARCHAR(100),
    entities JSONB,
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Correlations table (Patent #1)
CREATE TABLE IF NOT EXISTS ml.correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_name VARCHAR(255),
    domains TEXT[],
    confidence FLOAT,
    affected_resources INTEGER,
    insights TEXT,
    metadata JSONB,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit.logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(255),
    resource_type VARCHAR(100),
    resource_id VARCHAR(500),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cost optimization table
CREATE TABLE IF NOT EXISTS governance.cost_optimization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id UUID REFERENCES governance.resources(id),
    current_cost DECIMAL(10, 2),
    optimized_cost DECIMAL(10, 2),
    savings DECIMAL(10, 2),
    recommendations JSONB,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Security threats table
CREATE TABLE IF NOT EXISTS governance.security_threats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    threat_id VARCHAR(255) UNIQUE,
    severity VARCHAR(50),
    category VARCHAR(100),
    description TEXT,
    affected_resources JSONB,
    remediation JSONB,
    status VARCHAR(50),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_resources_subscription ON governance.resources(subscription_id);
CREATE INDEX IF NOT EXISTS idx_resources_type ON governance.resources(resource_type);
CREATE INDEX IF NOT EXISTS idx_compliance_resource ON governance.compliance_results(resource_id);
CREATE INDEX IF NOT EXISTS idx_predictions_resource ON ml.predictions(resource_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON ml.conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit.logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit.logs(created_at);

-- Insert sample data
INSERT INTO users (email, name, tenant_id, roles) VALUES
    ('admin@policycortex.com', 'System Admin', 'default', ARRAY['admin', 'user']),
    ('demo@policycortex.com', 'Demo User', 'default', ARRAY['user'])
ON CONFLICT (email) DO NOTHING;

-- Insert sample policies
INSERT INTO governance.policies (name, description, category, compliance_score) VALUES
    ('Require Tags', 'All resources must have environment and owner tags', 'Tagging', 0.94),
    ('Security Baseline', 'Resources must meet security baseline requirements', 'Security', 0.87),
    ('Cost Management', 'Resources must follow cost optimization guidelines', 'Cost', 0.82)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL ON SCHEMA governance TO postgres;
GRANT ALL ON SCHEMA ml TO postgres;
GRANT ALL ON SCHEMA audit TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA governance TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA audit TO postgres;