-- PolicyCortex Database Initialization Script
-- Creates all necessary tables and initial data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS compliance;
CREATE SCHEMA IF NOT EXISTS security;
CREATE SCHEMA IF NOT EXISTS finops;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO governance, compliance, security, finops, audit, public;

-- Governance Schema Tables

-- Policies table
CREATE TABLE IF NOT EXISTS governance.policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    severity VARCHAR(50),
    compliance_frameworks TEXT[],
    rules JSONB,
    metadata JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Resources table
CREATE TABLE IF NOT EXISTS governance.resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    azure_resource_id VARCHAR(500) UNIQUE NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    resource_group VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(255) NOT NULL,
    location VARCHAR(100),
    tags JSONB,
    properties JSONB,
    compliance_status VARCHAR(50),
    last_scanned TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance Schema Tables

-- Compliance assessments
CREATE TABLE IF NOT EXISTS compliance.assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES governance.resources(id),
    policy_id UUID REFERENCES governance.policies(id),
    compliance_state VARCHAR(50) NOT NULL,
    details JSONB,
    evidence JSONB,
    remediation_steps TEXT[],
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    next_assessment TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance frameworks
CREATE TABLE IF NOT EXISTS compliance.frameworks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    version VARCHAR(50),
    description TEXT,
    requirements JSONB,
    controls JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security Schema Tables

-- Security findings
CREATE TABLE IF NOT EXISTS security.findings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES governance.resources(id),
    finding_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    recommendation TEXT,
    attack_vector VARCHAR(255),
    cvss_score DECIMAL(3,1),
    cve_ids TEXT[],
    status VARCHAR(50) DEFAULT 'open',
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Attack paths
CREATE TABLE IF NOT EXISTS security.attack_paths (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    path_id VARCHAR(255) UNIQUE NOT NULL,
    source_node JSONB,
    target_node JSONB,
    path_nodes JSONB[],
    risk_score DECIMAL(5,2),
    exploitability DECIMAL(5,2),
    impact DECIMAL(5,2),
    mitigation_bundles JSONB,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_validated TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- FinOps Schema Tables

-- Cost analysis
CREATE TABLE IF NOT EXISTS finops.cost_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    subscription_id VARCHAR(100) NOT NULL,
    resource_id UUID REFERENCES governance.resources(id),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    actual_cost DECIMAL(15,2),
    forecasted_cost DECIMAL(15,2),
    budget DECIMAL(15,2),
    currency VARCHAR(10) DEFAULT 'USD',
    cost_breakdown JSONB,
    anomalies JSONB,
    optimization_opportunities JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optimization recommendations
CREATE TABLE IF NOT EXISTS finops.optimizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES governance.resources(id),
    optimization_type VARCHAR(100) NOT NULL,
    current_cost DECIMAL(15,2),
    optimized_cost DECIMAL(15,2),
    savings_amount DECIMAL(15,2),
    savings_percentage DECIMAL(5,2),
    confidence_score DECIMAL(5,2),
    recommendation TEXT,
    implementation_steps JSONB,
    risk_level VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    implemented_at TIMESTAMP WITH TIME ZONE
);

-- Audit Schema Tables

-- Audit logs
CREATE TABLE IF NOT EXISTS audit.logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(100),
    actor VARCHAR(255),
    actor_type VARCHAR(50),
    resource_id VARCHAR(500),
    resource_type VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    result VARCHAR(50),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    correlation_id UUID,
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_resources_subscription ON governance.resources(subscription_id);
CREATE INDEX idx_resources_type ON governance.resources(type);
CREATE INDEX idx_resources_compliance ON governance.resources(compliance_status);
CREATE INDEX idx_assessments_resource ON compliance.assessments(resource_id);
CREATE INDEX idx_assessments_policy ON compliance.assessments(policy_id);
CREATE INDEX idx_assessments_state ON compliance.assessments(compliance_state);
CREATE INDEX idx_findings_resource ON security.findings(resource_id);
CREATE INDEX idx_findings_severity ON security.findings(severity);
CREATE INDEX idx_findings_status ON security.findings(status);
CREATE INDEX idx_cost_subscription ON finops.cost_analysis(subscription_id);
CREATE INDEX idx_cost_period ON finops.cost_analysis(period_start, period_end);
CREATE INDEX idx_optimizations_resource ON finops.optimizations(resource_id);
CREATE INDEX idx_optimizations_status ON finops.optimizations(status);
CREATE INDEX idx_audit_event_type ON audit.logs(event_type);
CREATE INDEX idx_audit_actor ON audit.logs(actor);
CREATE INDEX idx_audit_occurred ON audit.logs(occurred_at);

-- Create views for common queries

-- Compliance dashboard view
CREATE OR REPLACE VIEW governance.compliance_dashboard AS
SELECT 
    r.subscription_id,
    r.resource_group,
    COUNT(DISTINCT r.id) as total_resources,
    COUNT(DISTINCT CASE WHEN r.compliance_status = 'compliant' THEN r.id END) as compliant_resources,
    COUNT(DISTINCT CASE WHEN r.compliance_status = 'non-compliant' THEN r.id END) as non_compliant_resources,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN r.compliance_status = 'compliant' THEN r.id END) / 
          NULLIF(COUNT(DISTINCT r.id), 0), 2) as compliance_percentage
FROM governance.resources r
GROUP BY r.subscription_id, r.resource_group;

-- Security risk view
CREATE OR REPLACE VIEW security.risk_summary AS
SELECT 
    r.subscription_id,
    COUNT(DISTINCT f.id) as total_findings,
    COUNT(DISTINCT CASE WHEN f.severity = 'critical' THEN f.id END) as critical_findings,
    COUNT(DISTINCT CASE WHEN f.severity = 'high' THEN f.id END) as high_findings,
    COUNT(DISTINCT CASE WHEN f.severity = 'medium' THEN f.id END) as medium_findings,
    COUNT(DISTINCT CASE WHEN f.severity = 'low' THEN f.id END) as low_findings,
    AVG(f.cvss_score) as avg_cvss_score
FROM governance.resources r
LEFT JOIN security.findings f ON r.id = f.resource_id
WHERE f.status = 'open'
GROUP BY r.subscription_id;

-- Cost optimization view
CREATE OR REPLACE VIEW finops.optimization_summary AS
SELECT 
    o.optimization_type,
    COUNT(*) as opportunity_count,
    SUM(o.savings_amount) as total_potential_savings,
    AVG(o.savings_percentage) as avg_savings_percentage,
    AVG(o.confidence_score) as avg_confidence
FROM finops.optimizations o
WHERE o.status = 'pending'
GROUP BY o.optimization_type;

-- Insert default compliance frameworks
INSERT INTO compliance.frameworks (name, version, description, requirements) VALUES
('CIS Azure Foundations', '2.0', 'Center for Internet Security Azure Foundations Benchmark', '{}'),
('PCI DSS', '4.0', 'Payment Card Industry Data Security Standard', '{}'),
('HIPAA', '2023', 'Health Insurance Portability and Accountability Act', '{}'),
('SOC 2', 'Type II', 'Service Organization Control 2', '{}'),
('ISO 27001', '2022', 'Information Security Management System', '{}'),
('GDPR', '2016/679', 'General Data Protection Regulation', '{}')
ON CONFLICT (name) DO NOTHING;

-- Grant permissions (adjust as needed)
GRANT USAGE ON SCHEMA governance, compliance, security, finops, audit TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA governance, compliance, security, finops TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO PUBLIC;
GRANT INSERT ON audit.logs TO PUBLIC;

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON governance.policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_resources_updated_at BEFORE UPDATE ON governance.resources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_frameworks_updated_at BEFORE UPDATE ON compliance.frameworks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();