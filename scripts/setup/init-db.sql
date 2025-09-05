-- PolicyCortex Database Initialization Script
-- This script creates the complete database schema for PolicyCortex
-- Includes all tables, indexes, functions, and initial data
-- Compatible with PostgreSQL 16+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create separate databases if they don't exist
SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Set session parameters for performance
SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';
SET random_page_cost = 1.1;
SET effective_io_concurrency = 200;

-- ============================================================
-- CORE SCHEMA CREATION
-- ============================================================

-- Drop existing schema if exists (for fresh installs)
DROP SCHEMA IF EXISTS policy_cortex CASCADE;
CREATE SCHEMA policy_cortex;
SET search_path = policy_cortex, public;

-- Create custom types
CREATE TYPE resource_status AS ENUM ('active', 'inactive', 'pending', 'error', 'unknown');
CREATE TYPE compliance_status AS ENUM ('compliant', 'non_compliant', 'unknown', 'not_applicable');
CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE audit_action AS ENUM ('create', 'update', 'delete', 'login', 'logout', 'approve', 'reject');
CREATE TYPE prediction_confidence AS ENUM ('very_low', 'low', 'medium', 'high', 'very_high');
CREATE TYPE workflow_status AS ENUM ('pending', 'in_progress', 'approved', 'rejected', 'completed', 'failed');

-- ============================================================
-- TENANT ISOLATION TABLES
-- ============================================================

-- Tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE NOT NULL,
    subscription_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Tenant users
CREATE TABLE tenant_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL, -- Azure AD User ID
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    roles JSONB DEFAULT '[]',
    permissions JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    UNIQUE(tenant_id, user_id)
);

-- ============================================================
-- RESOURCE MANAGEMENT TABLES
-- ============================================================

-- Azure resources
CREATE TABLE resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_id VARCHAR(500) NOT NULL, -- Azure Resource ID
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    location VARCHAR(50),
    resource_group VARCHAR(255),
    subscription_id VARCHAR(255) NOT NULL,
    status resource_status DEFAULT 'unknown',
    tags JSONB DEFAULT '{}',
    properties JSONB DEFAULT '{}',
    cost_month DECIMAL(10,2) DEFAULT 0,
    compliance_score DECIMAL(5,2) DEFAULT 0 CHECK (compliance_score >= 0 AND compliance_score <= 100),
    risk_score DECIMAL(5,2) DEFAULT 0 CHECK (risk_score >= 0 AND risk_score <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_scanned TIMESTAMP WITH TIME ZONE,
    UNIQUE(tenant_id, resource_id)
);

-- Resource dependencies (for correlation analysis)
CREATE TABLE resource_dependencies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    parent_resource_id UUID NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    child_resource_id UUID NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    dependency_type VARCHAR(50) NOT NULL, -- 'network', 'storage', 'compute', 'data', 'security'
    strength DECIMAL(3,2) DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    discovered_method VARCHAR(50) DEFAULT 'automatic',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, parent_resource_id, child_resource_id)
);

-- ============================================================
-- POLICY AND COMPLIANCE TABLES
-- ============================================================

-- Policy definitions
CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    policy_type VARCHAR(50) NOT NULL, -- 'built-in', 'custom', 'initiative'
    mode VARCHAR(20) DEFAULT 'Indexed', -- 'All', 'Indexed', 'NotSpecified'
    metadata JSONB DEFAULT '{}',
    parameters JSONB DEFAULT '{}',
    policy_rule JSONB NOT NULL,
    effect VARCHAR(50) DEFAULT 'audit', -- 'audit', 'deny', 'disabled', 'auditIfNotExists', 'deployIfNotExists'
    is_enabled BOOLEAN DEFAULT true,
    version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Policy assignments
CREATE TABLE policy_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    assignment_id VARCHAR(500) UNIQUE NOT NULL, -- Azure assignment ID
    display_name VARCHAR(255) NOT NULL,
    scope VARCHAR(500) NOT NULL, -- Resource group, subscription, etc.
    scope_type VARCHAR(50) NOT NULL, -- 'subscription', 'resourceGroup', 'resource'
    parameters JSONB DEFAULT '{}',
    identity JSONB DEFAULT '{}',
    location VARCHAR(50),
    not_scopes JSONB DEFAULT '[]',
    enforcement_mode VARCHAR(20) DEFAULT 'Default', -- 'Default', 'DoNotEnforce'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_by VARCHAR(255)
);

-- Compliance assessments
CREATE TABLE compliance_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_id UUID NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    policy_assignment_id UUID NOT NULL REFERENCES policy_assignments(id) ON DELETE CASCADE,
    assessment_id VARCHAR(500) UNIQUE NOT NULL,
    compliance_state compliance_status NOT NULL,
    result_description TEXT,
    result_details JSONB DEFAULT '{}',
    policy_evaluation_details JSONB DEFAULT '{}',
    assessed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- ML AND AI TABLES (Patent Implementation)
-- ============================================================

-- ML Models metadata
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'correlation', 'prediction', 'classification', 'nlp'
    version VARCHAR(20) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    model_path VARCHAR(500),
    training_data_hash VARCHAR(64),
    feature_importance JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'training',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    trained_at TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(tenant_id, name, version)
);

-- Feature store for ML
CREATE TABLE ml_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    feature_name VARCHAR(255) NOT NULL,
    feature_type VARCHAR(50) NOT NULL, -- 'categorical', 'numerical', 'text', 'boolean'
    feature_value JSONB NOT NULL,
    feature_metadata JSONB DEFAULT '{}',
    extraction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ttl_hours INTEGER DEFAULT 24,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Cross-domain correlations (Patent #1)
CREATE TABLE cross_domain_correlations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_resource_id UUID NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    target_resource_id UUID NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    correlation_type VARCHAR(100) NOT NULL, -- 'cost_impact', 'security_risk', 'performance_impact'
    correlation_strength DECIMAL(5,4) NOT NULL CHECK (correlation_strength >= -1 AND correlation_strength <= 1),
    confidence_score prediction_confidence NOT NULL,
    evidence JSONB DEFAULT '{}',
    impact_analysis JSONB DEFAULT '{}',
    discovered_by VARCHAR(100) DEFAULT 'ml_engine',
    discovery_method VARCHAR(100) NOT NULL,
    temporal_patterns JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Conversational AI sessions (Patent #2)
CREATE TABLE conversation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    context JSONB DEFAULT '{}',
    conversation_state JSONB DEFAULT '{}',
    intent_history JSONB DEFAULT '[]',
    entity_cache JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours')
);

-- Conversation messages
CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    message_type VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    intent VARCHAR(100),
    entities JSONB DEFAULT '{}',
    confidence_score DECIMAL(5,4),
    response_metadata JSONB DEFAULT '{}',
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Predictive compliance (Patent #4)
CREATE TABLE compliance_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_id UUID NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    prediction_type VARCHAR(100) NOT NULL, -- 'drift_detection', 'compliance_violation', 'cost_anomaly'
    predicted_value JSONB NOT NULL,
    confidence prediction_confidence NOT NULL,
    risk_factors JSONB DEFAULT '{}',
    recommended_actions JSONB DEFAULT '[]',
    shap_values JSONB DEFAULT '{}', -- SHAP explainability
    prediction_horizon_days INTEGER DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    prediction_for TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_outcome JSONB,
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5)
);

-- ============================================================
-- APPROVAL WORKFLOW TABLES
-- ============================================================

-- Approval workflows
CREATE TABLE approval_workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(100) NOT NULL, -- 'policy_exception', 'resource_deployment', 'access_request'
    approval_steps JSONB NOT NULL, -- Array of approval steps
    conditions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Approval requests
CREATE TABLE approval_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL REFERENCES approval_workflows(id) ON DELETE CASCADE,
    requester_id VARCHAR(255) NOT NULL,
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    request_type VARCHAR(100) NOT NULL,
    request_data JSONB NOT NULL,
    current_step INTEGER DEFAULT 1,
    status workflow_status DEFAULT 'pending',
    priority INTEGER DEFAULT 3 CHECK (priority >= 1 AND priority <= 5),
    due_date TIMESTAMP WITH TIME ZONE,
    reason TEXT,
    business_justification TEXT,
    risk_assessment JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255)
);

-- Approval actions
CREATE TABLE approval_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL REFERENCES approval_requests(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    approver_id VARCHAR(255) NOT NULL,
    action VARCHAR(20) NOT NULL, -- 'approve', 'reject', 'delegate', 'escalate'
    comments TEXT,
    action_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- AUDIT AND SECURITY TABLES
-- ============================================================

-- Audit log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    resource_id UUID REFERENCES resources(id) ON DELETE SET NULL,
    action audit_action NOT NULL,
    resource_type VARCHAR(100),
    resource_name VARCHAR(255),
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    correlation_id VARCHAR(255),
    severity severity_level DEFAULT 'low',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Security events
CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(100) NOT NULL, -- 'defender', 'sentinel', 'policy_cortex'
    resource_id UUID REFERENCES resources(id) ON DELETE SET NULL,
    severity severity_level NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    recommendations JSONB DEFAULT '[]',
    affected_resources JSONB DEFAULT '[]',
    remediation_status VARCHAR(50) DEFAULT 'open',
    event_data JSONB DEFAULT '{}',
    external_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255)
);

-- ============================================================
-- NOTIFICATION AND ALERTING TABLES
-- ============================================================

-- Notification channels
CREATE TABLE notification_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    channel_type VARCHAR(50) NOT NULL, -- 'email', 'teams', 'slack', 'webhook'
    configuration JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, name)
);

-- Alert rules
CREATE TABLE alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    rule_type VARCHAR(100) NOT NULL, -- 'compliance_drift', 'cost_anomaly', 'security_event'
    conditions JSONB NOT NULL,
    severity severity_level NOT NULL,
    notification_channels JSONB DEFAULT '[]', -- Array of channel IDs
    is_enabled BOOLEAN DEFAULT true,
    throttle_minutes INTEGER DEFAULT 60,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Notifications sent
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    alert_rule_id UUID REFERENCES alert_rules(id) ON DELETE SET NULL,
    channel_id UUID NOT NULL REFERENCES notification_channels(id) ON DELETE CASCADE,
    recipient VARCHAR(255) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    notification_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'sent', 'failed', 'retry'
    attempts INTEGER DEFAULT 0,
    sent_at TIMESTAMP WITH TIME ZONE,
    failed_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- PERFORMANCE AND METRICS TABLES
-- ============================================================

-- System metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'counter', 'gauge', 'histogram'
    value DECIMAL(15,4) NOT NULL,
    unit VARCHAR(20),
    dimensions JSONB DEFAULT '{}',
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    request_size_bytes INTEGER DEFAULT 0,
    response_size_bytes INTEGER DEFAULT 0,
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================================

-- Tenant isolation indexes
CREATE INDEX idx_resources_tenant_id ON resources(tenant_id);
CREATE INDEX idx_resources_tenant_resource_id ON resources(tenant_id, resource_id);
CREATE INDEX idx_resources_type_status ON resources(type, status);
CREATE INDEX idx_resources_updated_at ON resources(updated_at DESC);

-- Policy and compliance indexes
CREATE INDEX idx_policies_tenant_category ON policies(tenant_id, category);
CREATE INDEX idx_policy_assignments_tenant_scope ON policy_assignments(tenant_id, scope);
CREATE INDEX idx_compliance_assessments_resource ON compliance_assessments(resource_id, assessed_at DESC);
CREATE INDEX idx_compliance_assessments_policy ON compliance_assessments(policy_assignment_id, compliance_state);

-- ML and correlation indexes
CREATE INDEX idx_correlations_tenant_source ON cross_domain_correlations(tenant_id, source_resource_id);
CREATE INDEX idx_correlations_strength ON cross_domain_correlations(correlation_strength DESC);
CREATE INDEX idx_predictions_resource_date ON compliance_predictions(resource_id, prediction_for);
CREATE INDEX idx_predictions_confidence ON compliance_predictions(confidence, created_at DESC);

-- Audit and security indexes
CREATE INDEX idx_audit_log_tenant_user ON audit_log(tenant_id, user_id, created_at DESC);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_id, created_at DESC);
CREATE INDEX idx_security_events_tenant_severity ON security_events(tenant_id, severity, created_at DESC);

-- Conversation and session indexes
CREATE INDEX idx_conversation_sessions_user ON conversation_sessions(tenant_id, user_id, created_at DESC);
CREATE INDEX idx_conversation_messages_session ON conversation_messages(session_id, created_at);

-- Feature store indexes
CREATE INDEX idx_ml_features_resource ON ml_features(resource_id, feature_name);
CREATE INDEX idx_ml_features_tenant_name ON ml_features(tenant_id, feature_name, extraction_timestamp DESC);

-- Notification indexes
CREATE INDEX idx_notifications_status ON notifications(status, created_at);
CREATE INDEX idx_alert_rules_enabled ON alert_rules(tenant_id, is_enabled);

-- Performance monitoring indexes
CREATE INDEX idx_system_metrics_tenant_name ON system_metrics(tenant_id, metric_name, collected_at DESC);
CREATE INDEX idx_api_usage_tenant_endpoint ON api_usage(tenant_id, endpoint, created_at DESC);

-- Text search indexes using trigrams
CREATE INDEX idx_resources_name_trgm ON resources USING gin(name gin_trgm_ops);
CREATE INDEX idx_policies_name_trgm ON policies USING gin(display_name gin_trgm_ops);

-- JSON indexes for efficient querying
CREATE INDEX idx_resources_tags ON resources USING gin(tags);
CREATE INDEX idx_resources_properties ON resources USING gin(properties);
CREATE INDEX idx_correlation_evidence ON cross_domain_correlations USING gin(evidence);

-- ============================================================
-- CREATE FUNCTIONS AND TRIGGERS
-- ============================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_tenant_users_updated_at BEFORE UPDATE ON tenant_users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_resources_updated_at BEFORE UPDATE ON resources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_policy_assignments_updated_at BEFORE UPDATE ON policy_assignments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cross_domain_correlations_updated_at BEFORE UPDATE ON cross_domain_correlations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conversation_sessions_updated_at BEFORE UPDATE ON conversation_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_approval_workflows_updated_at BEFORE UPDATE ON approval_workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_approval_requests_updated_at BEFORE UPDATE ON approval_requests FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_security_events_updated_at BEFORE UPDATE ON security_events FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_notification_channels_updated_at BEFORE UPDATE ON notification_channels FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_alert_rules_updated_at BEFORE UPDATE ON alert_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate resource compliance score
CREATE OR REPLACE FUNCTION calculate_compliance_score(resource_uuid UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    compliant_count INTEGER;
    total_count INTEGER;
    score DECIMAL(5,2);
BEGIN
    SELECT 
        COUNT(CASE WHEN compliance_state = 'compliant' THEN 1 END),
        COUNT(*)
    INTO compliant_count, total_count
    FROM compliance_assessments 
    WHERE resource_id = resource_uuid;
    
    IF total_count = 0 THEN
        RETURN 0;
    END IF;
    
    score := (compliant_count::DECIMAL / total_count::DECIMAL) * 100;
    RETURN ROUND(score, 2);
END;
$$ LANGUAGE plpgsql;

-- Function to get resource risk score based on multiple factors
CREATE OR REPLACE FUNCTION calculate_risk_score(resource_uuid UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    compliance_factor DECIMAL(5,2);
    security_factor DECIMAL(5,2);
    prediction_factor DECIMAL(5,2);
    final_score DECIMAL(5,2);
BEGIN
    -- Compliance factor (inverse of compliance score)
    SELECT 100 - COALESCE(compliance_score, 0) INTO compliance_factor
    FROM resources WHERE id = resource_uuid;
    
    -- Security events factor
    SELECT CASE 
        WHEN COUNT(*) = 0 THEN 0
        WHEN COUNT(*) BETWEEN 1 AND 5 THEN 20
        WHEN COUNT(*) BETWEEN 6 AND 10 THEN 40
        ELSE 60
    END INTO security_factor
    FROM security_events 
    WHERE resource_id = resource_uuid 
    AND created_at > CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Prediction factor (based on high-risk predictions)
    SELECT CASE 
        WHEN COUNT(*) = 0 THEN 0
        ELSE 30
    END INTO prediction_factor
    FROM compliance_predictions 
    WHERE resource_id = resource_uuid 
    AND confidence IN ('high', 'very_high')
    AND created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    final_score := LEAST((compliance_factor * 0.5) + (security_factor * 0.3) + (prediction_factor * 0.2), 100);
    RETURN ROUND(final_score, 2);
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- INSERT INITIAL DATA
-- ============================================================

-- Insert default tenant (for development)
INSERT INTO tenants (name, domain, subscription_id, tenant_id, settings) VALUES
(
    'PolicyCortex Development',
    'dev.policycortex.com',
    '6dc7cfa2-0332-4740-98b6-bac9f1a23de9',
    'e1f3e196-aa55-4709-9c55-0e334c0b444f',
    '{"theme": "dark", "notifications": true, "auto_remediation": false}'
);

-- Insert default admin user
INSERT INTO tenant_users (tenant_id, user_id, email, name, roles, permissions) VALUES
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'admin@policycortex.com',
    'admin@policycortex.com',
    'PolicyCortex Administrator',
    '["admin", "security_manager", "compliance_officer"]',
    '{"read": "*", "write": "*", "delete": "*", "approve": "*"}'
);

-- Insert built-in policies (sample)
INSERT INTO policies (tenant_id, name, display_name, description, category, policy_type, policy_rule, effect) VALUES
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'require-resource-tags',
    'Require Resource Tags',
    'Ensures all resources have required tags like Environment, Owner, and CostCenter',
    'Tagging',
    'built-in',
    '{"if": {"anyOf": [{"field": "tags.Environment", "exists": "false"}, {"field": "tags.Owner", "exists": "false"}, {"field": "tags.CostCenter", "exists": "false"}]}, "then": {"effect": "audit"}}',
    'audit'
),
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'allowed-vm-skus',
    'Allowed Virtual Machine SKUs',
    'Restricts virtual machine SKUs to approved list for cost optimization',
    'Compute',
    'built-in',
    '{"if": {"allOf": [{"field": "type", "equals": "Microsoft.Compute/virtualMachines"}, {"not": {"field": "Microsoft.Compute/virtualMachines/sku.name", "in": ["Standard_B2s", "Standard_D2s_v3", "Standard_D4s_v3"]}}]}, "then": {"effect": "deny"}}',
    'deny'
),
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'require-https-storage',
    'Require HTTPS for Storage Accounts',
    'Ensures storage accounts only accept HTTPS traffic for security',
    'Security',
    'built-in',
    '{"if": {"allOf": [{"field": "type", "equals": "Microsoft.Storage/storageAccounts"}, {"field": "Microsoft.Storage/storageAccounts/supportsHttpsTrafficOnly", "notEquals": "true"}]}, "then": {"effect": "deny"}}',
    'deny'
);

-- Insert default approval workflow
INSERT INTO approval_workflows (tenant_id, name, description, workflow_type, approval_steps) VALUES
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'Policy Exception Request',
    'Standard workflow for requesting exceptions to security and compliance policies',
    'policy_exception',
    '[{"step": 1, "approver_roles": ["security_manager"], "required_count": 1}, {"step": 2, "approver_roles": ["compliance_officer"], "required_count": 1}]'
);

-- Insert default notification channel
INSERT INTO notification_channels (tenant_id, name, channel_type, configuration) VALUES
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'Admin Email',
    'email',
    '{"recipients": ["admin@policycortex.com"], "smtp_server": "localhost", "port": 587}'
);

-- Insert default alert rules
INSERT INTO alert_rules (tenant_id, name, description, rule_type, conditions, severity, notification_channels) VALUES
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'High Risk Resource Detected',
    'Alert when a resource has a risk score above 80',
    'risk_threshold',
    '{"metric": "risk_score", "operator": "greater_than", "threshold": 80}',
    'high',
    '[]'
),
(
    (SELECT id FROM tenants WHERE domain = 'dev.policycortex.com'),
    'Compliance Drift Detected',
    'Alert when compliance score drops below 70%',
    'compliance_drift',
    '{"metric": "compliance_score", "operator": "less_than", "threshold": 70}',
    'medium',
    '[]'
);

-- ============================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- ============================================================

-- Update table statistics
ANALYZE;

-- Set up automatic statistics collection
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_io_timing = on;

-- Configure autovacuum for better performance
ALTER SYSTEM SET autovacuum_max_workers = 4;
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 2000;

-- ============================================================
-- SECURITY SETTINGS
-- ============================================================

-- Create read-only role for reporting
CREATE ROLE policy_cortex_readonly;
GRANT USAGE ON SCHEMA policy_cortex TO policy_cortex_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA policy_cortex TO policy_cortex_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA policy_cortex GRANT SELECT ON TABLES TO policy_cortex_readonly;

-- Create application role with limited permissions
CREATE ROLE policy_cortex_app;
GRANT USAGE ON SCHEMA policy_cortex TO policy_cortex_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA policy_cortex TO policy_cortex_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA policy_cortex TO policy_cortex_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA policy_cortex GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO policy_cortex_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA policy_cortex GRANT USAGE, SELECT ON SEQUENCES TO policy_cortex_app;

-- ============================================================
-- FINAL SETUP COMMANDS
-- ============================================================

-- Reset search path
RESET search_path;

-- Create completion marker
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    installed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES 
('2.0.0', 'PolicyCortex complete database schema with patent implementations')
ON CONFLICT (version) DO UPDATE SET 
    installed_at = CURRENT_TIMESTAMP,
    description = EXCLUDED.description;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'PolicyCortex database initialization completed successfully';
    RAISE NOTICE 'Schema version: 2.0.0';
    RAISE NOTICE 'Tables created: %', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'policy_cortex');
    RAISE NOTICE 'Indexes created: %', (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'policy_cortex');
END
$$;