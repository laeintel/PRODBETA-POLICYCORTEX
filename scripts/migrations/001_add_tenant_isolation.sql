-- Migration: Add Tenant Isolation
-- Version: 001
-- Date: 2025-08-09
-- Description: Adds tenant_id to all tables and implements row-level security

-- Create tenants table
CREATE TABLE IF NOT EXISTS public.tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    azure_tenant_id VARCHAR(100),
    subscription_ids TEXT[],
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add tenant_id to all governance tables
ALTER TABLE governance.policies 
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_policies_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE governance.resources 
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_resources_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE governance.actions
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_actions_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE governance.exceptions
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_exceptions_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

-- Add tenant_id to compliance tables
ALTER TABLE compliance.violations
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_violations_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE compliance.evidence
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_evidence_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE compliance.controls
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_controls_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

-- Add tenant_id to security tables
ALTER TABLE security.findings
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_findings_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE security.incidents
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_incidents_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

-- Add tenant_id to finops tables
ALTER TABLE finops.cost_data
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_cost_data_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

ALTER TABLE finops.optimizations
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_optimizations_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

-- Add tenant_id to audit tables
ALTER TABLE audit.logs
    ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100),
    ADD CONSTRAINT fk_audit_logs_tenant FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);

-- Create indexes for tenant_id
CREATE INDEX IF NOT EXISTS idx_policies_tenant ON governance.policies(tenant_id);
CREATE INDEX IF NOT EXISTS idx_resources_tenant ON governance.resources(tenant_id);
CREATE INDEX IF NOT EXISTS idx_actions_tenant ON governance.actions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_exceptions_tenant ON governance.exceptions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_violations_tenant ON compliance.violations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_evidence_tenant ON compliance.evidence(tenant_id);
CREATE INDEX IF NOT EXISTS idx_controls_tenant ON compliance.controls(tenant_id);
CREATE INDEX IF NOT EXISTS idx_findings_tenant ON security.findings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_incidents_tenant ON security.incidents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_cost_data_tenant ON finops.cost_data(tenant_id);
CREATE INDEX IF NOT EXISTS idx_optimizations_tenant ON finops.optimizations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant ON audit.logs(tenant_id);

-- Enable Row Level Security
ALTER TABLE governance.policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.resources ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE governance.exceptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.violations ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.controls ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE finops.cost_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE finops.optimizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit.logs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Policies table
CREATE POLICY tenant_isolation_policies ON governance.policies
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Resources table  
CREATE POLICY tenant_isolation_resources ON governance.resources
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Actions table
CREATE POLICY tenant_isolation_actions ON governance.actions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Exceptions table
CREATE POLICY tenant_isolation_exceptions ON governance.exceptions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Violations table
CREATE POLICY tenant_isolation_violations ON compliance.violations
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Evidence table
CREATE POLICY tenant_isolation_evidence ON compliance.evidence
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Controls table
CREATE POLICY tenant_isolation_controls ON compliance.controls
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Findings table
CREATE POLICY tenant_isolation_findings ON security.findings
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Incidents table
CREATE POLICY tenant_isolation_incidents ON security.incidents
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Cost data table
CREATE POLICY tenant_isolation_cost_data ON finops.cost_data
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Optimizations table
CREATE POLICY tenant_isolation_optimizations ON finops.optimizations
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Audit logs table
CREATE POLICY tenant_isolation_audit_logs ON audit.logs
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::VARCHAR);

-- Grant permissions
GRANT ALL ON public.tenants TO policycortex_app;
GRANT USAGE ON SCHEMA governance, compliance, security, finops, audit TO policycortex_app;
GRANT ALL ON ALL TABLES IN SCHEMA governance, compliance, security, finops, audit TO policycortex_app;

-- Create function to set tenant context
CREATE OR REPLACE FUNCTION set_tenant_context(p_tenant_id VARCHAR)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_tenant', p_tenant_id, false);
END;
$$ LANGUAGE plpgsql;

-- Create default tenant for development
INSERT INTO public.tenants (tenant_id, name, azure_tenant_id)
VALUES ('default', 'Default Tenant', 'e1f3e196-aa55-4709-9c55-0e334c0b444f')
ON CONFLICT (tenant_id) DO NOTHING;