-- Tenant Isolation Migration
-- This migration adds comprehensive tenant isolation to all tables
-- ensuring data separation and security between different tenants

-- Add tenant_id to policies table if not exists
ALTER TABLE IF EXISTS policies 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) NOT NULL DEFAULT 'e1f3e196-aa55-4709-9c55-0e334c0b444f';

-- Add tenant_id to resources table if not exists
ALTER TABLE IF EXISTS resources 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) NOT NULL DEFAULT 'e1f3e196-aa55-4709-9c55-0e334c0b444f';

-- Add tenant_id to actions table if not exists
ALTER TABLE IF EXISTS actions 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) NOT NULL DEFAULT 'e1f3e196-aa55-4709-9c55-0e334c0b444f';

-- Add tenant_id to audit_logs table if not exists
ALTER TABLE IF EXISTS audit_logs 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) NOT NULL DEFAULT 'e1f3e196-aa55-4709-9c55-0e334c0b444f';

-- Add tenant_id to approvals table if not exists
ALTER TABLE IF EXISTS approvals 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) NOT NULL DEFAULT 'e1f3e196-aa55-4709-9c55-0e334c0b444f';

-- Add tenant_id to exceptions table if not exists
ALTER TABLE IF EXISTS exceptions 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) NOT NULL DEFAULT 'e1f3e196-aa55-4709-9c55-0e334c0b444f';

-- Create indexes for tenant_id on all tables for performance
CREATE INDEX IF NOT EXISTS idx_policies_tenant_id ON policies(tenant_id);
CREATE INDEX IF NOT EXISTS idx_resources_tenant_id ON resources(tenant_id);
CREATE INDEX IF NOT EXISTS idx_actions_tenant_id ON actions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_approvals_tenant_id ON approvals(tenant_id);
CREATE INDEX IF NOT EXISTS idx_exceptions_tenant_id ON exceptions(tenant_id);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_policies_tenant_status ON policies(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_resources_tenant_type ON resources(tenant_id, resource_type);
CREATE INDEX IF NOT EXISTS idx_actions_tenant_status ON actions(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_time ON audit_logs(tenant_id, created_at DESC);

-- Create Row Level Security (RLS) policies
-- Enable RLS on all tables
ALTER TABLE policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE resources ENABLE ROW LEVEL SECURITY;
ALTER TABLE actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE exceptions ENABLE ROW LEVEL SECURITY;

-- Create function to get current tenant from session
CREATE OR REPLACE FUNCTION current_tenant_id() 
RETURNS VARCHAR(255) AS $$
BEGIN
    -- Get tenant_id from session variable set by application
    RETURN current_setting('app.current_tenant_id', TRUE);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create RLS policies for each table
-- Policies table
CREATE POLICY tenant_isolation_select_policies ON policies
    FOR SELECT
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_insert_policies ON policies
    FOR INSERT
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_update_policies ON policies
    FOR UPDATE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL)
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_delete_policies ON policies
    FOR DELETE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Resources table
CREATE POLICY tenant_isolation_select_resources ON resources
    FOR SELECT
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_insert_resources ON resources
    FOR INSERT
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_update_resources ON resources
    FOR UPDATE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL)
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_delete_resources ON resources
    FOR DELETE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Actions table
CREATE POLICY tenant_isolation_select_actions ON actions
    FOR SELECT
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_insert_actions ON actions
    FOR INSERT
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_update_actions ON actions
    FOR UPDATE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL)
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_delete_actions ON actions
    FOR DELETE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Audit logs table (read-only for most operations)
CREATE POLICY tenant_isolation_select_audit_logs ON audit_logs
    FOR SELECT
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_insert_audit_logs ON audit_logs
    FOR INSERT
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Approvals table
CREATE POLICY tenant_isolation_select_approvals ON approvals
    FOR SELECT
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_insert_approvals ON approvals
    FOR INSERT
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_update_approvals ON approvals
    FOR UPDATE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL)
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Exceptions table
CREATE POLICY tenant_isolation_select_exceptions ON exceptions
    FOR SELECT
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_insert_exceptions ON exceptions
    FOR INSERT
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

CREATE POLICY tenant_isolation_update_exceptions ON exceptions
    FOR UPDATE
    USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL)
    WITH CHECK (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL);

-- Create tenant management table
CREATE TABLE IF NOT EXISTS tenants (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    azure_tenant_id VARCHAR(255) UNIQUE NOT NULL,
    subscription_ids TEXT[], -- Array of Azure subscription IDs
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Insert default tenant
INSERT INTO tenants (id, name, azure_tenant_id, subscription_ids)
VALUES (
    'e1f3e196-aa55-4709-9c55-0e334c0b444f',
    'Default Tenant',
    'e1f3e196-aa55-4709-9c55-0e334c0b444f',
    ARRAY['6dc7cfa2-0332-4740-98b6-bac9f1a23de9']
) ON CONFLICT (id) DO NOTHING;

-- Create function to validate tenant access
CREATE OR REPLACE FUNCTION validate_tenant_access(user_tenant_id VARCHAR(255), resource_tenant_id VARCHAR(255))
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if user's tenant matches resource tenant
    RETURN user_tenant_id = resource_tenant_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add trigger to automatically set tenant_id on insert
CREATE OR REPLACE FUNCTION set_tenant_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.tenant_id IS NULL THEN
        NEW.tenant_id := current_tenant_id();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for all tables
CREATE TRIGGER set_policies_tenant_id
    BEFORE INSERT ON policies
    FOR EACH ROW
    EXECUTE FUNCTION set_tenant_id();

CREATE TRIGGER set_resources_tenant_id
    BEFORE INSERT ON resources
    FOR EACH ROW
    EXECUTE FUNCTION set_tenant_id();

CREATE TRIGGER set_actions_tenant_id
    BEFORE INSERT ON actions
    FOR EACH ROW
    EXECUTE FUNCTION set_tenant_id();

CREATE TRIGGER set_audit_logs_tenant_id
    BEFORE INSERT ON audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION set_tenant_id();

CREATE TRIGGER set_approvals_tenant_id
    BEFORE INSERT ON approvals
    FOR EACH ROW
    EXECUTE FUNCTION set_tenant_id();

CREATE TRIGGER set_exceptions_tenant_id
    BEFORE INSERT ON exceptions
    FOR EACH ROW
    EXECUTE FUNCTION set_tenant_id();

-- Grant necessary permissions to application role
GRANT USAGE ON SCHEMA public TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;