-- PolicyCortex Row-Level Security Implementation
-- This script enables RLS for multi-tenant data isolation

-- Create tenant table if not exists
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    subdomain VARCHAR(100) UNIQUE,
    tier VARCHAR(50) NOT NULL DEFAULT 'basic' CHECK (tier IN ('basic', 'premium', 'enterprise')),
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deleted')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create tenant users table
CREATE TABLE IF NOT EXISTS tenant_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    role VARCHAR(100) NOT NULL DEFAULT 'user',
    permissions JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, user_id),
    UNIQUE(tenant_id, email)
);

-- Create index for performance
CREATE INDEX idx_tenant_users_tenant_id ON tenant_users(tenant_id);
CREATE INDEX idx_tenant_users_user_id ON tenant_users(user_id);
CREATE INDEX idx_tenant_users_email ON tenant_users(email);

-- Add tenant_id to all existing tables
DO $$
DECLARE
    tbl RECORD;
BEGIN
    FOR tbl IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename NOT IN ('tenants', 'tenant_users', 'migrations')
        AND tablename NOT LIKE 'pg_%'
    LOOP
        -- Check if tenant_id column already exists
        IF NOT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = tbl.tablename 
            AND column_name = 'tenant_id'
        ) THEN
            EXECUTE format('ALTER TABLE %I ADD COLUMN tenant_id UUID', tbl.tablename);
            EXECUTE format('CREATE INDEX idx_%I_tenant_id ON %I(tenant_id)', 
                          tbl.tablename, tbl.tablename);
        END IF;
    END LOOP;
END $$;

-- Enable RLS on all tables
DO $$
DECLARE
    tbl RECORD;
BEGIN
    FOR tbl IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename NOT IN ('migrations')
        AND tablename NOT LIKE 'pg_%'
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl.tablename);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', tbl.tablename);
    END LOOP;
END $$;

-- Create a function to get current tenant_id from session
CREATE OR REPLACE FUNCTION current_tenant_id() 
RETURNS UUID AS $$
BEGIN
    RETURN current_setting('app.current_tenant_id', true)::UUID;
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a function to set current tenant_id in session
CREATE OR REPLACE FUNCTION set_current_tenant(p_tenant_id UUID) 
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_tenant_id', p_tenant_id::text, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a function to validate tenant access
CREATE OR REPLACE FUNCTION validate_tenant_access(
    p_user_id VARCHAR(255),
    p_tenant_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    v_exists BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 
        FROM tenant_users 
        WHERE user_id = p_user_id 
        AND tenant_id = p_tenant_id
    ) INTO v_exists;
    
    RETURN v_exists;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO PUBLIC;
GRANT EXECUTE ON FUNCTION current_tenant_id() TO PUBLIC;
GRANT EXECUTE ON FUNCTION set_current_tenant(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION validate_tenant_access(VARCHAR, UUID) TO PUBLIC;