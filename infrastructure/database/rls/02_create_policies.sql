-- Create RLS policies for all tables

-- Helper function to create standard RLS policies
CREATE OR REPLACE FUNCTION create_tenant_rls_policies(p_table_name TEXT)
RETURNS VOID AS $$
DECLARE
    policy_select TEXT;
    policy_insert TEXT;
    policy_update TEXT;
    policy_delete TEXT;
BEGIN
    -- Drop existing policies if they exist
    EXECUTE format('DROP POLICY IF EXISTS tenant_select_policy ON %I', p_table_name);
    EXECUTE format('DROP POLICY IF EXISTS tenant_insert_policy ON %I', p_table_name);
    EXECUTE format('DROP POLICY IF EXISTS tenant_update_policy ON %I', p_table_name);
    EXECUTE format('DROP POLICY IF EXISTS tenant_delete_policy ON %I', p_table_name);
    
    -- Create SELECT policy
    policy_select := format(
        'CREATE POLICY tenant_select_policy ON %I 
         FOR SELECT 
         USING (tenant_id = current_tenant_id() OR current_tenant_id() IS NULL)',
        p_table_name
    );
    EXECUTE policy_select;
    
    -- Create INSERT policy
    policy_insert := format(
        'CREATE POLICY tenant_insert_policy ON %I 
         FOR INSERT 
         WITH CHECK (tenant_id = current_tenant_id())',
        p_table_name
    );
    EXECUTE policy_insert;
    
    -- Create UPDATE policy
    policy_update := format(
        'CREATE POLICY tenant_update_policy ON %I 
         FOR UPDATE 
         USING (tenant_id = current_tenant_id())
         WITH CHECK (tenant_id = current_tenant_id())',
        p_table_name
    );
    EXECUTE policy_update;
    
    -- Create DELETE policy
    policy_delete := format(
        'CREATE POLICY tenant_delete_policy ON %I 
         FOR DELETE 
         USING (tenant_id = current_tenant_id())',
        p_table_name
    );
    EXECUTE policy_delete;
    
    RAISE NOTICE 'Created RLS policies for table: %', p_table_name;
END;
$$ LANGUAGE plpgsql;

-- Apply RLS policies to all tenant-scoped tables
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
        AND EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = tablename 
            AND column_name = 'tenant_id'
        )
    LOOP
        PERFORM create_tenant_rls_policies(tbl.tablename);
    END LOOP;
END $$;

-- Special policies for tenants table
DROP POLICY IF EXISTS tenants_select_policy ON tenants;
DROP POLICY IF EXISTS tenants_insert_policy ON tenants;
DROP POLICY IF EXISTS tenants_update_policy ON tenants;
DROP POLICY IF EXISTS tenants_delete_policy ON tenants;

CREATE POLICY tenants_select_policy ON tenants
    FOR SELECT
    USING (
        id = current_tenant_id() 
        OR EXISTS (
            SELECT 1 FROM tenant_users 
            WHERE tenant_id = tenants.id 
            AND user_id = current_setting('app.current_user_id', true)
        )
    );

CREATE POLICY tenants_insert_policy ON tenants
    FOR INSERT
    WITH CHECK (
        current_setting('app.current_user_role', true) = 'super_admin'
    );

CREATE POLICY tenants_update_policy ON tenants
    FOR UPDATE
    USING (
        id = current_tenant_id()
        OR current_setting('app.current_user_role', true) = 'super_admin'
    )
    WITH CHECK (
        id = current_tenant_id()
        OR current_setting('app.current_user_role', true) = 'super_admin'
    );

CREATE POLICY tenants_delete_policy ON tenants
    FOR DELETE
    USING (
        current_setting('app.current_user_role', true) = 'super_admin'
    );

-- Special policies for tenant_users table
DROP POLICY IF EXISTS tenant_users_select_policy ON tenant_users;
DROP POLICY IF EXISTS tenant_users_insert_policy ON tenant_users;
DROP POLICY IF EXISTS tenant_users_update_policy ON tenant_users;
DROP POLICY IF EXISTS tenant_users_delete_policy ON tenant_users;

CREATE POLICY tenant_users_select_policy ON tenant_users
    FOR SELECT
    USING (
        tenant_id = current_tenant_id()
        OR user_id = current_setting('app.current_user_id', true)
    );

CREATE POLICY tenant_users_insert_policy ON tenant_users
    FOR INSERT
    WITH CHECK (
        tenant_id = current_tenant_id()
        AND (
            current_setting('app.current_user_role', true) IN ('admin', 'super_admin')
            OR current_setting('app.current_user_id', true) = user_id
        )
    );

CREATE POLICY tenant_users_update_policy ON tenant_users
    FOR UPDATE
    USING (
        tenant_id = current_tenant_id()
        AND (
            current_setting('app.current_user_role', true) IN ('admin', 'super_admin')
            OR user_id = current_setting('app.current_user_id', true)
        )
    )
    WITH CHECK (
        tenant_id = current_tenant_id()
        AND (
            current_setting('app.current_user_role', true) IN ('admin', 'super_admin')
            OR user_id = current_setting('app.current_user_id', true)
        )
    );

CREATE POLICY tenant_users_delete_policy ON tenant_users
    FOR DELETE
    USING (
        tenant_id = current_tenant_id()
        AND current_setting('app.current_user_role', true) IN ('admin', 'super_admin')
    );

-- Create bypass role for migrations and admin operations
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rls_bypass') THEN
        CREATE ROLE rls_bypass;
    END IF;
END $$;

GRANT ALL ON ALL TABLES IN SCHEMA public TO rls_bypass;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO rls_bypass;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO rls_bypass;

-- Grant bypass to specific tables for admin operations
ALTER TABLE tenants OWNER TO rls_bypass;
ALTER TABLE tenant_users OWNER TO rls_bypass;