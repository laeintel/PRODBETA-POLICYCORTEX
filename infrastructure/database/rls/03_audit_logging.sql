-- Create audit logging for tenant operations

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    user_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    table_name VARCHAR(100),
    record_id VARCHAR(255),
    old_data JSONB,
    new_data JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_table_name ON audit_logs(table_name);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_session_id ON audit_logs(session_id);

-- Enable RLS on audit_logs
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs FORCE ROW LEVEL SECURITY;

-- Create RLS policy for audit_logs (read-only for tenants)
CREATE POLICY audit_logs_select_policy ON audit_logs
    FOR SELECT
    USING (tenant_id = current_tenant_id());

-- Only system can insert audit logs
CREATE POLICY audit_logs_insert_policy ON audit_logs
    FOR INSERT
    WITH CHECK (false); -- Prevent direct inserts, use function instead

-- Function to log audit events
CREATE OR REPLACE FUNCTION log_audit_event(
    p_action VARCHAR(50),
    p_table_name VARCHAR(100),
    p_record_id VARCHAR(255),
    p_old_data JSONB DEFAULT NULL,
    p_new_data JSONB DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    v_audit_id UUID;
    v_tenant_id UUID;
    v_user_id VARCHAR(255);
    v_ip_address INET;
    v_user_agent TEXT;
    v_session_id VARCHAR(255);
BEGIN
    -- Get current context
    v_tenant_id := current_setting('app.current_tenant_id', true)::UUID;
    v_user_id := current_setting('app.current_user_id', true);
    v_ip_address := current_setting('app.client_ip', true)::INET;
    v_user_agent := current_setting('app.user_agent', true);
    v_session_id := current_setting('app.session_id', true);
    
    -- Insert audit log with SECURITY DEFINER to bypass RLS
    INSERT INTO audit_logs (
        tenant_id,
        user_id,
        action,
        table_name,
        record_id,
        old_data,
        new_data,
        ip_address,
        user_agent,
        session_id,
        metadata
    ) VALUES (
        v_tenant_id,
        v_user_id,
        p_action,
        p_table_name,
        p_record_id,
        p_old_data,
        p_new_data,
        v_ip_address,
        v_user_agent,
        v_session_id,
        p_metadata
    ) RETURNING id INTO v_audit_id;
    
    RETURN v_audit_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger function for automatic audit logging
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    v_old_data JSONB;
    v_new_data JSONB;
    v_record_id VARCHAR(255);
    v_action VARCHAR(50);
BEGIN
    -- Determine action
    IF TG_OP = 'INSERT' THEN
        v_action := 'INSERT';
        v_new_data := to_jsonb(NEW);
        v_old_data := NULL;
        v_record_id := COALESCE(NEW.id::text, 'unknown');
    ELSIF TG_OP = 'UPDATE' THEN
        v_action := 'UPDATE';
        v_new_data := to_jsonb(NEW);
        v_old_data := to_jsonb(OLD);
        v_record_id := COALESCE(NEW.id::text, OLD.id::text, 'unknown');
    ELSIF TG_OP = 'DELETE' THEN
        v_action := 'DELETE';
        v_new_data := NULL;
        v_old_data := to_jsonb(OLD);
        v_record_id := COALESCE(OLD.id::text, 'unknown');
    END IF;
    
    -- Log the audit event
    PERFORM log_audit_event(
        v_action,
        TG_TABLE_NAME,
        v_record_id,
        v_old_data,
        v_new_data,
        jsonb_build_object(
            'trigger_name', TG_NAME,
            'schema_name', TG_TABLE_SCHEMA,
            'operation', TG_OP
        )
    );
    
    -- Return appropriate value
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to add audit triggers to tables
CREATE OR REPLACE FUNCTION add_audit_trigger(p_table_name TEXT)
RETURNS VOID AS $$
DECLARE
    v_trigger_name TEXT;
BEGIN
    v_trigger_name := p_table_name || '_audit_trigger';
    
    -- Drop existing trigger if it exists
    EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', v_trigger_name, p_table_name);
    
    -- Create new trigger
    EXECUTE format(
        'CREATE TRIGGER %I
         AFTER INSERT OR UPDATE OR DELETE ON %I
         FOR EACH ROW EXECUTE FUNCTION audit_trigger_function()',
        v_trigger_name, p_table_name
    );
    
    RAISE NOTICE 'Added audit trigger to table: %', p_table_name;
END;
$$ LANGUAGE plpgsql;

-- Add audit triggers to critical tables
DO $$
DECLARE
    critical_tables TEXT[] := ARRAY[
        'tenants',
        'tenant_users',
        'resources',
        'policies',
        'compliance_reports',
        'cost_analysis',
        'security_alerts'
    ];
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY critical_tables
    LOOP
        -- Check if table exists before adding trigger
        IF EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = tbl
        ) THEN
            PERFORM add_audit_trigger(tbl);
        END IF;
    END LOOP;
END $$;

-- Create view for audit log analysis
CREATE OR REPLACE VIEW audit_summary AS
SELECT 
    tenant_id,
    DATE(created_at) as audit_date,
    action,
    table_name,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT session_id) as unique_sessions
FROM audit_logs
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY tenant_id, DATE(created_at), action, table_name;

-- Grant permissions
GRANT EXECUTE ON FUNCTION log_audit_event(VARCHAR, VARCHAR, VARCHAR, JSONB, JSONB, JSONB) TO PUBLIC;
GRANT SELECT ON audit_summary TO PUBLIC;