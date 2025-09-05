-- Migration: Create auth and RBAC tables
-- Required for T08 - Auth/RBAC implementation

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    azure_ad_id VARCHAR(255) UNIQUE,
    display_name VARCHAR(255),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Authentication
    is_active BOOLEAN DEFAULT TRUE,
    is_service_account BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    INDEX idx_users_email (email),
    INDEX idx_users_azure (azure_ad_id),
    INDEX idx_users_tenant (tenant_id)
);

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_name VARCHAR(100) NOT NULL,
    role_description TEXT,
    tenant_id VARCHAR(255),
    is_system_role BOOLEAN DEFAULT FALSE,
    
    -- Role hierarchy
    parent_role_id UUID REFERENCES roles(id),
    role_level INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(role_name, tenant_id),
    INDEX idx_roles_name (role_name),
    INDEX idx_roles_tenant (tenant_id)
);

-- Permissions table
CREATE TABLE IF NOT EXISTS permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    permission_name VARCHAR(255) NOT NULL UNIQUE,
    resource_type VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL, -- 'read', 'write', 'delete', 'execute'
    permission_description TEXT,
    
    INDEX idx_permissions_resource (resource_type),
    INDEX idx_permissions_action (action)
);

-- User roles mapping
CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    granted_by VARCHAR(255),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(user_id, role_id),
    INDEX idx_user_roles_user (user_id),
    INDEX idx_user_roles_role (role_id),
    INDEX idx_user_roles_expires (expires_at)
);

-- Role permissions mapping
CREATE TABLE IF NOT EXISTS role_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(role_id, permission_id),
    INDEX idx_role_permissions_role (role_id),
    INDEX idx_role_permissions_permission (permission_id)
);

-- Azure AD group mappings
CREATE TABLE IF NOT EXISTS azure_group_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    azure_group_id VARCHAR(255) NOT NULL,
    azure_group_name VARCHAR(255),
    role_id UUID NOT NULL REFERENCES roles(id),
    tenant_id VARCHAR(255) NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(azure_group_id, tenant_id),
    INDEX idx_azure_groups_id (azure_group_id),
    INDEX idx_azure_groups_role (role_id)
);

-- API keys for service accounts
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(255) NOT NULL UNIQUE, -- SHA-256 hash of the key
    key_prefix VARCHAR(20) NOT NULL, -- First 8 chars for identification
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(255),
    
    -- Restrictions
    allowed_ips TEXT[],
    allowed_origins TEXT[],
    rate_limit_per_minute INTEGER DEFAULT 60,
    
    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_api_keys_hash (key_hash),
    INDEX idx_api_keys_prefix (key_prefix),
    INDEX idx_api_keys_user (user_id),
    INDEX idx_api_keys_active (is_active)
);

-- Session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    
    -- Session details
    ip_address INET,
    user_agent TEXT,
    device_info JSONB,
    
    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    INDEX idx_sessions_token (session_token),
    INDEX idx_sessions_user (user_id),
    INDEX idx_sessions_active (is_active, expires_at)
);

-- Audit log for auth events
CREATE TABLE IF NOT EXISTS auth_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL, -- 'login', 'logout', 'permission_grant', 'permission_revoke', 'failed_login'
    event_details JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_auth_audit_user (user_id),
    INDEX idx_auth_audit_type (event_type),
    INDEX idx_auth_audit_timestamp (timestamp DESC)
);

-- Resource access control
CREATE TABLE IF NOT EXISTS resource_access (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    owner_id UUID REFERENCES users(id),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Access control
    is_public BOOLEAN DEFAULT FALSE,
    allowed_users UUID[],
    allowed_roles UUID[],
    denied_users UUID[],
    denied_roles UUID[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_resource_access_id (resource_id),
    INDEX idx_resource_access_type (resource_type),
    INDEX idx_resource_access_owner (owner_id),
    INDEX idx_resource_access_tenant (tenant_id)
);

-- Insert default system roles
INSERT INTO roles (role_name, role_description, is_system_role) VALUES
    ('admin', 'Full system administration access', TRUE),
    ('auditor', 'Read-only access for compliance auditing', TRUE),
    ('operator', 'Operational access for managing resources', TRUE),
    ('viewer', 'Read-only access to resources', TRUE)
ON CONFLICT (role_name, tenant_id) DO NOTHING;

-- Insert default permissions
INSERT INTO permissions (permission_name, resource_type, action) VALUES
    ('resources.read', 'resources', 'read'),
    ('resources.write', 'resources', 'write'),
    ('resources.delete', 'resources', 'delete'),
    ('compliance.read', 'compliance', 'read'),
    ('compliance.write', 'compliance', 'write'),
    ('predictions.read', 'predictions', 'read'),
    ('predictions.write', 'predictions', 'write'),
    ('evidence.read', 'evidence', 'read'),
    ('evidence.write', 'evidence', 'write'),
    ('evidence.verify', 'evidence', 'execute'),
    ('costs.read', 'costs', 'read'),
    ('costs.write', 'costs', 'write'),
    ('admin.users', 'admin', 'write'),
    ('admin.roles', 'admin', 'write'),
    ('admin.audit', 'admin', 'read')
ON CONFLICT (permission_name) DO NOTHING;

-- Function to check user permission
CREATE OR REPLACE FUNCTION check_user_permission(
    p_user_id UUID,
    p_resource_type VARCHAR(100),
    p_action VARCHAR(50)
)
RETURNS BOOLEAN AS $$
DECLARE
    has_permission BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1
        FROM users u
        JOIN user_roles ur ON u.id = ur.user_id
        JOIN role_permissions rp ON ur.role_id = rp.role_id
        JOIN permissions p ON rp.permission_id = p.id
        WHERE u.id = p_user_id
        AND p.resource_type = p_resource_type
        AND p.action = p_action
        AND u.is_active = TRUE
        AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
    ) INTO has_permission;
    
    RETURN has_permission;
END;
$$ LANGUAGE plpgsql;

-- Function to get user roles
CREATE OR REPLACE FUNCTION get_user_roles(p_user_id UUID)
RETURNS TABLE(role_name VARCHAR, role_description TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT r.role_name, r.role_description
    FROM roles r
    JOIN user_roles ur ON r.id = ur.role_id
    WHERE ur.user_id = p_user_id
    AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP);
END;
$$ LANGUAGE plpgsql;

-- Function to audit auth events
CREATE OR REPLACE FUNCTION audit_auth_event(
    p_user_id UUID,
    p_event_type VARCHAR(50),
    p_event_details JSONB,
    p_ip_address INET,
    p_user_agent TEXT,
    p_success BOOLEAN,
    p_error_message TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    audit_id UUID;
BEGIN
    INSERT INTO auth_audit_log (
        user_id,
        event_type,
        event_details,
        ip_address,
        user_agent,
        success,
        error_message
    ) VALUES (
        p_user_id,
        p_event_type,
        p_event_details,
        p_ip_address,
        p_user_agent,
        p_success,
        p_error_message
    ) RETURNING id INTO audit_id;
    
    -- Update user last login if successful login
    IF p_event_type = 'login' AND p_success THEN
        UPDATE users 
        SET last_login = CURRENT_TIMESTAMP,
            login_count = login_count + 1
        WHERE id = p_user_id;
    END IF;
    
    RETURN audit_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update user timestamp
CREATE OR REPLACE FUNCTION update_user_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_update_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_user_timestamp();

-- View for user permissions
CREATE OR REPLACE VIEW user_permissions_view AS
SELECT 
    u.id as user_id,
    u.email,
    u.display_name,
    r.role_name,
    p.permission_name,
    p.resource_type,
    p.action
FROM users u
JOIN user_roles ur ON u.id = ur.user_id
JOIN roles r ON ur.role_id = r.id
JOIN role_permissions rp ON r.id = rp.role_id
JOIN permissions p ON rp.permission_id = p.id
WHERE u.is_active = TRUE
AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP);

-- Materialized view for permission cache
CREATE MATERIALIZED VIEW IF NOT EXISTS permission_cache AS
SELECT 
    user_id,
    array_agg(DISTINCT permission_name) as permissions,
    array_agg(DISTINCT role_name) as roles
FROM user_permissions_view
GROUP BY user_id;

CREATE UNIQUE INDEX idx_permission_cache_user ON permission_cache(user_id);

-- Function to refresh permission cache
CREATE OR REPLACE FUNCTION refresh_permission_cache()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY permission_cache;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON users TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON roles TO policycortex_app;
-- GRANT SELECT ON permissions TO policycortex_app;
-- GRANT SELECT, INSERT, DELETE ON user_roles TO policycortex_app;
-- GRANT SELECT, INSERT, DELETE ON role_permissions TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON api_keys TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON user_sessions TO policycortex_app;
-- GRANT INSERT ON auth_audit_log TO policycortex_app;
-- GRANT SELECT ON user_permissions_view TO policycortex_app;
-- GRANT SELECT ON permission_cache TO policycortex_app;