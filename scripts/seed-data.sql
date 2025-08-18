-- PolicyCortex Seed Data SQL

-- Create tables if not exists
CREATE TABLE IF NOT EXISTS organizations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(50) PRIMARY KEY,
    organization_id VARCHAR(50),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS azure_subscriptions (
    id VARCHAR(50) PRIMARY KEY,
    organization_id VARCHAR(50),
    subscription_id VARCHAR(100),
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policies (
    id VARCHAR(50) PRIMARY KEY,
    organization_id VARCHAR(50),
    name VARCHAR(255),
    category VARCHAR(50),
    severity VARCHAR(50),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resources (
    id VARCHAR(50) PRIMARY KEY,
    subscription_id VARCHAR(50),
    resource_id TEXT,
    name VARCHAR(255),
    type VARCHAR(255),
    location VARCHAR(50),
    tags JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS compliance_results (
    id VARCHAR(50) PRIMARY KEY,
    policy_id VARCHAR(50),
    resource_id VARCHAR(50),
    status VARCHAR(50),
    reason TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS achievements (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    points INTEGER,
    icon VARCHAR(10),
    criteria JSONB
);

CREATE TABLE IF NOT EXISTS user_achievements (
    user_id VARCHAR(50),
    achievement_id VARCHAR(50),
    earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, achievement_id)
);

CREATE TABLE IF NOT EXISTS policy_templates (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    category VARCHAR(50),
    author VARCHAR(255),
    downloads INTEGER DEFAULT 0,
    rating DECIMAL(2,1),
    price DECIMAL(10,2),
    template JSONB
);

-- Clear existing data
TRUNCATE TABLE 
    organizations, users, azure_subscriptions, policies, resources, 
    compliance_results, achievements, user_achievements, policy_templates 
CASCADE;

-- Insert Organizations
INSERT INTO organizations (id, name, tier, created_at) VALUES
    ('org-1', 'Contoso Corporation', 'enterprise', NOW()),
    ('org-2', 'Fabrikam Industries', 'professional', NOW()),
    ('org-3', 'Adventure Works', 'starter', NOW());

-- Insert Users
INSERT INTO users (id, organization_id, email, name, role, created_at) VALUES
    ('user-1', 'org-1', 'admin@contoso.com', 'Admin User', 'admin', NOW()),
    ('user-2', 'org-1', 'analyst@contoso.com', 'Policy Analyst', 'analyst', NOW()),
    ('user-3', 'org-2', 'admin@fabrikam.com', 'Fabrikam Admin', 'admin', NOW());

-- Insert Azure Subscriptions
INSERT INTO azure_subscriptions (id, organization_id, subscription_id, name, created_at) VALUES
    ('sub-1', 'org-1', 'sub-contoso-prod', 'Contoso Production', NOW()),
    ('sub-2', 'org-1', 'sub-contoso-dev', 'Contoso Development', NOW()),
    ('sub-3', 'org-2', 'sub-fabrikam-prod', 'Fabrikam Production', NOW());

-- Insert Policies
INSERT INTO policies (id, organization_id, name, category, severity, status, created_at) VALUES
    ('pol-1', 'org-1', 'Require HTTPS for Storage Accounts', 'Security', 'high', 'active', NOW()),
    ('pol-2', 'org-1', 'Enforce Tagging Standards', 'Governance', 'medium', 'active', NOW()),
    ('pol-3', 'org-1', 'Restrict VM SKUs', 'Cost', 'medium', 'active', NOW()),
    ('pol-4', 'org-2', 'Require MFA for Admin Accounts', 'Security', 'critical', 'active', NOW()),
    ('pol-5', 'org-2', 'Backup Policy for Databases', 'Resilience', 'high', 'draft', NOW());

-- Insert Resources
INSERT INTO resources (id, subscription_id, resource_id, name, type, location, tags, created_at) VALUES
    ('res-1', 'sub-1', '/subscriptions/sub-1/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stcontoso01', 
     'stcontoso01', 'Microsoft.Storage/storageAccounts', 'eastus', '{"env": "prod", "dept": "finance"}', NOW()),
    ('res-2', 'sub-1', '/subscriptions/sub-1/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-web-01', 
     'vm-web-01', 'Microsoft.Compute/virtualMachines', 'eastus', '{"env": "prod", "dept": "it"}', NOW()),
    ('res-3', 'sub-2', '/subscriptions/sub-2/resourceGroups/rg-dev/providers/Microsoft.Sql/servers/sql-dev-01', 
     'sql-dev-01', 'Microsoft.Sql/servers', 'westus', '{"env": "dev"}', NOW());

-- Insert Compliance Results
INSERT INTO compliance_results (id, policy_id, resource_id, status, reason, checked_at) VALUES
    ('comp-1', 'pol-1', 'res-1', 'compliant', 'HTTPS is enforced', NOW()),
    ('comp-2', 'pol-2', 'res-1', 'compliant', 'All required tags present', NOW()),
    ('comp-3', 'pol-2', 'res-2', 'non_compliant', 'Missing required tag: cost-center', NOW()),
    ('comp-4', 'pol-3', 'res-2', 'compliant', 'VM SKU is in allowed list', NOW());

-- Insert Achievements (Gamification)
INSERT INTO achievements (id, name, description, points, icon, criteria) VALUES
    ('ach-1', 'First Policy', 'Create your first policy', 10, '🎯', '{"type": "policy_count", "value": 1}'),
    ('ach-2', 'Compliance Champion', 'Achieve 95% compliance', 50, '🏆', '{"type": "compliance_rate", "value": 95}'),
    ('ach-3', 'Cost Saver', 'Save $1000 through optimization', 100, '💰', '{"type": "cost_saved", "value": 1000}'),
    ('ach-4', 'Security Expert', 'Fix 10 security issues', 25, '🛡️', '{"type": "security_fixes", "value": 10}');

-- Insert User Achievements
INSERT INTO user_achievements (user_id, achievement_id, earned_at) VALUES
    ('user-1', 'ach-1', NOW() - INTERVAL '7 days'),
    ('user-1', 'ach-4', NOW() - INTERVAL '3 days'),
    ('user-2', 'ach-1', NOW() - INTERVAL '5 days');

-- Insert Policy Templates (Marketplace)
INSERT INTO policy_templates (id, name, description, category, author, downloads, rating, price, template) VALUES
    ('tpl-1', 'CIS Azure Foundations Benchmark', 'Complete CIS benchmark implementation', 'Security', 'PolicyCortex', 1250, 4.8, 0, '{}'),
    ('tpl-2', 'FinOps Cost Optimization Pack', 'Comprehensive cost optimization policies', 'Cost', 'Community', 890, 4.6, 49.99, '{}'),
    ('tpl-3', 'HIPAA Compliance Suite', 'Healthcare compliance policies', 'Compliance', 'HealthTech Corp', 450, 4.9, 199.99, '{}');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_org ON users(organization_id);
CREATE INDEX IF NOT EXISTS idx_policies_org ON policies(organization_id);
CREATE INDEX IF NOT EXISTS idx_resources_sub ON resources(subscription_id);
CREATE INDEX IF NOT EXISTS idx_compliance_policy ON compliance_results(policy_id);
CREATE INDEX IF NOT EXISTS idx_compliance_resource ON compliance_results(resource_id);