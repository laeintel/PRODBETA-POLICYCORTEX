-- Migration: Create P&L and cost analysis tables
-- Required for T07 - P&L Forecast API/UI implementation

-- Policy cost impact table
CREATE TABLE IF NOT EXISTS policy_cost_impact (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id VARCHAR(255) NOT NULL,
    policy_name VARCHAR(500) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Month-to-date savings
    mtd_savings DECIMAL(15,2) DEFAULT 0,
    mtd_costs DECIMAL(15,2) DEFAULT 0,
    mtd_net DECIMAL(15,2) GENERATED ALWAYS AS (mtd_savings - mtd_costs) STORED,
    
    -- 90-day projections
    projected_savings_30d DECIMAL(15,2) DEFAULT 0,
    projected_savings_60d DECIMAL(15,2) DEFAULT 0,
    projected_savings_90d DECIMAL(15,2) DEFAULT 0,
    projected_costs_90d DECIMAL(15,2) DEFAULT 0,
    
    -- Compliance metrics
    compliance_rate DECIMAL(5,4) DEFAULT 0 CHECK (compliance_rate >= 0 AND compliance_rate <= 1),
    resources_compliant INTEGER DEFAULT 0,
    resources_total INTEGER DEFAULT 0,
    
    -- Time tracking
    calculation_date DATE NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(policy_id, tenant_id, calculation_date),
    INDEX idx_cost_policy (policy_id),
    INDEX idx_cost_tenant (tenant_id),
    INDEX idx_cost_date (calculation_date DESC)
);

-- Resource cost tracking
CREATE TABLE IF NOT EXISTS resource_costs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Current costs
    daily_cost DECIMAL(12,2) DEFAULT 0,
    monthly_cost DECIMAL(15,2) DEFAULT 0,
    annual_cost DECIMAL(15,2) DEFAULT 0,
    
    -- Optimization potential
    potential_savings_daily DECIMAL(12,2) DEFAULT 0,
    potential_savings_monthly DECIMAL(15,2) DEFAULT 0,
    optimization_score DECIMAL(5,4) DEFAULT 0,
    
    -- Policy compliance impact
    compliant_policies TEXT[],
    non_compliant_policies TEXT[],
    compliance_cost_impact DECIMAL(12,2) DEFAULT 0,
    
    -- Metadata
    cost_center VARCHAR(255),
    department VARCHAR(255),
    owner_email VARCHAR(255),
    tags JSONB,
    
    last_analyzed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_resource_costs_id (resource_id),
    INDEX idx_resource_costs_tenant (tenant_id),
    INDEX idx_resource_costs_type (resource_type),
    INDEX idx_resource_costs_analyzed (last_analyzed DESC)
);

-- Cost anomalies detection
CREATE TABLE IF NOT EXISTS cost_anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500),
    tenant_id VARCHAR(255) NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL, -- 'spike', 'unusual_pattern', 'threshold_breach'
    
    -- Anomaly details
    expected_cost DECIMAL(12,2) NOT NULL,
    actual_cost DECIMAL(12,2) NOT NULL,
    deviation_percentage DECIMAL(8,2) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    
    -- Time window
    anomaly_start TIMESTAMP WITH TIME ZONE NOT NULL,
    anomaly_end TIMESTAMP WITH TIME ZONE,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Resolution
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    INDEX idx_anomalies_resource (resource_id),
    INDEX idx_anomalies_tenant (tenant_id),
    INDEX idx_anomalies_detected (detected_at DESC),
    INDEX idx_anomalies_unresolved (is_resolved, tenant_id)
);

-- Budget tracking
CREATE TABLE IF NOT EXISTS budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    budget_name VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    scope_type VARCHAR(50) NOT NULL, -- 'tenant', 'department', 'resource_group', 'tag'
    scope_value VARCHAR(500),
    
    -- Budget amounts
    monthly_budget DECIMAL(15,2) NOT NULL,
    quarterly_budget DECIMAL(15,2),
    annual_budget DECIMAL(15,2),
    
    -- Current spending
    current_month_spent DECIMAL(15,2) DEFAULT 0,
    current_quarter_spent DECIMAL(15,2) DEFAULT 0,
    current_year_spent DECIMAL(15,2) DEFAULT 0,
    
    -- Alerts
    alert_threshold_percentage INTEGER DEFAULT 80,
    critical_threshold_percentage INTEGER DEFAULT 95,
    last_alert_sent TIMESTAMP WITH TIME ZONE,
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(budget_name, tenant_id),
    INDEX idx_budgets_tenant (tenant_id),
    INDEX idx_budgets_scope (scope_type, scope_value),
    INDEX idx_budgets_active (is_active)
);

-- Cost optimization recommendations
CREATE TABLE IF NOT EXISTS cost_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    recommendation_type VARCHAR(100) NOT NULL,
    
    -- Recommendation details
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    estimated_savings_monthly DECIMAL(12,2) NOT NULL,
    implementation_effort VARCHAR(20), -- 'low', 'medium', 'high'
    risk_level VARCHAR(20), -- 'low', 'medium', 'high'
    
    -- Implementation
    implementation_steps JSONB,
    automation_available BOOLEAN DEFAULT FALSE,
    pr_url TEXT,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'in_progress', 'implemented', 'dismissed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    implemented_at TIMESTAMP WITH TIME ZONE,
    dismissed_at TIMESTAMP WITH TIME ZONE,
    dismissed_reason TEXT,
    
    INDEX idx_recommendations_resource (resource_id),
    INDEX idx_recommendations_tenant (tenant_id),
    INDEX idx_recommendations_status (status),
    INDEX idx_recommendations_savings (estimated_savings_monthly DESC)
);

-- Historical P&L data for trending
CREATE TABLE IF NOT EXISTS pnl_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    snapshot_date DATE NOT NULL,
    
    -- Aggregated values
    total_mtd_savings DECIMAL(15,2) NOT NULL,
    total_mtd_costs DECIMAL(15,2) NOT NULL,
    total_projected_90d DECIMAL(15,2) NOT NULL,
    
    -- Policy breakdown
    policy_details JSONB NOT NULL,
    
    -- Compliance metrics
    overall_compliance DECIMAL(5,4) NOT NULL,
    compliant_resources INTEGER NOT NULL,
    total_resources INTEGER NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(tenant_id, snapshot_date),
    INDEX idx_pnl_history_tenant (tenant_id),
    INDEX idx_pnl_history_date (snapshot_date DESC)
);

-- Function to calculate P&L for a tenant
CREATE OR REPLACE FUNCTION calculate_tenant_pnl(tenant VARCHAR(255), as_of_date DATE DEFAULT CURRENT_DATE)
RETURNS TABLE(
    policy_id VARCHAR,
    policy_name VARCHAR,
    mtd_savings DECIMAL,
    projected_90d DECIMAL,
    compliance_rate DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pci.policy_id,
        pci.policy_name,
        pci.mtd_savings,
        pci.projected_savings_90d,
        pci.compliance_rate
    FROM policy_cost_impact pci
    WHERE pci.tenant_id = tenant
    AND pci.calculation_date = as_of_date
    ORDER BY pci.mtd_savings DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to detect cost anomalies
CREATE OR REPLACE FUNCTION detect_cost_anomaly(
    p_resource_id VARCHAR(500),
    p_current_cost DECIMAL(12,2),
    p_tenant_id VARCHAR(255)
)
RETURNS BOOLEAN AS $$
DECLARE
    avg_cost DECIMAL(12,2);
    std_dev DECIMAL(12,2);
    z_score DECIMAL(8,2);
BEGIN
    -- Calculate historical average and standard deviation
    SELECT 
        AVG(daily_cost),
        STDDEV(daily_cost)
    INTO avg_cost, std_dev
    FROM resource_costs
    WHERE resource_id = p_resource_id
    AND last_analyzed >= CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    IF avg_cost IS NULL OR std_dev IS NULL OR std_dev = 0 THEN
        RETURN FALSE;
    END IF;
    
    -- Calculate Z-score
    z_score := ABS((p_current_cost - avg_cost) / std_dev);
    
    -- Anomaly if Z-score > 3 (99.7% confidence)
    IF z_score > 3 THEN
        INSERT INTO cost_anomalies (
            resource_id,
            tenant_id,
            anomaly_type,
            expected_cost,
            actual_cost,
            deviation_percentage,
            confidence_score,
            anomaly_start
        ) VALUES (
            p_resource_id,
            p_tenant_id,
            'spike',
            avg_cost,
            p_current_cost,
            ((p_current_cost - avg_cost) / avg_cost) * 100,
            LEAST(z_score / 10, 1.0),
            CURRENT_TIMESTAMP
        );
        RETURN TRUE;
    END IF;
    
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update P&L history daily
CREATE OR REPLACE FUNCTION snapshot_daily_pnl()
RETURNS void AS $$
BEGIN
    INSERT INTO pnl_history (
        tenant_id,
        snapshot_date,
        total_mtd_savings,
        total_mtd_costs,
        total_projected_90d,
        policy_details,
        overall_compliance,
        compliant_resources,
        total_resources
    )
    SELECT 
        tenant_id,
        CURRENT_DATE,
        SUM(mtd_savings),
        SUM(mtd_costs),
        SUM(projected_savings_90d),
        jsonb_agg(jsonb_build_object(
            'policy_id', policy_id,
            'policy_name', policy_name,
            'mtd_savings', mtd_savings,
            'projected_90d', projected_savings_90d,
            'compliance', compliance_rate
        )),
        AVG(compliance_rate),
        SUM(resources_compliant),
        SUM(resources_total)
    FROM policy_cost_impact
    WHERE calculation_date = CURRENT_DATE
    GROUP BY tenant_id
    ON CONFLICT (tenant_id, snapshot_date) DO UPDATE
    SET 
        total_mtd_savings = EXCLUDED.total_mtd_savings,
        total_mtd_costs = EXCLUDED.total_mtd_costs,
        total_projected_90d = EXCLUDED.total_projected_90d,
        policy_details = EXCLUDED.policy_details,
        overall_compliance = EXCLUDED.overall_compliance;
END;
$$ LANGUAGE plpgsql;

-- View for P&L dashboard
CREATE OR REPLACE VIEW pnl_dashboard AS
SELECT 
    pci.tenant_id,
    pci.calculation_date,
    COUNT(DISTINCT pci.policy_id) as total_policies,
    SUM(pci.mtd_savings) as total_mtd_savings,
    SUM(pci.mtd_costs) as total_mtd_costs,
    SUM(pci.mtd_net) as total_mtd_net,
    SUM(pci.projected_savings_90d) as total_projected_90d,
    AVG(pci.compliance_rate) as avg_compliance_rate,
    SUM(pci.resources_compliant) as total_compliant_resources,
    SUM(pci.resources_total) as total_resources
FROM policy_cost_impact pci
WHERE pci.calculation_date = CURRENT_DATE
GROUP BY pci.tenant_id, pci.calculation_date;

-- Materialized view for cost trends
CREATE MATERIALIZED VIEW IF NOT EXISTS cost_trends AS
SELECT 
    tenant_id,
    DATE_TRUNC('month', snapshot_date) as month,
    AVG(total_mtd_savings) as avg_monthly_savings,
    AVG(total_projected_90d) as avg_projected_90d,
    AVG(overall_compliance) as avg_compliance,
    MAX(total_mtd_savings) as max_monthly_savings,
    MIN(total_mtd_savings) as min_monthly_savings
FROM pnl_history
GROUP BY tenant_id, DATE_TRUNC('month', snapshot_date);

CREATE INDEX idx_cost_trends_tenant ON cost_trends(tenant_id);
CREATE INDEX idx_cost_trends_month ON cost_trends(month DESC);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON policy_cost_impact TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON resource_costs TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON cost_anomalies TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON budgets TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON cost_recommendations TO policycortex_app;
-- GRANT SELECT, INSERT ON pnl_history TO policycortex_app;
-- GRANT SELECT ON pnl_dashboard TO policycortex_app;
-- GRANT SELECT ON cost_trends TO policycortex_app;