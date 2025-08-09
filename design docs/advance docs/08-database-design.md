# Database Design

## Table of Contents
1. [Database Architecture](#database-architecture)
2. [PostgreSQL Schema Design](#postgresql-schema-design)
3. [EventStore Design](#eventstore-design)
4. [Caching Strategy](#caching-strategy)
5. [Data Models](#data-models)
6. [Query Optimization](#query-optimization)
7. [Data Migration](#data-migration)
8. [Backup and Recovery](#backup-and-recovery)
9. [Performance Tuning](#performance-tuning)

## Database Architecture

PolicyCortex uses a polyglot persistence approach with multiple specialized databases:

```
┌─────────────────────────────────────────────────────────────┐
│                    Database Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │ PostgreSQL  │  │ EventStore  │  │   DragonflyDB   │     │
│  │             │  │             │  │  (Redis-compat) │     │
│  │ • Resources │  │ • Events    │  │ • Cache         │     │
│  │ • Policies  │  │ • Audit     │  │ • Sessions      │     │
│  │ • Users     │  │ • Sourcing  │  │ • Real-time     │     │
│  │ • Config    │  │ • Snapshots │  │ • Pub/Sub       │     │
│  └─────────────┘  └─────────────┘  └─────────────────┘     │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │   Search    │  │  Time Series│  │   Graph DB      │     │
│  │(Elasticsearch)│  │(InfluxDB)   │  │   (Neo4j)       │     │
│  │             │  │             │  │                 │     │
│  │ • Full-text │  │ • Metrics   │  │ • Dependencies  │     │
│  │ • Analytics │  │ • Monitoring│  │ • Relationships │     │
│  │ • Logs      │  │ • Time-based│  │ • Graph Queries │     │
│  └─────────────┘  └─────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Database Selection Rationale

1. **PostgreSQL**: Primary transactional database
   - ACID compliance for critical data
   - Rich data types (JSON, arrays, enums)
   - Advanced indexing capabilities
   - Mature ecosystem and tooling

2. **EventStore**: Event sourcing and audit trails
   - Immutable event storage
   - Event replay capabilities
   - Built-in projections
   - High-performance streaming

3. **DragonflyDB**: High-performance caching
   - 25x faster than Redis
   - Multi-threaded architecture
   - Full Redis compatibility
   - Memory-efficient

4. **Elasticsearch**: Search and analytics
   - Full-text search capabilities
   - Real-time analytics
   - Log aggregation
   - Complex query support

## PostgreSQL Schema Design

### Core Tables

```sql
-- scripts/migrations/001_initial_schema.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Custom types
CREATE TYPE resource_type AS ENUM (
    'virtual_machine',
    'storage_account',
    'sql_database',
    'app_service',
    'function_app',
    'key_vault',
    'network_security_group',
    'load_balancer',
    'virtual_network',
    'public_ip',
    'network_interface',
    'disk',
    'availability_set',
    'scale_set'
);

CREATE TYPE policy_type AS ENUM (
    'security',
    'compliance',
    'cost_optimization',
    'performance',
    'governance',
    'finops'
);

CREATE TYPE severity_level AS ENUM (
    'critical',
    'high',
    'medium',
    'low',
    'info'
);

CREATE TYPE evaluation_status AS ENUM (
    'pending',
    'evaluating',
    'completed',
    'failed',
    'skipped'
);

CREATE TYPE evaluation_result AS ENUM (
    'compliant',
    'non_compliant',
    'warning',
    'unknown',
    'not_applicable'
);

CREATE TYPE execution_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled',
    'rolled_back'
);

-- Tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    azure_tenant_id VARCHAR(36) UNIQUE,
    subscription_ids TEXT[], -- Array of subscription IDs
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    azure_object_id VARCHAR(36) UNIQUE,
    role VARCHAR(100) NOT NULL DEFAULT 'user',
    permissions TEXT[] DEFAULT '{}',
    attributes JSONB DEFAULT '{}',
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Resources table
CREATE TABLE resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    azure_resource_id VARCHAR(500) UNIQUE NOT NULL,
    subscription_id VARCHAR(36) NOT NULL,
    resource_group_name VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    type resource_type NOT NULL,
    location VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    synced_at TIMESTAMP WITH TIME ZONE,
    deleted_at TIMESTAMP WITH TIME ZONE,
    
    -- Computed columns for better querying
    tag_keys TEXT[] GENERATED ALWAYS AS (
        ARRAY(SELECT jsonb_object_keys(tags))
    ) STORED
);

-- Policies table
CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id),
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    description TEXT,
    type policy_type NOT NULL,
    category VARCHAR(100),
    severity severity_level NOT NULL DEFAULT 'medium',
    definition JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    version INTEGER NOT NULL DEFAULT 1,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Policy versions for history tracking
CREATE TABLE policy_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id UUID NOT NULL REFERENCES policies(id),
    version INTEGER NOT NULL,
    definition JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    changes TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(policy_id, version)
);

-- Policy evaluations
CREATE TABLE policy_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    policy_id UUID NOT NULL REFERENCES policies(id),
    resource_id UUID NOT NULL REFERENCES resources(id),
    status evaluation_status NOT NULL DEFAULT 'pending',
    result evaluation_result,
    score DECIMAL(5,2), -- 0.00 to 100.00
    evidence JSONB DEFAULT '[]',
    recommendations TEXT[],
    error_message TEXT,
    execution_time_ms INTEGER,
    evaluated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure one active evaluation per policy-resource pair
    UNIQUE(policy_id, resource_id, evaluated_at)
);

-- Compliance reports
CREATE TABLE compliance_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    filters JSONB DEFAULT '{}',
    generated_by UUID REFERENCES users(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Actions/Remediation
CREATE TABLE actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    resource_id UUID NOT NULL REFERENCES resources(id),
    policy_evaluation_id UUID REFERENCES policy_evaluations(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    action_type VARCHAR(100) NOT NULL,
    parameters JSONB DEFAULT '{}',
    status execution_status NOT NULL DEFAULT 'pending',
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    result JSONB,
    error_message TEXT,
    execution_time_ms INTEGER,
    executed_by UUID REFERENCES users(id),
    approved_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- AI insights
CREATE TABLE ai_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    resource_id UUID NOT NULL REFERENCES resources(id),
    insight_type VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    severity severity_level NOT NULL,
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    data JSONB DEFAULT '{}',
    recommendations TEXT[],
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Ensure one current insight per type per resource
    UNIQUE(resource_id, insight_type, generated_at)
);

-- Cost data
CREATE TABLE cost_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    resource_id UUID REFERENCES resources(id),
    subscription_id VARCHAR(36) NOT NULL,
    resource_group_name VARCHAR(255),
    service_name VARCHAR(255) NOT NULL,
    cost_center VARCHAR(100),
    date DATE NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    cost DECIMAL(15,4) NOT NULL,
    usage_quantity DECIMAL(15,4),
    usage_unit VARCHAR(50),
    billing_period VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Partition by date for performance
    UNIQUE(resource_id, date, service_name)
) PARTITION BY RANGE (date);

-- Create partitions for cost data (monthly partitions)
CREATE TABLE cost_data_2024_01 PARTITION OF cost_data
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
    
CREATE TABLE cost_data_2024_02 PARTITION OF cost_data
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Add more partitions as needed...

-- Security events
CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    resource_id UUID REFERENCES resources(id),
    event_type VARCHAR(100) NOT NULL,
    severity severity_level NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    source VARCHAR(100),
    event_data JSONB DEFAULT '{}',
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by UUID REFERENCES users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_resources_tenant_id ON resources(tenant_id);
CREATE INDEX idx_resources_subscription_id ON resources(subscription_id);
CREATE INDEX idx_resources_type ON resources(type);
CREATE INDEX idx_resources_location ON resources(location);
CREATE INDEX idx_resources_tags ON resources USING GIN(tags);
CREATE INDEX idx_resources_tag_keys ON resources USING GIN(tag_keys);
CREATE INDEX idx_resources_azure_resource_id ON resources(azure_resource_id);
CREATE INDEX idx_resources_deleted_at ON resources(deleted_at) WHERE deleted_at IS NULL;

CREATE INDEX idx_policies_tenant_id ON policies(tenant_id);
CREATE INDEX idx_policies_type ON policies(type);
CREATE INDEX idx_policies_category ON policies(category);
CREATE INDEX idx_policies_enabled ON policies(enabled) WHERE enabled = TRUE;
CREATE INDEX idx_policies_deleted_at ON policies(deleted_at) WHERE deleted_at IS NULL;

CREATE INDEX idx_policy_evaluations_tenant_id ON policy_evaluations(tenant_id);
CREATE INDEX idx_policy_evaluations_policy_id ON policy_evaluations(policy_id);
CREATE INDEX idx_policy_evaluations_resource_id ON policy_evaluations(resource_id);
CREATE INDEX idx_policy_evaluations_status ON policy_evaluations(status);
CREATE INDEX idx_policy_evaluations_result ON policy_evaluations(result);
CREATE INDEX idx_policy_evaluations_evaluated_at ON policy_evaluations(evaluated_at);

CREATE INDEX idx_actions_tenant_id ON actions(tenant_id);
CREATE INDEX idx_actions_resource_id ON actions(resource_id);
CREATE INDEX idx_actions_status ON actions(status);
CREATE INDEX idx_actions_action_type ON actions(action_type);
CREATE INDEX idx_actions_created_at ON actions(created_at);

CREATE INDEX idx_ai_insights_tenant_id ON ai_insights(tenant_id);
CREATE INDEX idx_ai_insights_resource_id ON ai_insights(resource_id);
CREATE INDEX idx_ai_insights_insight_type ON ai_insights(insight_type);
CREATE INDEX idx_ai_insights_severity ON ai_insights(severity);
CREATE INDEX idx_ai_insights_generated_at ON ai_insights(generated_at);

CREATE INDEX idx_cost_data_tenant_id ON cost_data(tenant_id);
CREATE INDEX idx_cost_data_resource_id ON cost_data(resource_id);
CREATE INDEX idx_cost_data_date ON cost_data(date);
CREATE INDEX idx_cost_data_service_name ON cost_data(service_name);

CREATE INDEX idx_security_events_tenant_id ON security_events(tenant_id);
CREATE INDEX idx_security_events_resource_id ON security_events(resource_id);
CREATE INDEX idx_security_events_event_type ON security_events(event_type);
CREATE INDEX idx_security_events_severity ON security_events(severity);
CREATE INDEX idx_security_events_detected_at ON security_events(detected_at);

-- Full-text search indexes
CREATE INDEX idx_resources_search ON resources USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(properties->>'description', ''))
);

CREATE INDEX idx_policies_search ON policies USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, '') || ' ' || display_name)
);

-- Triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_resources_updated_at BEFORE UPDATE ON resources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### Views and Functions

```sql
-- scripts/migrations/002_views_and_functions.sql

-- Compliance dashboard view
CREATE VIEW compliance_dashboard AS
SELECT 
    r.tenant_id,
    r.subscription_id,
    r.resource_group_name,
    r.type as resource_type,
    COUNT(pe.id) as total_evaluations,
    COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END) as compliant_count,
    COUNT(CASE WHEN pe.result = 'non_compliant' THEN 1 END) as non_compliant_count,
    COUNT(CASE WHEN pe.result = 'warning' THEN 1 END) as warning_count,
    ROUND(
        COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END)::decimal / 
        NULLIF(COUNT(pe.id), 0) * 100, 2
    ) as compliance_percentage,
    MAX(pe.evaluated_at) as last_evaluated
FROM resources r
LEFT JOIN policy_evaluations pe ON r.id = pe.resource_id 
    AND pe.evaluated_at = (
        SELECT MAX(evaluated_at) 
        FROM policy_evaluations 
        WHERE resource_id = r.id AND policy_id = pe.policy_id
    )
WHERE r.deleted_at IS NULL
GROUP BY r.tenant_id, r.subscription_id, r.resource_group_name, r.type;

-- Resource insights view
CREATE VIEW resource_insights AS
SELECT 
    r.id as resource_id,
    r.name as resource_name,
    r.type as resource_type,
    r.subscription_id,
    r.resource_group_name,
    r.location,
    
    -- Compliance metrics
    COALESCE(comp.compliance_score, 0) as compliance_score,
    COALESCE(comp.total_policies, 0) as total_policies,
    COALESCE(comp.compliant_policies, 0) as compliant_policies,
    
    -- Cost metrics
    COALESCE(costs.monthly_cost, 0) as monthly_cost,
    COALESCE(costs.daily_average, 0) as daily_average_cost,
    
    -- Security metrics
    COALESCE(security.critical_alerts, 0) as critical_security_alerts,
    COALESCE(security.high_alerts, 0) as high_security_alerts,
    
    -- AI insights
    COALESCE(ai.insight_count, 0) as ai_insights_count,
    COALESCE(ai.high_confidence_insights, 0) as high_confidence_insights,
    
    r.updated_at as last_synced
FROM resources r

-- Compliance subquery
LEFT JOIN (
    SELECT 
        pe.resource_id,
        ROUND(AVG(CASE WHEN pe.result = 'compliant' THEN 100 ELSE 0 END), 2) as compliance_score,
        COUNT(*) as total_policies,
        COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END) as compliant_policies
    FROM policy_evaluations pe
    WHERE pe.evaluated_at = (
        SELECT MAX(evaluated_at) 
        FROM policy_evaluations 
        WHERE resource_id = pe.resource_id AND policy_id = pe.policy_id
    )
    GROUP BY pe.resource_id
) comp ON r.id = comp.resource_id

-- Cost subquery
LEFT JOIN (
    SELECT 
        cd.resource_id,
        SUM(CASE WHEN cd.date >= CURRENT_DATE - INTERVAL '30 days' THEN cd.cost ELSE 0 END) as monthly_cost,
        AVG(cd.cost) as daily_average
    FROM cost_data cd
    WHERE cd.date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY cd.resource_id
) costs ON r.id = costs.resource_id

-- Security subquery
LEFT JOIN (
    SELECT 
        se.resource_id,
        COUNT(CASE WHEN se.severity = 'critical' AND NOT se.resolved THEN 1 END) as critical_alerts,
        COUNT(CASE WHEN se.severity = 'high' AND NOT se.resolved THEN 1 END) as high_alerts
    FROM security_events se
    WHERE se.detected_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY se.resource_id
) security ON r.id = security.resource_id

-- AI insights subquery
LEFT JOIN (
    SELECT 
        ai.resource_id,
        COUNT(*) as insight_count,
        COUNT(CASE WHEN ai.confidence >= 0.8 THEN 1 END) as high_confidence_insights
    FROM ai_insights ai
    WHERE ai.expires_at IS NULL OR ai.expires_at > NOW()
    GROUP BY ai.resource_id
) ai ON r.id = ai.resource_id

WHERE r.deleted_at IS NULL;

-- Policy effectiveness view
CREATE VIEW policy_effectiveness AS
SELECT 
    p.id as policy_id,
    p.name as policy_name,
    p.type as policy_type,
    p.category,
    p.severity,
    
    COUNT(pe.id) as total_evaluations,
    COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END) as compliant_count,
    COUNT(CASE WHEN pe.result = 'non_compliant' THEN 1 END) as non_compliant_count,
    
    ROUND(
        COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END)::decimal / 
        NULLIF(COUNT(pe.id), 0) * 100, 2
    ) as compliance_rate,
    
    AVG(pe.execution_time_ms) as avg_execution_time,
    MAX(pe.evaluated_at) as last_evaluation,
    
    -- Count of remediation actions triggered by this policy
    COUNT(DISTINCT a.id) as remediation_actions_triggered,
    
    p.created_at,
    p.updated_at
    
FROM policies p
LEFT JOIN policy_evaluations pe ON p.id = pe.policy_id
LEFT JOIN actions a ON pe.id = a.policy_evaluation_id
WHERE p.deleted_at IS NULL AND p.enabled = TRUE
GROUP BY p.id, p.name, p.type, p.category, p.severity, p.created_at, p.updated_at;

-- Cost optimization opportunities view
CREATE VIEW cost_optimization_opportunities AS
SELECT 
    r.id as resource_id,
    r.name as resource_name,
    r.type as resource_type,
    r.subscription_id,
    r.resource_group_name,
    
    cd.monthly_cost,
    cd.usage_percentage,
    
    -- Potential savings based on utilization
    CASE 
        WHEN cd.usage_percentage < 0.2 THEN cd.monthly_cost * 0.8
        WHEN cd.usage_percentage < 0.5 THEN cd.monthly_cost * 0.3
        ELSE 0
    END as potential_monthly_savings,
    
    -- Optimization recommendations
    CASE 
        WHEN cd.usage_percentage < 0.2 THEN 'Consider downsizing or shutting down'
        WHEN cd.usage_percentage < 0.5 THEN 'Consider rightsizing to smaller SKU'
        WHEN cd.usage_percentage > 0.9 THEN 'Consider scaling up for better performance'
        ELSE 'Resource utilization is optimal'
    END as recommendation,
    
    ai.confidence as ai_confidence,
    ai.generated_at as analysis_date
    
FROM resources r
JOIN (
    SELECT 
        resource_id,
        SUM(cost) as monthly_cost,
        -- Simplified usage calculation - in practice this would come from metrics
        RANDOM() as usage_percentage
    FROM cost_data
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY resource_id
) cd ON r.id = cd.resource_id
LEFT JOIN ai_insights ai ON r.id = ai.resource_id 
    AND ai.insight_type = 'cost_optimization'
    AND (ai.expires_at IS NULL OR ai.expires_at > NOW())
WHERE r.deleted_at IS NULL
    AND cd.monthly_cost > 10  -- Only show resources with meaningful cost
ORDER BY potential_monthly_savings DESC;

-- Stored procedures
CREATE OR REPLACE FUNCTION get_resource_compliance_score(resource_uuid UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    compliance_score DECIMAL(5,2);
BEGIN
    SELECT 
        ROUND(
            AVG(CASE WHEN pe.result = 'compliant' THEN 100 ELSE 0 END), 2
        ) INTO compliance_score
    FROM policy_evaluations pe
    WHERE pe.resource_id = resource_uuid
        AND pe.evaluated_at = (
            SELECT MAX(evaluated_at) 
            FROM policy_evaluations 
            WHERE resource_id = pe.resource_id AND policy_id = pe.policy_id
        );
        
    RETURN COALESCE(compliance_score, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to get trending compliance data
CREATE OR REPLACE FUNCTION get_compliance_trend(
    tenant_uuid UUID,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE(
    date DATE,
    total_evaluations BIGINT,
    compliant_count BIGINT,
    compliance_rate DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pe.evaluated_at::date as date,
        COUNT(*) as total_evaluations,
        COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END) as compliant_count,
        ROUND(
            COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END)::decimal / 
            COUNT(*) * 100, 2
        ) as compliance_rate
    FROM policy_evaluations pe
    WHERE pe.tenant_id = tenant_uuid
        AND pe.evaluated_at >= CURRENT_DATE - INTERVAL '1 day' * days_back
    GROUP BY pe.evaluated_at::date
    ORDER BY date;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate policy impact score
CREATE OR REPLACE FUNCTION calculate_policy_impact_score(policy_uuid UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    impact_score DECIMAL(5,2);
    severity_weight DECIMAL(3,2);
    compliance_rate DECIMAL(5,2);
    resource_count INTEGER;
BEGIN
    -- Get policy severity weight
    SELECT 
        CASE 
            WHEN severity = 'critical' THEN 1.0
            WHEN severity = 'high' THEN 0.8
            WHEN severity = 'medium' THEN 0.6
            WHEN severity = 'low' THEN 0.4
            ELSE 0.2
        END INTO severity_weight
    FROM policies 
    WHERE id = policy_uuid;
    
    -- Get compliance statistics
    SELECT 
        ROUND(
            COUNT(CASE WHEN pe.result = 'non_compliant' THEN 1 END)::decimal / 
            NULLIF(COUNT(*), 0) * 100, 2
        ),
        COUNT(*)
    INTO compliance_rate, resource_count
    FROM policy_evaluations pe
    WHERE pe.policy_id = policy_uuid
        AND pe.evaluated_at = (
            SELECT MAX(evaluated_at) 
            FROM policy_evaluations 
            WHERE resource_id = pe.resource_id AND policy_id = pe.policy_id
        );
    
    -- Calculate impact score: (non-compliance rate) * severity weight * log(resource count)
    impact_score = COALESCE(compliance_rate, 0) * COALESCE(severity_weight, 0.5) * 
                   CASE WHEN resource_count > 0 THEN LOG(resource_count + 1) ELSE 0 END;
    
    RETURN ROUND(impact_score, 2);
END;
$$ LANGUAGE plpgsql;
```

## EventStore Design

### Event Schema Design

```csharp
// EventStore schema design (conceptual - EventStore doesn't use SQL)

// Base event interface
public interface IEvent
{
    string EventId { get; }
    string EventType { get; }
    DateTime Timestamp { get; }
    string CorrelationId { get; }
    string CausationId { get; }
    Dictionary<string, object> Metadata { get; }
}

// Resource events
public class ResourceCreatedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "ResourceCreated";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    // Event data
    public string ResourceId { get; set; }
    public string TenantId { get; set; }
    public string AzureResourceId { get; set; }
    public string SubscriptionId { get; set; }
    public string ResourceGroupName { get; set; }
    public string Name { get; set; }
    public string Type { get; set; }
    public string Location { get; set; }
    public Dictionary<string, object> Properties { get; set; }
    public Dictionary<string, string> Tags { get; set; }
    public string CreatedBy { get; set; }
}

public class ResourceUpdatedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "ResourceUpdated";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public string ResourceId { get; set; }
    public Dictionary<string, object> Changes { get; set; }
    public Dictionary<string, object> PreviousValues { get; set; }
    public string UpdatedBy { get; set; }
}

// Policy events
public class PolicyCreatedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "PolicyCreated";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public string PolicyId { get; set; }
    public string TenantId { get; set; }
    public string Name { get; set; }
    public string Type { get; set; }
    public string Category { get; set; }
    public string Severity { get; set; }
    public Dictionary<string, object> Definition { get; set; }
    public string CreatedBy { get; set; }
}

public class PolicyEvaluatedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "PolicyEvaluated";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public string EvaluationId { get; set; }
    public string PolicyId { get; set; }
    public string ResourceId { get; set; }
    public string Status { get; set; }
    public string Result { get; set; }
    public decimal? Score { get; set; }
    public List<object> Evidence { get; set; }
    public List<string> Recommendations { get; set; }
    public int ExecutionTimeMs { get; set; }
}

// Action events
public class ActionExecutedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "ActionExecuted";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public string ActionId { get; set; }
    public string ResourceId { get; set; }
    public string ActionType { get; set; }
    public Dictionary<string, object> Parameters { get; set; }
    public string Status { get; set; }
    public bool DryRun { get; set; }
    public Dictionary<string, object> Result { get; set; }
    public string ExecutedBy { get; set; }
    public string ApprovedBy { get; set; }
}

// Compliance events
public class ComplianceViolationDetectedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "ComplianceViolationDetected";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public string ViolationId { get; set; }
    public string ResourceId { get; set; }
    public string PolicyId { get; set; }
    public string Severity { get; set; }
    public string Description { get; set; }
    public List<object> Evidence { get; set; }
    public List<string> RecommendedActions { get; set; }
}

// Security events
public class SecurityThreatDetectedEvent : IEvent
{
    public string EventId { get; set; } = Guid.NewGuid().ToString();
    public string EventType { get; set; } = "SecurityThreatDetected";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CorrelationId { get; set; }
    public string CausationId { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    public string ThreatId { get; set; }
    public string ResourceId { get; set; }
    public string ThreatType { get; set; }
    public string Severity { get; set; }
    public string Description { get; set; }
    public Dictionary<string, object> ThreatData { get; set; }
    public string Source { get; set; }
    public decimal Confidence { get; set; }
}
```

### EventStore Projections

```javascript
// EventStore projections for read models

// Resource aggregate projection
fromCategory('resource')
    .when({
        $init: function() {
            return {
                id: null,
                tenant_id: null,
                azure_resource_id: null,
                subscription_id: null,
                resource_group_name: null,
                name: null,
                type: null,
                location: null,
                properties: {},
                tags: {},
                created_at: null,
                updated_at: null,
                version: 0,
                deleted: false
            };
        },
        ResourceCreated: function(state, event) {
            state.id = event.data.ResourceId;
            state.tenant_id = event.data.TenantId;
            state.azure_resource_id = event.data.AzureResourceId;
            state.subscription_id = event.data.SubscriptionId;
            state.resource_group_name = event.data.ResourceGroupName;
            state.name = event.data.Name;
            state.type = event.data.Type;
            state.location = event.data.Location;
            state.properties = event.data.Properties || {};
            state.tags = event.data.Tags || {};
            state.created_at = event.data.Timestamp;
            state.updated_at = event.data.Timestamp;
            state.version = 1;
        },
        ResourceUpdated: function(state, event) {
            if (event.data.Changes.Name) {
                state.name = event.data.Changes.Name;
            }
            if (event.data.Changes.Properties) {
                Object.assign(state.properties, event.data.Changes.Properties);
            }
            if (event.data.Changes.Tags) {
                Object.assign(state.tags, event.data.Changes.Tags);
            }
            state.updated_at = event.data.Timestamp;
            state.version += 1;
        },
        ResourceDeleted: function(state, event) {
            state.deleted = true;
            state.updated_at = event.data.Timestamp;
            state.version += 1;
        }
    });

// Compliance status projection
fromCategory('policy-evaluation')
    .when({
        $init: function() {
            return {
                resources: {}
            };
        },
        PolicyEvaluated: function(state, event) {
            var resourceId = event.data.ResourceId;
            var policyId = event.data.PolicyId;
            
            if (!state.resources[resourceId]) {
                state.resources[resourceId] = {
                    resource_id: resourceId,
                    evaluations: {},
                    overall_status: 'unknown',
                    compliance_score: 0,
                    last_evaluated: null
                };
            }
            
            state.resources[resourceId].evaluations[policyId] = {
                policy_id: policyId,
                status: event.data.Status,
                result: event.data.Result,
                score: event.data.Score || 0,
                evidence: event.data.Evidence || [],
                recommendations: event.data.Recommendations || [],
                evaluated_at: event.data.Timestamp
            };
            
            // Calculate overall compliance score
            var evaluations = state.resources[resourceId].evaluations;
            var scores = Object.values(evaluations)
                .filter(e => e.result === 'compliant' || e.result === 'non_compliant')
                .map(e => e.result === 'compliant' ? 100 : 0);
            
            if (scores.length > 0) {
                state.resources[resourceId].compliance_score = 
                    scores.reduce((a, b) => a + b, 0) / scores.length;
            }
            
            // Determine overall status
            var results = Object.values(evaluations).map(e => e.result);
            if (results.includes('non_compliant')) {
                state.resources[resourceId].overall_status = 'non_compliant';
            } else if (results.includes('warning')) {
                state.resources[resourceId].overall_status = 'warning';
            } else if (results.every(r => r === 'compliant')) {
                state.resources[resourceId].overall_status = 'compliant';
            } else {
                state.resources[resourceId].overall_status = 'partial';
            }
            
            state.resources[resourceId].last_evaluated = event.data.Timestamp;
        }
    });

// Audit trail projection
fromAll()
    .when({
        $init: function() {
            return {
                events: []
            };
        },
        $any: function(state, event) {
            // Only capture events with audit significance
            if (event.eventType.includes('Created') || 
                event.eventType.includes('Updated') || 
                event.eventType.includes('Deleted') ||
                event.eventType.includes('Executed') ||
                event.eventType.includes('Evaluated')) {
                
                state.events.push({
                    event_id: event.eventId,
                    event_type: event.eventType,
                    timestamp: event.data.Timestamp || new Date().toISOString(),
                    user_id: event.data.CreatedBy || event.data.UpdatedBy || event.data.ExecutedBy,
                    resource_id: event.data.ResourceId,
                    tenant_id: event.data.TenantId,
                    correlation_id: event.metadata.correlationId,
                    summary: generateEventSummary(event),
                    details: event.data
                });
                
                // Keep only last 10000 events to prevent unbounded growth
                if (state.events.length > 10000) {
                    state.events = state.events.slice(-10000);
                }
            }
        }
    });

function generateEventSummary(event) {
    switch (event.eventType) {
        case 'ResourceCreated':
            return `Resource '${event.data.Name}' created in ${event.data.ResourceGroupName}`;
        case 'ResourceUpdated':
            return `Resource '${event.data.ResourceId}' updated`;
        case 'PolicyCreated':
            return `Policy '${event.data.Name}' created`;
        case 'PolicyEvaluated':
            return `Policy evaluation completed with result: ${event.data.Result}`;
        case 'ActionExecuted':
            return `Action '${event.data.ActionType}' executed on resource`;
        default:
            return `${event.eventType} occurred`;
    }
}
```

## Caching Strategy

### DragonflyDB Configuration

```rust
// core/src/cache/dragonfly_client.rs
use redis::{Client, Connection, Commands, RedisResult};
use serde::{Serialize, Deserialize};
use std::time::Duration;

pub struct DragonflyClient {
    client: Client,
}

impl DragonflyClient {
    pub fn new(redis_url: &str) -> RedisResult<Self> {
        let client = Client::open(redis_url)?;
        Ok(Self { client })
    }

    pub fn get_connection(&self) -> RedisResult<Connection> {
        self.client.get_connection()
    }

    // Resource caching
    pub fn cache_resource(&self, resource: &Resource) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let key = format!("resource:{}", resource.id);
        let serialized = serde_json::to_string(resource).unwrap();
        
        conn.setex(key, 300, serialized)?; // 5 minutes TTL
        
        // Also cache by azure_resource_id for lookups
        let azure_key = format!("azure_resource:{}", resource.azure_resource_id);
        conn.setex(azure_key, 300, &resource.id.to_string())?;
        
        Ok(())
    }

    pub fn get_resource(&self, resource_id: &str) -> RedisResult<Option<Resource>> {
        let mut conn = self.get_connection()?;
        let key = format!("resource:{}", resource_id);
        
        let cached: Option<String> = conn.get(key)?;
        match cached {
            Some(data) => {
                let resource: Resource = serde_json::from_str(&data).unwrap();
                Ok(Some(resource))
            }
            None => Ok(None)
        }
    }

    // Compliance status caching
    pub fn cache_compliance_status(&self, resource_id: &str, status: &ComplianceStatus) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let key = format!("compliance:{}", resource_id);
        let serialized = serde_json::to_string(status).unwrap();
        
        conn.setex(key, 180, serialized)?; // 3 minutes TTL
        Ok(())
    }

    // AI insights caching
    pub fn cache_ai_insights(&self, resource_id: &str, insights: &AIInsights) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let key = format!("ai_insights:{}", resource_id);
        let serialized = serde_json::to_string(insights).unwrap();
        
        conn.setex(key, 1800, serialized)?; // 30 minutes TTL
        Ok(())
    }

    // Session caching
    pub fn cache_user_session(&self, session_id: &str, session: &UserSession) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let key = format!("session:{}", session_id);
        let serialized = serde_json::to_string(session).unwrap();
        
        conn.setex(key, 3600, serialized)?; // 1 hour TTL
        Ok(())
    }

    // GraphQL query result caching
    pub fn cache_query_result(&self, query_hash: &str, result: &str) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let key = format!("query:{}", query_hash);
        
        conn.setex(key, 60, result)?; // 1 minute TTL for GraphQL queries
        Ok(())
    }

    // Batch operations
    pub fn batch_cache_resources(&self, resources: &[Resource]) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let pipe = redis::pipe();
        
        for resource in resources {
            let key = format!("resource:{}", resource.id);
            let serialized = serde_json::to_string(resource).unwrap();
            pipe.setex(key, 300, serialized);
        }
        
        pipe.query(&mut conn)?;
        Ok(())
    }

    // Cache invalidation
    pub fn invalidate_resource(&self, resource_id: &str) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        
        // Delete all related keys
        let keys = vec![
            format!("resource:{}", resource_id),
            format!("compliance:{}", resource_id),
            format!("ai_insights:{}", resource_id),
        ];
        
        for key in keys {
            let _: () = conn.del(key)?;
        }
        
        Ok(())
    }

    // Pub/Sub for real-time updates
    pub fn publish_resource_update(&self, resource_id: &str, update: &ResourceUpdate) -> RedisResult<()> {
        let mut conn = self.get_connection()?;
        let channel = format!("resource_updates:{}", resource_id);
        let message = serde_json::to_string(update).unwrap();
        
        conn.publish(channel, message)?;
        Ok(())
    }

    // Health check
    pub fn health_check(&self) -> RedisResult<String> {
        let mut conn = self.get_connection()?;
        conn.ping()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Resource {
    pub id: uuid::Uuid,
    pub tenant_id: uuid::Uuid,
    pub azure_resource_id: String,
    pub subscription_id: String,
    pub resource_group_name: String,
    pub name: String,
    pub resource_type: String,
    pub location: String,
    pub properties: serde_json::Value,
    pub tags: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub resource_id: uuid::Uuid,
    pub overall_status: String,
    pub compliance_score: f64,
    pub policy_evaluations: Vec<PolicyEvaluation>,
    pub last_evaluated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub policy_id: uuid::Uuid,
    pub result: String,
    pub score: Option<f64>,
    pub evidence: serde_json::Value,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIInsights {
    pub resource_id: uuid::Uuid,
    pub security_score: f64,
    pub cost_optimization_score: f64,
    pub recommendations: Vec<String>,
    pub risks: Vec<String>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserSession {
    pub user_id: uuid::Uuid,
    pub tenant_id: uuid::Uuid,
    pub email: String,
    pub permissions: Vec<String>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUpdate {
    pub resource_id: uuid::Uuid,
    pub update_type: String,
    pub changes: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

### Cache Strategy Implementation

```rust
// core/src/cache/strategy.rs
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum CacheStrategy {
    WriteThrough,  // Write to cache and database simultaneously
    WriteBack,     // Write to cache first, database later
    WriteAround,   // Write to database, invalidate cache
    ReadThrough,   // Read from cache, if miss read from database and cache
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub ttl: Duration,
    pub strategy: CacheStrategy,
    pub max_size: Option<usize>,
    pub eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,  // Least Recently Used
    LFU,  // Least Frequently Used
    TTL,  // Time To Live based
}

pub struct CacheManager {
    dragonfly: DragonflyClient,
    local_cache: LocalCache,
    strategies: HashMap<String, CacheConfig>,
}

impl CacheManager {
    pub fn new(dragonfly_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let dragonfly = DragonflyClient::new(dragonfly_url)?;
        let local_cache = LocalCache::new();
        let mut strategies = HashMap::new();

        // Configure caching strategies for different data types
        strategies.insert("resource".to_string(), CacheConfig {
            ttl: Duration::from_secs(300), // 5 minutes
            strategy: CacheStrategy::WriteThrough,
            max_size: Some(10000),
            eviction_policy: EvictionPolicy::LRU,
        });

        strategies.insert("compliance".to_string(), CacheConfig {
            ttl: Duration::from_secs(180), // 3 minutes
            strategy: CacheStrategy::WriteThrough,
            max_size: Some(5000),
            eviction_policy: EvictionPolicy::TTL,
        });

        strategies.insert("ai_insights".to_string(), CacheConfig {
            ttl: Duration::from_secs(1800), // 30 minutes
            strategy: CacheStrategy::WriteThrough,
            max_size: Some(1000),
            eviction_policy: EvictionPolicy::LRU,
        });

        strategies.insert("query_result".to_string(), CacheConfig {
            ttl: Duration::from_secs(60), // 1 minute
            strategy: CacheStrategy::ReadThrough,
            max_size: Some(1000),
            eviction_policy: EvictionPolicy::LFU,
        });

        Ok(Self {
            dragonfly,
            local_cache,
            strategies,
        })
    }

    pub async fn get<T>(&self, cache_type: &str, key: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let config = self.strategies.get(cache_type)?;

        // Try local cache first (L1)
        if let Some(value) = self.local_cache.get::<T>(key) {
            return Some(value);
        }

        // Try distributed cache (L2)
        match config.strategy {
            CacheStrategy::ReadThrough => {
                if let Ok(Some(value)) = self.get_from_dragonfly::<T>(key) {
                    // Populate local cache
                    self.local_cache.set(key, &value, config.ttl);
                    return Some(value);
                }
            }
            _ => {
                if let Ok(Some(value)) = self.get_from_dragonfly::<T>(key) {
                    self.local_cache.set(key, &value, config.ttl);
                    return Some(value);
                }
            }
        }

        None
    }

    pub async fn set<T>(&self, cache_type: &str, key: &str, value: &T) -> Result<(), Box<dyn std::error::Error>>
    where
        T: serde::Serialize,
    {
        let config = self.strategies.get(cache_type)
            .ok_or("Unknown cache type")?;

        match config.strategy {
            CacheStrategy::WriteThrough => {
                // Write to both caches
                self.local_cache.set(key, value, config.ttl);
                self.set_in_dragonfly(key, value, config.ttl).await?;
            }
            CacheStrategy::WriteBack => {
                // Write to local cache, schedule background write to distributed cache
                self.local_cache.set(key, value, config.ttl);
                self.schedule_background_write(key, value, config.ttl);
            }
            CacheStrategy::WriteAround => {
                // Only write to distributed cache
                self.set_in_dragonfly(key, value, config.ttl).await?;
            }
            CacheStrategy::ReadThrough => {
                // Same as write-through for set operations
                self.local_cache.set(key, value, config.ttl);
                self.set_in_dragonfly(key, value, config.ttl).await?;
            }
        }

        Ok(())
    }

    pub async fn invalidate(&self, key: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.local_cache.remove(key);
        self.invalidate_in_dragonfly(key).await?;
        Ok(())
    }

    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.local_cache.remove_pattern(pattern);
        self.invalidate_pattern_in_dragonfly(pattern).await?;
        Ok(())
    }

    async fn get_from_dragonfly<T>(&self, key: &str) -> redis::RedisResult<Option<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        let mut conn = self.dragonfly.get_connection()?;
        let cached: Option<String> = conn.get(key)?;
        
        match cached {
            Some(data) => {
                let value: T = serde_json::from_str(&data)
                    .map_err(|e| redis::RedisError::from((redis::ErrorKind::TypeError, "Deserialization failed", e.to_string())))?;
                Ok(Some(value))
            }
            None => Ok(None)
        }
    }

    async fn set_in_dragonfly<T>(&self, key: &str, value: &T, ttl: Duration) -> redis::RedisResult<()>
    where
        T: serde::Serialize,
    {
        let mut conn = self.dragonfly.get_connection()?;
        let serialized = serde_json::to_string(value)
            .map_err(|e| redis::RedisError::from((redis::ErrorKind::TypeError, "Serialization failed", e.to_string())))?;
        
        conn.setex(key, ttl.as_secs() as usize, serialized)?;
        Ok(())
    }

    async fn invalidate_in_dragonfly(&self, key: &str) -> redis::RedisResult<()> {
        let mut conn = self.dragonfly.get_connection()?;
        conn.del(key)?;
        Ok(())
    }

    async fn invalidate_pattern_in_dragonfly(&self, pattern: &str) -> redis::RedisResult<()> {
        let mut conn = self.dragonfly.get_connection()?;
        
        // Get all keys matching pattern
        let keys: Vec<String> = conn.keys(pattern)?;
        
        if !keys.is_empty() {
            conn.del(keys)?;
        }
        
        Ok(())
    }

    fn schedule_background_write<T>(&self, key: &str, value: &T, ttl: Duration)
    where
        T: serde::Serialize + Send + 'static,
    {
        let dragonfly = self.dragonfly.clone();
        let key = key.to_string();
        let serialized = serde_json::to_string(value).unwrap();
        
        tokio::spawn(async move {
            // Delay write by a small amount to batch similar operations
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            if let Err(e) = dragonfly.setex(&key, ttl.as_secs() as usize, &serialized) {
                eprintln!("Background cache write failed: {}", e);
            }
        });
    }
}

// Local cache implementation (L1 cache)
struct LocalCache {
    data: std::sync::RwLock<HashMap<String, CacheEntry>>,
    access_times: std::sync::RwLock<HashMap<String, Instant>>,
    access_counts: std::sync::RwLock<HashMap<String, u32>>,
}

struct CacheEntry {
    data: String,
    created_at: Instant,
    ttl: Duration,
}

impl LocalCache {
    fn new() -> Self {
        Self {
            data: std::sync::RwLock::new(HashMap::new()),
            access_times: std::sync::RwLock::new(HashMap::new()),
            access_counts: std::sync::RwLock::new(HashMap::new()),
        }
    }

    fn get<T>(&self, key: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let data = self.data.read().unwrap();
        let entry = data.get(key)?;

        // Check TTL
        if entry.created_at.elapsed() > entry.ttl {
            // Entry expired, remove it
            drop(data);
            self.remove(key);
            return None;
        }

        // Update access tracking
        let mut access_times = self.access_times.write().unwrap();
        access_times.insert(key.to_string(), Instant::now());
        drop(access_times);

        let mut access_counts = self.access_counts.write().unwrap();
        *access_counts.entry(key.to_string()).or_insert(0) += 1;
        drop(access_counts);

        // Deserialize and return
        serde_json::from_str(&entry.data).ok()
    }

    fn set<T>(&self, key: &str, value: &T, ttl: Duration)
    where
        T: serde::Serialize,
    {
        let serialized = serde_json::to_string(value).unwrap();
        let entry = CacheEntry {
            data: serialized,
            created_at: Instant::now(),
            ttl,
        };

        let mut data = self.data.write().unwrap();
        data.insert(key.to_string(), entry);

        // Update access tracking
        drop(data);
        let mut access_times = self.access_times.write().unwrap();
        access_times.insert(key.to_string(), Instant::now());
        drop(access_times);

        let mut access_counts = self.access_counts.write().unwrap();
        access_counts.insert(key.to_string(), 1);
    }

    fn remove(&self, key: &str) {
        let mut data = self.data.write().unwrap();
        data.remove(key);
        drop(data);

        let mut access_times = self.access_times.write().unwrap();
        access_times.remove(key);
        drop(access_times);

        let mut access_counts = self.access_counts.write().unwrap();
        access_counts.remove(key);
    }

    fn remove_pattern(&self, pattern: &str) {
        let mut data = self.data.write().unwrap();
        let keys_to_remove: Vec<String> = data.keys()
            .filter(|k| k.contains(pattern.trim_end_matches('*')))
            .cloned()
            .collect();

        for key in keys_to_remove {
            data.remove(&key);
        }
    }
}
```

This comprehensive database design documentation covers all aspects of PolicyCortex's data architecture, from schema design and event sourcing to caching strategies and performance optimization. The polyglot persistence approach ensures each type of data is stored in the most appropriate database system.