-- Patent #4: Predictive Policy Compliance Engine Database Schema
-- This schema implements the multi-modal data requirements specified in Patent #4
-- Supports configuration telemetry, policy definitions, historical violations, and model metadata

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =====================================================
-- CONFIGURATIONS TABLE
-- Stores resource configuration snapshots with temporal indexing for drift detection
-- =====================================================
CREATE TABLE IF NOT EXISTS configurations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    configuration_data JSONB NOT NULL,
    configuration_hash VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    drift_score FLOAT DEFAULT 0,
    baseline_id UUID REFERENCES configurations(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for configurations
CREATE TABLE configurations_2025_01 PARTITION OF configurations
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE configurations_2025_02 PARTITION OF configurations
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE configurations_2025_03 PARTITION OF configurations
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Indexes for configurations
CREATE INDEX idx_configurations_tenant_resource ON configurations(tenant_id, resource_id, timestamp DESC);
CREATE INDEX idx_configurations_hash ON configurations(configuration_hash);
CREATE INDEX idx_configurations_timestamp ON configurations(timestamp DESC);
CREATE INDEX idx_configurations_drift ON configurations(drift_score) WHERE drift_score > 0;
CREATE INDEX idx_configurations_data_gin ON configurations USING GIN (configuration_data);

-- =====================================================
-- POLICIES TABLE
-- Stores policy definitions with versioning and inheritance relationships
-- =====================================================
CREATE TABLE IF NOT EXISTS policies (
    policy_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    policy_name VARCHAR(255) NOT NULL,
    policy_definition JSONB NOT NULL,
    policy_type VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    parent_policy_id UUID REFERENCES policies(policy_id),
    effective_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expiration_date TIMESTAMPTZ,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    enabled BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for policies
CREATE INDEX idx_policies_tenant ON policies(tenant_id, enabled);
CREATE INDEX idx_policies_effective ON policies(effective_date, expiration_date);
CREATE INDEX idx_policies_parent ON policies(parent_policy_id) WHERE parent_policy_id IS NOT NULL;
CREATE INDEX idx_policies_type ON policies(policy_type);
CREATE INDEX idx_policies_definition_gin ON policies USING GIN (policy_definition);

-- =====================================================
-- VIOLATIONS TABLE
-- Stores historical violation events with detailed context for model training
-- =====================================================
CREATE TABLE IF NOT EXISTS violations (
    violation_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    policy_id UUID NOT NULL REFERENCES policies(policy_id),
    tenant_id VARCHAR(255) NOT NULL,
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    detection_time TIMESTAMPTZ NOT NULL,
    resolution_time TIMESTAMPTZ,
    violation_context JSONB NOT NULL,
    remediation_actions JSONB DEFAULT '[]',
    false_positive BOOLEAN DEFAULT false,
    expert_validated BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (detection_time);

-- Create monthly partitions for violations
CREATE TABLE violations_2025_01 PARTITION OF violations
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE violations_2025_02 PARTITION OF violations
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE violations_2025_03 PARTITION OF violations
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Indexes for violations
CREATE INDEX idx_violations_tenant_resource ON violations(tenant_id, resource_id, detection_time DESC);
CREATE INDEX idx_violations_policy ON violations(policy_id);
CREATE INDEX idx_violations_severity ON violations(severity, detection_time DESC);
CREATE INDEX idx_violations_unresolved ON violations(resolution_time) WHERE resolution_time IS NULL;
CREATE INDEX idx_violations_context_gin ON violations USING GIN (violation_context);

-- =====================================================
-- PREDICTIONS TABLE
-- Stores model predictions with confidence scores and explainability data
-- =====================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    policy_id UUID REFERENCES policies(policy_id),
    tenant_id VARCHAR(255) NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    prediction_horizon_hours INTEGER NOT NULL,
    violation_probability FLOAT NOT NULL CHECK (violation_probability >= 0 AND violation_probability <= 1),
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    risk_level VARCHAR(20) NOT NULL,
    explanation_data JSONB NOT NULL,
    shap_values JSONB,
    attention_weights JSONB,
    model_version VARCHAR(50) NOT NULL,
    remediation_suggested JSONB,
    actual_outcome BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (prediction_time);

-- Create monthly partitions for predictions
CREATE TABLE predictions_2025_01 PARTITION OF predictions
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE predictions_2025_02 PARTITION OF predictions
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE predictions_2025_03 PARTITION OF predictions
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Indexes for predictions
CREATE INDEX idx_predictions_tenant_resource ON predictions(tenant_id, resource_id, prediction_time DESC);
CREATE INDEX idx_predictions_risk ON predictions(risk_level, violation_probability DESC);
CREATE INDEX idx_predictions_confidence ON predictions(confidence_score DESC);
CREATE INDEX idx_predictions_model ON predictions(model_version);
CREATE INDEX idx_predictions_outcome ON predictions(actual_outcome) WHERE actual_outcome IS NOT NULL;

-- =====================================================
-- MODEL_METADATA TABLE
-- Stores information about trained models, versions, and performance metrics
-- =====================================================
CREATE TABLE IF NOT EXISTS model_metadata (
    model_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    training_data_hash VARCHAR(64) NOT NULL,
    hyperparameters JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    feature_importance JSONB,
    deployment_status VARCHAR(50) NOT NULL DEFAULT 'inactive',
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    false_positive_rate FLOAT,
    training_duration_seconds INTEGER,
    model_size_mb FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

-- Indexes for model_metadata
CREATE INDEX idx_model_metadata_tenant ON model_metadata(tenant_id, deployment_status);
CREATE INDEX idx_model_metadata_type ON model_metadata(model_type, version);
CREATE INDEX idx_model_metadata_performance ON model_metadata(accuracy DESC, false_positive_rate);
CREATE UNIQUE INDEX idx_model_metadata_active ON model_metadata(tenant_id, model_type) 
    WHERE deployment_status = 'active';

-- =====================================================
-- FEEDBACK TABLE
-- Stores human feedback on prediction accuracy and model performance
-- =====================================================
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    prediction_id UUID REFERENCES predictions(prediction_id),
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL,
    accuracy_rating INTEGER CHECK (accuracy_rating >= 1 AND accuracy_rating <= 5),
    was_accurate BOOLEAN,
    comments TEXT,
    suggested_action JSONB,
    feedback_time TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT false,
    incorporated_in_model VARCHAR(50)
);

-- Indexes for feedback
CREATE INDEX idx_feedback_tenant ON feedback(tenant_id, feedback_time DESC);
CREATE INDEX idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX idx_feedback_unprocessed ON feedback(processed) WHERE processed = false;
CREATE INDEX idx_feedback_rating ON feedback(accuracy_rating);

-- =====================================================
-- DRIFT_BASELINES TABLE
-- Stores baseline configurations for drift detection
-- =====================================================
CREATE TABLE IF NOT EXISTS drift_baselines (
    baseline_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    baseline_config JSONB NOT NULL,
    baseline_hash VARCHAR(64) NOT NULL,
    statistical_properties JSONB NOT NULL,
    control_limits JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    active BOOLEAN DEFAULT true
);

-- Indexes for drift_baselines
CREATE INDEX idx_drift_baselines_resource ON drift_baselines(tenant_id, resource_id, active);
CREATE INDEX idx_drift_baselines_hash ON drift_baselines(baseline_hash);

-- =====================================================
-- FEATURE_STORE TABLE
-- Stores engineered features for model training and inference
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_store (
    feature_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    feature_vector JSONB NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for feature_store
CREATE TABLE feature_store_2025_01 PARTITION OF feature_store
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE feature_store_2025_02 PARTITION OF feature_store
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Indexes for feature_store
CREATE INDEX idx_feature_store_lookup ON feature_store(tenant_id, resource_id, timestamp DESC);
CREATE INDEX idx_feature_store_type ON feature_store(feature_type, timestamp DESC);

-- =====================================================
-- MODEL_TRAINING_JOBS TABLE
-- Tracks model training jobs and their status
-- =====================================================
CREATE TABLE IF NOT EXISTS model_training_jobs (
    job_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    job_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    trigger_type VARCHAR(50) NOT NULL,
    hyperparameters JSONB NOT NULL,
    training_data_config JSONB NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    resulting_model_id UUID REFERENCES model_metadata(model_id),
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for model_training_jobs
CREATE INDEX idx_training_jobs_tenant ON model_training_jobs(tenant_id, job_status);
CREATE INDEX idx_training_jobs_status ON model_training_jobs(job_status, created_at DESC);

-- =====================================================
-- EXPLANATION_CACHE TABLE
-- Caches SHAP explanations for performance optimization
-- =====================================================
CREATE TABLE IF NOT EXISTS explanation_cache (
    cache_id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    prediction_id UUID REFERENCES predictions(prediction_id),
    explanation_type VARCHAR(50) NOT NULL,
    explanation_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Indexes for explanation_cache
CREATE INDEX idx_explanation_cache_prediction ON explanation_cache(prediction_id);
CREATE INDEX idx_explanation_cache_expiry ON explanation_cache(expires_at);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_configurations_updated_at BEFORE UPDATE ON configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate drift score
CREATE OR REPLACE FUNCTION calculate_drift_score(
    current_config JSONB,
    baseline_config JSONB
) RETURNS FLOAT AS $$
DECLARE
    drift_score FLOAT := 0;
    key_count INTEGER := 0;
BEGIN
    -- Simple drift calculation (can be enhanced with ML model)
    SELECT COUNT(*), AVG(
        CASE 
            WHEN current_config->key != baseline_config->key THEN 1
            ELSE 0
        END
    ) INTO key_count, drift_score
    FROM jsonb_object_keys(current_config) AS key;
    
    RETURN COALESCE(drift_score, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old partitions
CREATE OR REPLACE FUNCTION cleanup_old_partitions(
    months_to_keep INTEGER DEFAULT 6
) RETURNS void AS $$
DECLARE
    partition_name TEXT;
    cutoff_date DATE;
BEGIN
    cutoff_date := DATE_TRUNC('month', NOW() - INTERVAL '1 month' * months_to_keep);
    
    FOR partition_name IN 
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' 
        AND (tablename LIKE 'configurations_%' 
             OR tablename LIKE 'violations_%' 
             OR tablename LIKE 'predictions_%'
             OR tablename LIKE 'feature_store_%')
    LOOP
        -- Extract date from partition name and drop if older than cutoff
        -- Implementation depends on partition naming convention
        NULL; -- Placeholder for actual drop logic
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for active predictions with high risk
CREATE OR REPLACE VIEW high_risk_predictions AS
SELECT 
    p.*,
    pol.policy_name,
    pol.severity as policy_severity
FROM predictions p
JOIN policies pol ON p.policy_id = pol.policy_id
WHERE p.risk_level IN ('high', 'critical')
    AND p.violation_probability > 0.7
    AND p.prediction_time > NOW() - INTERVAL '24 hours'
ORDER BY p.violation_probability DESC, p.confidence_score DESC;

-- View for drift analysis
CREATE OR REPLACE VIEW configuration_drift_analysis AS
SELECT 
    c.resource_id,
    c.tenant_id,
    c.drift_score,
    c.timestamp,
    db.baseline_hash,
    db.control_limits
FROM configurations c
LEFT JOIN drift_baselines db ON 
    c.resource_id = db.resource_id 
    AND c.tenant_id = db.tenant_id 
    AND db.active = true
WHERE c.drift_score > 0
ORDER BY c.drift_score DESC;

-- View for model performance tracking
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    tenant_id,
    model_type,
    version,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    false_positive_rate,
    deployment_status,
    created_at
FROM model_metadata
WHERE deployment_status IN ('active', 'testing')
ORDER BY tenant_id, model_type, created_at DESC;

-- =====================================================
-- PERMISSIONS
-- =====================================================

-- Create roles for different access levels
CREATE ROLE ml_trainer;
CREATE ROLE ml_predictor;
CREATE ROLE ml_admin;

-- Grant appropriate permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ml_predictor;
GRANT INSERT ON predictions, feedback, explanation_cache TO ml_predictor;

GRANT ALL ON ALL TABLES IN SCHEMA public TO ml_trainer;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO ml_trainer;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ml_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ml_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO ml_admin;

-- =====================================================
-- INITIAL DATA AND CONSTRAINTS
-- =====================================================

-- Add check constraints for data quality
ALTER TABLE predictions ADD CONSTRAINT check_horizon 
    CHECK (prediction_horizon_hours BETWEEN 1 AND 168);

ALTER TABLE model_metadata ADD CONSTRAINT check_metrics
    CHECK (accuracy >= 0 AND accuracy <= 1 AND 
           false_positive_rate >= 0 AND false_positive_rate <= 1);

-- Create composite indexes for common query patterns
CREATE INDEX idx_predictions_recent_high_risk ON predictions(tenant_id, risk_level, prediction_time DESC)
    WHERE risk_level IN ('high', 'critical') AND prediction_time > NOW() - INTERVAL '7 days';

CREATE INDEX idx_violations_recent_unresolved ON violations(tenant_id, severity, detection_time DESC)
    WHERE resolution_time IS NULL AND detection_time > NOW() - INTERVAL '30 days';

-- Add comments for documentation
COMMENT ON TABLE configurations IS 'Stores resource configuration snapshots for drift detection and compliance analysis';
COMMENT ON TABLE policies IS 'Policy definitions with versioning and inheritance for compliance rules';
COMMENT ON TABLE violations IS 'Historical violation events used for model training and validation';
COMMENT ON TABLE predictions IS 'ML model predictions with confidence scores and explainability data';
COMMENT ON TABLE model_metadata IS 'Trained model information, versions, and performance metrics';
COMMENT ON TABLE feedback IS 'Human feedback for continuous learning and model improvement';

-- Create notification trigger for high-risk predictions
CREATE OR REPLACE FUNCTION notify_high_risk_prediction() RETURNS trigger AS $$
BEGIN
    IF NEW.risk_level IN ('high', 'critical') AND NEW.violation_probability > 0.8 THEN
        PERFORM pg_notify('high_risk_prediction', json_build_object(
            'prediction_id', NEW.prediction_id,
            'resource_id', NEW.resource_id,
            'risk_level', NEW.risk_level,
            'probability', NEW.violation_probability
        )::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER high_risk_prediction_notify
    AFTER INSERT ON predictions
    FOR EACH ROW EXECUTE FUNCTION notify_high_risk_prediction();