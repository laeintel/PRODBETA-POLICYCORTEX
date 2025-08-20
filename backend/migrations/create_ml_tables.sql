-- Patent #4: Predictive Policy Compliance Engine - Database Schema
-- ML-specific tables for model management, predictions, and feedback
-- Author: PolicyCortex Engineering Team
-- Date: January 2025

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS ml_feedback CASCADE;
DROP TABLE IF EXISTS ml_predictions CASCADE;
DROP TABLE IF EXISTS ml_training_jobs CASCADE;
DROP TABLE IF EXISTS ml_models CASCADE;
DROP TABLE IF EXISTS ml_configurations CASCADE;
DROP TABLE IF EXISTS ml_feature_store CASCADE;
DROP TABLE IF EXISTS ml_drift_metrics CASCADE;

-- ML Configurations table
CREATE TABLE ml_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    configuration JSONB NOT NULL,
    features JSONB,
    policy_context JSONB,
    baseline_config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_ml_config_resource (resource_id),
    INDEX idx_ml_config_tenant (tenant_id),
    INDEX idx_ml_config_created (created_at DESC)
);

-- ML Models table
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'ensemble', 'vae', 'prophet'
    version VARCHAR(20) NOT NULL,
    parameters JSONB NOT NULL,
    metrics JSONB NOT NULL, -- accuracy, precision, recall, f1, etc.
    model_path TEXT, -- S3/Azure blob storage path
    encrypted_model BYTEA, -- Encrypted model parameters
    encryption_key_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'training', 'retired', 'failed'
    training_job_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP WITH TIME ZONE,
    retired_at TIMESTAMP WITH TIME ZONE,
    INDEX idx_ml_model_tenant (tenant_id),
    INDEX idx_ml_model_status (status),
    INDEX idx_ml_model_type (model_type),
    INDEX idx_ml_model_version (version)
);

-- ML Predictions table
CREATE TABLE ml_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id VARCHAR(255) UNIQUE NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    violation_probability DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    time_to_violation_hours DECIMAL(10,2),
    confidence_score DECIMAL(5,4) NOT NULL,
    confidence_interval_lower DECIMAL(5,4),
    confidence_interval_upper DECIMAL(5,4),
    risk_level VARCHAR(20) NOT NULL, -- 'critical', 'high', 'medium', 'low'
    risk_score DECIMAL(5,4),
    impact_factors JSONB,
    uncertainty_sources JSONB,
    recommendations TEXT[],
    features_used JSONB,
    shap_values JSONB,
    attention_weights JSONB,
    inference_time_ms DECIMAL(10,2) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    remediation_status VARCHAR(50), -- 'pending', 'in_progress', 'completed', 'failed'
    remediation_id UUID,
    INDEX idx_ml_pred_resource (resource_id),
    INDEX idx_ml_pred_tenant (tenant_id),
    INDEX idx_ml_pred_timestamp (prediction_timestamp DESC),
    INDEX idx_ml_pred_risk (risk_level),
    INDEX idx_ml_pred_probability (violation_probability DESC),
    INDEX idx_ml_pred_expires (expires_at)
);

-- ML Training Jobs table
CREATE TABLE ml_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    trigger_reason VARCHAR(100) NOT NULL, -- 'scheduled', 'drift_detected', 'manual', 'feedback_threshold'
    status VARCHAR(50) NOT NULL, -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    training_config JSONB NOT NULL,
    hyperparameters JSONB,
    dataset_info JSONB,
    metrics JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    resource_usage JSONB, -- CPU, memory, GPU usage
    cost_estimate DECIMAL(10,2),
    INDEX idx_ml_job_tenant (tenant_id),
    INDEX idx_ml_job_status (status),
    INDEX idx_ml_job_created (created_at DESC)
);

-- ML Feedback table
CREATE TABLE ml_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id VARCHAR(255) UNIQUE NOT NULL,
    prediction_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL, -- 'correct', 'incorrect', 'false_positive', 'false_negative'
    correct_label BOOLEAN,
    accuracy_rating DECIMAL(3,2), -- 0.00 to 5.00
    comments TEXT,
    user_id VARCHAR(255) NOT NULL,
    user_role VARCHAR(50),
    processed BOOLEAN DEFAULT FALSE,
    impact_on_model VARCHAR(50), -- 'retrain_triggered', 'weight_adjusted', 'ignored'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    INDEX idx_ml_feedback_prediction (prediction_id),
    INDEX idx_ml_feedback_tenant (tenant_id),
    INDEX idx_ml_feedback_processed (processed),
    INDEX idx_ml_feedback_created (created_at DESC),
    FOREIGN KEY (prediction_id) REFERENCES ml_predictions(prediction_id)
);

-- Feature Store table
CREATE TABLE ml_feature_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_set_id VARCHAR(255) UNIQUE NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    feature_type VARCHAR(50) NOT NULL, -- 'configuration', 'temporal', 'contextual', 'policy'
    features JSONB NOT NULL,
    feature_version INTEGER DEFAULT 1,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    quality_score DECIMAL(3,2), -- 0.00 to 1.00
    completeness DECIMAL(3,2), -- 0.00 to 1.00
    INDEX idx_ml_feature_resource (resource_id),
    INDEX idx_ml_feature_tenant (tenant_id),
    INDEX idx_ml_feature_type (feature_type),
    INDEX idx_ml_feature_computed (computed_at DESC)
);

-- Drift Metrics table
CREATE TABLE ml_drift_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_id VARCHAR(255) UNIQUE NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    drift_type VARCHAR(50) NOT NULL, -- 'data', 'concept', 'prediction'
    drift_score DECIMAL(10,4) NOT NULL,
    drift_velocity DECIMAL(10,6), -- Rate of change
    reconstruction_error DECIMAL(10,6), -- For VAE
    psi_score DECIMAL(10,6), -- Population Stability Index
    ks_statistic DECIMAL(10,6), -- Kolmogorov-Smirnov
    wasserstein_distance DECIMAL(10,6),
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_level VARCHAR(20), -- 'info', 'warning', 'critical'
    time_to_violation_hours DECIMAL(10,2),
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    baseline_comparison JSONB,
    feature_contributions JSONB,
    INDEX idx_ml_drift_resource (resource_id),
    INDEX idx_ml_drift_tenant (tenant_id),
    INDEX idx_ml_drift_model (model_id),
    INDEX idx_ml_drift_measured (measured_at DESC),
    INDEX idx_ml_drift_alert (alert_triggered, alert_level)
);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_ml_configurations_updated_at BEFORE UPDATE ON ml_configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create partitioning for high-volume tables
-- Partition ml_predictions by month
CREATE TABLE ml_predictions_2025_01 PARTITION OF ml_predictions
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE ml_predictions_2025_02 PARTITION OF ml_predictions
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Add indexes for common query patterns
CREATE INDEX idx_ml_pred_recent_high_risk ON ml_predictions(tenant_id, risk_level, prediction_timestamp DESC)
    WHERE risk_level IN ('critical', 'high');

CREATE INDEX idx_ml_drift_recent_alerts ON ml_drift_metrics(tenant_id, alert_triggered, measured_at DESC)
    WHERE alert_triggered = TRUE;

-- Create views for common queries
CREATE OR REPLACE VIEW v_recent_predictions AS
SELECT 
    p.*,
    m.model_name,
    m.model_type,
    m.metrics->>'accuracy' as model_accuracy
FROM ml_predictions p
JOIN ml_models m ON p.model_id = m.model_id
WHERE p.prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY p.prediction_timestamp DESC;

CREATE OR REPLACE VIEW v_model_performance AS
SELECT 
    m.model_id,
    m.model_name,
    m.tenant_id,
    m.metrics->>'accuracy' as accuracy,
    m.metrics->>'precision' as precision,
    m.metrics->>'recall' as recall,
    m.metrics->>'f1_score' as f1_score,
    COUNT(DISTINCT p.id) as total_predictions,
    AVG(p.confidence_score) as avg_confidence,
    AVG(p.inference_time_ms) as avg_inference_time
FROM ml_models m
LEFT JOIN ml_predictions p ON m.model_id = p.model_id
WHERE m.status = 'active'
GROUP BY m.model_id, m.model_name, m.tenant_id, m.metrics;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO policycortex_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO policycortex_app;