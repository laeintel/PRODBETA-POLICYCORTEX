-- Migration: Create predictions and ML model tables
-- Required for T06 - Predictions/Explain implementation

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    resource_id VARCHAR(500) NOT NULL,
    control_family VARCHAR(100) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL, -- 'compliance', 'drift', 'cost', 'security'
    will_fail BOOLEAN NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    eta TIMESTAMP WITH TIME ZONE,
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    
    -- Feature importance (SHAP values)
    top_features JSONB,
    feature_values JSONB,
    shap_values JSONB,
    
    -- Remediation
    recommended_action TEXT,
    fix_branch VARCHAR(255),
    pull_request_url TEXT,
    
    INDEX idx_predictions_tenant (tenant_id),
    INDEX idx_predictions_resource (resource_id),
    INDEX idx_predictions_control (control_family),
    INDEX idx_predictions_created (created_at DESC),
    INDEX idx_predictions_eta (eta),
    INDEX idx_predictions_risk (risk_score DESC)
);

-- ML model registry
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'isolation_forest', 'autoencoder', 'ensemble'
    model_path TEXT,
    model_metadata JSONB,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_date TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT FALSE,
    
    UNIQUE(model_name, model_version),
    INDEX idx_models_name (model_name),
    INDEX idx_models_version (model_version),
    INDEX idx_models_active (is_active)
);

-- Feature store for ML
CREATE TABLE IF NOT EXISTS feature_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) NOT NULL,
    feature_set VARCHAR(100) NOT NULL,
    feature_data JSONB NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(255),
    
    -- Feature versioning
    feature_version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,
    
    INDEX idx_features_resource (resource_id),
    INDEX idx_features_set (feature_set),
    INDEX idx_features_computed (computed_at DESC),
    INDEX idx_features_tenant (tenant_id)
);

-- Training data samples
CREATE TABLE IF NOT EXISTS training_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) NOT NULL,
    feature_vector JSONB NOT NULL,
    label BOOLEAN NOT NULL, -- True = will fail, False = compliant
    label_confidence DECIMAL(5,4),
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    used_in_training BOOLEAN DEFAULT FALSE,
    model_version VARCHAR(50),
    tenant_id VARCHAR(255),
    
    INDEX idx_training_resource (resource_id),
    INDEX idx_training_collected (collected_at DESC),
    INDEX idx_training_used (used_in_training),
    INDEX idx_training_tenant (tenant_id)
);

-- Prediction feedback for continuous learning
CREATE TABLE IF NOT EXISTS prediction_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES predictions(id),
    actual_outcome BOOLEAN,
    feedback_type VARCHAR(50), -- 'correct', 'false_positive', 'false_negative'
    feedback_text TEXT,
    feedback_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    provided_by VARCHAR(255),
    
    INDEX idx_feedback_prediction (prediction_id),
    INDEX idx_feedback_type (feedback_type),
    INDEX idx_feedback_timestamp (feedback_timestamp DESC)
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id),
    metric_date DATE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    average_confidence DECIMAL(5,4),
    average_latency_ms INTEGER,
    
    UNIQUE(model_id, metric_date),
    INDEX idx_metrics_model (model_id),
    INDEX idx_metrics_date (metric_date DESC)
);

-- Drift detection records
CREATE TABLE IF NOT EXISTS drift_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id VARCHAR(500) NOT NULL,
    drift_type VARCHAR(50) NOT NULL, -- 'config', 'performance', 'data', 'concept'
    drift_score DECIMAL(5,4) NOT NULL,
    baseline_snapshot JSONB,
    current_snapshot JSONB,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    auto_remediated BOOLEAN DEFAULT FALSE,
    remediation_id UUID,
    
    INDEX idx_drift_resource (resource_id),
    INDEX idx_drift_type (drift_type),
    INDEX idx_drift_detected (detected_at DESC),
    INDEX idx_drift_score (drift_score DESC)
);

-- Explanation cache for SHAP values
CREATE TABLE IF NOT EXISTS explanation_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES predictions(id),
    explanation_type VARCHAR(50) NOT NULL, -- 'shap', 'lime', 'anchor'
    explanation_data JSONB NOT NULL,
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cache_ttl_seconds INTEGER DEFAULT 3600,
    
    UNIQUE(prediction_id, explanation_type),
    INDEX idx_explanation_prediction (prediction_id),
    INDEX idx_explanation_cached (cached_at)
);

-- Function to calculate model accuracy
CREATE OR REPLACE FUNCTION calculate_model_accuracy(model_uuid UUID, days_back INTEGER DEFAULT 30)
RETURNS TABLE(
    accuracy DECIMAL,
    precision_val DECIMAL,
    recall_val DECIMAL,
    f1 DECIMAL
) AS $$
DECLARE
    tp INTEGER;
    tn INTEGER;
    fp INTEGER;
    fn INTEGER;
    total INTEGER;
BEGIN
    SELECT 
        SUM(CASE WHEN pf.actual_outcome = true AND p.will_fail = true THEN 1 ELSE 0 END),
        SUM(CASE WHEN pf.actual_outcome = false AND p.will_fail = false THEN 1 ELSE 0 END),
        SUM(CASE WHEN pf.actual_outcome = false AND p.will_fail = true THEN 1 ELSE 0 END),
        SUM(CASE WHEN pf.actual_outcome = true AND p.will_fail = false THEN 1 ELSE 0 END)
    INTO tp, tn, fp, fn
    FROM predictions p
    JOIN prediction_feedback pf ON p.id = pf.prediction_id
    WHERE p.model_version = (SELECT model_version FROM ml_models WHERE id = model_uuid)
    AND p.created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' * days_back;
    
    total := tp + tn + fp + fn;
    
    IF total > 0 THEN
        accuracy := (tp + tn)::DECIMAL / total;
        
        IF tp + fp > 0 THEN
            precision_val := tp::DECIMAL / (tp + fp);
        ELSE
            precision_val := 0;
        END IF;
        
        IF tp + fn > 0 THEN
            recall_val := tp::DECIMAL / (tp + fn);
        ELSE
            recall_val := 0;
        END IF;
        
        IF precision_val + recall_val > 0 THEN
            f1 := 2 * (precision_val * recall_val) / (precision_val + recall_val);
        ELSE
            f1 := 0;
        END IF;
    ELSE
        accuracy := 0;
        precision_val := 0;
        recall_val := 0;
        f1 := 0;
    END IF;
    
    RETURN QUERY SELECT accuracy, precision_val, recall_val, f1;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update model metrics daily
CREATE OR REPLACE FUNCTION update_model_metrics()
RETURNS void AS $$
DECLARE
    model RECORD;
    metrics RECORD;
BEGIN
    FOR model IN SELECT id, model_version FROM ml_models WHERE is_active = true
    LOOP
        SELECT * INTO metrics FROM calculate_model_accuracy(model.id, 1);
        
        INSERT INTO model_metrics (
            model_id, metric_date, 
            correct_predictions,
            false_positives,
            false_negatives,
            average_confidence
        )
        VALUES (
            model.id,
            CURRENT_DATE,
            COALESCE(metrics.accuracy * 100, 0)::INTEGER,
            0, -- Calculate separately if needed
            0, -- Calculate separately if needed
            (SELECT AVG(confidence) FROM predictions 
             WHERE model_version = model.model_version 
             AND created_at >= CURRENT_DATE)
        )
        ON CONFLICT (model_id, metric_date) 
        DO UPDATE SET
            correct_predictions = EXCLUDED.correct_predictions,
            average_confidence = EXCLUDED.average_confidence;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- View for prediction success rate by control family
CREATE OR REPLACE VIEW prediction_success_by_control AS
SELECT 
    p.control_family,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN pf.feedback_type = 'correct' THEN 1 ELSE 0 END) as correct_predictions,
    AVG(p.confidence) as avg_confidence,
    AVG(p.risk_score) as avg_risk_score
FROM predictions p
LEFT JOIN prediction_feedback pf ON p.id = pf.prediction_id
GROUP BY p.control_family;

-- Materialized view for ML performance dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS ml_performance_dashboard AS
SELECT 
    m.model_name,
    m.model_version,
    m.model_type,
    COUNT(DISTINCT p.id) as total_predictions_30d,
    AVG(p.confidence) as avg_confidence_30d,
    COUNT(DISTINCT pf.id) as feedback_count_30d,
    SUM(CASE WHEN pf.feedback_type = 'correct' THEN 1 ELSE 0 END)::DECIMAL / 
        NULLIF(COUNT(DISTINCT pf.id), 0) as accuracy_30d
FROM ml_models m
LEFT JOIN predictions p ON p.model_version = m.model_version 
    AND p.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
LEFT JOIN prediction_feedback pf ON pf.prediction_id = p.id
WHERE m.is_active = true
GROUP BY m.model_name, m.model_version, m.model_type;

CREATE INDEX idx_ml_dashboard_model ON ml_performance_dashboard(model_name);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT ON predictions TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON ml_models TO policycortex_app;
-- GRANT SELECT, INSERT, UPDATE ON feature_store TO policycortex_app;
-- GRANT SELECT, INSERT ON training_samples TO policycortex_app;
-- GRANT SELECT, INSERT ON prediction_feedback TO policycortex_app;
-- GRANT SELECT, INSERT ON model_metrics TO policycortex_app;
-- GRANT SELECT, INSERT ON drift_records TO policycortex_app;
-- GRANT SELECT, INSERT ON explanation_cache TO policycortex_app;
-- GRANT SELECT ON prediction_success_by_control TO policycortex_app;
-- GRANT SELECT ON ml_performance_dashboard TO policycortex_app;