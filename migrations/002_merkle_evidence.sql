-- Migration: Create Merkle tree and evidence tables
-- Required for T05 - Evidence/Merkle implementation

-- Evidence artifacts table
CREATE TABLE IF NOT EXISTS evidence_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type VARCHAR(100) NOT NULL,
    artifact_data JSONB NOT NULL,
    content_hash VARCHAR(64) NOT NULL, -- SHA-256 hash
    merkle_root VARCHAR(64),
    merkle_proof JSONB,
    signature TEXT,
    signer_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(255),
    resource_id VARCHAR(500),
    policy_id VARCHAR(255),
    
    -- Indexes for query performance
    INDEX idx_evidence_hash (content_hash),
    INDEX idx_evidence_merkle (merkle_root),
    INDEX idx_evidence_timestamp (timestamp DESC),
    INDEX idx_evidence_tenant (tenant_id),
    INDEX idx_evidence_resource (resource_id),
    INDEX idx_evidence_policy (policy_id)
);

-- Merkle trees table
CREATE TABLE IF NOT EXISTS merkle_trees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tree_root VARCHAR(64) NOT NULL UNIQUE,
    tree_height INTEGER NOT NULL,
    leaf_count INTEGER NOT NULL,
    tree_data JSONB NOT NULL, -- Stores complete tree structure
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    finalized_at TIMESTAMP WITH TIME ZONE,
    is_finalized BOOLEAN DEFAULT FALSE,
    tenant_id VARCHAR(255),
    
    INDEX idx_merkle_root (tree_root),
    INDEX idx_merkle_created (created_at DESC),
    INDEX idx_merkle_tenant (tenant_id)
);

-- Merkle tree leaves table
CREATE TABLE IF NOT EXISTS merkle_leaves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tree_id UUID NOT NULL REFERENCES merkle_trees(id) ON DELETE CASCADE,
    leaf_index INTEGER NOT NULL,
    leaf_hash VARCHAR(64) NOT NULL,
    artifact_id UUID REFERENCES evidence_artifacts(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(tree_id, leaf_index),
    INDEX idx_leaves_tree (tree_id),
    INDEX idx_leaves_hash (leaf_hash)
);

-- Verification records table
CREATE TABLE IF NOT EXISTS verification_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID REFERENCES evidence_artifacts(id),
    merkle_root VARCHAR(64) NOT NULL,
    verification_hash VARCHAR(64) NOT NULL,
    verification_result BOOLEAN NOT NULL,
    verification_proof JSONB,
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    verified_by VARCHAR(255),
    verification_method VARCHAR(50), -- 'online', 'offline', 'external'
    
    INDEX idx_verification_artifact (artifact_id),
    INDEX idx_verification_root (merkle_root),
    INDEX idx_verification_time (verified_at DESC)
);

-- Export records for tracking evidence exports
CREATE TABLE IF NOT EXISTS evidence_exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_type VARCHAR(50) NOT NULL, -- 'json', 'csv', 'pdf'
    export_format_version VARCHAR(20) DEFAULT '1.0',
    merkle_root VARCHAR(64) NOT NULL,
    artifacts_included INTEGER NOT NULL,
    export_data JSONB,
    export_hash VARCHAR(64) NOT NULL,
    exported_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    exported_by VARCHAR(255),
    tenant_id VARCHAR(255),
    
    INDEX idx_exports_merkle (merkle_root),
    INDEX idx_exports_time (exported_at DESC),
    INDEX idx_exports_tenant (tenant_id)
);

-- Compliance evidence chain table
CREATE TABLE IF NOT EXISTS compliance_chain (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id VARCHAR(255) NOT NULL,
    sequence_number INTEGER NOT NULL,
    previous_hash VARCHAR(64),
    current_hash VARCHAR(64) NOT NULL,
    evidence_data JSONB NOT NULL,
    block_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(255),
    
    UNIQUE(chain_id, sequence_number),
    INDEX idx_chain_id (chain_id),
    INDEX idx_chain_seq (sequence_number),
    INDEX idx_chain_hash (current_hash)
);

-- Function to calculate Merkle root
CREATE OR REPLACE FUNCTION calculate_merkle_root(leaf_hashes TEXT[])
RETURNS TEXT AS $$
DECLARE
    current_level TEXT[];
    next_level TEXT[];
    i INTEGER;
    combined TEXT;
BEGIN
    IF array_length(leaf_hashes, 1) = 0 THEN
        RETURN NULL;
    END IF;
    
    current_level := leaf_hashes;
    
    WHILE array_length(current_level, 1) > 1 LOOP
        next_level := ARRAY[]::TEXT[];
        
        FOR i IN 1..array_length(current_level, 1) BY 2 LOOP
            IF i + 1 <= array_length(current_level, 1) THEN
                -- Canonical pairing: sort hashes before combining
                IF current_level[i] < current_level[i + 1] THEN
                    combined := current_level[i] || current_level[i + 1];
                ELSE
                    combined := current_level[i + 1] || current_level[i];
                END IF;
            ELSE
                -- Odd number of elements, duplicate last one
                combined := current_level[i] || current_level[i];
            END IF;
            
            -- In production, this should call a proper SHA-256 function
            next_level := array_append(next_level, encode(sha256(combined::bytea), 'hex'));
        END LOOP;
        
        current_level := next_level;
    END LOOP;
    
    RETURN current_level[1];
END;
$$ LANGUAGE plpgsql;

-- Function to verify Merkle proof
CREATE OR REPLACE FUNCTION verify_merkle_proof(
    leaf_hash TEXT,
    merkle_root TEXT,
    proof JSONB
)
RETURNS BOOLEAN AS $$
DECLARE
    current_hash TEXT;
    proof_element JSONB;
    sibling_hash TEXT;
    position TEXT;
BEGIN
    current_hash := leaf_hash;
    
    FOR proof_element IN SELECT * FROM jsonb_array_elements(proof)
    LOOP
        sibling_hash := proof_element->>'hash';
        position := proof_element->>'position';
        
        IF position = 'left' THEN
            current_hash := encode(sha256((sibling_hash || current_hash)::bytea), 'hex');
        ELSE
            current_hash := encode(sha256((current_hash || sibling_hash)::bytea), 'hex');
        END IF;
    END LOOP;
    
    RETURN current_hash = merkle_root;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update Merkle tree when leaves are added
CREATE OR REPLACE FUNCTION update_merkle_tree()
RETURNS TRIGGER AS $$
DECLARE
    leaf_hashes TEXT[];
    new_root TEXT;
BEGIN
    -- Get all leaf hashes for this tree
    SELECT ARRAY_AGG(leaf_hash ORDER BY leaf_index)
    INTO leaf_hashes
    FROM merkle_leaves
    WHERE tree_id = NEW.tree_id;
    
    -- Calculate new Merkle root
    new_root := calculate_merkle_root(leaf_hashes);
    
    -- Update the tree
    UPDATE merkle_trees
    SET tree_root = new_root,
        leaf_count = array_length(leaf_hashes, 1)
    WHERE id = NEW.tree_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER merkle_tree_update
    AFTER INSERT ON merkle_leaves
    FOR EACH ROW
    EXECUTE FUNCTION update_merkle_tree();

-- View for latest verification status
CREATE OR REPLACE VIEW evidence_verification_status AS
SELECT 
    ea.id,
    ea.artifact_type,
    ea.content_hash,
    ea.merkle_root,
    ea.timestamp,
    vr.verification_result,
    vr.verified_at,
    vr.verified_by
FROM evidence_artifacts ea
LEFT JOIN LATERAL (
    SELECT verification_result, verified_at, verified_by
    FROM verification_records
    WHERE artifact_id = ea.id
    ORDER BY verified_at DESC
    LIMIT 1
) vr ON TRUE;

-- Materialized view for evidence statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS evidence_statistics AS
SELECT 
    artifact_type,
    COUNT(*) as artifact_count,
    DATE_TRUNC('day', timestamp) as evidence_date,
    COUNT(DISTINCT tenant_id) as unique_tenants,
    COUNT(DISTINCT policy_id) as unique_policies
FROM evidence_artifacts
GROUP BY artifact_type, DATE_TRUNC('day', timestamp);

CREATE INDEX idx_evidence_stats_type ON evidence_statistics(artifact_type);
CREATE INDEX idx_evidence_stats_date ON evidence_statistics(evidence_date DESC);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT ON evidence_artifacts TO policycortex_app;
-- GRANT SELECT, INSERT ON merkle_trees TO policycortex_app;
-- GRANT SELECT, INSERT ON merkle_leaves TO policycortex_app;
-- GRANT SELECT, INSERT ON verification_records TO policycortex_app;
-- GRANT SELECT, INSERT ON evidence_exports TO policycortex_app;
-- GRANT SELECT, INSERT ON compliance_chain TO policycortex_app;
-- GRANT SELECT ON evidence_verification_status TO policycortex_app;
-- GRANT SELECT ON evidence_statistics TO policycortex_app;