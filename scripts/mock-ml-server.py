#!/usr/bin/env python3
"""
Mock ML Server for PolicyCortex Patent #4 API Testing
Provides mock implementations of all ML endpoints for testing purposes
"""

from flask import Flask, jsonify, request
from datetime import datetime
import random
import uuid

app = Flask(__name__)

# Mock data storage
predictions = []
feedback_items = []

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return 'OK', 200

@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.14.3'
    })

@app.route('/api/v1/predictions', methods=['GET'])
def get_predictions():
    """Get all predictions"""
    mock_predictions = [
        {
            'id': f'pred-{i}',
            'resource_id': f'resource-{i}',
            'prediction_type': 'compliance_drift',
            'risk_score': random.uniform(0.1, 0.9),
            'confidence': random.uniform(0.7, 0.99),
            'timestamp': datetime.utcnow().isoformat(),
            'risk_level': random.choice(['low', 'medium', 'high']),
            'recommendations': [
                'Review security group rules',
                'Update compliance policies'
            ]
        }
        for i in range(5)
    ]
    return jsonify(mock_predictions)

@app.route('/api/v1/predictions', methods=['POST'])
def create_prediction():
    """Create a new prediction"""
    data = request.json
    new_prediction = {
        'id': f'pred-{uuid.uuid4().hex[:8]}',
        'resource_id': data.get('resource_id', 'unknown'),
        'prediction_type': data.get('prediction_type', 'compliance_drift'),
        'risk_score': random.uniform(0.1, 0.9),
        'confidence': random.uniform(0.7, 0.99),
        'timestamp': datetime.utcnow().isoformat()
    }
    predictions.append(new_prediction)
    return jsonify(new_prediction), 201

@app.route('/api/v1/predictions/risk-score/<resource_id>', methods=['GET'])
def get_risk_score(resource_id):
    """Get risk score for a specific resource"""
    return jsonify({
        'resource_id': resource_id,
        'risk_score': random.uniform(0.1, 0.9),
        'risk_level': random.choice(['low', 'medium', 'high', 'critical']),
        'contributing_factors': [
            {'factor': 'Configuration drift', 'weight': 0.35},
            {'factor': 'Policy violations', 'weight': 0.25},
            {'factor': 'Security vulnerabilities', 'weight': 0.40}
        ],
        'recommendations': [
            'Apply latest security patches',
            'Review and update access controls',
            'Enable encryption at rest'
        ],
        'confidence': random.uniform(0.85, 0.99),
        'last_updated': datetime.utcnow().isoformat()
    })

@app.route('/api/v1/predictions/remediate/<resource_id>', methods=['POST'])
def trigger_remediation(resource_id):
    """Trigger remediation for a resource"""
    data = request.json
    return jsonify({
        'remediation_id': f'rem-{uuid.uuid4().hex[:8]}',
        'resource_id': resource_id,
        'status': 'initiated',
        'initiated_at': datetime.utcnow().isoformat(),
        'action_type': data.get('action_type', 'auto_remediate'),
        'estimated_completion': '2 minutes'
    }), 202

@app.route('/api/v1/ml/predict/<resource_id>', methods=['GET'])
def ml_predict(resource_id):
    """ML prediction for a specific resource"""
    return jsonify({
        'resource_id': resource_id,
        'prediction': {
            'violation_probability': random.uniform(0.1, 0.9),
            'time_to_violation_hours': random.uniform(1, 72),
            'confidence_score': random.uniform(0.85, 0.99),
            'model_version': 'v2.1.0'
        },
        'risk_score': random.uniform(0.1, 0.9),
        'confidence': random.uniform(0.85, 0.99),
        'recommendations': [
            'Monitor resource for configuration changes',
            'Review compliance policies',
            'Schedule maintenance window'
        ],
        'inference_time_ms': random.uniform(10, 50)
    })

@app.route('/api/v1/ml/metrics', methods=['GET'])
def get_ml_metrics():
    """Get ML model metrics"""
    return jsonify({
        'accuracy': 0.992,
        'precision': 0.987,
        'recall': 0.983,
        'f1_score': 0.985,
        'auc_roc': 0.996,
        'model_version': 'v2.1.0',
        'last_trained': '2025-01-15T10:30:00Z',
        'training_samples': 1500000,
        'validation_samples': 300000,
        'test_samples': 200000,
        'false_positive_rate': 0.018,
        'false_negative_rate': 0.022,
        'models': {
            'lstm': {'accuracy': 0.989, 'weight': 0.3},
            'isolation_forest': {'accuracy': 0.991, 'weight': 0.4},
            'autoencoder': {'accuracy': 0.988, 'weight': 0.3}
        }
    })

@app.route('/api/v1/ml/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get SHAP feature importance analysis"""
    return jsonify({
        'global_importance': {
            'cpu_utilization': 0.342,
            'memory_usage': 0.298,
            'network_throughput': 0.156,
            'error_rate': 0.089,
            'compliance_score': 0.067,
            'configuration_changes': 0.048
        },
        'feature_impacts': [
            {'name': 'cpu_utilization', 'impact': 0.342, 'direction': 'positive'},
            {'name': 'memory_usage', 'impact': 0.298, 'direction': 'positive'},
            {'name': 'network_throughput', 'impact': 0.156, 'direction': 'negative'}
        ],
        'model_interpretation': {
            'confidence': 0.92,
            'explanation': 'Resource utilization metrics are primary indicators of compliance drift'
        },
        'features': [
            {'name': 'cpu_utilization', 'importance': 0.342},
            {'name': 'memory_usage', 'importance': 0.298},
            {'name': 'network_throughput', 'importance': 0.156}
        ]
    })

@app.route('/api/v1/ml/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for ML model improvement"""
    data = request.json
    feedback_id = f'feedback-{uuid.uuid4().hex[:8]}'
    feedback_items.append({
        'id': feedback_id,
        **data,
        'processed_at': datetime.utcnow().isoformat()
    })
    return jsonify({
        'feedback_id': feedback_id,
        'status': 'accepted',
        'processed_at': datetime.utcnow().isoformat(),
        'message': 'Feedback received and queued for processing'
    }), 201

@app.route('/api/v1/configurations/drift-analysis', methods=['POST'])
def analyze_drift():
    """Analyze configuration drift"""
    data = request.json
    
    # Simulate drift detection
    has_drift = random.choice([True, False])
    drift_score = random.uniform(0.1, 0.9) if has_drift else random.uniform(0, 0.3)
    
    return jsonify({
        'drift_score': drift_score,
        'drift_detected': has_drift,
        'drift_details': [
            {'field': 'vm_size', 'current': 'Standard_D4s_v3', 'baseline': 'Standard_D2s_v3', 'severity': 'medium'},
            {'field': 'tags.compliance', 'current': None, 'baseline': 'required', 'severity': 'high'}
        ] if has_drift else [],
        'recommendations': [
            'Revert VM size to baseline configuration',
            'Add missing compliance tag'
        ] if has_drift else ['Configuration is within acceptable drift tolerance'],
        'analysis_timestamp': datetime.utcnow().isoformat()
    }), 201

@app.route('/api/v1/ml/anomalies', methods=['GET'])
def detect_anomalies():
    """Detect anomalies in resources"""
    return jsonify({
        'anomalies': [
            {
                'resource_id': f'resource-{i}',
                'anomaly_score': random.uniform(0.7, 0.95),
                'type': random.choice(['behavioral', 'statistical', 'contextual']),
                'detected_at': datetime.utcnow().isoformat(),
                'description': 'Unusual spike in API calls detected'
            }
            for i in range(3)
        ],
        'total_resources_analyzed': 150,
        'anomalies_detected': 3
    })

@app.route('/api/v1/ml/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining"""
    data = request.json
    return jsonify({
        'job_id': f'retrain-{uuid.uuid4().hex[:8]}',
        'status': 'queued',
        'model_type': data.get('model_type', 'compliance_prediction'),
        'initiated_at': datetime.utcnow().isoformat(),
        'estimated_duration': '45 minutes',
        'message': 'Retraining job successfully queued'
    }), 202

@app.route('/api/v1/correlations', methods=['GET'])
def get_correlations():
    """Get cross-domain correlations (Patent #4)"""
    return jsonify({
        'correlations': [
            {
                'id': f'corr-{i}',
                'domains': ['security', 'compliance'],
                'correlation_strength': random.uniform(0.7, 0.95),
                'pattern': 'Security group changes correlate with compliance drift',
                'resources_affected': random.randint(10, 50),
                'risk_impact': random.choice(['low', 'medium', 'high'])
            }
            for i in range(5)
        ],
        'analysis_timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/v1/conversation', methods=['POST'])
def process_conversation():
    """Process conversational AI request (Patent #2)"""
    data = request.json
    message = data.get('message', '')
    
    # Mock NLP response
    return jsonify({
        'response': f'Based on your query about "{message[:50]}...", I found 3 compliance risks affecting 12 virtual machines. The primary concern is missing encryption at rest policies.',
        'intent': 'query_compliance_risks',
        'entities': [
            {'type': 'resource_type', 'value': 'virtual_machines'},
            {'type': 'risk_type', 'value': 'compliance'}
        ],
        'confidence': 0.94,
        'suggestions': [
            'View detailed compliance report',
            'Apply recommended remediations',
            'Schedule compliance review'
        ],
        'session_id': data.get('context', {}).get('session_id', 'default')
    }), 200

if __name__ == '__main__':
    print("Starting Mock ML Server on port 8081...")
    print("This mock server provides all Patent #4 ML endpoints for testing")
    app.run(host='0.0.0.0', port=8081, debug=False)