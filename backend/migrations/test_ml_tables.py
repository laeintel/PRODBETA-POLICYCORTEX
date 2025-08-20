"""
Test ML database tables with sample data
"""
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import json
from datetime import datetime, timedelta
import uuid
import random

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'policycortex',
    'user': 'postgres',
    'password': 'postgres'
}

def generate_test_data():
    """Generate sample test data for ML tables"""
    
    # Sample tenant and resource IDs
    tenant_id = "tenant_001"
    resource_id = f"rg-prod-{uuid.uuid4().hex[:8]}"
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
    feedback_id = f"feedback_{uuid.uuid4().hex[:8]}"
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    feature_set_id = f"features_{uuid.uuid4().hex[:8]}"
    metric_id = f"metric_{uuid.uuid4().hex[:8]}"
    
    return {
        'tenant_id': tenant_id,
        'resource_id': resource_id,
        'model_id': model_id,
        'prediction_id': prediction_id,
        'feedback_id': feedback_id,
        'job_id': job_id,
        'feature_set_id': feature_set_id,
        'metric_id': metric_id
    }

def test_ml_configurations(cursor, data):
    """Test ML configurations table"""
    print("\n1. Testing ml_configurations table...")
    
    insert_sql = """
        INSERT INTO ml_configurations 
        (resource_id, tenant_id, configuration, features, policy_context, baseline_config)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, created_at;
    """
    
    config_data = (
        data['resource_id'],
        data['tenant_id'],
        Json({
            "max_cpu": 80,
            "max_memory": 90,
            "auto_scale": True,
            "min_instances": 2,
            "max_instances": 10
        }),
        Json({
            "cpu_usage_avg": 65.5,
            "memory_usage_avg": 72.3,
            "request_rate": 1500,
            "error_rate": 0.02
        }),
        Json({
            "compliance_framework": "ISO27001",
            "data_classification": "confidential",
            "region": "eastus"
        }),
        Json({
            "baseline_cpu": 50,
            "baseline_memory": 60
        })
    )
    
    cursor.execute(insert_sql, config_data)
    result = cursor.fetchone()
    print(f"   Inserted configuration with ID: {result['id']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_configurations WHERE resource_id = %s", (data['resource_id'],))
    config = cursor.fetchone()
    print(f"   Retrieved configuration for resource: {config['resource_id']}")
    return True

def test_ml_models(cursor, data):
    """Test ML models table"""
    print("\n2. Testing ml_models table...")
    
    insert_sql = """
        INSERT INTO ml_models 
        (model_id, tenant_id, model_name, model_type, version, parameters, metrics, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at;
    """
    
    model_data = (
        data['model_id'],
        data['tenant_id'],
        "PolicyCompliancePredictor",
        "ensemble",
        "1.0.0",
        Json({
            "lstm_layers": 3,
            "lstm_units": 512,
            "dropout": 0.2,
            "attention_heads": 8,
            "learning_rate": 0.001
        }),
        Json({
            "accuracy": 0.992,
            "precision": 0.985,
            "recall": 0.978,
            "f1_score": 0.981,
            "auc_roc": 0.996
        }),
        "active"
    )
    
    cursor.execute(insert_sql, model_data)
    result = cursor.fetchone()
    print(f"   Inserted model with ID: {result['id']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_models WHERE model_id = %s", (data['model_id'],))
    model = cursor.fetchone()
    print(f"   Retrieved model: {model['model_name']} v{model['version']}")
    print(f"   Model accuracy: {model['metrics']['accuracy']}")
    return True

def test_ml_predictions(cursor, data):
    """Test ML predictions table"""
    print("\n3. Testing ml_predictions table...")
    
    insert_sql = """
        INSERT INTO ml_predictions 
        (prediction_id, resource_id, tenant_id, model_id, violation_probability, 
         time_to_violation_hours, confidence_score, risk_level, risk_score,
         recommendations, features_used, shap_values, inference_time_ms, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, prediction_timestamp;
    """
    
    prediction_data = (
        data['prediction_id'],
        data['resource_id'],
        data['tenant_id'],
        data['model_id'],
        0.8734,  # violation_probability
        12.5,    # time_to_violation_hours
        0.925,   # confidence_score
        "high",  # risk_level
        0.85,    # risk_score
        ["Enable auto-scaling", "Increase memory allocation", "Review security policies"],
        Json({
            "cpu_usage": 0.82,
            "memory_usage": 0.91,
            "network_latency": 150,
            "error_rate": 0.05
        }),
        Json({
            "cpu_usage": 0.35,
            "memory_usage": 0.42,
            "network_latency": 0.08,
            "error_rate": 0.15
        }),
        45.67,   # inference_time_ms
        "1.0.0"  # model_version
    )
    
    cursor.execute(insert_sql, prediction_data)
    result = cursor.fetchone()
    print(f"   Inserted prediction with ID: {result['id']}")
    print(f"   Prediction timestamp: {result['prediction_timestamp']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_predictions WHERE prediction_id = %s", (data['prediction_id'],))
    prediction = cursor.fetchone()
    print(f"   Violation probability: {prediction['violation_probability']}")
    print(f"   Risk level: {prediction['risk_level']}")
    print(f"   Recommendations: {prediction['recommendations']}")
    return True

def test_ml_training_jobs(cursor, data):
    """Test ML training jobs table"""
    print("\n4. Testing ml_training_jobs table...")
    
    insert_sql = """
        INSERT INTO ml_training_jobs 
        (job_id, tenant_id, model_type, trigger_reason, status, training_config,
         hyperparameters, dataset_info, started_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at;
    """
    
    job_data = (
        data['job_id'],
        data['tenant_id'],
        "ensemble",
        "scheduled",
        "running",
        Json({
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping": True
        }),
        Json({
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy"
        }),
        Json({
            "total_samples": 50000,
            "features": 45,
            "train_samples": 40000,
            "val_samples": 10000
        }),
        datetime.now()
    )
    
    cursor.execute(insert_sql, job_data)
    result = cursor.fetchone()
    print(f"   Inserted training job with ID: {result['id']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_training_jobs WHERE job_id = %s", (data['job_id'],))
    job = cursor.fetchone()
    print(f"   Job status: {job['status']}")
    print(f"   Trigger reason: {job['trigger_reason']}")
    return True

def test_ml_feedback(cursor, data):
    """Test ML feedback table"""
    print("\n5. Testing ml_feedback table...")
    
    insert_sql = """
        INSERT INTO ml_feedback 
        (feedback_id, prediction_id, tenant_id, feedback_type, correct_label,
         accuracy_rating, comments, user_id, user_role)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at;
    """
    
    feedback_data = (
        data['feedback_id'],
        data['prediction_id'],
        data['tenant_id'],
        "correct",
        True,
        4.5,
        "Prediction was accurate. Resource did violate policy within predicted timeframe.",
        "user_admin_001",
        "CloudArchitect"
    )
    
    cursor.execute(insert_sql, feedback_data)
    result = cursor.fetchone()
    print(f"   Inserted feedback with ID: {result['id']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_feedback WHERE feedback_id = %s", (data['feedback_id'],))
    feedback = cursor.fetchone()
    print(f"   Feedback type: {feedback['feedback_type']}")
    print(f"   Accuracy rating: {feedback['accuracy_rating']}")
    return True

def test_ml_feature_store(cursor, data):
    """Test ML feature store table"""
    print("\n6. Testing ml_feature_store table...")
    
    insert_sql = """
        INSERT INTO ml_feature_store 
        (feature_set_id, resource_id, tenant_id, feature_type, features, 
         quality_score, completeness)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, computed_at;
    """
    
    feature_data = (
        data['feature_set_id'],
        data['resource_id'],
        data['tenant_id'],
        "temporal",
        Json({
            "cpu_usage_1h": [65, 70, 68, 72, 75],
            "memory_usage_1h": [80, 82, 85, 83, 84],
            "request_rate_1h": [1200, 1350, 1400, 1380, 1420],
            "error_rate_1h": [0.01, 0.02, 0.015, 0.018, 0.02],
            "latency_p95_1h": [120, 125, 130, 128, 135]
        }),
        0.95,
        0.98
    )
    
    cursor.execute(insert_sql, feature_data)
    result = cursor.fetchone()
    print(f"   Inserted feature set with ID: {result['id']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_feature_store WHERE feature_set_id = %s", (data['feature_set_id'],))
    features = cursor.fetchone()
    print(f"   Feature type: {features['feature_type']}")
    print(f"   Quality score: {features['quality_score']}")
    return True

def test_ml_drift_metrics(cursor, data):
    """Test ML drift metrics table"""
    print("\n7. Testing ml_drift_metrics table...")
    
    insert_sql = """
        INSERT INTO ml_drift_metrics 
        (metric_id, resource_id, tenant_id, model_id, drift_type, drift_score,
         drift_velocity, reconstruction_error, psi_score, alert_triggered, alert_level)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, measured_at;
    """
    
    drift_data = (
        data['metric_id'],
        data['resource_id'],
        data['tenant_id'],
        data['model_id'],
        "data",
        0.0234,
        0.000125,
        0.0189,
        0.0145,
        False,
        "info"
    )
    
    cursor.execute(insert_sql, drift_data)
    result = cursor.fetchone()
    print(f"   Inserted drift metric with ID: {result['id']}")
    
    # Read back
    cursor.execute("SELECT * FROM ml_drift_metrics WHERE metric_id = %s", (data['metric_id'],))
    drift = cursor.fetchone()
    print(f"   Drift type: {drift['drift_type']}")
    print(f"   Drift score: {drift['drift_score']}")
    print(f"   Alert level: {drift['alert_level']}")
    return True

def test_views(cursor):
    """Test database views"""
    print("\n8. Testing database views...")
    
    # Test v_recent_predictions view
    cursor.execute("SELECT COUNT(*) as count FROM v_recent_predictions")
    result = cursor.fetchone()
    count = result['count']
    print(f"   v_recent_predictions: {count} predictions in last 24 hours")
    
    # Test v_model_performance view
    cursor.execute("SELECT COUNT(*) as count FROM v_model_performance")
    result = cursor.fetchone()
    count = result['count']
    print(f"   v_model_performance: {count} active models")
    
    return True

def test_complex_queries(cursor, data):
    """Test complex queries and relationships"""
    print("\n9. Testing complex queries...")
    
    # Query 1: Get predictions with feedback
    query1 = """
        SELECT 
            p.prediction_id,
            p.violation_probability,
            p.risk_level,
            f.feedback_type,
            f.accuracy_rating
        FROM ml_predictions p
        LEFT JOIN ml_feedback f ON p.prediction_id = f.prediction_id
        WHERE p.tenant_id = %s
        ORDER BY p.prediction_timestamp DESC
        LIMIT 5;
    """
    
    cursor.execute(query1, (data['tenant_id'],))
    results = cursor.fetchall()
    print(f"   Found {len(results)} predictions with feedback")
    
    # Query 2: Get model performance metrics
    query2 = """
        SELECT 
            m.model_name,
            m.model_type,
            m.metrics->>'accuracy' as accuracy,
            COUNT(p.id) as prediction_count,
            AVG(p.confidence_score) as avg_confidence
        FROM ml_models m
        LEFT JOIN ml_predictions p ON m.model_id = p.model_id
        WHERE m.tenant_id = %s
        GROUP BY m.model_id, m.model_name, m.model_type, m.metrics;
    """
    
    cursor.execute(query2, (data['tenant_id'],))
    results = cursor.fetchall()
    for result in results:
        print(f"   Model: {result['model_name']} - Accuracy: {result['accuracy']}, Predictions: {result['prediction_count']}")
    
    # Query 3: Check drift alerts
    query3 = """
        SELECT 
            COUNT(*) as total_metrics,
            SUM(CASE WHEN alert_triggered THEN 1 ELSE 0 END) as alerts_triggered,
            AVG(drift_score) as avg_drift_score
        FROM ml_drift_metrics
        WHERE tenant_id = %s
        AND measured_at >= CURRENT_TIMESTAMP - INTERVAL '7 days';
    """
    
    cursor.execute(query3, (data['tenant_id'],))
    result = cursor.fetchone()
    print(f"   Drift metrics (7 days): Total: {result['total_metrics']}, Alerts: {result['alerts_triggered']}")
    
    return True

def main():
    """Main test function"""
    conn = None
    cursor = None
    
    try:
        # Connect to database
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
        conn.autocommit = False  # Use transaction
        cursor = conn.cursor()
        
        print("Connection successful!")
        print("-" * 50)
        
        # Generate test data
        test_data = generate_test_data()
        
        # Run tests
        all_tests_passed = True
        
        tests = [
            test_ml_configurations,
            test_ml_models,
            test_ml_predictions,
            test_ml_training_jobs,
            test_ml_feedback,
            test_ml_feature_store,
            test_ml_drift_metrics,
            test_views,
            test_complex_queries
        ]
        
        for test_func in tests:
            try:
                if test_func == test_views:
                    result = test_func(cursor)
                elif test_func == test_complex_queries:
                    result = test_func(cursor, test_data)
                else:
                    result = test_func(cursor, test_data)
                
                if not result:
                    all_tests_passed = False
            except Exception as e:
                print(f"   ERROR: {e}")
                all_tests_passed = False
        
        # Commit all test data
        conn.commit()
        
        print("\n" + "=" * 50)
        if all_tests_passed:
            print("ALL TESTS PASSED!")
            print("ML tables are working correctly.")
        else:
            print("SOME TESTS FAILED!")
            print("Please check the errors above.")
        
        # Show summary
        print("\nDatabase Summary:")
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM ml_configurations) as configurations,
                (SELECT COUNT(*) FROM ml_models) as models,
                (SELECT COUNT(*) FROM ml_predictions) as predictions,
                (SELECT COUNT(*) FROM ml_training_jobs) as training_jobs,
                (SELECT COUNT(*) FROM ml_feedback) as feedback,
                (SELECT COUNT(*) FROM ml_feature_store) as features,
                (SELECT COUNT(*) FROM ml_drift_metrics) as drift_metrics;
        """)
        
        summary = cursor.fetchone()
        print(f"  Configurations: {summary['configurations']}")
        print(f"  Models: {summary['models']}")
        print(f"  Predictions: {summary['predictions']}")
        print(f"  Training Jobs: {summary['training_jobs']}")
        print(f"  Feedback: {summary['feedback']}")
        print(f"  Feature Sets: {summary['features']}")
        print(f"  Drift Metrics: {summary['drift_metrics']}")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
        
    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    main()