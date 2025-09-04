"""
Test ML Integration Script
Tests that the ML models are working correctly through the simple_ml_service
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_ml_service import simple_ml_service
import json
from datetime import datetime

def test_compliance_prediction():
    """Test compliance prediction"""
    print("\n=== Testing Compliance Prediction ===")
    
    # Test different scenarios
    test_cases = [
        {
            "name": "Compliant Resource",
            "data": {
                "id": "vm-001",
                "type": "VM",
                "encryption_enabled": True,
                "backup_enabled": True,
                "monitoring_enabled": True,
                "public_access": False,
                "tags": {"Environment": "Production", "Owner": "TeamA", "CostCenter": "CC001"},
                "configuration": {"tls_version": "1.3", "firewall": "enabled"},
                "age_days": 30,
                "modifications_last_30_days": 2
            }
        },
        {
            "name": "Non-Compliant Resource",
            "data": {
                "id": "storage-002",
                "type": "Storage",
                "encryption_enabled": False,
                "backup_enabled": False,
                "monitoring_enabled": False,
                "public_access": True,
                "tags": {},
                "configuration": {},
                "age_days": 180,
                "modifications_last_30_days": 0
            }
        },
        {
            "name": "Needs Review Resource",
            "data": {
                "id": "db-003",
                "type": "Database",
                "encryption_enabled": True,
                "backup_enabled": False,
                "monitoring_enabled": True,
                "public_access": False,
                "tags": {"Environment": "Dev"},
                "configuration": {"tls_version": "1.2"},
                "age_days": 60,
                "modifications_last_30_days": 5
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        result = simple_ml_service.predict_compliance(test_case['data'])
        
        print(f"  Resource ID: {result['resource_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Recommendations: {len(result['recommendations'])} items")
        for rec in result['recommendations'][:2]:
            print(f"    - {rec}")

def test_anomaly_detection():
    """Test anomaly detection"""
    print("\n=== Testing Anomaly Detection ===")
    
    # Normal data with some anomalies
    metrics = [
        {"timestamp": "2024-01-01T00:00:00", "value": 50, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T01:00:00", "value": 52, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T02:00:00", "value": 48, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T03:00:00", "value": 250, "resource_count": 10, "alert_count": 5},  # Anomaly
        {"timestamp": "2024-01-01T04:00:00", "value": 51, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T05:00:00", "value": 49, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T06:00:00", "value": -20, "resource_count": 10, "alert_count": 2},  # Anomaly
        {"timestamp": "2024-01-01T07:00:00", "value": 53, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T08:00:00", "value": 47, "resource_count": 10, "alert_count": 0},
        {"timestamp": "2024-01-01T09:00:00", "value": 52, "resource_count": 10, "alert_count": 0},
    ]
    
    result = simple_ml_service.detect_anomalies(metrics)
    
    print(f"\nAnomalies Detected: {result['anomalies_detected']} out of {result['total_points']} points")
    print(f"Anomaly Rate: {result['anomaly_rate']:.2%}")
    print(f"Summary: {result['summary']}")
    
    if result['anomalies']:
        print("\nDetected Anomalies:")
        for anomaly in result['anomalies'][:3]:
            print(f"  - Time: {anomaly['timestamp']}, Value: {anomaly['value']}, Severity: {anomaly['severity']}")

def test_cost_optimization():
    """Test cost optimization"""
    print("\n=== Testing Cost Optimization ===")
    
    test_scenarios = [
        {
            "name": "Underutilized Resources",
            "data": {
                "cpu_utilization": 10,
                "memory_utilization": 15,
                "storage_utilization": 30,
                "network_utilization": 5,
                "monthly_cost": 1000,
                "compute_cost": 600,
                "storage_cost": 300,
                "network_cost": 100,
                "instance_count": 10,
                "average_instance_age_days": 180,
                "reserved_instances": False,
                "spot_instances": False
            }
        },
        {
            "name": "Well-Optimized Resources",
            "data": {
                "cpu_utilization": 75,
                "memory_utilization": 80,
                "storage_utilization": 70,
                "network_utilization": 60,
                "monthly_cost": 800,
                "compute_cost": 500,
                "storage_cost": 200,
                "network_cost": 100,
                "instance_count": 3,
                "average_instance_age_days": 90,
                "reserved_instances": True,
                "spot_instances": False
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        result = simple_ml_service.optimize_costs(scenario['data'])
        
        print(f"  Current Monthly Cost: ${result['current_monthly_cost']:.2f}")
        print(f"  Optimized Cost: ${result['predicted_monthly_cost']:.2f}")
        print(f"  Estimated Savings: ${result['estimated_savings']:.2f} ({result['savings_percentage']:.1f}%)")
        print(f"  Recommendations: {len(result['recommendations'])} items")
        
        for rec in result['recommendations'][:3]:
            print(f"    - {rec['action']}: ${rec['estimated_savings']:.2f} ({rec['priority']} priority)")
            print(f"      {rec['description']}")

def main():
    """Run all tests"""
    print("="*60)
    print("ML Integration Test Suite")
    print("="*60)
    
    try:
        # Check that models are loaded
        print(f"\nModels loaded: {len(simple_ml_service.models)}")
        print(f"Model directory: {simple_ml_service.config.model_dir}")
        print(f"Available models: {', '.join(simple_ml_service.models.keys())}")
        
        # Run tests
        test_compliance_prediction()
        test_anomaly_detection()
        test_cost_optimization()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())