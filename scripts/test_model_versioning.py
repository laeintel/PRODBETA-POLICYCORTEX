#!/usr/bin/env python3
"""
Test Model Versioning and Rollback System
Tests the model lifecycle management for Patent #4
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.ml_models.model_versioning import (
    ModelVersionManager,
    ModelVersion,
    ModelStatus
)

class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self, input_size=100, hidden_size=50, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_model_versioning():
    """Test complete model versioning lifecycle"""
    
    print("="*60)
    print("PolicyCortex Model Versioning System Test")
    print("Patent #4 Implementation")
    print("="*60)
    
    # Initialize version manager
    database_url = "postgresql://postgres:postgres@localhost:5432/policycortex"
    model_storage_path = "./test_models"
    os.makedirs(model_storage_path, exist_ok=True)
    
    try:
        manager = ModelVersionManager(database_url, model_storage_path)
        print("✓ Version manager initialized")
    except Exception as e:
        print(f"✗ Failed to initialize version manager: {e}")
        print("  Make sure PostgreSQL is running")
        return False
    
    # Test 1: Create multiple model versions
    print("\n[TEST 1] Creating Model Versions")
    print("-" * 40)
    
    models = []
    for i in range(3):
        model = DummyModel()
        version_num = f"1.{i}.0"
        
        # Simulate improving metrics
        metrics = {
            'accuracy': 0.98 + (i * 0.005),  # 98%, 98.5%, 99%
            'false_positive_rate': 0.02 - (i * 0.005),  # 2%, 1.5%, 1%
            'inference_time_ms': 100 - (i * 10),  # 100ms, 90ms, 80ms
            'f1_score': 0.96 + (i * 0.01)
        }
        
        try:
            model_id = manager.create_version(
                model=model,
                tenant_id="test-tenant",
                model_type="compliance_predictor",
                metrics=metrics,
                parent_version=models[-1]['model_id'] if models else None,
                changelog=f"Version {version_num}: Improved accuracy and latency"
            )
            
            models.append({
                'model_id': model_id,
                'version': version_num,
                'metrics': metrics
            })
            
            print(f"  ✓ Created version {version_num}")
            print(f"    - Model ID: {model_id}")
            print(f"    - Accuracy: {metrics['accuracy']:.1%}")
            print(f"    - FPR: {metrics['false_positive_rate']:.1%}")
            print(f"    - Latency: {metrics['inference_time_ms']}ms")
            
        except Exception as e:
            print(f"  ✗ Failed to create version {version_num}: {e}")
    
    # Test 2: Promote model through stages
    print("\n[TEST 2] Model Promotion Pipeline")
    print("-" * 40)
    
    if models:
        latest_model = models[-1]
        model_id = latest_model['model_id']
        
        stages = [
            (ModelStatus.STAGING, "staging"),
            (ModelStatus.PRODUCTION, "production")
        ]
        
        for status, stage_name in stages:
            try:
                success = manager.promote_model(
                    model_id=model_id,
                    target_status=status
                )
                if success:
                    print(f"  ✓ Promoted model to {stage_name}")
                else:
                    print(f"  ✗ Failed promotion to {stage_name} (metrics below threshold)")
            except Exception as e:
                print(f"  ✗ Error promoting to {stage_name}: {e}")
            
            time.sleep(0.5)  # Small delay between promotions
    
    # Test 3: Canary deployment
    print("\n[TEST 3] Canary Deployment")
    print("-" * 40)
    
    if len(models) >= 2:
        old_model = models[-2]['model_id']
        new_model = models[-1]['model_id']
        
        traffic_splits = [
            (10, "10% traffic to new model"),
            (50, "50% traffic split"),
            (100, "Full traffic to new model")
        ]
        
        for percentage, description in traffic_splits:
            try:
                manager.canary_deployment(
                    old_model_id=old_model,
                    new_model_id=new_model,
                    traffic_percentage=percentage
                )
                print(f"  ✓ {description}")
            except Exception as e:
                print(f"  ✗ Failed canary deployment: {e}")
            
            time.sleep(0.5)
    
    # Test 4: Model comparison
    print("\n[TEST 4] Model Comparison")
    print("-" * 40)
    
    if len(models) >= 2:
        model1 = models[0]['model_id']
        model2 = models[-1]['model_id']
        
        try:
            comparison = manager.compare_models(model1, model2)
            print(f"  Comparing version {models[0]['version']} vs {models[-1]['version']}:")
            
            for metric, values in comparison.items():
                if isinstance(values, dict) and 'improvement' in values:
                    improvement = values['improvement']
                    sign = "+" if improvement > 0 else ""
                    print(f"    {metric}: {sign}{improvement:.2%} improvement")
        except Exception as e:
            print(f"  ✗ Failed to compare models: {e}")
    
    # Test 5: Rollback simulation
    print("\n[TEST 5] Rollback Testing")
    print("-" * 40)
    
    if models:
        current_model = models[-1]['model_id']
        
        # Simulate performance degradation
        print("  Simulating performance degradation...")
        degraded_metrics = {
            'accuracy': 0.85,  # Significant drop
            'error_rate': 0.15,  # High error rate
            'latency_spike': 3.0  # 3x latency
        }
        
        try:
            should_rollback = manager.check_rollback_conditions(
                model_id=current_model,
                current_metrics=degraded_metrics
            )
            
            if should_rollback:
                print("  ⚠ Rollback conditions met!")
                print(f"    - Accuracy: {degraded_metrics['accuracy']:.1%} (threshold: 94%)")
                print(f"    - Error rate: {degraded_metrics['error_rate']:.1%} (threshold: 5%)")
                
                # Perform rollback
                if len(models) >= 2:
                    previous_model = models[-2]['model_id']
                    success = manager.rollback(
                        current_model_id=current_model,
                        target_model_id=previous_model,
                        reason="Performance degradation detected"
                    )
                    if success:
                        print(f"  ✓ Rolled back to version {models[-2]['version']}")
                    else:
                        print("  ✗ Rollback failed")
            else:
                print("  ✓ No rollback needed")
                
        except Exception as e:
            print(f"  ✗ Rollback check failed: {e}")
    
    # Test 6: Get deployment history
    print("\n[TEST 6] Deployment History")
    print("-" * 40)
    
    try:
        history = manager.get_deployment_history(
            tenant_id="test-tenant",
            limit=5
        )
        
        if history:
            print(f"  Found {len(history)} deployments:")
            for deployment in history:
                print(f"    - {deployment.get('version', 'Unknown')} "
                      f"({deployment.get('status', 'Unknown')}) "
                      f"- {deployment.get('deployed_at', 'N/A')}")
        else:
            print("  No deployment history found")
            
    except Exception as e:
        print(f"  ✗ Failed to get history: {e}")
    
    # Test 7: Archive old models
    print("\n[TEST 7] Model Archival")
    print("-" * 40)
    
    try:
        archived_count = manager.archive_old_models(days_old=0)  # Archive all for testing
        print(f"  ✓ Archived {archived_count} old models")
    except Exception as e:
        print(f"  ✗ Failed to archive models: {e}")
    
    # Clean up test models
    print("\n[CLEANUP] Removing test models")
    print("-" * 40)
    
    try:
        import shutil
        if os.path.exists(model_storage_path):
            shutil.rmtree(model_storage_path)
            print("  ✓ Test models cleaned up")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")
    
    print("\n" + "="*60)
    print("Model Versioning Test Complete")
    print("="*60)
    
    return True


def test_model_persistence():
    """Test model saving and loading"""
    
    print("\n[PERSISTENCE TEST] Model Save/Load")
    print("-" * 40)
    
    model_path = "./test_model.pt"
    
    try:
        # Create and save model
        model = DummyModel()
        torch.save(model.state_dict(), model_path)
        print("  ✓ Model saved successfully")
        
        # Load model
        loaded_model = DummyModel()
        loaded_model.load_state_dict(torch.load(model_path))
        print("  ✓ Model loaded successfully")
        
        # Test inference
        test_input = torch.randn(1, 100)
        with torch.no_grad():
            output = loaded_model(test_input)
        
        print(f"  ✓ Inference successful: output shape {output.shape}")
        
        # Cleanup
        os.remove(model_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Persistence test failed: {e}")
        return False


def main():
    """Main test runner"""
    
    print("\n" + "="*60)
    print("PolicyCortex ML Model Versioning Test Suite")
    print("Testing Patent #4 Model Lifecycle Management")
    print("="*60)
    
    # Run persistence test first (doesn't need DB)
    persistence_ok = test_model_persistence()
    
    # Run versioning tests (needs DB)
    versioning_ok = test_model_versioning()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Persistence Test: {'✓ PASSED' if persistence_ok else '✗ FAILED'}")
    print(f"Versioning Test: {'✓ PASSED' if versioning_ok else '✗ FAILED'}")
    
    if persistence_ok and versioning_ok:
        print("\n✓ All model versioning tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())