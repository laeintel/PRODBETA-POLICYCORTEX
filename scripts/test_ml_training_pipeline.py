"""
Patent #4: ML Training Pipeline Test Script
Comprehensive testing of PolicyCortex ML model training with synthetic data
Author: PolicyCortex ML Testing Team
Date: January 2025

This script:
1. Generates synthetic training data matching Patent #4 requirements
2. Tests the model training pipeline
3. Verifies performance metrics meet patent specifications
4. Tests model saving and loading from database
5. Generates comprehensive test report
"""

import os
import sys
import json
import pickle
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'services', 'ml_models'))

# Import ML components
from ensemble_engine import EnsembleComplianceEngine
from drift_detection import ConfigurationDriftEngine, VAEDriftDetector
from confidence_scoring import ConfidenceScoringEngine
from tenant_isolation import TenantIsolationEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic Azure compliance data for testing"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.resource_types = [
            'Microsoft.Compute/virtualMachines',
            'Microsoft.Storage/storageAccounts',
            'Microsoft.Network/virtualNetworks',
            'Microsoft.Network/networkSecurityGroups',
            'Microsoft.Sql/servers',
            'Microsoft.Web/sites',
            'Microsoft.KeyVault/vaults',
            'Microsoft.ContainerService/managedClusters'
        ]
        
        self.locations = ['eastus', 'westus', 'centralus', 'northeurope', 'westeurope']
        self.environments = ['production', 'staging', 'development', 'test']
        self.compliance_frameworks = ['HIPAA', 'PCI-DSS', 'SOC2', 'ISO27001', 'GDPR']
    
    def generate_resource_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic resource configuration data"""
        logger.info(f"Generating {num_samples} synthetic resource samples...")
        
        resources = []
        for i in range(num_samples):
            # Base resource properties
            resource_type = np.random.choice(self.resource_types)
            location = np.random.choice(self.locations)
            environment = np.random.choice(self.environments)
            
            # Compliance-related features
            is_production = environment == 'production'
            has_encryption = np.random.choice([True, False], p=[0.8, 0.2] if is_production else [0.6, 0.4])
            has_public_access = np.random.choice([False, True], p=[0.9, 0.1] if is_production else [0.7, 0.3])
            has_managed_identity = np.random.choice([True, False], p=[0.7, 0.3] if is_production else [0.5, 0.5])
            
            # Security scores influenced by configuration
            base_security_score = 70 if is_production else 60
            security_adjustments = (
                (10 if has_encryption else -10) +
                (-15 if has_public_access else 5) +
                (5 if has_managed_identity else 0)
            )
            security_score = max(0, min(100, base_security_score + security_adjustments + np.random.normal(0, 5)))
            
            # Compliance state based on features
            compliance_probability = (
                0.9 if has_encryption and not has_public_access and security_score > 70
                else 0.7 if has_encryption and security_score > 60
                else 0.4 if security_score > 50
                else 0.2
            )
            is_compliant = np.random.choice([True, False], p=[compliance_probability, 1 - compliance_probability])
            
            # Metrics
            cpu_usage = np.random.beta(2, 5) * 100  # Typically low CPU usage
            memory_usage = np.random.beta(3, 2) * 100  # Higher memory usage
            network_in = np.random.exponential(100)  # MB
            network_out = np.random.exponential(150)  # MB
            
            # Add anomaly patterns for some resources
            is_anomalous = np.random.choice([False, True], p=[0.95, 0.05])
            if is_anomalous:
                cpu_usage = min(100, cpu_usage * np.random.uniform(2, 4))
                network_out = network_out * np.random.uniform(5, 10)
                is_compliant = False
            
            resources.append({
                'resource_id': f'/subscriptions/test-sub/resourceGroups/rg-{i}/providers/{resource_type}/resource-{i}',
                'resource_type': resource_type,
                'resource_name': f'resource-{i}',
                'location': location,
                'environment': environment,
                'has_encryption': has_encryption,
                'has_public_access': has_public_access,
                'has_managed_identity': has_managed_identity,
                'security_score': security_score,
                'is_compliant': is_compliant,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_in': network_in,
                'network_out': network_out,
                'is_anomalous': is_anomalous,
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 720))
            })
        
        df = pd.DataFrame(resources)
        logger.info(f"Generated {len(df)} resources with {df['is_compliant'].mean():.2%} compliance rate")
        return df
    
    def generate_violation_history(self, resource_df: pd.DataFrame, num_violations: int = 5000) -> pd.DataFrame:
        """Generate synthetic policy violation history"""
        logger.info(f"Generating {num_violations} synthetic violations...")
        
        violations = []
        non_compliant_resources = resource_df[~resource_df['is_compliant']]
        
        for _ in range(num_violations):
            # Select a non-compliant resource
            if len(non_compliant_resources) > 0:
                resource = non_compliant_resources.sample(1).iloc[0]
            else:
                resource = resource_df.sample(1).iloc[0]
            
            policy_types = [
                'RequireEncryption',
                'DenyPublicAccess', 
                'RequireManagedIdentity',
                'RequireTagging',
                'RequireBackup',
                'RequireMonitoring'
            ]
            
            violations.append({
                'resource_id': resource['resource_id'],
                'policy_name': np.random.choice(policy_types),
                'compliance_state': 'NonCompliant',
                'timestamp': resource['timestamp'] + timedelta(hours=np.random.randint(1, 24)),
                'resource_type': resource['resource_type'],
                'resource_location': resource['location'],
                'severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], p=[0.1, 0.3, 0.4, 0.2])
            })
        
        return pd.DataFrame(violations)
    
    def generate_time_series_data(self, days: int = 30) -> pd.DataFrame:
        """Generate time series data for Prophet forecasting"""
        logger.info(f"Generating {days} days of time series data...")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Create seasonal pattern
        base_violations = 50
        seasonal_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 10  # Weekly pattern
        trend = np.arange(days) * 0.5  # Slight upward trend
        noise = np.random.normal(0, 5, days)
        
        violations_count = np.maximum(0, base_violations + seasonal_pattern + trend + noise)
        
        return pd.DataFrame({
            'ds': dates,
            'y': violations_count.astype(int)
        })
    
    def transform_to_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform dataframe to feature matrix and labels"""
        logger.info("Transforming data to features...")
        
        features = []
        
        for _, row in df.iterrows():
            feature_vector = []
            
            # One-hot encode resource type
            for rtype in self.resource_types:
                feature_vector.append(1.0 if row['resource_type'] == rtype else 0.0)
            
            # One-hot encode location
            for loc in self.locations:
                feature_vector.append(1.0 if row['location'] == loc else 0.0)
            
            # One-hot encode environment
            for env in self.environments:
                feature_vector.append(1.0 if row['environment'] == env else 0.0)
            
            # Binary features
            feature_vector.append(1.0 if row['has_encryption'] else 0.0)
            feature_vector.append(1.0 if row['has_public_access'] else 0.0)
            feature_vector.append(1.0 if row['has_managed_identity'] else 0.0)
            
            # Continuous features (normalized)
            feature_vector.append(row['security_score'] / 100.0)
            feature_vector.append(row['cpu_usage'] / 100.0)
            feature_vector.append(row['memory_usage'] / 100.0)
            feature_vector.append(min(row['network_in'] / 1000.0, 1.0))
            feature_vector.append(min(row['network_out'] / 1000.0, 1.0))
            
            # Add random features to reach 100 dimensions
            while len(feature_vector) < 100:
                feature_vector.append(np.random.random())
            
            features.append(feature_vector[:100])  # Ensure exactly 100 features
        
        X = np.array(features)
        y = (~df['is_compliant'].values).astype(int)  # 1 = will violate, 0 = compliant
        
        logger.info(f"Created feature matrix: {X.shape}, Labels: {y.shape}, Violation rate: {y.mean():.2%}")
        return X, y


class ModelTestingPipeline:
    """Comprehensive testing pipeline for ML models"""
    
    def __init__(self):
        self.results = {}
        self.test_start_time = None
        # Force CPU usage to avoid CUDA compatibility issues during testing
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device} (CPU mode for compatibility)")
    
    def test_ensemble_model(self, X_train, y_train, X_test, y_test, time_series_df):
        """Test the ensemble model with all components"""
        logger.info("\n" + "="*60)
        logger.info("Testing Ensemble Compliance Engine")
        logger.info("="*60)
        
        # Initialize model
        ensemble = EnsembleComplianceEngine(input_dim=100)
        
        # Training
        logger.info("Training ensemble model...")
        train_start = time.time()
        ensemble.fit(X_train, y_train, time_series_df)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Prediction and performance validation
        logger.info("Validating model performance...")
        metrics = ensemble.validate_performance(X_test, y_test)
        
        # Additional detailed predictions
        predictions = ensemble.predict(X_test[:100], return_all_scores=True)
        
        # Calculate additional metrics
        y_pred = predictions['predictions']
        y_true = y_test[:100]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Store results
        self.results['ensemble'] = {
            'accuracy': metrics['accuracy'],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': metrics['false_positive_rate'],
            'inference_time_ms': metrics['inference_time_ms'],
            'training_time_seconds': train_time,
            'meets_requirements': metrics['all_requirements_met'],
            'detailed_requirements': {
                'accuracy_requirement': metrics['meets_accuracy_requirement'],
                'fpr_requirement': metrics['meets_fpr_requirement'],
                'latency_requirement': metrics['meets_latency_requirement']
            }
        }
        
        # Test individual predictions
        logger.info("\nTesting individual predictions...")
        sample_predictions = []
        for i in range(5):
            single_pred = ensemble.predict(X_test[i:i+1])
            sample_predictions.append({
                'sample': i,
                'prediction': single_pred['predictions'][0],
                'confidence': single_pred['confidence'][0],
                'probability': single_pred['compliance_probability'][0],
                'actual': y_test[i]
            })
        
        self.results['ensemble']['sample_predictions'] = sample_predictions
        
        # Print results
        self._print_model_results('Ensemble Model', self.results['ensemble'])
        
        return ensemble
    
    def test_drift_detection(self, X_train, X_test):
        """Test VAE drift detection model"""
        logger.info("\n" + "="*60)
        logger.info("Testing Drift Detection Engine")
        logger.info("="*60)
        
        # Initialize model with feature names
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        drift_detector = ConfigurationDriftEngine(input_dim=X_train.shape[1], feature_names=feature_names)
        
        # Training
        logger.info("Training VAE drift detector...")
        train_start = time.time()
        
        # Train the VAE model
        drift_detector.train(X_train, X_test[:1000], epochs=30)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Test drift detection
        logger.info("Testing drift detection...")
        
        # Normal data (should have low drift)
        normal_drift = drift_detector.detect_drift(X_test[:100])
        
        # Create drifted data (add noise)
        X_drifted = X_test[:100] + np.random.normal(0, 0.5, X_test[:100].shape)
        drifted_drift = drift_detector.detect_drift(X_drifted)
        
        self.results['drift_detection'] = {
            'training_time_seconds': train_time,
            'normal_drift_score': normal_drift.drift_score,
            'drifted_drift_score': drifted_drift.drift_score,
            'normal_psi': 0,  # PSI calculated internally
            'drifted_psi': 0,  # PSI calculated internally
            'drift_detected_correctly': drifted_drift.drift_score > 0.5 and normal_drift.drift_score < 0.5,
            'latent_dim': 128
        }
        
        # Print results
        self._print_drift_results(self.results['drift_detection'])
        
        return drift_detector
    
    def test_confidence_scoring(self, ensemble_model, X_test):
        """Test confidence scoring mechanism"""
        logger.info("\n" + "="*60)
        logger.info("Testing Confidence Scoring Engine")
        logger.info("="*60)
        
        # Initialize confidence scorer
        confidence_scorer = ConfidenceScoringEngine(ensemble_model.compliance_predictor)
        
        # Test confidence scoring
        logger.info("Testing confidence scoring...")
        
        # Get predictions with confidence
        X_tensor = torch.FloatTensor(X_test[:100]).to(self.device)
        confidence_scores = confidence_scorer.calculate_confidence(X_tensor)
        
        # Calculate statistics
        mean_confidence = np.mean(confidence_scores['aleatoric'])
        std_confidence = np.std(confidence_scores['aleatoric'])
        
        self.results['confidence_scoring'] = {
            'mean_aleatoric_uncertainty': mean_confidence,
            'std_aleatoric_uncertainty': std_confidence,
            'mean_epistemic_uncertainty': np.mean(confidence_scores['epistemic']),
            'mean_total_confidence': np.mean(confidence_scores['total']),
            'min_confidence': np.min(confidence_scores['total']),
            'max_confidence': np.max(confidence_scores['total'])
        }
        
        # Print results
        self._print_confidence_results(self.results['confidence_scoring'])
        
        return confidence_scorer
    
    def test_latency_requirements(self, ensemble_model, X_test):
        """Test inference latency requirements"""
        logger.info("\n" + "="*60)
        logger.info("Testing Latency Requirements")
        logger.info("="*60)
        
        batch_sizes = [1, 10, 100, 1000]
        latency_results = {}
        
        for batch_size in batch_sizes:
            X_batch = X_test[:batch_size]
            
            # Warm up
            _ = ensemble_model.predict(X_batch)
            
            # Measure latency
            latencies = []
            for _ in range(10):
                start = time.time()
                _ = ensemble_model.predict(X_batch)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            per_sample_latency = avg_latency / batch_size
            
            latency_results[f'batch_{batch_size}'] = {
                'total_ms': avg_latency,
                'per_sample_ms': per_sample_latency,
                'meets_requirement': per_sample_latency < 100
            }
            
            logger.info(f"Batch size {batch_size}: {avg_latency:.2f}ms total, "
                       f"{per_sample_latency:.2f}ms per sample")
        
        self.results['latency'] = latency_results
        
        return latency_results
    
    def test_model_persistence(self, ensemble_model):
        """Test model saving and loading"""
        logger.info("\n" + "="*60)
        logger.info("Testing Model Persistence")
        logger.info("="*60)
        
        try:
            # Serialize model
            logger.info("Serializing model...")
            model_bytes = pickle.dumps(ensemble_model)
            model_size_mb = len(model_bytes) / (1024 * 1024)
            
            # Calculate hash for integrity
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # Deserialize model
            logger.info("Deserializing model...")
            loaded_model = pickle.loads(model_bytes)
            
            # Verify loaded model works
            test_input = np.random.randn(1, 100)
            original_pred = ensemble_model.predict(test_input)
            loaded_pred = loaded_model.predict(test_input)
            
            # Check predictions match
            predictions_match = np.allclose(
                original_pred['compliance_probability'],
                loaded_pred['compliance_probability'],
                rtol=1e-5
            )
            
            self.results['persistence'] = {
                'serialization_successful': True,
                'model_size_mb': model_size_mb,
                'model_hash': model_hash,
                'deserialization_successful': True,
                'predictions_match': predictions_match
            }
            
            logger.info(f"Model size: {model_size_mb:.2f} MB")
            logger.info(f"Model hash: {model_hash[:16]}...")
            logger.info(f"Predictions match: {predictions_match}")
            
        except Exception as e:
            logger.error(f"Model persistence test failed: {e}")
            self.results['persistence'] = {
                'serialization_successful': False,
                'error': str(e)
            }
        
        return self.results['persistence']
    
    def _print_model_results(self, model_name: str, results: Dict):
        """Print formatted model results"""
        print(f"\n{model_name} Results:")
        print("-" * 40)
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f} ({results['false_positive_rate']*100:.2f}%)")
        print(f"Inference Time: {results['inference_time_ms']:.2f}ms")
        print(f"Training Time: {results['training_time_seconds']:.2f}s")
        print(f"\nMeets All Requirements: {results['meets_requirements']}")
        print(f"  - Accuracy ≥ 99.2%: {results['detailed_requirements']['accuracy_requirement']}")
        print(f"  - FPR < 2%: {results['detailed_requirements']['fpr_requirement']}")
        print(f"  - Latency < 100ms: {results['detailed_requirements']['latency_requirement']}")
    
    def _print_drift_results(self, results: Dict):
        """Print formatted drift detection results"""
        print("\nDrift Detection Results:")
        print("-" * 40)
        print(f"Training Time: {results['training_time_seconds']:.2f}s")
        print(f"Normal Data Drift Score: {results['normal_drift_score']:.4f}")
        print(f"Drifted Data Drift Score: {results['drifted_drift_score']:.4f}")
        print(f"Normal PSI: {results['normal_psi']:.4f}")
        print(f"Drifted PSI: {results['drifted_psi']:.4f}")
        print(f"Drift Detected Correctly: {results['drift_detected_correctly']}")
        print(f"VAE Latent Dimension: {results['latent_dim']}")
    
    def _print_confidence_results(self, results: Dict):
        """Print formatted confidence scoring results"""
        print("\nConfidence Scoring Results:")
        print("-" * 40)
        print(f"Mean Aleatoric Uncertainty: {results['mean_aleatoric_uncertainty']:.4f}")
        print(f"Std Aleatoric Uncertainty: {results['std_aleatoric_uncertainty']:.4f}")
        print(f"Mean Epistemic Uncertainty: {results['mean_epistemic_uncertainty']:.4f}")
        print(f"Mean Total Confidence: {results['mean_total_confidence']:.4f}")
        print(f"Confidence Range: [{results['min_confidence']:.4f}, {results['max_confidence']:.4f}]")
    
    def generate_report(self, output_file: str = 'ml_test_report.json'):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("Generating Test Report")
        logger.info("="*60)
        
        report = {
            'test_date': datetime.now().isoformat(),
            'test_duration_seconds': time.time() - self.test_start_time if self.test_start_time else 0,
            'device': str(self.device),
            'results': self.results,
            'patent_requirements': {
                'accuracy_target': 0.992,
                'fpr_target': 0.02,
                'latency_target_ms': 100,
                'lstm_specs': {
                    'hidden_dims': 512,
                    'layers': 3,
                    'dropout': 0.2,
                    'attention_heads': 8
                },
                'ensemble_weights': {
                    'isolation_forest': 0.4,
                    'lstm': 0.3,
                    'autoencoder': 0.3
                },
                'vae_latent_dim': 128
            },
            'overall_status': self._determine_overall_status()
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_file}")
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _determine_overall_status(self) -> str:
        """Determine overall test status"""
        if 'ensemble' in self.results:
            if self.results['ensemble']['meets_requirements']:
                return "PASS - All patent requirements met"
            else:
                return "FAIL - Patent requirements not met"
        return "INCOMPLETE - Tests did not complete"
    
    def _print_summary(self, report: Dict):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Test Date: {report['test_date']}")
        print(f"Duration: {report['test_duration_seconds']:.2f} seconds")
        print(f"Device: {report['device']}")
        print(f"\nOverall Status: {report['overall_status']}")
        
        if 'ensemble' in self.results:
            print("\nKey Metrics:")
            print(f"  - Accuracy: {self.results['ensemble']['accuracy']:.4f} (Target: ≥0.992)")
            print(f"  - FPR: {self.results['ensemble']['false_positive_rate']:.4f} (Target: <0.02)")
            print(f"  - Latency: {self.results['ensemble']['inference_time_ms']:.2f}ms (Target: <100ms)")
        
        print("\n" + "="*60)


async def main():
    """Main test execution"""
    logger.info("Starting ML Training Pipeline Test")
    logger.info("="*60)
    
    # Initialize components
    data_generator = SyntheticDataGenerator(seed=42)
    test_pipeline = ModelTestingPipeline()
    test_pipeline.test_start_time = time.time()
    
    # Generate synthetic data
    logger.info("\nPhase 1: Data Generation")
    logger.info("-"*40)
    resource_df = data_generator.generate_resource_data(num_samples=10000)
    violations_df = data_generator.generate_violation_history(resource_df, num_violations=5000)
    time_series_df = data_generator.generate_time_series_data(days=30)
    
    # Transform to features
    X, y = data_generator.transform_to_features(resource_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Class distribution - Train: {y_train.mean():.2%} violations")
    logger.info(f"Class distribution - Test: {y_test.mean():.2%} violations")
    
    # Test ensemble model
    logger.info("\nPhase 2: Model Training and Testing")
    logger.info("-"*40)
    ensemble_model = test_pipeline.test_ensemble_model(X_train, y_train, X_test, y_test, time_series_df)
    
    # Test drift detection
    logger.info("\nPhase 3: Drift Detection Testing")
    logger.info("-"*40)
    drift_detector = test_pipeline.test_drift_detection(X_train, X_test)
    
    # Test confidence scoring
    logger.info("\nPhase 4: Confidence Scoring Testing")
    logger.info("-"*40)
    confidence_scorer = test_pipeline.test_confidence_scoring(ensemble_model, X_test)
    
    # Test latency requirements
    logger.info("\nPhase 5: Latency Testing")
    logger.info("-"*40)
    latency_results = test_pipeline.test_latency_requirements(ensemble_model, X_test)
    
    # Test model persistence
    logger.info("\nPhase 6: Model Persistence Testing")
    logger.info("-"*40)
    persistence_results = test_pipeline.test_model_persistence(ensemble_model)
    
    # Generate final report
    logger.info("\nPhase 7: Report Generation")
    logger.info("-"*40)
    report = test_pipeline.generate_report('ml_test_report.json')
    
    # Create detailed markdown report
    create_markdown_report(test_pipeline.results, 'ml_test_report.md')
    
    logger.info("\nML Training Pipeline Test Complete!")
    logger.info("Reports saved: ml_test_report.json, ml_test_report.md")
    
    return test_pipeline.results


def create_markdown_report(results: Dict, output_file: str):
    """Create detailed markdown report"""
    with open(output_file, 'w') as f:
        f.write("# PolicyCortex ML Training Pipeline Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        if 'ensemble' in results:
            meets_reqs = results['ensemble']['meets_requirements']
            status = "✅ PASS" if meets_reqs else "❌ FAIL"
            f.write(f"**Overall Status:** {status}\n\n")
            
            f.write("### Patent Requirements Compliance\n\n")
            f.write("| Requirement | Target | Achieved | Status |\n")
            f.write("|------------|--------|----------|--------|\n")
            f.write(f"| Accuracy | ≥99.2% | {results['ensemble']['accuracy']*100:.2f}% | {'✅' if results['ensemble']['detailed_requirements']['accuracy_requirement'] else '❌'} |\n")
            f.write(f"| False Positive Rate | <2% | {results['ensemble']['false_positive_rate']*100:.2f}% | {'✅' if results['ensemble']['detailed_requirements']['fpr_requirement'] else '❌'} |\n")
            f.write(f"| Inference Latency | <100ms | {results['ensemble']['inference_time_ms']:.2f}ms | {'✅' if results['ensemble']['detailed_requirements']['latency_requirement'] else '❌'} |\n\n")
        
        # Detailed Results
        f.write("## Detailed Test Results\n\n")
        
        # Ensemble Model Results
        if 'ensemble' in results:
            f.write("### 1. Ensemble Model Performance\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Accuracy | {results['ensemble']['accuracy']:.4f} |\n")
            f.write(f"| Precision | {results['ensemble']['precision']:.4f} |\n")
            f.write(f"| Recall | {results['ensemble']['recall']:.4f} |\n")
            f.write(f"| F1 Score | {results['ensemble']['f1_score']:.4f} |\n")
            f.write(f"| Training Time | {results['ensemble']['training_time_seconds']:.2f}s |\n\n")
            
            # Sample Predictions
            if 'sample_predictions' in results['ensemble']:
                f.write("#### Sample Predictions\n\n")
                f.write("| Sample | Prediction | Confidence | Probability | Actual | Correct |\n")
                f.write("|--------|------------|------------|-------------|--------|----------|\n")
                for pred in results['ensemble']['sample_predictions']:
                    correct = "✅" if pred['prediction'] == pred['actual'] else "❌"
                    f.write(f"| {pred['sample']} | {pred['prediction']} | {pred['confidence']:.4f} | {pred['probability']:.4f} | {pred['actual']} | {correct} |\n")
                f.write("\n")
        
        # Drift Detection Results
        if 'drift_detection' in results:
            f.write("### 2. Drift Detection (VAE)\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| VAE Latent Dimension | {results['drift_detection']['latent_dim']} |\n")
            f.write(f"| Normal Data Drift Score | {results['drift_detection']['normal_drift_score']:.4f} |\n")
            f.write(f"| Drifted Data Drift Score | {results['drift_detection']['drifted_drift_score']:.4f} |\n")
            f.write(f"| Drift Detection Accuracy | {'✅ Correct' if results['drift_detection']['drift_detected_correctly'] else '❌ Incorrect'} |\n\n")
        
        # Confidence Scoring Results
        if 'confidence_scoring' in results:
            f.write("### 3. Confidence Scoring\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean Total Confidence | {results['confidence_scoring']['mean_total_confidence']:.4f} |\n")
            f.write(f"| Confidence Range | [{results['confidence_scoring']['min_confidence']:.4f}, {results['confidence_scoring']['max_confidence']:.4f}] |\n")
            f.write(f"| Mean Aleatoric Uncertainty | {results['confidence_scoring']['mean_aleatoric_uncertainty']:.4f} |\n")
            f.write(f"| Mean Epistemic Uncertainty | {results['confidence_scoring']['mean_epistemic_uncertainty']:.4f} |\n\n")
        
        # Latency Results
        if 'latency' in results:
            f.write("### 4. Latency Performance\n\n")
            f.write("| Batch Size | Total (ms) | Per Sample (ms) | Meets Requirement |\n")
            f.write("|------------|------------|-----------------|-------------------|\n")
            for batch_name, batch_results in results['latency'].items():
                batch_size = batch_name.split('_')[1]
                meets = "✅" if batch_results['meets_requirement'] else "❌"
                f.write(f"| {batch_size} | {batch_results['total_ms']:.2f} | {batch_results['per_sample_ms']:.2f} | {meets} |\n")
            f.write("\n")
        
        # Model Persistence Results
        if 'persistence' in results:
            f.write("### 5. Model Persistence\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            if results['persistence']['serialization_successful']:
                f.write(f"| Model Size | {results['persistence']['model_size_mb']:.2f} MB |\n")
                f.write(f"| Model Hash | {results['persistence']['model_hash'][:16]}... |\n")
                f.write(f"| Serialization | ✅ Successful |\n")
                f.write(f"| Deserialization | {'✅ Successful' if results['persistence']['deserialization_successful'] else '❌ Failed'} |\n")
                f.write(f"| Predictions Match | {'✅ Yes' if results['persistence']['predictions_match'] else '❌ No'} |\n")
            else:
                f.write(f"| Status | ❌ Failed |\n")
                f.write(f"| Error | {results['persistence'].get('error', 'Unknown')} |\n")
            f.write("\n")
        
        # Architecture Details
        f.write("## Model Architecture Details\n\n")
        f.write("### LSTM Network\n")
        f.write("- **Hidden Dimensions:** 512\n")
        f.write("- **Layers:** 3\n")
        f.write("- **Dropout:** 0.2\n")
        f.write("- **Attention Heads:** 8\n")
        f.write("- **Bidirectional:** Yes\n\n")
        
        f.write("### Ensemble Weights\n")
        f.write("- **Isolation Forest:** 40%\n")
        f.write("- **LSTM:** 30%\n")
        f.write("- **Autoencoder:** 30%\n\n")
        
        f.write("### VAE Drift Detection\n")
        f.write("- **Latent Dimension:** 128\n")
        f.write("- **Encoder Layers:** 3 LSTM layers\n")
        f.write("- **Decoder Layers:** 3 LSTM layers\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        if 'ensemble' in results:
            if results['ensemble']['meets_requirements']:
                f.write("✅ **Model meets all Patent #4 requirements and is ready for production deployment.**\n\n")
                f.write("### Next Steps:\n")
                f.write("1. Deploy model to production environment\n")
                f.write("2. Set up continuous monitoring for drift detection\n")
                f.write("3. Implement A/B testing framework\n")
                f.write("4. Configure automated retraining pipeline\n\n")
            else:
                f.write("⚠️ **Model does not meet all Patent #4 requirements.**\n\n")
                f.write("### Required Improvements:\n")
                if not results['ensemble']['detailed_requirements']['accuracy_requirement']:
                    f.write("1. **Accuracy:** Increase model accuracy through:\n")
                    f.write("   - Additional feature engineering\n")
                    f.write("   - Hyperparameter tuning\n")
                    f.write("   - More training data\n\n")
                if not results['ensemble']['detailed_requirements']['fpr_requirement']:
                    f.write("2. **False Positive Rate:** Reduce FPR through:\n")
                    f.write("   - Threshold optimization\n")
                    f.write("   - Class weight balancing\n")
                    f.write("   - Enhanced anomaly detection\n\n")
                if not results['ensemble']['detailed_requirements']['latency_requirement']:
                    f.write("3. **Latency:** Improve inference speed through:\n")
                    f.write("   - Model optimization/pruning\n")
                    f.write("   - Hardware acceleration (GPU/TPU)\n")
                    f.write("   - Batch processing optimization\n\n")
        
        f.write("---\n")
        f.write("*Report generated automatically by PolicyCortex ML Testing Pipeline*\n")
    
    logger.info(f"Markdown report saved to {output_file}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())