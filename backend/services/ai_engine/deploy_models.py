"""
Model Deployment Script for PolicyCortex
Trains and deploys ML models for production use
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_ml_service import SimpleMlService, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Handles model training and deployment"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the model deployer
        
        Args:
            model_dir: Directory to save trained models
        """
        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(os.path.dirname(__file__), "models_cache")
        
        # Ensure model directory exists
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ML service with auto-train disabled
        config = ModelConfig(
            model_dir=self.model_dir,
            auto_train=False  # We'll train manually
        )
        self.ml_service = SimpleMlService(config)
        
        logger.info(f"Model deployer initialized with model directory: {self.model_dir}")
    
    def train_all_models(self):
        """Train all models"""
        logger.info("Starting training of all models...")
        
        results = {
            "training_started": datetime.utcnow().isoformat(),
            "models": {}
        }
        
        # Train compliance model
        logger.info("Training compliance prediction model...")
        try:
            self.ml_service._train_compliance_model()
            results["models"]["compliance"] = {
                "status": "success",
                "metadata": self.ml_service.model_metadata.get("compliance", {})
            }
            logger.info("[OK] Compliance model trained successfully")
        except Exception as e:
            logger.error(f"[FAIL] Failed to train compliance model: {e}")
            results["models"]["compliance"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model...")
        try:
            self.ml_service._train_anomaly_model()
            results["models"]["anomaly"] = {
                "status": "success",
                "metadata": {}
            }
            logger.info("[OK] Anomaly detection model trained successfully")
        except Exception as e:
            logger.error(f"[FAIL] Failed to train anomaly model: {e}")
            results["models"]["anomaly"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Train cost optimization model
        logger.info("Training cost optimization model...")
        try:
            self.ml_service._train_cost_model()
            results["models"]["cost"] = {
                "status": "success",
                "metadata": {}
            }
            logger.info("[OK] Cost optimization model trained successfully")
        except Exception as e:
            logger.error(f"[FAIL] Failed to train cost model: {e}")
            results["models"]["cost"] = {
                "status": "failed",
                "error": str(e)
            }
        
        results["training_completed"] = datetime.utcnow().isoformat()
        
        # Save training results
        results_file = os.path.join(self.model_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to: {results_file}")
        
        return results
    
    def validate_models(self):
        """Validate that models are working correctly"""
        logger.info("Validating trained models...")
        
        validation_results = {
            "validation_started": datetime.utcnow().isoformat(),
            "models": {}
        }
        
        # Test compliance prediction
        logger.info("Testing compliance prediction...")
        try:
            test_resource = {
                "id": "test-001",
                "type": "VM",
                "encryption_enabled": True,
                "backup_enabled": True,
                "monitoring_enabled": True,
                "public_access": False,
                "tags": {"Environment": "Test"},
                "age_days": 30
            }
            
            result = self.ml_service.predict_compliance(test_resource)
            
            if result and "status" in result:
                validation_results["models"]["compliance"] = {
                    "status": "passed",
                    "test_result": result
                }
                logger.info(f"[OK] Compliance prediction test passed: {result['status']}")
            else:
                validation_results["models"]["compliance"] = {
                    "status": "failed",
                    "error": "Invalid prediction result"
                }
                logger.error("[FAIL] Compliance prediction test failed")
        except Exception as e:
            validation_results["models"]["compliance"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"[FAIL] Compliance prediction test failed: {e}")
        
        # Test anomaly detection
        logger.info("Testing anomaly detection...")
        try:
            test_metrics = [
                {"timestamp": "2024-01-01T00:00:00", "value": 50},
                {"timestamp": "2024-01-01T01:00:00", "value": 55},
                {"timestamp": "2024-01-01T02:00:00", "value": 200},  # Anomaly
                {"timestamp": "2024-01-01T03:00:00", "value": 52},
                {"timestamp": "2024-01-01T04:00:00", "value": 48}
            ]
            
            result = self.ml_service.detect_anomalies(test_metrics)
            
            if result and "anomalies_detected" in result:
                validation_results["models"]["anomaly"] = {
                    "status": "passed",
                    "test_result": {
                        "anomalies_detected": result["anomalies_detected"],
                        "total_points": result["total_points"]
                    }
                }
                logger.info(f"[OK] Anomaly detection test passed: {result['anomalies_detected']} anomalies found")
            else:
                validation_results["models"]["anomaly"] = {
                    "status": "failed",
                    "error": "Invalid detection result"
                }
                logger.error("[FAIL] Anomaly detection test failed")
        except Exception as e:
            validation_results["models"]["anomaly"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"[FAIL] Anomaly detection test failed: {e}")
        
        # Test cost optimization
        logger.info("Testing cost optimization...")
        try:
            test_usage = {
                "cpu_utilization": 15,
                "memory_utilization": 25,
                "storage_utilization": 40,
                "network_utilization": 10,
                "monthly_cost": 1000,
                "instance_count": 5,
                "average_instance_age_days": 120
            }
            
            result = self.ml_service.optimize_costs(test_usage)
            
            if result and "estimated_savings" in result:
                validation_results["models"]["cost"] = {
                    "status": "passed",
                    "test_result": {
                        "estimated_savings": result["estimated_savings"],
                        "recommendations": len(result.get("recommendations", []))
                    }
                }
                logger.info(f"[OK] Cost optimization test passed: ${result['estimated_savings']:.2f} potential savings")
            else:
                validation_results["models"]["cost"] = {
                    "status": "failed",
                    "error": "Invalid optimization result"
                }
                logger.error("[FAIL] Cost optimization test failed")
        except Exception as e:
            validation_results["models"]["cost"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"[FAIL] Cost optimization test failed: {e}")
        
        validation_results["validation_completed"] = datetime.utcnow().isoformat()
        
        # Save validation results
        validation_file = os.path.join(self.model_dir, "validation_results.json")
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to: {validation_file}")
        
        # Check overall status
        all_passed = all(
            model_result.get("status") == "passed"
            for model_result in validation_results["models"].values()
        )
        
        if all_passed:
            logger.info("[OK] All models validated successfully")
        else:
            logger.warning("[WARNING] Some models failed validation")
        
        return validation_results
    
    def get_model_info(self):
        """Get information about deployed models"""
        info = {
            "model_directory": self.model_dir,
            "models": {}
        }
        
        # Check for each model file
        model_files = {
            "compliance": "compliance_model.pkl",
            "anomaly": "anomaly_model.pkl",
            "cost": "cost_model.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                file_stats = os.stat(model_path)
                info["models"][model_name] = {
                    "exists": True,
                    "file_size": file_stats.st_size,
                    "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "path": model_path
                }
                
                # Add metadata if available
                metadata_file = os.path.join(self.model_dir, f"{model_name}_metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        info["models"][model_name]["metadata"] = json.load(f)
            else:
                info["models"][model_name] = {
                    "exists": False,
                    "path": model_path
                }
        
        return info

def main():
    """Main entry point for the deployment script"""
    parser = argparse.ArgumentParser(description="Deploy ML models for PolicyCortex")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory to save trained models",
        default=None
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train all models"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate deployed models"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about deployed models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train and validate all models"
    )
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = ModelDeployer(args.model_dir)
    
    # Execute requested actions
    if args.all or (args.train and args.validate):
        logger.info("=== Training and Validating All Models ===")
        training_results = deployer.train_all_models()
        print("\nTraining Summary:")
        for model, result in training_results["models"].items():
            status = "[OK]" if result["status"] == "success" else "[FAIL]"
            print(f"  {status} {model}: {result['status']}")
        
        print("\n" + "="*50 + "\n")
        
        validation_results = deployer.validate_models()
        print("\nValidation Summary:")
        for model, result in validation_results["models"].items():
            status = "[OK]" if result["status"] == "passed" else "[FAIL]"
            print(f"  {status} {model}: {result['status']}")
    
    elif args.train:
        logger.info("=== Training All Models ===")
        results = deployer.train_all_models()
        print("\nTraining Summary:")
        for model, result in results["models"].items():
            status = "[OK]" if result["status"] == "success" else "[FAIL]"
            print(f"  {status} {model}: {result['status']}")
    
    elif args.validate:
        logger.info("=== Validating Deployed Models ===")
        results = deployer.validate_models()
        print("\nValidation Summary:")
        for model, result in results["models"].items():
            status = "[OK]" if result["status"] == "passed" else "[FAIL]"
            print(f"  {status} {model}: {result['status']}")
    
    elif args.info:
        logger.info("=== Model Deployment Information ===")
        info = deployer.get_model_info()
        print(f"\nModel Directory: {info['model_directory']}")
        print("\nDeployed Models:")
        for model, details in info["models"].items():
            if details["exists"]:
                print(f"  [OK] {model}:")
                print(f"    - File size: {details['file_size']:,} bytes")
                print(f"    - Modified: {details['modified']}")
                if "metadata" in details:
                    metadata = details["metadata"]
                    if "accuracy" in metadata:
                        print(f"    - Accuracy: {metadata['accuracy']:.3f}")
                    if "f1_score" in metadata:
                        print(f"    - F1 Score: {metadata['f1_score']:.3f}")
            else:
                print(f"  [FAIL] {model}: Not deployed")
    
    else:
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python deploy_models.py --all     # Train and validate all models")
        print("  python deploy_models.py --train   # Train models only")
        print("  python deploy_models.py --validate # Validate existing models")
        print("  python deploy_models.py --info    # Show deployment status")

if __name__ == "__main__":
    main()