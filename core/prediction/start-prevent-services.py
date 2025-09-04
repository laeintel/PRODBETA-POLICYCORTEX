#!/usr/bin/env python3
"""
PolicyCortex PREVENT Pillar - Service Orchestrator
Starts all prediction services and coordinates between components
"""

import os
import sys
import asyncio
import subprocess
import signal
from pathlib import Path
import logging
from typing import List, Dict, Any
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreventServicesOrchestrator:
    """Orchestrates PREVENT pillar services"""
    
    def __init__(self):
        self.services = []
        self.base_dir = Path(__file__).parent
        self.pids = {}
        
    def start_prediction_engine(self) -> subprocess.Popen:
        """Start the ML prediction engine"""
        logger.info("Starting Prediction Engine...")
        
        cmd = [
            sys.executable,
            str(self.base_dir / "ml-engine" / "predictor.py")
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.base_dir)
        )
        
        self.pids["prediction_engine"] = process.pid
        logger.info(f"Prediction Engine started (PID: {process.pid})")
        return process
    
    def start_autofix_generator(self) -> subprocess.Popen:
        """Start the auto-fix generator"""
        logger.info("Starting Auto-Fix Generator...")
        
        cmd = [
            sys.executable,
            str(self.base_dir / "auto-fixer" / "generator.py")
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.base_dir)
        )
        
        self.pids["autofix_generator"] = process.pid
        logger.info(f"Auto-Fix Generator started (PID: {process.pid})")
        return process
    
    def start_drift_detector(self) -> subprocess.Popen:
        """Start the Rust drift detector"""
        logger.info("Starting Drift Detector...")
        
        # Note: This would start the Rust service in production
        # For now, we'll simulate it
        logger.info("Drift Detector integration pending (Rust service)")
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        import aiohttp
        
        health_status = {
            "prediction_engine": False,
            "autofix_generator": False,
            "drift_detector": False
        }
        
        async with aiohttp.ClientSession() as session:
            # Check prediction engine
            try:
                async with session.get("http://localhost:8001/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        health_status["prediction_engine"] = data.get("status") == "healthy"
            except:
                pass
            
            # Check auto-fix generator
            try:
                async with session.get("http://localhost:8002/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        health_status["autofix_generator"] = data.get("status") == "healthy"
            except:
                pass
        
        return health_status
    
    async def wait_for_services(self, timeout: int = 30):
        """Wait for all services to be healthy"""
        logger.info("Waiting for services to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            health = await self.health_check()
            
            if health["prediction_engine"] and health["autofix_generator"]:
                logger.info("All services are healthy!")
                return True
            
            await asyncio.sleep(2)
        
        logger.warning("Some services failed to start within timeout")
        return False
    
    async def run_demo(self):
        """Run a demo prediction and fix generation"""
        import aiohttp
        
        logger.info("\n" + "="*50)
        logger.info("Running PREVENT Pillar Demo")
        logger.info("="*50 + "\n")
        
        async with aiohttp.ClientSession() as session:
            # 1. Generate predictions
            logger.info("1. Generating 7-day predictions...")
            
            prediction_request = {
                "subscription_ids": ["demo-sub-001", "demo-sub-002"],
                "violation_types": ["data_encryption", "network_security", "cost_overrun"],
                "include_low_confidence": False
            }
            
            try:
                async with session.post(
                    "http://localhost:8001/api/v1/predict/forecast",
                    json=prediction_request
                ) as resp:
                    if resp.status == 200:
                        predictions = await resp.json()
                        logger.info(f"   Generated {predictions['total_predictions']} predictions")
                        logger.info(f"   High risk violations: {predictions['high_risk_count']}")
                        logger.info(f"   Inference time: {predictions['inference_time_ms']}ms")
                        
                        # Show sample forecast cards
                        if predictions['forecast_cards']:
                            card = predictions['forecast_cards'][0]
                            logger.info(f"\n   Sample Forecast Card:")
                            logger.info(f"   - Violation: {card['violation_type']}")
                            logger.info(f"   - Resource: {card['resource_name']}")
                            logger.info(f"   - Probability: {card['probability']*100:.1f}%")
                            logger.info(f"   - ETA: {card['eta_days']} days")
                            logger.info(f"   - Confidence: {card['confidence']}")
            except Exception as e:
                logger.error(f"Failed to generate predictions: {e}")
            
            # 2. Get MTTP metrics
            logger.info("\n2. Fetching Mean Time To Prevention metrics...")
            
            try:
                async with session.get("http://localhost:8001/api/v1/predict/mttp") as resp:
                    if resp.status == 200:
                        mttp = await resp.json()
                        logger.info(f"   MTTP: {mttp['mean_time_to_prevention_hours']} hours")
                        logger.info(f"   Prevented violations (7d): {mttp['prevented_violations_last_7_days']}")
                        logger.info(f"   Success rate: {mttp['prevention_success_rate']*100:.1f}%")
            except Exception as e:
                logger.error(f"Failed to get MTTP metrics: {e}")
            
            # 3. Generate auto-fix
            logger.info("\n3. Generating auto-fix for sample violation...")
            
            fix_request = {
                "violation_type": "data_encryption",
                "resource_id": "/subscriptions/demo-sub-001/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/storage001",
                "resource_type": "Microsoft.Storage/storageAccounts",
                "subscription_id": "demo-sub-001",
                "remediation_type": "terraform",
                "violation_details": {
                    "storage_account_name": "storage001",
                    "resource_group": "rg-prod",
                    "location": "eastus"
                },
                "create_pr": False  # Don't create actual PR in demo
            }
            
            try:
                async with session.post(
                    "http://localhost:8002/api/v1/predict/fix",
                    json=fix_request
                ) as resp:
                    if resp.status == 200:
                        fix = await resp.json()
                        logger.info(f"   Fix generated: {fix['fix_id']}")
                        logger.info(f"   Type: {fix['remediation_type']}")
                        logger.info(f"   File: {fix['file_path']}")
                        logger.info(f"   Estimated time: {fix['estimated_time_minutes']} minutes")
                        logger.info(f"\n   Instructions:")
                        for i, instruction in enumerate(fix['instructions'], 1):
                            logger.info(f"   {instruction}")
            except Exception as e:
                logger.error(f"Failed to generate fix: {e}")
            
            # 4. Show available fixes
            logger.info("\n4. Available fix templates...")
            
            try:
                async with session.get("http://localhost:8002/api/v1/predict/fixes/available") as resp:
                    if resp.status == 200:
                        fixes = await resp.json()
                        logger.info(f"   Total templates: {fixes['total']}")
                        for fix_template in fixes['fixes'][:3]:
                            logger.info(f"   - {fix_template['name']} ({fix_template['violation_type']})")
            except Exception as e:
                logger.error(f"Failed to get available fixes: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("Demo completed!")
        logger.info("="*50 + "\n")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutting down services...")
        
        for name, pid in self.pids.items():
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Stopped {name} (PID: {pid})")
            except:
                pass
        
        sys.exit(0)
    
    async def start(self):
        """Start all PREVENT services"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Starting PolicyCortex PREVENT Pillar Services")
        logger.info("=" * 50)
        
        # Start services
        prediction_process = self.start_prediction_engine()
        autofix_process = self.start_autofix_generator()
        drift_process = self.start_drift_detector()
        
        # Store processes
        if prediction_process:
            self.services.append(prediction_process)
        if autofix_process:
            self.services.append(autofix_process)
        if drift_process:
            self.services.append(drift_process)
        
        # Wait for services to be ready
        await self.wait_for_services()
        
        # Run demo
        await asyncio.sleep(3)
        await self.run_demo()
        
        # Service URLs
        logger.info("\nService Endpoints:")
        logger.info("-" * 30)
        logger.info("Prediction Engine: http://localhost:8001")
        logger.info("  - POST /api/v1/predict/forecast")
        logger.info("  - GET  /api/v1/predict/cards")
        logger.info("  - GET  /api/v1/predict/mttp")
        logger.info("\nAuto-Fix Generator: http://localhost:8002")
        logger.info("  - POST /api/v1/predict/fix")
        logger.info("  - GET  /api/v1/predict/fixes/available")
        logger.info("-" * 30)
        
        logger.info("\nPress Ctrl+C to stop all services")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                # Periodic health check
                health = await self.health_check()
                if not all(health.values()):
                    logger.warning(f"Service health check: {health}")
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    orchestrator = PreventServicesOrchestrator()
    asyncio.run(orchestrator.start())