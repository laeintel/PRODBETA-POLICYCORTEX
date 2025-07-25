#!/usr/bin/env python3
"""
Startup script for Notification service.
This script helps debug container startup issues.
"""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        'ENVIRONMENT',
        'SERVICE_NAME',
        'PORT',
        'JWT_SECRET_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("All required environment variables are set")
    return True

def check_python_path():
    """Check Python path and module availability."""
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Try to import key modules
    try:
        import fastapi
        logger.info(f"FastAPI version: {fastapi.__version__}")
    except ImportError as e:
        logger.error(f"Failed to import FastAPI: {e}")
        return False
    
    try:
        import uvicorn
        logger.info(f"Uvicorn version: {uvicorn.__version__}")
    except ImportError as e:
        logger.error(f"Failed to import Uvicorn: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    logger.info("Starting Notification service...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check Python path and modules
    if not check_python_path():
        sys.exit(1)
    
    logger.info("Startup checks passed, starting application...")
    
    # Import and run the main application
    try:
        from main import app
        import uvicorn
        
        port = int(os.getenv('PORT', 8005))
        host = os.getenv('SERVICE_HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 