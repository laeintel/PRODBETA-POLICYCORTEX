#!/usr/bin/env python
"""
Start the API Gateway server for local development
"""
import os
import sys
import uvicorn

# Set environment variables
os.environ['SERVICE_PORT'] = '8010'
os.environ['ENVIRONMENT'] = 'development'
os.environ['SERVICE_NAME'] = 'api_gateway'
os.environ['JWT_SECRET_KEY'] = 'dev-secret-key-change-in-production'
os.environ['LOG_LEVEL'] = 'debug'
os.environ['AZURE_SUBSCRIPTION_ID'] = '205b477d-17e7-4b3b-92c1-32cf02626b78'

print("Starting API Gateway on port 8010...")
print(f"Environment: {os.environ.get('ENVIRONMENT')}")
print(f"Service: {os.environ.get('SERVICE_NAME')}")
print(f"Log Level: {os.environ.get('LOG_LEVEL')}")
print()

if __name__ == "__main__":
    # Run the simple development server
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="debug"
    )