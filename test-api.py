import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Set required environment variables
os.environ.update({
    'ENVIRONMENT': 'development',
    'SERVICE_NAME': 'api_gateway',
    'JWT_SECRET_KEY': 'test-key',
    'SQL_SERVER': 'localhost',
    'SQL_USERNAME': 'sa',
    'SQL_PASSWORD': 'password',
    'SQL_DATABASE': 'test',
    'COSMOS_ENDPOINT': 'https://localhost:8081/',
    'COSMOS_KEY': 'dummy-key',
    'COSMOS_DATABASE': 'test',
    'REDIS_CONNECTION_STRING': 'localhost:6379',
    'AZURE_CLIENT_ID': 'dummy',
    'AZURE_TENANT_ID': 'dummy',
    'AZURE_CLIENT_SECRET': 'dummy',
    'AZURE_SUBSCRIPTION_ID': 'dummy',
    'AZURE_INTEGRATION_URL': 'http://localhost:8001',
    'AI_ENGINE_URL': 'http://localhost:8002',
    'DATA_PROCESSING_URL': 'http://localhost:8003',
    'CONVERSATION_URL': 'http://localhost:8004',
    'NOTIFICATION_URL': 'http://localhost:8005',
})

print("Testing API Gateway imports...")
try:
    from backend.services.api_gateway.main import app
    print("✓ Successfully imported API Gateway app")
    print(f"  App type: {type(app)}")
    print(f"  App title: {app.title}")
except Exception as e:
    print(f"✗ Failed to import API Gateway: {e}")
    import traceback
    traceback.print_exc()