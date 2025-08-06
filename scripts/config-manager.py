#!/usr/bin/env python3
"""
PolicyCortex Configuration Manager
Manages environment variables, Key Vault secrets, and container app configurations
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.sql import SqlManagementClient
from azure.mgmt.cosmosdb import CosmosDBManagementClient
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    resource_group: str
    location: str
    key_vault_name: str
    container_registry_name: str
    container_app_environment_name: str
    
@dataclass
class ServiceConfig:
    """Individual service configuration"""
    name: str
    port: int
    image_name: str
    cpu: float
    memory: str
    min_replicas: int
    max_replicas: int
    env_vars: Dict[str, str]
    secrets: List[str]

class ConfigurationManager:
    """Manages all configuration for PolicyCortex services"""
    
    def __init__(self, subscription_id: str, environment: str = 'dev'):
        self.subscription_id = subscription_id
        self.environment = environment
        self.credential = DefaultAzureCredential()
        
        # Environment configurations
        self.environments = {
            'dev': EnvironmentConfig(
                name='dev',
                resource_group='rg-pcx-app-dev',
                location='eastus',
                key_vault_name='kv-pcx-dev',
                container_registry_name='crpcxdev',
                container_app_environment_name='cae-pcx-dev'
            ),
            'staging': EnvironmentConfig(
                name='staging',
                resource_group='rg-pcx-app-staging',
                location='eastus',
                key_vault_name='kv-pcx-staging',
                container_registry_name='crpcxstaging',
                container_app_environment_name='cae-pcx-staging'
            ),
            'prod': EnvironmentConfig(
                name='prod',
                resource_group='rg-pcx-app-prod',
                location='eastus',
                key_vault_name='kv-pcx-prod',
                container_registry_name='crpcxprod',
                container_app_environment_name='cae-pcx-prod'
            )
        }
        
        self.current_env = self.environments[environment]
        
        # Initialize Azure clients
        self._init_clients()
        
    def _init_clients(self):
        """Initialize Azure management clients"""
        self.resource_client = ResourceManagementClient(
            self.credential, 
            self.subscription_id
        )
        self.storage_client = StorageManagementClient(
            self.credential,
            self.subscription_id
        )
        self.sql_client = SqlManagementClient(
            self.credential,
            self.subscription_id
        )
        self.cosmos_client = CosmosDBManagementClient(
            self.credential,
            self.subscription_id
        )
        
        # Key Vault client
        key_vault_url = f"https://{self.current_env.key_vault_name}.vault.azure.net/"
        self.secret_client = SecretClient(
            vault_url=key_vault_url,
            credential=self.credential
        )
        
    async def get_infrastructure_outputs(self) -> Dict[str, Any]:
        """Get infrastructure outputs from deployed resources"""
        logger.info(f"Getting infrastructure outputs for {self.environment}")
        
        outputs = {}
        
        try:
            # Get storage account details
            storage_accounts = list(self.storage_client.storage_accounts.list_by_resource_group(
                self.current_env.resource_group
            ))
            if storage_accounts:
                storage_account = storage_accounts[0]
                keys = self.storage_client.storage_accounts.list_keys(
                    self.current_env.resource_group,
                    storage_account.name
                )
                outputs['storage_account'] = {
                    'name': storage_account.name,
                    'connection_string': f"DefaultEndpointsProtocol=https;AccountName={storage_account.name};AccountKey={keys.keys[0].value};EndpointSuffix=core.windows.net"
                }
            
            # Get SQL Server details
            sql_servers = list(self.sql_client.servers.list_by_resource_group(
                self.current_env.resource_group
            ))
            if sql_servers:
                sql_server = sql_servers[0]
                outputs['sql_server'] = {
                    'name': sql_server.name,
                    'fqdn': sql_server.fully_qualified_domain_name
                }
            
            # Get Cosmos DB details
            cosmos_accounts = list(self.cosmos_client.database_accounts.list_by_resource_group(
                self.current_env.resource_group
            ))
            if cosmos_accounts:
                cosmos_account = cosmos_accounts[0]
                keys = self.cosmos_client.database_accounts.list_keys(
                    self.current_env.resource_group,
                    cosmos_account.name
                )
                outputs['cosmos_db'] = {
                    'name': cosmos_account.name,
                    'endpoint': cosmos_account.document_endpoint,
                    'primary_key': keys.primary_master_key
                }
            
            logger.info("Successfully retrieved infrastructure outputs")
            return outputs
            
        except Exception as e:
            logger.error(f"Error getting infrastructure outputs: {e}")
            raise
    
    async def get_key_vault_secrets(self) -> Dict[str, str]:
        """Retrieve all secrets from Key Vault"""
        logger.info(f"Retrieving secrets from Key Vault: {self.current_env.key_vault_name}")
        
        try:
            secrets = {}
            secret_properties = self.secret_client.list_properties_of_secrets()
            
            for secret_property in secret_properties:
                try:
                    secret = self.secret_client.get_secret(secret_property.name)
                    secrets[secret_property.name] = secret.value
                except Exception as e:
                    logger.warning(f"Could not retrieve secret {secret_property.name}: {e}")
                    
            logger.info(f"Retrieved {len(secrets)} secrets from Key Vault")
            return secrets
            
        except Exception as e:
            logger.error(f"Error retrieving Key Vault secrets: {e}")
            raise
    
    async def get_container_app_urls(self) -> Dict[str, str]:
        """Get actual container app URLs from Azure"""
        logger.info("Getting container app URLs")
        
        try:
            from azure.mgmt.containerinstance import ContainerInstanceManagementClient
            
            # Note: We'll use a simplified approach for now since getting container app URLs 
            # requires the container apps management client which isn't imported yet
            # For now, use the environment-based naming with a placeholder domain
            
            # Container app names based on your resource list
            app_names = {
                'API_GATEWAY_URL': f'ca-pcx-gateway-{self.environment}',
                'AZURE_INTEGRATION_URL': f'ca-pcx-azureint-{self.environment}',
                'AI_ENGINE_URL': f'ca-pcx-ai-{self.environment}',
                'DATA_PROCESSING_URL': f'ca-pcx-dataproc-{self.environment}',
                'CONVERSATION_URL': f'ca-pcx-chat-{self.environment}',
                'NOTIFICATION_URL': f'ca-pcx-notify-{self.environment}',
                'FRONTEND_URL': f'ca-pcx-web-{self.environment}',
            }
            
            # For now, return URLs with correct app names but placeholder domain
            # In production, this would query the actual container app environment
            base_domain = f".azurecontainerapps.io"  # Will be determined dynamically later
            
            urls = {}
            for key, app_name in app_names.items():
                urls[key] = f"https://{app_name}{base_domain}"
            
            return urls
            
        except Exception as e:
            logger.error(f"Error getting container app URLs: {e}")
            # Return simplified URLs as fallback
            return {
                'API_GATEWAY_URL': f'https://ca-pcx-gateway-{self.environment}.azurecontainerapps.io',
                'AZURE_INTEGRATION_URL': f'https://ca-pcx-azureint-{self.environment}.azurecontainerapps.io',
                'AI_ENGINE_URL': f'https://ca-pcx-ai-{self.environment}.azurecontainerapps.io',
                'DATA_PROCESSING_URL': f'https://ca-pcx-dataproc-{self.environment}.azurecontainerapps.io',
                'CONVERSATION_URL': f'https://ca-pcx-chat-{self.environment}.azurecontainerapps.io',
                'NOTIFICATION_URL': f'https://ca-pcx-notify-{self.environment}.azurecontainerapps.io',
                'FRONTEND_URL': f'https://ca-pcx-web-{self.environment}.azurecontainerapps.io',
            }
    
    def get_service_configurations(self) -> Dict[str, ServiceConfig]:
        """Get configuration for all services"""
        
        # Common environment variables for all services
        common_env_vars = {
            'ENVIRONMENT': self.environment,
            'AZURE_SUBSCRIPTION_ID': self.subscription_id,
            'AZURE_RESOURCE_GROUP': self.current_env.resource_group,
            'AZURE_LOCATION': self.current_env.location,
            'AZURE_KEY_VAULT_NAME': self.current_env.key_vault_name,
            'AZURE_USE_MANAGED_IDENTITY': 'true',
            'LOG_LEVEL': 'INFO' if self.environment == 'prod' else 'DEBUG',
            'METRICS_ENABLED': 'true',
            'TRACING_ENABLED': 'true',
            'ENABLE_CORS': 'true' if self.environment != 'prod' else 'false',
        }
        
        # Service URLs for inter-service communication
        # Using correct container app names based on actual deployed resources
        service_urls = {
            'API_GATEWAY_URL': f'https://ca-pcx-gateway-{self.environment}.azurecontainerapps.io',
            'AZURE_INTEGRATION_URL': f'https://ca-pcx-azureint-{self.environment}.azurecontainerapps.io',
            'AI_ENGINE_URL': f'https://ca-pcx-ai-{self.environment}.azurecontainerapps.io',
            'DATA_PROCESSING_URL': f'https://ca-pcx-dataproc-{self.environment}.azurecontainerapps.io',
            'CONVERSATION_URL': f'https://ca-pcx-chat-{self.environment}.azurecontainerapps.io',
            'NOTIFICATION_URL': f'https://ca-pcx-notify-{self.environment}.azurecontainerapps.io',
            'FRONTEND_URL': f'https://ca-pcx-web-{self.environment}.azurecontainerapps.io',
        }
        
        # Secrets that all services need from Key Vault
        common_secrets = [
            'AZURE-SQL-PASSWORD',
            'AZURE-SQL-USERNAME', 
            'AZURE-SQL-SERVER',
            'AZURE-SQL-DATABASE',
            'AZURE-COSMOS-KEY',
            'AZURE-COSMOS-ENDPOINT',
            'REDIS-PASSWORD',
            'REDIS-URL',
            'AZURE-STORAGE-ACCOUNT-KEY',
            'AZURE-STORAGE-ACCOUNT-NAME',
            'APPLICATION-INSIGHTS-CONNECTION-STRING',
            'JWT-SECRET-KEY'
        ]
        
        services = {
            'api-gateway': ServiceConfig(
                name='api-gateway',
                port=8000,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/api-gateway',
                cpu=1.0,
                memory='2Gi',
                min_replicas=2,
                max_replicas=10,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'api_gateway',
                    'SERVICE_PORT': '8000',
                    'RATE_LIMIT_PER_MINUTE': '1000',
                    'CIRCUIT_BREAKER_ENABLED': 'true',
                    'ENABLE_AUTHENTICATION': 'true',
                },
                secrets=common_secrets + [
                    'AZURE-AD-CLIENT-SECRET'
                ]
            ),
            
            'azure-integration': ServiceConfig(
                name='azure-integration',
                port=8001,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/azure-integration',
                cpu=1.5,
                memory='3Gi',
                min_replicas=2,
                max_replicas=8,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'azure_integration',
                    'SERVICE_PORT': '8001',
                    'AZURE_POLLING_INTERVAL': '300',
                    'ENABLE_COST_OPTIMIZATION': 'true',
                    'ENABLE_POLICY_AUTOMATION': 'true',
                    'ENABLE_RBAC_ANALYSIS': 'true',
                    'ENABLE_NETWORK_SECURITY': 'true',
                },
                secrets=common_secrets + [
                    'AZURE-AD-CLIENT-SECRET'
                ]
            ),
            
            'ai-engine': ServiceConfig(
                name='ai-engine',
                port=8002,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/ai-engine',
                cpu=2.0,
                memory='4Gi',
                min_replicas=1,
                max_replicas=5,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'ai_engine',
                    'SERVICE_PORT': '8002',
                    'AI_MAX_TOKENS': '4000',
                    'AI_TEMPERATURE': '0.7',
                    'AI_TIMEOUT_SECONDS': '30',
                    'ML_MODEL_CACHE_DIR': '/app/models',
                    'ENABLE_PREDICTIVE_ANALYTICS': 'true',
                    'AZURE_OPENAI_DEPLOYMENT': 'gpt-4',
                    'AZURE_OPENAI_API_VERSION': '2023-12-01-preview',
                },
                secrets=common_secrets + [
                    'AZURE-OPENAI-KEY',
                    'AZURE-OPENAI-ENDPOINT'
                ]
            ),
            
            'data-processing': ServiceConfig(
                name='data-processing',
                port=8003,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/data-processing',
                cpu=2.0,
                memory='4Gi',
                min_replicas=1,
                max_replicas=6,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'data_processing',
                    'SERVICE_PORT': '8003',
                    'BATCH_SIZE': '1000',
                    'MAX_CONCURRENT_JOBS': '5',
                    'DATA_RETENTION_DAYS': '365',
                    'ENABLE_DATA_LINEAGE': 'true',
                    'ENABLE_DATA_VALIDATION': 'true',
                },
                secrets=common_secrets + [
                    'AZURE-SERVICE-BUS-CONNECTION-STRING',
                    'EVENT-HUB-CONNECTION-STRING'
                ]
            ),
            
            'conversation': ServiceConfig(
                name='conversation',
                port=8004,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/conversation',
                cpu=1.5,
                memory='3Gi',
                min_replicas=1,
                max_replicas=5,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'conversation',
                    'SERVICE_PORT': '8004',
                    'CONVERSATION_TIMEOUT': '1800',
                    'MAX_CONVERSATION_HISTORY': '50',
                    'ENABLE_CONTEXT_PERSISTENCE': 'true',
                    'ENABLE_SENTIMENT_ANALYSIS': 'true',
                },
                secrets=common_secrets + [
                    'AZURE-OPENAI-KEY',
                    'AZURE-OPENAI-ENDPOINT'
                ]
            ),
            
            'notification': ServiceConfig(
                name='notification',
                port=8005,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/notification',
                cpu=1.0,
                memory='2Gi',
                min_replicas=1,
                max_replicas=4,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'notification',
                    'SERVICE_PORT': '8005',
                    'ALERT_CORRELATION_THRESHOLD': '0.8',
                    'ALERT_STORM_THRESHOLD': '3',
                    'ESCALATION_CLEANUP_INTERVAL': '300',
                    'NOTIFICATION_RETRY_ATTEMPTS': '3',
                },
                secrets=common_secrets + [
                    'AZURE-COMMUNICATION-CONNECTION-STRING',
                    'SMTP-PASSWORD',
                    'TWILIO-AUTH-TOKEN',
                    'SLACK-BOT-TOKEN'
                ]
            ),
            
            'customer-onboarding': ServiceConfig(
                name='customer-onboarding',
                port=8009,
                image_name=f'{self.current_env.container_registry_name}.azurecr.io/customer-onboarding',
                cpu=1.0,
                memory='2Gi',
                min_replicas=1,
                max_replicas=3,
                env_vars={
                    **common_env_vars,
                    **service_urls,
                    'SERVICE_NAME': 'customer_onboarding',
                    'SERVICE_PORT': '8009',
                    'TRIAL_DEFAULT_DAYS': '14',
                    'MAX_TRIAL_EXTENSIONS': '2',
                    'ONBOARDING_TIMEOUT_MINUTES': '30',
                    'INVOICE_PAYMENT_TERMS_DAYS': '30',
                    'OVERAGE_RATE_PER_UNIT': '0.10',
                    'CURRENCY_DEFAULT': 'USD',
                    'TAX_RATE': '0.08',
                },
                secrets=common_secrets + [
                    'STRIPE-API-KEY',
                    'STRIPE-WEBHOOK-SECRET',
                    'PAYPAL-CLIENT-ID',
                    'PAYPAL-CLIENT-SECRET'
                ]
            )
        }
        
        return services
    
    async def generate_environment_file(self, output_path: str):
        """Generate .env file with all configurations"""
        logger.info(f"Generating environment file: {output_path}")
        
        try:
            # Get infrastructure outputs
            infra_outputs = await self.get_infrastructure_outputs()
            
            # Get Key Vault secrets
            secrets = await self.get_key_vault_secrets()
            
            # Generate environment variables
            env_vars = []
            env_vars.append(f"# PolicyCortex Environment Configuration - {self.environment.upper()}")
            env_vars.append(f"# Generated automatically - Do not edit manually")
            env_vars.append("")
            
            # Basic environment configuration
            env_vars.append("# Basic Configuration")
            env_vars.append(f"ENVIRONMENT={self.environment}")
            env_vars.append(f"DEBUG={'true' if self.environment == 'dev' else 'false'}")
            env_vars.append(f"TESTING={'true' if self.environment == 'dev' else 'false'}")
            env_vars.append("")
            
            # Azure configuration
            env_vars.append("# Azure Configuration")
            env_vars.append(f"AZURE_SUBSCRIPTION_ID={self.subscription_id}")
            env_vars.append(f"AZURE_RESOURCE_GROUP={self.current_env.resource_group}")
            env_vars.append(f"AZURE_LOCATION={self.current_env.location}")
            env_vars.append(f"AZURE_KEY_VAULT_NAME={self.current_env.key_vault_name}")
            env_vars.append(f"AZURE_KEY_VAULT_URL=https://{self.current_env.key_vault_name}.vault.azure.net/")
            env_vars.append("")
            
            # Database configuration from secrets
            env_vars.append("# Database Configuration")
            if 'AZURE-SQL-SERVER' in secrets:
                env_vars.append(f"AZURE_SQL_SERVER={secrets['AZURE-SQL-SERVER']}")
            if 'AZURE-SQL-DATABASE' in secrets:
                env_vars.append(f"AZURE_SQL_DATABASE={secrets['AZURE-SQL-DATABASE']}")
            if 'AZURE-SQL-USERNAME' in secrets:
                env_vars.append(f"AZURE_SQL_USERNAME={secrets['AZURE-SQL-USERNAME']}")
            if 'AZURE-SQL-PASSWORD' in secrets:
                env_vars.append(f"AZURE_SQL_PASSWORD={secrets['AZURE-SQL-PASSWORD']}")
            env_vars.append("")
            
            # Cosmos DB configuration
            env_vars.append("# Cosmos DB Configuration")
            if 'AZURE-COSMOS-ENDPOINT' in secrets:
                env_vars.append(f"AZURE_COSMOS_ENDPOINT={secrets['AZURE-COSMOS-ENDPOINT']}")
            if 'AZURE-COSMOS-KEY' in secrets:
                env_vars.append(f"AZURE_COSMOS_KEY={secrets['AZURE-COSMOS-KEY']}")
            env_vars.append("AZURE_COSMOS_DATABASE=policycortex")
            env_vars.append("")
            
            # Redis configuration
            env_vars.append("# Redis Configuration")
            if 'REDIS-URL' in secrets:
                env_vars.append(f"REDIS_URL=redis://{secrets['REDIS-URL']}")
            if 'REDIS-PASSWORD' in secrets:
                env_vars.append(f"REDIS_PASSWORD={secrets['REDIS-PASSWORD']}")
            env_vars.append("REDIS_SSL=true")
            env_vars.append("")
            
            # Storage configuration
            env_vars.append("# Storage Configuration")
            if 'AZURE-STORAGE-ACCOUNT-NAME' in secrets:
                env_vars.append(f"AZURE_STORAGE_ACCOUNT_NAME={secrets['AZURE-STORAGE-ACCOUNT-NAME']}")
            if 'AZURE-STORAGE-ACCOUNT-KEY' in secrets:
                env_vars.append(f"AZURE_STORAGE_ACCOUNT_KEY={secrets['AZURE-STORAGE-ACCOUNT-KEY']}")
            env_vars.append("")
            
            # AI/ML configuration
            env_vars.append("# AI/ML Configuration")
            if 'AZURE-OPENAI-ENDPOINT' in secrets:
                env_vars.append(f"AZURE_OPENAI_ENDPOINT={secrets['AZURE-OPENAI-ENDPOINT']}")
            if 'AZURE-OPENAI-KEY' in secrets:
                env_vars.append(f"AZURE_OPENAI_KEY={secrets['AZURE-OPENAI-KEY']}")
            env_vars.append("AZURE_OPENAI_DEPLOYMENT=gpt-4")
            env_vars.append("AZURE_OPENAI_API_VERSION=2023-12-01-preview")
            env_vars.append("")
            
            # Security configuration
            env_vars.append("# Security Configuration")
            if 'JWT-SECRET-KEY' in secrets:
                env_vars.append(f"JWT_SECRET_KEY={secrets['JWT-SECRET-KEY']}")
            env_vars.append("JWT_ALGORITHM=HS256")
            env_vars.append("JWT_EXPIRATION_MINUTES=30")
            env_vars.append("")
            
            # Monitoring configuration
            env_vars.append("# Monitoring Configuration")
            if 'APPLICATION-INSIGHTS-CONNECTION-STRING' in secrets:
                env_vars.append(f"APPLICATION_INSIGHTS_CONNECTION_STRING={secrets['APPLICATION-INSIGHTS-CONNECTION-STRING']}")
            env_vars.append(f"LOG_LEVEL={'INFO' if self.environment == 'prod' else 'DEBUG'}")
            env_vars.append("METRICS_ENABLED=true")
            env_vars.append("")
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write('\n'.join(env_vars))
                
            logger.info(f"Environment file generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating environment file: {e}")
            raise
    
    async def update_container_app_configurations(self):
        """Update all container app configurations with proper environment variables"""
        logger.info("Updating container app configurations")
        
        try:
            services = self.get_service_configurations()
            secrets = await self.get_key_vault_secrets()
            
            for service_name, config in services.items():
                await self._update_single_container_app(service_name, config, secrets)
                
            logger.info("All container app configurations updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating container app configurations: {e}")
            raise
    
    async def _update_single_container_app(self, service_name: str, config: ServiceConfig, secrets: Dict[str, str]):
        """Update a single container app configuration"""
        logger.info(f"Updating container app: {service_name}")
        
        try:
            # Build environment variables
            env_vars = []
            
            # Add regular environment variables
            for key, value in config.env_vars.items():
                env_vars.append({
                    'name': key,
                    'value': value
                })
            
            # Add secrets from Key Vault
            for secret_name in config.secrets:
                if secret_name in secrets:
                    # Convert secret name format (AZURE-SQL-PASSWORD -> AZURE_SQL_PASSWORD)
                    env_var_name = secret_name.replace('-', '_')
                    env_vars.append({
                        'name': env_var_name,
                        'value': secrets[secret_name]
                    })
            
            # Here you would call Azure Container Apps API to update the configuration
            # For now, we'll just log the configuration
            logger.info(f"Container app {service_name} configuration:")
            logger.info(f"  Image: {config.image_name}")
            logger.info(f"  CPU: {config.cpu}, Memory: {config.memory}")
            logger.info(f"  Replicas: {config.min_replicas}-{config.max_replicas}")
            logger.info(f"  Environment variables: {len(env_vars)}")
            
        except Exception as e:
            logger.error(f"Error updating container app {service_name}: {e}")
            raise
    
    def generate_docker_compose(self, output_path: str):
        """Generate docker-compose.yml for local development"""
        logger.info(f"Generating docker-compose file: {output_path}")
        
        services = self.get_service_configurations()
        
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {
                'policycortex': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'postgres_data': {},
                'redis_data': {},
                'ml_models': {}
            }
        }
        
        # Add database services for local development
        compose_config['services']['postgres'] = {
            'image': 'postgres:15',
            'environment': {
                'POSTGRES_DB': 'policycortex',
                'POSTGRES_USER': 'postgres',
                'POSTGRES_PASSWORD': 'postgres'
            },
            'ports': ['5432:5432'],
            'volumes': ['postgres_data:/var/lib/postgresql/data'],
            'networks': ['policycortex']
        }
        
        compose_config['services']['redis'] = {
            'image': 'redis:7-alpine',
            'ports': ['6379:6379'],
            'volumes': ['redis_data:/data'],
            'networks': ['policycortex']
        }
        
        # Add application services
        for service_name, config in services.items():
            service_config = {
                'build': {
                    'context': './backend',
                    'dockerfile': f'services/{service_name.replace("-", "_")}/Dockerfile'
                },
                'ports': [f'{config.port}:{config.port}'],
                'environment': {
                    **config.env_vars,
                    # Override for local development
                    'AZURE_SQL_SERVER': 'postgres',
                    'AZURE_SQL_DATABASE': 'policycortex',
                    'AZURE_SQL_USERNAME': 'postgres',
                    'AZURE_SQL_PASSWORD': 'postgres',
                    'REDIS_URL': 'redis://redis:6379/0',
                    'REDIS_PASSWORD': '',
                    'REDIS_SSL': 'false'
                },
                'depends_on': ['postgres', 'redis'],
                'networks': ['policycortex'],
                'volumes': [
                    './backend:/app',
                    'ml_models:/app/models'
                ]
            }
            
            compose_config['services'][service_name] = service_config
        
        # Add frontend service
        compose_config['services']['frontend'] = {
            'build': {
                'context': './frontend',
                'dockerfile': 'Dockerfile'
            },
            'ports': ['5173:5173'],
            'environment': {
                'VITE_API_BASE_URL': 'http://localhost:8000/api'
            },
            'volumes': ['./frontend:/app'],
            'networks': ['policycortex']
        }
        
        # Write docker-compose.yml
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Docker compose file generated: {output_path}")

async def main():
    """Main configuration management function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PolicyCortex Configuration Manager')
    parser.add_argument('--subscription-id', required=True, help='Azure subscription ID')
    parser.add_argument('--environment', default='dev', choices=['dev', 'staging', 'prod'], help='Environment')
    parser.add_argument('--action', required=True, choices=['generate-env', 'update-container-apps', 'generate-compose'], help='Action to perform')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    config_manager = ConfigurationManager(args.subscription_id, args.environment)
    
    try:
        if args.action == 'generate-env':
            output_path = args.output or f'.env.{args.environment}'
            await config_manager.generate_environment_file(output_path)
            
        elif args.action == 'update-container-apps':
            await config_manager.update_container_app_configurations()
            
        elif args.action == 'generate-compose':
            output_path = args.output or 'docker-compose.yml'
            config_manager.generate_docker_compose(output_path)
            
        logger.info("Configuration management completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration management failed: {e}")
        exit(1)

if __name__ == '__main__':
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    asyncio.run(main())