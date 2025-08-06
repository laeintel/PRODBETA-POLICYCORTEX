"""
Shared configuration module for PolicyCortex microservices.
Provides environment-based configuration management using Pydantic Settings.
"""

import os
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import Field
from pydantic import validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    # SQL Database (Azure SQL Database)
    sql_server: str = Field(default="localhost", env="AZURE_SQL_SERVER")
    sql_database: str = Field("policycortex", env="AZURE_SQL_DATABASE")
    sql_username: str = Field(default="test", env="AZURE_SQL_USERNAME")
    sql_password: str = Field(default="test", env="AZURE_SQL_PASSWORD")
    sql_driver: str = Field("ODBC Driver 18 for SQL Server", env="AZURE_SQL_DRIVER")
    sql_port: int = Field(1433, env="AZURE_SQL_PORT")
    sql_pool_size: int = Field(20, env="SQL_POOL_SIZE")
    sql_max_overflow: int = Field(30, env="SQL_MAX_OVERFLOW")

    # Cosmos DB (NoSQL)
    cosmos_endpoint: str = Field(default="https://test.documents.azure.com:443/", env="AZURE_COSMOS_ENDPOINT")
    cosmos_key: str = Field(default="test-cosmos-key", env="AZURE_COSMOS_KEY")
    cosmos_database: str = Field("policycortex", env="AZURE_COSMOS_DATABASE")

    # Redis Cache
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(False, env="REDIS_SSL")
    redis_pool_size: int = Field(20, env="REDIS_POOL_SIZE")

    @property
    def sql_connection_string(self) -> str:
        """Generate SQL Server connection string."""
        return (
            f"mssql+pyodbc://{self.sql_username}:{self.sql_password}@"
            f"{self.sql_server}:{self.sql_port}/{self.sql_database}"
            f"?driver={self.sql_driver.replace(' ', '+')}&TrustServerCertificate=yes"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False


class AzureConfig(BaseSettings):
    """Azure services configuration."""

    # Core Azure Configuration
    subscription_id: str = Field(default="test-subscription-id", env="AZURE_SUBSCRIPTION_ID")
    tenant_id: str = Field(default="test-tenant-id", env="AZURE_TENANT_ID")
    client_id: str = Field(default="test-client-id", env="AZURE_CLIENT_ID")
    client_secret: str = Field(default="test-client-secret", env="AZURE_CLIENT_SECRET")
    resource_group: str = Field(default="rg-test", env="AZURE_RESOURCE_GROUP")
    location: str = Field("eastus", env="AZURE_LOCATION")

    # Key Vault
    key_vault_name: str = Field(default="kv-test", env="AZURE_KEY_VAULT_NAME")
    key_vault_url: Optional[str] = Field(None, env="AZURE_KEY_VAULT_URL")

    # Storage Account
    storage_account_name: str = Field(default="teststore", env="AZURE_STORAGE_ACCOUNT_NAME")
    storage_account_key: Optional[str] = Field(None, env="AZURE_STORAGE_ACCOUNT_KEY")

    # Service Bus
    service_bus_namespace: str = Field(default="test-servicebus", env="AZURE_SERVICE_BUS_NAMESPACE")
    service_bus_connection_string: Optional[str] = Field(
        None, env="AZURE_SERVICE_BUS_CONNECTION_STRING"
    )

    # Application Insights
    application_insights_key: Optional[str] = Field(None, env="APPLICATION_INSIGHTS_KEY")
    application_insights_connection_string: Optional[str] = Field(
        None, env="APPLICATION_INSIGHTS_CONNECTION_STRING"
    )

    # Machine Learning
    ml_workspace_name: str = Field(default="test-ml-workspace", env="AZURE_ML_WORKSPACE_NAME")
    ml_resource_group: Optional[str] = Field(None, env="AZURE_ML_RESOURCE_GROUP")

    @validator("key_vault_url", always=True)
    def generate_key_vault_url(cls, v, values):
        """Generate Key Vault URL if not provided."""
        if v is None and "key_vault_name" in values:
            return f"https://{values['key_vault_name']}.vault.azure.net/"
        return v

    @validator("ml_resource_group", always=True)
    def default_ml_resource_group(cls, v, values):
        """Use main resource group if ML resource group not specified."""
        return v or values.get("resource_group")

    class Config:
        env_file = ".env"
        case_sensitive = False


class AIConfig(BaseSettings):
    """AI/ML configuration."""

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(None, env="OPENAI_API_BASE")
    openai_api_version: str = Field("2023-12-01-preview", env="OPENAI_API_VERSION")

    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: Optional[str] = Field(None, env="AZURE_OPENAI_KEY")
    azure_openai_deployment: str = Field("gpt-4", env="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field("2023-12-01-preview", env="AZURE_OPENAI_API_VERSION")

    # Model Configuration
    max_tokens: int = Field(4000, env="AI_MAX_TOKENS")
    temperature: float = Field(0.7, env="AI_TEMPERATURE")
    timeout_seconds: int = Field(30, env="AI_TIMEOUT_SECONDS")

    # ML Model Paths
    model_registry_uri: Optional[str] = Field(None, env="ML_MODEL_REGISTRY_URI")
    model_cache_dir: str = Field("./models", env="ML_MODEL_CACHE_DIR")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",),
    }


class SecurityConfig(BaseSettings):
    """Security and authentication configuration."""

    # JWT Configuration
    jwt_secret_key: str = Field(default="test-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(30, env="JWT_EXPIRATION_MINUTES")
    jwt_refresh_expiration_days: int = Field(7, env="JWT_REFRESH_EXPIRATION_DAYS")

    # API Keys and Secrets
    api_key_header: str = Field("X-API-Key", env="API_KEY_HEADER")
    rate_limit_per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")

    # CORS Configuration
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(["*"], env="CORS_HEADERS")

    # Azure AD Configuration
    azure_ad_tenant_id: Optional[str] = Field(None, env="AZURE_AD_TENANT_ID")
    azure_ad_client_id: Optional[str] = Field(None, env="AZURE_AD_CLIENT_ID")
    azure_ad_client_secret: Optional[str] = Field(None, env="AZURE_AD_CLIENT_SECRET")

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


class ServiceConfig(BaseSettings):
    """Individual service configuration."""

    # Service Identity
    service_name: str = Field(default="test-service", env="SERVICE_NAME")
    service_version: str = Field("1.0.0", env="SERVICE_VERSION")
    service_port: int = Field(8000, env="SERVICE_PORT")
    service_host: str = Field("0.0.0.0", env="SERVICE_HOST")

    # Worker Configuration
    workers: int = Field(4, env="WORKERS")
    worker_timeout: int = Field(30, env="WORKER_TIMEOUT")
    max_requests: int = Field(1000, env="MAX_REQUESTS")
    max_requests_jitter: int = Field(100, env="MAX_REQUESTS_JITTER")

    # Health Check Configuration
    health_check_path: str = Field("/health", env="HEALTH_CHECK_PATH")
    readiness_check_path: str = Field("/ready", env="READINESS_CHECK_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    # Logging Configuration
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(None, env="LOG_FILE")

    # Metrics Configuration
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    metrics_port: int = Field(9090, env="METRICS_PORT")
    metrics_path: str = Field("/metrics", env="METRICS_PATH")

    # Tracing Configuration
    tracing_enabled: bool = Field(True, env="TRACING_ENABLED")
    tracing_sample_rate: float = Field(0.1, env="TRACING_SAMPLE_RATE")
    jaeger_endpoint: Optional[str] = Field(None, env="JAEGER_ENDPOINT")

    # Performance Monitoring
    slow_query_threshold: float = Field(1.0, env="SLOW_QUERY_THRESHOLD")
    enable_sql_tracing: bool = Field(True, env="ENABLE_SQL_TRACING")

    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")

    # Component Configurations
    database: DatabaseConfig = DatabaseConfig()
    azure: AzureConfig = AzureConfig()
    ai: AIConfig = AIConfig()
    security: SecurityConfig = SecurityConfig()
    service: ServiceConfig = ServiceConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    # Feature Flags
    enable_cost_optimization: bool = Field(True, env="ENABLE_COST_OPTIMIZATION")
    enable_policy_automation: bool = Field(True, env="ENABLE_POLICY_AUTOMATION")
    enable_rbac_analysis: bool = Field(True, env="ENABLE_RBAC_ANALYSIS")
    enable_network_security: bool = Field(True, env="ENABLE_NETWORK_SECURITY")
    enable_predictive_analytics: bool = Field(True, env="ENABLE_PREDICTIVE_ANALYTICS")

    # External Service URLs
    api_gateway_url: str = Field("http://localhost:8000", env="API_GATEWAY_URL")
    azure_integration_url: str = Field("http://localhost:8001", env="AZURE_INTEGRATION_URL")
    ai_engine_url: str = Field("http://localhost:8002", env="AI_ENGINE_URL")
    data_processing_url: str = Field("http://localhost:8003", env="DATA_PROCESSING_URL")
    conversation_url: str = Field("http://localhost:8004", env="CONVERSATION_URL")
    notification_url: str = Field("http://localhost:8005", env="NOTIFICATION_URL")

    @validator("debug", always=True)
    def set_debug_from_environment(cls, v, values):
        """Set debug mode based on environment."""
        if values.get("environment") == Environment.DEVELOPMENT:
            return True
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING or self.testing


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
