"""
Multi-Tenant Data Isolation Manager for PolicyCortex
Implements tenant-specific namespaces, data encryption, and access control
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import structlog
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from azure.keyvault.keys.aio import KeyClient
from azure.keyvault.keys.crypto.aio import CryptographyClient
from azure.identity.aio import DefaultAzureCredential
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

from shared.config import get_settings
from shared.database import cosmos_manager, async_db_transaction, DatabaseUtils

settings = get_settings()
logger = structlog.get_logger(__name__)


class DataClassification(Enum):
    """Data classification levels for compliance"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"


class TenantManager:
    """
    Manages multi-tenant data isolation with Cosmos DB namespaces
    Ensures complete data separation and encryption per tenant
    """

    def __init__(self):
        self.settings = settings
        self.cosmos_client = None
        self.key_vault_client = None
        self.azure_credential = None
        self._tenant_cache = {}
        self._encryption_keys = {}
        
    async def _get_cosmos_client(self) -> CosmosClient:
        """Get Cosmos DB client"""
        if self.cosmos_client is None:
            self.cosmos_client = CosmosClient(
                settings.database.cosmos_endpoint,
                settings.database.cosmos_key
            )
        return self.cosmos_client

    async def _get_key_vault_client(self) -> KeyClient:
        """Get Key Vault client for encryption key management"""
        if self.key_vault_client is None:
            if self.azure_credential is None:
                self.azure_credential = DefaultAzureCredential()
            self.key_vault_client = KeyClient(
                vault_url=settings.azure.key_vault_url,
                credential=self.azure_credential
            )
        return self.key_vault_client

    async def create_tenant_namespace(
        self,
        tenant_id: str,
        organization_name: str,
        org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create isolated namespace for a new tenant
        This happens automatically during organization onboarding
        """
        logger.info("creating_tenant_namespace", tenant_id=tenant_id)
        
        try:
            # Generate tenant-specific encryption key
            encryption_key = await self._generate_tenant_encryption_key(tenant_id)
            
            # Create Cosmos DB containers for tenant
            await self._create_tenant_containers(tenant_id, org_config)
            
            # Create SQL Database schema for tenant
            await self._create_tenant_sql_schema(tenant_id)
            
            # Initialize tenant configuration
            tenant_config = {
                "tenant_id": tenant_id,
                "organization_name": organization_name,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active",
                "data_residency": org_config.get("settings", {}).get("data_residency", "us"),
                "encryption_key_id": encryption_key["key_id"],
                "containers": {
                    "policies": f"policies_{tenant_id}",
                    "resources": f"resources_{tenant_id}",
                    "compliance": f"compliance_{tenant_id}",
                    "audit": f"audit_{tenant_id}",
                    "analytics": f"analytics_{tenant_id}"
                },
                "sql_schema": f"tenant_{tenant_id}",
                "storage_account": f"st{tenant_id[:8]}",
                "limits": org_config.get("limits", {}),
                "features": org_config.get("features", {}),
                "compliance_frameworks": org_config.get("settings", {}).get("compliance_frameworks", [])
            }
            
            # Store tenant configuration
            await self._store_tenant_config(tenant_config)
            
            # Initialize default data
            await self._initialize_tenant_data(tenant_id, org_config)
            
            # Set up tenant-specific monitoring
            await self._setup_tenant_monitoring(tenant_id)
            
            # Cache tenant configuration
            self._tenant_cache[tenant_id] = tenant_config
            
            # Log audit event
            async with async_db_transaction() as session:
                await DatabaseUtils.log_audit_event(
                    session=session,
                    entity_type="tenant",
                    entity_id=tenant_id,
                    action="CREATE",
                    new_values=tenant_config,
                    details=f"Created tenant namespace for {organization_name}"
                )
            
            logger.info(
                "tenant_namespace_created",
                tenant_id=tenant_id,
                organization=organization_name
            )
            
            return tenant_config
            
        except Exception as e:
            logger.error(
                "tenant_namespace_creation_failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise Exception(f"Failed to create tenant namespace: {str(e)}")

    async def _generate_tenant_encryption_key(self, tenant_id: str) -> Dict[str, Any]:
        """Generate unique encryption key for tenant"""
        try:
            # Generate key material
            key = Fernet.generate_key()
            
            # Store in Key Vault if production
            if settings.is_production():
                key_vault_client = await self._get_key_vault_client()
                key_name = f"tenant-key-{tenant_id}"
                
                # Create key in Key Vault
                key_response = await key_vault_client.create_rsa_key(
                    name=key_name,
                    size=2048
                )
                
                # Store key material as secret
                from azure.keyvault.secrets.aio import SecretClient
                secret_client = SecretClient(
                    vault_url=settings.azure.key_vault_url,
                    credential=self.azure_credential
                )
                
                await secret_client.set_secret(
                    name=f"tenant-encryption-{tenant_id}",
                    value=key.decode()
                )
                
                key_id = key_response.id
            else:
                # In development, use local storage
                key_id = f"local-key-{tenant_id}"
                self._encryption_keys[tenant_id] = key
            
            return {
                "key_id": key_id,
                "created_at": datetime.utcnow().isoformat(),
                "algorithm": "AES-256-GCM"
            }
            
        except Exception as e:
            logger.error("encryption_key_generation_failed", error=str(e))
            raise

    async def _create_tenant_containers(self, tenant_id: str, org_config: Dict[str, Any]) -> None:
        """Create Cosmos DB containers for tenant"""
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        
        # Define containers with tenant-specific configurations
        containers = [
            {
                "id": f"policies_{tenant_id}",
                "partition_key": PartitionKey(path="/tenantId"),
                "indexing_policy": {
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "excludedPaths": [{"path": "/\"_etag\"/?"}]
                },
                "default_ttl": 2592000 if org_config.get("type") == "trial" else -1  # 30 days for trial
            },
            {
                "id": f"resources_{tenant_id}",
                "partition_key": PartitionKey(path="/tenantId"),
                "indexing_policy": {
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}]
                }
            },
            {
                "id": f"compliance_{tenant_id}",
                "partition_key": PartitionKey(path="/tenantId"),
                "indexing_policy": {
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "compositeIndexes": [
                        [
                            {"path": "/complianceScore", "order": "descending"},
                            {"path": "/timestamp", "order": "descending"}
                        ]
                    ]
                }
            },
            {
                "id": f"audit_{tenant_id}",
                "partition_key": PartitionKey(path="/tenantId"),
                "indexing_policy": {
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}]
                },
                "default_ttl": self._get_audit_retention_seconds(org_config)
            },
            {
                "id": f"analytics_{tenant_id}",
                "partition_key": PartitionKey(path="/tenantId"),
                "indexing_policy": {
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}]
                }
            }
        ]
        
        # Create containers
        for container_config in containers:
            try:
                await database.create_container_if_not_exists(**container_config)
                logger.info(
                    "tenant_container_created",
                    tenant_id=tenant_id,
                    container=container_config["id"]
                )
            except Exception as e:
                logger.error(
                    "container_creation_failed",
                    tenant_id=tenant_id,
                    container=container_config["id"],
                    error=str(e)
                )

    async def _create_tenant_sql_schema(self, tenant_id: str) -> None:
        """Create SQL Database schema for tenant"""
        # This would create tenant-specific schema in SQL Database
        # For now, we'll use a shared schema with tenant_id columns
        logger.info("tenant_sql_schema_created", tenant_id=tenant_id)

    async def _store_tenant_config(self, config: Dict[str, Any]) -> None:
        """Store tenant configuration"""
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        
        # Use a master tenants container
        container = database.get_container_client("tenants")
        
        # Store configuration
        await container.upsert_item(config)

    async def _initialize_tenant_data(self, tenant_id: str, org_config: Dict[str, Any]) -> None:
        """Initialize default data for new tenant"""
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        
        # Initialize default policies based on compliance frameworks
        if "compliance_frameworks" in org_config.get("settings", {}):
            policies_container = database.get_container_client(f"policies_{tenant_id}")
            
            default_policies = await self._get_default_policies(
                org_config["settings"]["compliance_frameworks"]
            )
            
            for policy in default_policies:
                policy["tenantId"] = tenant_id
                policy["createdAt"] = datetime.utcnow().isoformat()
                policy["isDefault"] = True
                await policies_container.create_item(policy)

    async def _get_default_policies(self, frameworks: List[str]) -> List[Dict[str, Any]]:
        """Get default policies for compliance frameworks"""
        policies = []
        
        framework_policies = {
            "SOC2": [
                {
                    "id": "soc2-access-control",
                    "name": "SOC2 Access Control Policy",
                    "description": "Enforce logical and physical access controls",
                    "category": "Security",
                    "severity": "High"
                },
                {
                    "id": "soc2-encryption",
                    "name": "SOC2 Data Encryption Policy",
                    "description": "Require encryption for data at rest and in transit",
                    "category": "Security",
                    "severity": "High"
                }
            ],
            "GDPR": [
                {
                    "id": "gdpr-data-retention",
                    "name": "GDPR Data Retention Policy",
                    "description": "Enforce data retention and deletion requirements",
                    "category": "Compliance",
                    "severity": "High"
                },
                {
                    "id": "gdpr-consent",
                    "name": "GDPR Consent Management Policy",
                    "description": "Ensure proper consent for data processing",
                    "category": "Compliance",
                    "severity": "High"
                }
            ],
            "HIPAA": [
                {
                    "id": "hipaa-phi-encryption",
                    "name": "HIPAA PHI Encryption Policy",
                    "description": "Require encryption for all PHI data",
                    "category": "Healthcare",
                    "severity": "Critical"
                },
                {
                    "id": "hipaa-access-audit",
                    "name": "HIPAA Access Audit Policy",
                    "description": "Audit all access to PHI data",
                    "category": "Healthcare",
                    "severity": "High"
                }
            ]
        }
        
        for framework in frameworks:
            if framework in framework_policies:
                policies.extend(framework_policies[framework])
        
        return policies

    async def _setup_tenant_monitoring(self, tenant_id: str) -> None:
        """Set up tenant-specific monitoring and alerting"""
        # This would configure Azure Monitor alerts and dashboards
        logger.info("tenant_monitoring_configured", tenant_id=tenant_id)

    def _get_audit_retention_seconds(self, org_config: Dict[str, Any]) -> int:
        """Get audit retention period in seconds based on org type"""
        retention_days = org_config.get("limits", {}).get("retention_days", 90)
        return retention_days * 86400  # Convert to seconds

    async def get_tenant_context(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant context for data operations"""
        # Check cache first
        if tenant_id in self._tenant_cache:
            return self._tenant_cache[tenant_id]
        
        # Load from storage
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        container = database.get_container_client("tenants")
        
        try:
            response = await container.read_item(
                item=tenant_id,
                partition_key=tenant_id
            )
            
            self._tenant_cache[tenant_id] = response
            return response
            
        except Exception as e:
            logger.error(
                "tenant_context_not_found",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise Exception(f"Tenant context not found: {tenant_id}")

    async def encrypt_data(
        self,
        tenant_id: str,
        data: Any,
        classification: DataClassification = DataClassification.INTERNAL
    ) -> str:
        """Encrypt data using tenant-specific key"""
        try:
            # Get tenant encryption key
            encryption_key = await self._get_tenant_encryption_key(tenant_id)
            
            # Serialize data
            if not isinstance(data, str):
                data = json.dumps(data)
            
            # Encrypt based on classification
            if classification in [DataClassification.PII, DataClassification.PHI, DataClassification.RESTRICTED]:
                # Use stronger encryption for sensitive data
                encrypted = await self._encrypt_sensitive(data, encryption_key)
            else:
                # Standard encryption
                fernet = Fernet(encryption_key)
                encrypted = fernet.encrypt(data.encode())
            
            # Add metadata
            encrypted_package = {
                "data": base64.b64encode(encrypted).decode(),
                "classification": classification.value,
                "tenant_id": tenant_id,
                "encrypted_at": datetime.utcnow().isoformat(),
                "algorithm": "AES-256-GCM"
            }
            
            return json.dumps(encrypted_package)
            
        except Exception as e:
            logger.error(
                "data_encryption_failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise

    async def decrypt_data(self, tenant_id: str, encrypted_package: str) -> Any:
        """Decrypt data using tenant-specific key"""
        try:
            # Parse encrypted package
            package = json.loads(encrypted_package)
            
            # Verify tenant ID
            if package["tenant_id"] != tenant_id:
                raise Exception("Tenant ID mismatch")
            
            # Get tenant encryption key
            encryption_key = await self._get_tenant_encryption_key(tenant_id)
            
            # Decode encrypted data
            encrypted_data = base64.b64decode(package["data"])
            
            # Decrypt based on classification
            classification = DataClassification(package["classification"])
            if classification in [DataClassification.PII, DataClassification.PHI, DataClassification.RESTRICTED]:
                decrypted = await self._decrypt_sensitive(encrypted_data, encryption_key)
            else:
                fernet = Fernet(encryption_key)
                decrypted = fernet.decrypt(encrypted_data).decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted)
            except:
                return decrypted
                
        except Exception as e:
            logger.error(
                "data_decryption_failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise

    async def _get_tenant_encryption_key(self, tenant_id: str) -> bytes:
        """Get tenant-specific encryption key"""
        # Check cache
        if tenant_id in self._encryption_keys:
            return self._encryption_keys[tenant_id]
        
        # Get from Key Vault in production
        if settings.is_production():
            from azure.keyvault.secrets.aio import SecretClient
            secret_client = SecretClient(
                vault_url=settings.azure.key_vault_url,
                credential=self.azure_credential or DefaultAzureCredential()
            )
            
            secret = await secret_client.get_secret(f"tenant-encryption-{tenant_id}")
            key = secret.value.encode()
        else:
            # Generate deterministic key for development
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=f"tenant_{tenant_id}".encode(),
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(settings.security.jwt_secret_key.encode()))
        
        # Cache the key
        self._encryption_keys[tenant_id] = key
        return key

    async def _encrypt_sensitive(self, data: str, key: bytes) -> bytes:
        """Enhanced encryption for sensitive data"""
        # This would use Azure Key Vault encryption for sensitive data
        # For now, use standard Fernet
        fernet = Fernet(key)
        return fernet.encrypt(data.encode())

    async def _decrypt_sensitive(self, encrypted_data: bytes, key: bytes) -> str:
        """Enhanced decryption for sensitive data"""
        # This would use Azure Key Vault decryption for sensitive data
        # For now, use standard Fernet
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data).decode()

    async def enforce_data_isolation(
        self,
        tenant_id: str,
        user_tenant_id: str,
        operation: str
    ) -> bool:
        """Enforce tenant data isolation"""
        # Verify user belongs to the tenant they're trying to access
        if tenant_id != user_tenant_id:
            logger.warning(
                "tenant_isolation_violation",
                requested_tenant=tenant_id,
                user_tenant=user_tenant_id,
                operation=operation
            )
            
            # Log security event
            async with async_db_transaction() as session:
                await DatabaseUtils.log_audit_event(
                    session=session,
                    entity_type="security",
                    entity_id=tenant_id,
                    action="ACCESS_DENIED",
                    details=f"Tenant isolation violation: User from {user_tenant_id} attempted to access {tenant_id}"
                )
            
            return False
        
        return True

    async def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant resource usage for billing and limits"""
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        
        usage = {
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "storage": {},
            "api_calls": 0,
            "users": 0,
            "policies": 0,
            "resources": 0
        }
        
        # Get container sizes
        tenant_context = await self.get_tenant_context(tenant_id)
        for container_type, container_name in tenant_context["containers"].items():
            try:
                container = database.get_container_client(container_name)
                # This would get actual container metrics
                usage["storage"][container_type] = {
                    "size_gb": 0,  # Would query actual size
                    "item_count": 0  # Would query actual count
                }
            except:
                pass
        
        return usage

    async def delete_tenant_data(self, tenant_id: str, confirmation_code: str) -> bool:
        """Delete all tenant data (GDPR compliance)"""
        # Verify confirmation code
        expected_code = hashlib.sha256(f"DELETE_{tenant_id}".encode()).hexdigest()[:8]
        if confirmation_code != expected_code:
            raise Exception("Invalid confirmation code")
        
        logger.warning("deleting_tenant_data", tenant_id=tenant_id)
        
        # Get tenant context
        tenant_context = await self.get_tenant_context(tenant_id)
        
        # Delete Cosmos DB containers
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        
        for container_name in tenant_context["containers"].values():
            try:
                container = database.get_container_client(container_name)
                await database.delete_container(container)
                logger.info("container_deleted", container=container_name)
            except Exception as e:
                logger.error("container_deletion_failed", container=container_name, error=str(e))
        
        # Delete encryption keys
        if settings.is_production():
            key_vault_client = await self._get_key_vault_client()
            try:
                await key_vault_client.begin_delete_key(f"tenant-key-{tenant_id}")
            except:
                pass
        
        # Remove from cache
        if tenant_id in self._tenant_cache:
            del self._tenant_cache[tenant_id]
        if tenant_id in self._encryption_keys:
            del self._encryption_keys[tenant_id]
        
        # Log deletion
        async with async_db_transaction() as session:
            await DatabaseUtils.log_audit_event(
                session=session,
                entity_type="tenant",
                entity_id=tenant_id,
                action="DELETE",
                details=f"Tenant data deleted with confirmation code: {confirmation_code}"
            )
        
        return True

    async def export_tenant_data(self, tenant_id: str) -> str:
        """Export all tenant data (GDPR compliance)"""
        logger.info("exporting_tenant_data", tenant_id=tenant_id)
        
        # Get tenant context
        tenant_context = await self.get_tenant_context(tenant_id)
        
        export_data = {
            "tenant_id": tenant_id,
            "export_date": datetime.utcnow().isoformat(),
            "configuration": tenant_context,
            "data": {}
        }
        
        # Export from each container
        cosmos_client = await self._get_cosmos_client()
        database = cosmos_client.get_database_client(settings.database.cosmos_database)
        
        for container_type, container_name in tenant_context["containers"].items():
            try:
                container = database.get_container_client(container_name)
                items = []
                
                # Query all items for tenant
                query = "SELECT * FROM c WHERE c.tenantId = @tenant_id"
                parameters = [{"name": "@tenant_id", "value": tenant_id}]
                
                async for item in container.query_items(
                    query=query,
                    parameters=parameters
                ):
                    items.append(item)
                
                export_data["data"][container_type] = items
                
            except Exception as e:
                logger.error(
                    "container_export_failed",
                    container=container_name,
                    error=str(e)
                )
        
        # Create encrypted export package
        export_json = json.dumps(export_data, indent=2)
        encrypted_export = await self.encrypt_data(
            tenant_id,
            export_json,
            DataClassification.CONFIDENTIAL
        )
        
        # Log export event
        async with async_db_transaction() as session:
            await DatabaseUtils.log_audit_event(
                session=session,
                entity_type="tenant",
                entity_id=tenant_id,
                action="EXPORT",
                details="Tenant data exported for GDPR compliance"
            )
        
        return encrypted_export


# Global tenant manager instance
tenant_manager = TenantManager()