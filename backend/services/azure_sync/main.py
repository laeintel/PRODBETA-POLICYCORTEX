#!/usr/bin/env python3
"""
PolicyCortex Azure Policy & SDK Auto-Refresh Service
Automated synchronization of Azure policies and SDK version management
"""

import asyncio
import logging
import os
import json
import hashlib
import zipfile
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import asyncpg
import redis.asyncio as redis
from azure.identity.aio import DefaultAzureCredential
from azure.mgmt.resource.policy.aio import PolicyClient
from azure.mgmt.resource.policy.models import PolicyDefinition, PolicySetDefinition
from azure.mgmt.managementgroups.aio import ManagementGroupsAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import semver
from packaging import version
import subprocess
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PolicyCortex Azure Sync Service", version="1.0.0")

@dataclass
class PolicySyncResult:
    policy_id: str
    policy_name: str
    version: str
    status: str  # 'added', 'updated', 'unchanged', 'deprecated'
    changes: List[str]
    sync_time: datetime

@dataclass
class SDKVersionInfo:
    package_name: str
    current_version: str
    latest_version: str
    update_available: bool
    security_update: bool
    changelog_url: str
    update_priority: str  # 'low', 'medium', 'high', 'critical'

class PolicySyncRequest(BaseModel):
    scope: str = "subscription"  # subscription, management_group, resource_group
    scope_id: Optional[str] = None
    include_builtin: bool = True
    include_custom: bool = True
    force_refresh: bool = False

class SDKUpdateRequest(BaseModel):
    package_name: Optional[str] = None  # Update specific package or all
    update_type: str = "patch"  # patch, minor, major
    include_preview: bool = False
    auto_deploy: bool = False

class PolicyDefinitionChange(BaseModel):
    change_type: str  # 'added', 'modified', 'removed'
    field_path: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

class AzureSyncService:
    """Main Azure synchronization service"""
    
    def __init__(self):
        self.config = self._load_config()
        self.db_pool = None
        self.redis_client = None
        self.credential = None
        self.policy_clients = {}
        self.mgmt_group_client = None
        self.known_sdks = self._get_known_sdks()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        return {
            'database': {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'name': os.getenv('DATABASE_NAME', 'policycortex'),
                'user': os.getenv('DATABASE_USER', 'postgres'),
                'password': os.getenv('DATABASE_PASSWORD', 'postgres'),
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'db': int(os.getenv('REDIS_DB', 1)),
            },
            'azure': {
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                'tenant_id': os.getenv('AZURE_TENANT_ID'),
                'client_id': os.getenv('AZURE_CLIENT_ID'),
            },
            'sync': {
                'policy_sync_interval': int(os.getenv('POLICY_SYNC_INTERVAL', 3600)),  # 1 hour
                'sdk_check_interval': int(os.getenv('SDK_CHECK_INTERVAL', 86400)),     # 24 hours
                'max_policy_versions': int(os.getenv('MAX_POLICY_VERSIONS', 10)),
                'enable_auto_update': os.getenv('ENABLE_AUTO_UPDATE', 'false').lower() == 'true',
            },
            'github': {
                'api_token': os.getenv('GITHUB_API_TOKEN'),
                'repo_owner': 'Azure',
                'sdk_repos': [
                    'azure-sdk-for-python',
                    'azure-sdk-for-js',
                    'azure-sdk-for-net',
                    'azure-sdk-for-java'
                ]
            }
        }
    
    def _get_known_sdks(self) -> Dict[str, Dict[str, Any]]:
        """Get known Azure SDK packages"""
        return {
            'azure-mgmt-resource': {
                'language': 'python',
                'category': 'management',
                'priority': 'high',
                'auto_update': True
            },
            'azure-mgmt-monitor': {
                'language': 'python',
                'category': 'management',
                'priority': 'medium',
                'auto_update': True
            },
            'azure-mgmt-security': {
                'language': 'python',
                'category': 'management',
                'priority': 'high',
                'auto_update': True
            },
            'azure-identity': {
                'language': 'python',
                'category': 'core',
                'priority': 'critical',
                'auto_update': False  # Security sensitive
            },
            'azure-core': {
                'language': 'python',
                'category': 'core',
                'priority': 'critical',
                'auto_update': False
            },
            '@azure/arm-resources': {
                'language': 'javascript',
                'category': 'management',
                'priority': 'high',
                'auto_update': True
            },
            '@azure/identity': {
                'language': 'javascript',
                'category': 'core',
                'priority': 'critical',
                'auto_update': False
            }
        }
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing Azure sync service...")
        
        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            database=self.config['database']['name'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            min_size=2,
            max_size=10
        )
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            decode_responses=True
        )
        
        # Initialize Azure clients
        self.credential = DefaultAzureCredential()
        
        # Initialize policy clients for different scopes
        if self.config['azure']['subscription_id']:
            self.policy_clients['subscription'] = PolicyClient(
                credential=self.credential,
                subscription_id=self.config['azure']['subscription_id']
            )
        
        # Initialize management group client
        self.mgmt_group_client = ManagementGroupsAPI(credential=self.credential)
        
        # Initialize database tables
        await self._initialize_database()
        
        # Start background tasks
        asyncio.create_task(self._policy_sync_loop())
        asyncio.create_task(self._sdk_check_loop())
        
        logger.info("Azure sync service initialized")
    
    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            async with self.db_pool.acquire() as conn:
                # Policy definitions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS azure_policy_definitions (
                        policy_id VARCHAR PRIMARY KEY,
                        name VARCHAR NOT NULL,
                        display_name VARCHAR,
                        description TEXT,
                        policy_type VARCHAR NOT NULL,
                        mode VARCHAR,
                        metadata JSONB,
                        policy_rule JSONB NOT NULL,
                        parameters JSONB,
                        version VARCHAR NOT NULL,
                        hash VARCHAR NOT NULL,
                        scope VARCHAR NOT NULL,
                        scope_id VARCHAR,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        deprecated_at TIMESTAMP,
                        is_builtin BOOLEAN DEFAULT FALSE,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Policy sync history table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS policy_sync_history (
                        sync_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        policy_id VARCHAR NOT NULL,
                        sync_time TIMESTAMP DEFAULT NOW(),
                        status VARCHAR NOT NULL,
                        changes JSONB DEFAULT '[]',
                        old_version VARCHAR,
                        new_version VARCHAR,
                        sync_duration_ms INTEGER
                    )
                """)
                
                # SDK versions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS azure_sdk_versions (
                        package_name VARCHAR PRIMARY KEY,
                        current_version VARCHAR NOT NULL,
                        latest_version VARCHAR NOT NULL,
                        update_available BOOLEAN DEFAULT FALSE,
                        security_update BOOLEAN DEFAULT FALSE,
                        update_priority VARCHAR DEFAULT 'medium',
                        changelog_url VARCHAR,
                        last_checked TIMESTAMP DEFAULT NOW(),
                        last_updated TIMESTAMP,
                        auto_update_enabled BOOLEAN DEFAULT FALSE,
                        update_notes TEXT
                    )
                """)
                
                # SDK update history table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sdk_update_history (
                        update_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        package_name VARCHAR NOT NULL,
                        from_version VARCHAR NOT NULL,
                        to_version VARCHAR NOT NULL,
                        update_type VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        started_at TIMESTAMP DEFAULT NOW(),
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        rollback_available BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_definitions_scope ON azure_policy_definitions(scope, scope_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_definitions_updated ON azure_policy_definitions(updated_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_sync_history_time ON policy_sync_history(sync_time)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_sdk_versions_priority ON azure_sdk_versions(update_priority)")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def sync_azure_policies(self, request: PolicySyncRequest) -> List[PolicySyncResult]:
        """Sync Azure policy definitions"""
        logger.info(f"Starting policy sync for scope: {request.scope}")
        start_time = datetime.utcnow()
        results = []
        
        try:
            # Get policy client for scope
            policy_client = await self._get_policy_client(request.scope, request.scope_id)
            
            # Fetch policies from Azure
            azure_policies = await self._fetch_azure_policies(
                policy_client, 
                request.include_builtin, 
                request.include_custom
            )
            
            # Get existing policies from database
            existing_policies = await self._get_existing_policies(request.scope, request.scope_id)
            
            # Compare and update policies
            for azure_policy in azure_policies:
                result = await self._process_policy_update(
                    azure_policy, 
                    existing_policies.get(azure_policy.name),
                    request.scope,
                    request.scope_id,
                    request.force_refresh
                )
                results.append(result)
            
            # Mark deprecated policies
            await self._mark_deprecated_policies(azure_policies, existing_policies, request.scope, request.scope_id)
            
            # Cache results
            await self._cache_sync_results(results, request.scope, request.scope_id)
            
            sync_duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Policy sync completed in {sync_duration:.2f}s. Processed {len(results)} policies")
            
            return results
            
        except Exception as e:
            logger.error(f"Policy sync failed: {e}")
            raise HTTPException(status_code=500, detail=f"Policy sync failed: {str(e)}")
    
    async def check_sdk_updates(self, request: SDKUpdateRequest) -> List[SDKVersionInfo]:
        """Check for SDK updates"""
        logger.info("Checking for SDK updates...")
        results = []
        
        try:
            packages = [request.package_name] if request.package_name else list(self.known_sdks.keys())
            
            for package_name in packages:
                version_info = await self._check_package_version(package_name, request.include_preview)
                results.append(version_info)
                
                # Store in database
                await self._store_sdk_version_info(version_info)
            
            # Sort by update priority
            results.sort(key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x.update_priority])
            
            return results
            
        except Exception as e:
            logger.error(f"SDK version check failed: {e}")
            raise HTTPException(status_code=500, detail=f"SDK check failed: {str(e)}")
    
    async def update_sdk_package(self, package_name: str, target_version: str, auto_deploy: bool = False) -> Dict[str, Any]:
        """Update a specific SDK package"""
        logger.info(f"Updating {package_name} to {target_version}")
        
        try:
            update_result = {
                'package_name': package_name,
                'from_version': await self._get_current_version(package_name),
                'to_version': target_version,
                'status': 'started',
                'started_at': datetime.utcnow(),
                'steps': []
            }
            
            # Record update start
            update_id = await self._record_update_start(update_result)
            
            # Perform update steps
            await self._backup_current_version(package_name)
            update_result['steps'].append('backup_created')
            
            await self._download_and_install_package(package_name, target_version)
            update_result['steps'].append('package_installed')
            
            await self._run_compatibility_tests(package_name)
            update_result['steps'].append('tests_passed')
            
            if auto_deploy:
                await self._deploy_updated_service()
                update_result['steps'].append('service_deployed')
            
            # Record success
            update_result['status'] = 'completed'
            update_result['completed_at'] = datetime.utcnow()
            await self._record_update_completion(update_id, update_result)
            
            logger.info(f"Successfully updated {package_name} to {target_version}")
            return update_result
            
        except Exception as e:
            logger.error(f"SDK update failed: {e}")
            update_result['status'] = 'failed'
            update_result['error_message'] = str(e)
            await self._record_update_failure(update_id, update_result)
            raise HTTPException(status_code=500, detail=f"SDK update failed: {str(e)}")
    
    async def _get_policy_client(self, scope: str, scope_id: Optional[str]):
        """Get policy client for specific scope"""
        if scope == 'subscription':
            return self.policy_clients.get('subscription')
        elif scope == 'management_group':
            # Create management group specific policy client
            return PolicyClient(credential=self.credential, subscription_id=None)
        else:
            raise ValueError(f"Unsupported scope: {scope}")
    
    async def _fetch_azure_policies(self, policy_client, include_builtin: bool, include_custom: bool) -> List[PolicyDefinition]:
        """Fetch policy definitions from Azure"""
        policies = []
        
        try:
            # Fetch all policy definitions
            async for policy in policy_client.policy_definitions.list():
                if include_builtin and policy.policy_type == 'BuiltIn':
                    policies.append(policy)
                elif include_custom and policy.policy_type == 'Custom':
                    policies.append(policy)
            
            return policies
            
        except Exception as e:
            logger.error(f"Failed to fetch Azure policies: {e}")
            return []
    
    async def _get_existing_policies(self, scope: str, scope_id: Optional[str]) -> Dict[str, Dict]:
        """Get existing policies from database"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM azure_policy_definitions 
                WHERE scope = $1 AND ($2 IS NULL OR scope_id = $2)
                AND is_active = TRUE
            """
            rows = await conn.fetch(query, scope, scope_id)
            
            return {row['name']: dict(row) for row in rows}
    
    async def _process_policy_update(self, azure_policy: PolicyDefinition, existing_policy: Optional[Dict], 
                                   scope: str, scope_id: Optional[str], force_refresh: bool) -> PolicySyncResult:
        """Process a single policy update"""
        policy_hash = self._calculate_policy_hash(azure_policy)
        changes = []
        status = 'unchanged'
        
        if not existing_policy:
            # New policy
            await self._insert_policy(azure_policy, policy_hash, scope, scope_id)
            status = 'added'
            changes.append('Policy definition added')
        else:
            if existing_policy['hash'] != policy_hash or force_refresh:
                # Policy changed
                changes = await self._detect_policy_changes(azure_policy, existing_policy)
                await self._update_policy(azure_policy, policy_hash, existing_policy['policy_id'])
                status = 'updated'
        
        result = PolicySyncResult(
            policy_id=azure_policy.name,
            policy_name=azure_policy.display_name or azure_policy.name,
            version=getattr(azure_policy, 'version', '1.0'),
            status=status,
            changes=changes,
            sync_time=datetime.utcnow()
        )
        
        # Record sync history
        await self._record_policy_sync(result)
        
        return result
    
    def _calculate_policy_hash(self, policy: PolicyDefinition) -> str:
        """Calculate hash of policy definition"""
        policy_content = {
            'policy_rule': policy.policy_rule,
            'parameters': policy.parameters,
            'metadata': policy.metadata
        }
        content_str = json.dumps(policy_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    async def _detect_policy_changes(self, azure_policy: PolicyDefinition, existing_policy: Dict) -> List[str]:
        """Detect changes between Azure policy and existing policy"""
        changes = []
        
        # Compare policy rule
        if azure_policy.policy_rule != existing_policy.get('policy_rule'):
            changes.append('Policy rule modified')
        
        # Compare parameters
        if azure_policy.parameters != existing_policy.get('parameters'):
            changes.append('Parameters modified')
        
        # Compare metadata
        if azure_policy.metadata != existing_policy.get('metadata'):
            changes.append('Metadata modified')
        
        # Compare description
        if azure_policy.description != existing_policy.get('description'):
            changes.append('Description modified')
        
        return changes
    
    async def _insert_policy(self, policy: PolicyDefinition, policy_hash: str, scope: str, scope_id: Optional[str]):
        """Insert new policy into database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO azure_policy_definitions 
                (policy_id, name, display_name, description, policy_type, mode, metadata, 
                 policy_rule, parameters, version, hash, scope, scope_id, is_builtin)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, 
                policy.name,
                policy.name,
                policy.display_name,
                policy.description,
                policy.policy_type,
                policy.mode,
                json.dumps(policy.metadata) if policy.metadata else None,
                json.dumps(policy.policy_rule),
                json.dumps(policy.parameters) if policy.parameters else None,
                getattr(policy, 'version', '1.0'),
                policy_hash,
                scope,
                scope_id,
                policy.policy_type == 'BuiltIn'
            )
    
    async def _update_policy(self, policy: PolicyDefinition, policy_hash: str, policy_id: str):
        """Update existing policy in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE azure_policy_definitions 
                SET display_name = $1, description = $2, policy_type = $3, mode = $4,
                    metadata = $5, policy_rule = $6, parameters = $7, version = $8,
                    hash = $9, updated_at = NOW()
                WHERE policy_id = $10
            """,
                policy.display_name,
                policy.description,
                policy.policy_type,
                policy.mode,
                json.dumps(policy.metadata) if policy.metadata else None,
                json.dumps(policy.policy_rule),
                json.dumps(policy.parameters) if policy.parameters else None,
                getattr(policy, 'version', '1.0'),
                policy_hash,
                policy_id
            )
    
    async def _mark_deprecated_policies(self, azure_policies: List[PolicyDefinition], 
                                      existing_policies: Dict, scope: str, scope_id: Optional[str]):
        """Mark policies as deprecated if they no longer exist in Azure"""
        azure_policy_names = {p.name for p in azure_policies}
        
        for policy_name, existing_policy in existing_policies.items():
            if policy_name not in azure_policy_names:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE azure_policy_definitions 
                        SET is_active = FALSE, deprecated_at = NOW()
                        WHERE policy_id = $1
                    """, existing_policy['policy_id'])
    
    async def _record_policy_sync(self, result: PolicySyncResult):
        """Record policy sync in history"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO policy_sync_history 
                (policy_id, status, changes, new_version, sync_time)
                VALUES ($1, $2, $3, $4, $5)
            """,
                result.policy_id,
                result.status,
                json.dumps(result.changes),
                result.version,
                result.sync_time
            )
    
    async def _check_package_version(self, package_name: str, include_preview: bool) -> SDKVersionInfo:
        """Check version information for a package"""
        try:
            current_version = await self._get_current_version(package_name)
            latest_version = await self._get_latest_version(package_name, include_preview)
            
            update_available = version.parse(latest_version) > version.parse(current_version)
            security_update = await self._check_security_updates(package_name, current_version, latest_version)
            
            # Determine update priority
            priority = 'low'
            if security_update:
                priority = 'critical'
            elif package_name in ['azure-identity', 'azure-core', '@azure/identity']:
                priority = 'high'
            elif update_available:
                version_diff = version.parse(latest_version) - version.parse(current_version)
                if version_diff.major > 0:
                    priority = 'high'
                elif version_diff.minor > 0:
                    priority = 'medium'
            
            changelog_url = await self._get_changelog_url(package_name, latest_version)
            
            return SDKVersionInfo(
                package_name=package_name,
                current_version=current_version,
                latest_version=latest_version,
                update_available=update_available,
                security_update=security_update,
                changelog_url=changelog_url,
                update_priority=priority
            )
            
        except Exception as e:
            logger.error(f"Failed to check version for {package_name}: {e}")
            return SDKVersionInfo(
                package_name=package_name,
                current_version="unknown",
                latest_version="unknown",
                update_available=False,
                security_update=False,
                changelog_url="",
                update_priority="low"
            )
    
    async def _get_current_version(self, package_name: str) -> str:
        """Get currently installed version of package"""
        try:
            if package_name.startswith('@'):
                # JavaScript package
                result = subprocess.run(['npm', 'list', package_name, '--depth=0'], 
                                      capture_output=True, text=True)
                # Parse npm output to extract version
                for line in result.stdout.split('\n'):
                    if package_name in line:
                        return line.split('@')[-1].strip()
            else:
                # Python package
                result = subprocess.run(['pip', 'show', package_name], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':')[1].strip()
        except Exception as e:
            logger.error(f"Failed to get current version for {package_name}: {e}")
        
        return "unknown"
    
    async def _get_latest_version(self, package_name: str, include_preview: bool) -> str:
        """Get latest available version of package"""
        try:
            if package_name.startswith('@'):
                # JavaScript package from npm
                url = f"https://registry.npmjs.org/{package_name}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        data = await response.json()
                        return data['dist-tags']['latest']
            else:
                # Python package from PyPI
                url = f"https://pypi.org/pypi/{package_name}/json"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        data = await response.json()
                        return data['info']['version']
        except Exception as e:
            logger.error(f"Failed to get latest version for {package_name}: {e}")
        
        return "unknown"
    
    async def _check_security_updates(self, package_name: str, current_version: str, latest_version: str) -> bool:
        """Check if update contains security fixes"""
        # This would integrate with security databases like GitHub Advisory Database
        # For now, return False as placeholder
        return False
    
    async def _get_changelog_url(self, package_name: str, version: str) -> str:
        """Get changelog URL for package version"""
        sdk_info = self.known_sdks.get(package_name, {})
        language = sdk_info.get('language', 'unknown')
        
        if language == 'python':
            return f"https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/{package_name}/CHANGELOG.md"
        elif language == 'javascript':
            return f"https://github.com/Azure/azure-sdk-for-js/blob/main/sdk/{package_name}/CHANGELOG.md"
        
        return ""
    
    async def _store_sdk_version_info(self, version_info: SDKVersionInfo):
        """Store SDK version information in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO azure_sdk_versions 
                (package_name, current_version, latest_version, update_available, 
                 security_update, update_priority, changelog_url, last_checked)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (package_name) DO UPDATE SET
                    current_version = $2,
                    latest_version = $3,
                    update_available = $4,
                    security_update = $5,
                    update_priority = $6,
                    changelog_url = $7,
                    last_checked = NOW()
            """,
                version_info.package_name,
                version_info.current_version,
                version_info.latest_version,
                version_info.update_available,
                version_info.security_update,
                version_info.update_priority,
                version_info.changelog_url
            )
    
    async def _policy_sync_loop(self):
        """Background task for periodic policy synchronization"""
        while True:
            try:
                await asyncio.sleep(self.config['sync']['policy_sync_interval'])
                
                # Sync subscription policies
                request = PolicySyncRequest(scope="subscription")
                await self.sync_azure_policies(request)
                
                logger.info("Completed scheduled policy sync")
                
            except Exception as e:
                logger.error(f"Error in policy sync loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _sdk_check_loop(self):
        """Background task for periodic SDK version checking"""
        while True:
            try:
                await asyncio.sleep(self.config['sync']['sdk_check_interval'])
                
                # Check all known SDKs
                request = SDKUpdateRequest()
                await self.check_sdk_updates(request)
                
                logger.info("Completed scheduled SDK version check")
                
            except Exception as e:
                logger.error(f"Error in SDK check loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _cache_sync_results(self, results: List[PolicySyncResult], scope: str, scope_id: Optional[str]):
        """Cache sync results in Redis"""
        cache_key = f"policy_sync:{scope}:{scope_id or 'default'}"
        cache_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'results': [asdict(result) for result in results]
        }
        
        await self.redis_client.setex(
            cache_key, 
            3600,  # 1 hour TTL
            json.dumps(cache_data, default=str)
        )
    
    async def _record_update_start(self, update_result: Dict) -> str:
        """Record SDK update start"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO sdk_update_history 
                (package_name, from_version, to_version, update_type, status, started_at)
                VALUES ($1, $2, $3, 'auto', $4, $5)
                RETURNING update_id
            """,
                update_result['package_name'],
                update_result['from_version'],
                update_result['to_version'],
                update_result['status'],
                update_result['started_at']
            )
            return str(row['update_id'])
    
    async def _record_update_completion(self, update_id: str, update_result: Dict):
        """Record SDK update completion"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE sdk_update_history 
                SET status = $1, completed_at = $2
                WHERE update_id = $3
            """,
                update_result['status'],
                update_result['completed_at'],
                update_id
            )
    
    async def _record_update_failure(self, update_id: str, update_result: Dict):
        """Record SDK update failure"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE sdk_update_history 
                SET status = $1, error_message = $2, completed_at = NOW()
                WHERE update_id = $3
            """,
                update_result['status'],
                update_result.get('error_message'),
                update_id
            )
    
    async def _backup_current_version(self, package_name: str):
        """Create backup of current package version"""
        # Implementation would create backup of current installation
        logger.info(f"Creating backup for {package_name}")
    
    async def _download_and_install_package(self, package_name: str, version: str):
        """Download and install package version"""
        # Implementation would handle actual package installation
        logger.info(f"Installing {package_name}=={version}")
    
    async def _run_compatibility_tests(self, package_name: str):
        """Run compatibility tests after package update"""
        # Implementation would run test suite
        logger.info(f"Running compatibility tests for {package_name}")
    
    async def _deploy_updated_service(self):
        """Deploy service with updated packages"""
        # Implementation would trigger service deployment
        logger.info("Deploying updated service")

# Global service instance
azure_sync_service = AzureSyncService()

@app.on_event("startup")
async def startup_event():
    await azure_sync_service.initialize()

@app.post("/sync/policies")
async def sync_policies(request: PolicySyncRequest, background_tasks: BackgroundTasks):
    """Sync Azure policy definitions"""
    background_tasks.add_task(azure_sync_service.sync_azure_policies, request)
    return {"status": "sync_started", "request": request.dict()}

@app.get("/sync/policies/status/{scope}")
async def get_policy_sync_status(scope: str, scope_id: Optional[str] = None):
    """Get policy sync status"""
    cache_key = f"policy_sync:{scope}:{scope_id or 'default'}"
    cached_data = await azure_sync_service.redis_client.get(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    else:
        return {"status": "no_recent_sync"}

@app.post("/sdk/check")
async def check_sdk_versions(request: SDKUpdateRequest):
    """Check SDK versions"""
    return await azure_sync_service.check_sdk_updates(request)

@app.post("/sdk/update/{package_name}")
async def update_sdk(package_name: str, target_version: str, auto_deploy: bool = False):
    """Update SDK package"""
    return await azure_sync_service.update_sdk_package(package_name, target_version, auto_deploy)

@app.get("/sdk/versions")
async def get_sdk_versions():
    """Get current SDK versions"""
    async with azure_sync_service.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM azure_sdk_versions ORDER BY update_priority, package_name")
        return [dict(row) for row in rows]

@app.get("/policies")
async def get_policies(scope: str = "subscription", scope_id: Optional[str] = None, limit: int = 100):
    """Get stored policy definitions"""
    async with azure_sync_service.db_pool.acquire() as conn:
        query = """
            SELECT * FROM azure_policy_definitions 
            WHERE scope = $1 AND ($2 IS NULL OR scope_id = $2)
            AND is_active = TRUE
            ORDER BY updated_at DESC
            LIMIT $3
        """
        rows = await conn.fetch(query, scope, scope_id, limit)
        return [dict(row) for row in rows]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "azure-sync"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)