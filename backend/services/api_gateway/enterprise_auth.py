"""
Enterprise Authentication Manager for PolicyCortex
Implements automatic organization detection, multi-tenant support, and enterprise SSO
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
import jwt
import redis.asyncio as redis
import structlog
from azure.identity.aio import ClientSecretCredential, DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient
from azure.monitor.opentelemetry import configure_azure_monitor
from jose import JWTError
from jose import jwt as jose_jwt
from msal import ConfidentialClientApplication
from opentelemetry import trace
from shared.config import get_settings
from shared.database import AuditLog, DatabaseUtils, async_db_transaction

from .models import UserInfo

settings = get_settings()
logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class AuthenticationMethod(Enum):
    """Supported authentication methods"""

    AZURE_AD = "azure_ad"
    SAML = "saml"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    INTERNAL = "internal"


class OrganizationType(Enum):
    """Organization types for different pricing tiers"""

    ENTERPRISE = "enterprise"
    PROFESSIONAL = "professional"
    STARTER = "starter"
    TRIAL = "trial"


class Role(Enum):
    """Predefined system roles aligned with B2B requirements"""

    GLOBAL_ADMIN = "global_admin"
    POLICY_ADMINISTRATOR = "policy_administrator"
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_ANALYST = "risk_analyst"
    EXECUTIVE_VIEWER = "executive_viewer"
    DEPARTMENT_MANAGER = "department_manager"
    TEAM_MEMBER = "team_member"
    VIEWER = "viewer"


class EnterpriseAuthManager:
    """
    Enterprise-grade authentication manager with B2B features
    Implements stupid simplicity while maintaining advanced capabilities
    """

    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.azure_credential = None
        self.key_vault_client = None
        self.msal_app = None
        self._org_cache = {}
        self._domain_patterns = self._initialize_domain_patterns()

        # Initialize Azure Monitor for audit logging
        if settings.is_production():
            configure_azure_monitor(
                connection_string=settings.azure.application_insights_connection_string
            )

    def _initialize_domain_patterns(self) -> Dict[str, str]:
        """Initialize common enterprise domain patterns for auto-detection"""
        return {
            r".*\.microsoft\.com$": "microsoft",
            r".*\.google\.com$": "google",
            r".*\.amazon\.com$": "amazon",
            r".*\.ibm\.com$": "ibm",
            r".*\.oracle\.com$": "oracle",
            r".*\.salesforce\.com$": "salesforce",
            r".*\.gov$": "government",
            r".*\.edu$": "education",
            r".*\.org$": "nonprofit",
        }

    async def detect_organization(self, email: str) -> Dict[str, Any]:
        """
        Automatically detect organization from email domain
        Returns organization configuration with zero user configuration required
        """
        with tracer.start_as_current_span("detect_organization") as span:
            span.set_attribute("email_domain", email.split("@")[1])

            domain = email.split("@")[1].lower()

            # Check cache first
            if domain in self._org_cache:
                logger.info("organization_detected_from_cache", domain=domain)
                return self._org_cache[domain]

            # Check if organization is already registered
            org_config = await self._lookup_organization_config(domain)
            if org_config:
                self._org_cache[domain] = org_config
                return org_config

            # Auto-detect organization type and settings
            org_type = self._detect_organization_type(domain)
            auth_method = await self._detect_authentication_method(domain)

            org_config = {
                "domain": domain,
                "name": self._generate_org_name(domain),
                "type": org_type.value,
                "authentication_method": auth_method.value,
                "tenant_id": await self._get_or_create_tenant_id(domain),
                "settings": {
                    "sso_enabled": auth_method != AuthenticationMethod.INTERNAL,
                    "mfa_required": org_type == OrganizationType.ENTERPRISE,
                    "session_timeout_minutes": (
                        480 if org_type == OrganizationType.ENTERPRISE else 120
                    ),
                    "password_policy": self._get_password_policy(org_type),
                    "data_residency": await self._detect_data_residency(domain),
                    "compliance_frameworks": self._get_compliance_frameworks(org_type),
                },
                "features": self._get_org_features(org_type),
                "limits": self._get_org_limits(org_type),
            }

            # Cache the configuration
            self._org_cache[domain] = org_config
            await self._persist_organization_config(org_config)

            logger.info(
                "organization_auto_detected",
                domain=domain,
                org_type=org_type.value,
                auth_method=auth_method.value,
            )

            return org_config

    async def _lookup_organization_config(self, domain: str) -> Optional[Dict[str, Any]]:
        """Look up existing organization configuration"""
        redis_client = await self._get_redis_client()
        config_key = f"org_config:{domain}"

        config_data = await redis_client.get(config_key)
        if config_data:
            return json.loads(config_data)

        return None

    def _detect_organization_type(self, domain: str) -> OrganizationType:
        """Detect organization type based on domain patterns"""
        # Fortune 500 companies get enterprise
        fortune_500_domains = ["microsoft.com", "google.com", "amazon.com", "apple.com"]
        if any(domain.endswith(d) for d in fortune_500_domains):
            return OrganizationType.ENTERPRISE

        # Government and education get enterprise features
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return OrganizationType.ENTERPRISE

        # Check for corporate indicators
        corporate_indicators = [".com", ".corp", ".biz"]
        if any(domain.endswith(ind) for ind in corporate_indicators):
            # Check domain age and size for classification
            parts = domain.split(".")
            if len(parts) > 2 or len(parts[0]) > 10:
                return OrganizationType.PROFESSIONAL
            return OrganizationType.STARTER

        # Non-profits get professional tier
        if domain.endswith(".org"):
            return OrganizationType.PROFESSIONAL

        # Default to starter for unknown domains
        return OrganizationType.STARTER

    async def _detect_authentication_method(self, domain: str) -> AuthenticationMethod:
        """Auto-detect the authentication method for the domain"""
        try:
            # Try Azure AD discovery first
            if await self._check_azure_ad_tenant(domain):
                return AuthenticationMethod.AZURE_AD

            # Check for SAML metadata
            if await self._check_saml_metadata(domain):
                return AuthenticationMethod.SAML

            # Check for OAuth2 configuration
            if await self._check_oauth2_config(domain):
                return AuthenticationMethod.OAUTH2

            # Default to internal authentication
            return AuthenticationMethod.INTERNAL

        except Exception as e:
            logger.warning("auth_method_detection_failed", domain=domain, error=str(e))
            return AuthenticationMethod.INTERNAL

    async def _check_azure_ad_tenant(self, domain: str) -> bool:
        """Check if domain has Azure AD tenant"""
        try:
            async with httpx.AsyncClient() as client:
                # Check OpenID configuration
                openid_url = (
                    f"https://login.microsoftonline.com/{domain}/.well-known/openid-configuration"
                )
                response = await client.get(openid_url, timeout=5.0)
                return response.status_code == 200
        except:
            return False

    async def _check_saml_metadata(self, domain: str) -> bool:
        """Check for SAML metadata endpoint"""
        common_saml_paths = ["/saml/metadata", "/sso/saml/metadata", "/auth/saml/metadata"]

        try:
            async with httpx.AsyncClient() as client:
                for path in common_saml_paths:
                    url = f"https://{domain}{path}"
                    response = await client.head(url, timeout=3.0)
                    if response.status_code == 200:
                        return True
        except:
            pass

        return False

    async def _check_oauth2_config(self, domain: str) -> bool:
        """Check for OAuth2 configuration"""
        try:
            async with httpx.AsyncClient() as client:
                # Check for well-known OAuth2 configuration
                oauth_url = f"https://{domain}/.well-known/oauth-authorization-server"
                response = await client.get(oauth_url, timeout=3.0)
                return response.status_code == 200
        except:
            return False

    async def _get_or_create_tenant_id(self, domain: str) -> str:
        """Get or create a unique tenant ID for the organization"""
        # Generate deterministic tenant ID from domain
        tenant_id = hashlib.sha256(f"tenant_{domain}".encode()).hexdigest()[:32]

        # Store tenant mapping
        redis_client = await self._get_redis_client()
        await redis_client.set(f"tenant_domain:{tenant_id}", domain)
        await redis_client.set(f"domain_tenant:{domain}", tenant_id)

        return tenant_id

    def _generate_org_name(self, domain: str) -> str:
        """Generate organization name from domain"""
        # Remove common TLDs and format
        name = domain.split(".")[0]
        # Convert to title case and replace hyphens/underscores
        name = name.replace("-", " ").replace("_", " ").title()
        return name

    async def _detect_data_residency(self, domain: str) -> str:
        """Detect data residency requirements based on domain"""
        # Map TLDs to regions
        tld_regions = {
            ".eu": "europe",
            ".uk": "uk",
            ".de": "germany",
            ".fr": "france",
            ".ca": "canada",
            ".au": "australia",
            ".jp": "japan",
            ".cn": "china",
            ".in": "india",
            ".gov": "us-gov",
        }

        for tld, region in tld_regions.items():
            if domain.endswith(tld):
                return region

        # Default to US
        return "us"

    def _get_password_policy(self, org_type: OrganizationType) -> Dict[str, Any]:
        """Get password policy based on organization type"""
        policies = {
            OrganizationType.ENTERPRISE: {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True,
                "max_age_days": 90,
                "history_count": 12,
                "lockout_attempts": 5,
            },
            OrganizationType.PROFESSIONAL: {
                "min_length": 10,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True,
                "max_age_days": 180,
                "history_count": 6,
                "lockout_attempts": 5,
            },
            OrganizationType.STARTER: {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": False,
                "max_age_days": 365,
                "history_count": 3,
                "lockout_attempts": 10,
            },
            OrganizationType.TRIAL: {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": False,
                "max_age_days": 0,
                "history_count": 0,
                "lockout_attempts": 10,
            },
        }

        return policies.get(org_type, policies[OrganizationType.STARTER])

    def _get_compliance_frameworks(self, org_type: OrganizationType) -> List[str]:
        """Get compliance frameworks based on organization type"""
        frameworks = {
            OrganizationType.ENTERPRISE: [
                "SOC2",
                "ISO27001",
                "GDPR",
                "HIPAA",
                "CCPA",
                "PCI-DSS",
                "NIST",
                "FedRAMP",
            ],
            OrganizationType.PROFESSIONAL: ["SOC2", "ISO27001", "GDPR", "CCPA"],
            OrganizationType.STARTER: ["SOC2", "GDPR"],
            OrganizationType.TRIAL: [],
        }

        return frameworks.get(org_type, [])

    def _get_org_features(self, org_type: OrganizationType) -> Dict[str, bool]:
        """Get enabled features based on organization type"""
        features = {
            OrganizationType.ENTERPRISE: {
                "unlimited_users": True,
                "custom_roles": True,
                "api_access": True,
                "advanced_analytics": True,
                "ai_predictions": True,
                "custom_policies": True,
                "white_labeling": True,
                "dedicated_support": True,
                "sla_guarantee": True,
                "data_export": True,
                "audit_logs": True,
                "multi_region": True,
            },
            OrganizationType.PROFESSIONAL: {
                "unlimited_users": False,
                "custom_roles": True,
                "api_access": True,
                "advanced_analytics": True,
                "ai_predictions": True,
                "custom_policies": True,
                "white_labeling": False,
                "dedicated_support": False,
                "sla_guarantee": True,
                "data_export": True,
                "audit_logs": True,
                "multi_region": False,
            },
            OrganizationType.STARTER: {
                "unlimited_users": False,
                "custom_roles": False,
                "api_access": False,
                "advanced_analytics": False,
                "ai_predictions": True,
                "custom_policies": False,
                "white_labeling": False,
                "dedicated_support": False,
                "sla_guarantee": False,
                "data_export": True,
                "audit_logs": True,
                "multi_region": False,
            },
            OrganizationType.TRIAL: {
                "unlimited_users": False,
                "custom_roles": False,
                "api_access": False,
                "advanced_analytics": False,
                "ai_predictions": True,
                "custom_policies": False,
                "white_labeling": False,
                "dedicated_support": False,
                "sla_guarantee": False,
                "data_export": False,
                "audit_logs": True,
                "multi_region": False,
            },
        }

        return features.get(org_type, features[OrganizationType.STARTER])

    def _get_org_limits(self, org_type: OrganizationType) -> Dict[str, int]:
        """Get organization limits based on type"""
        limits = {
            OrganizationType.ENTERPRISE: {
                "max_users": -1,  # Unlimited
                "max_policies": -1,
                "max_resources": -1,
                "max_api_calls_per_month": -1,
                "max_storage_gb": 10000,
                "max_concurrent_sessions": 10000,
                "retention_days": 2555,  # 7 years
            },
            OrganizationType.PROFESSIONAL: {
                "max_users": 500,
                "max_policies": 1000,
                "max_resources": 10000,
                "max_api_calls_per_month": 1000000,
                "max_storage_gb": 1000,
                "max_concurrent_sessions": 500,
                "retention_days": 365,
            },
            OrganizationType.STARTER: {
                "max_users": 50,
                "max_policies": 100,
                "max_resources": 1000,
                "max_api_calls_per_month": 100000,
                "max_storage_gb": 100,
                "max_concurrent_sessions": 50,
                "retention_days": 90,
            },
            OrganizationType.TRIAL: {
                "max_users": 5,
                "max_policies": 10,
                "max_resources": 100,
                "max_api_calls_per_month": 1000,
                "max_storage_gb": 10,
                "max_concurrent_sessions": 5,
                "retention_days": 30,
            },
        }

        return limits.get(org_type, limits[OrganizationType.STARTER])

    async def _persist_organization_config(self, config: Dict[str, Any]) -> None:
        """Persist organization configuration to storage"""
        redis_client = await self._get_redis_client()

        # Store in Redis with long TTL
        config_key = f"org_config:{config['domain']}"
        await redis_client.set(config_key, json.dumps(config), ex=86400 * 30)  # 30 days

        # Log audit event
        async with async_db_transaction() as session:
            await DatabaseUtils.log_audit_event(
                session=session,
                entity_type="organization",
                entity_id=config["tenant_id"],
                action="CREATE",
                new_values=config,
                details=f"Auto-detected organization: {config['name']}",
            )

    async def authenticate_user(
        self,
        email: str,
        password: Optional[str] = None,
        token: Optional[str] = None,
        auth_code: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Authenticate user with automatic method detection
        Returns (user_info, tokens)
        """
        with tracer.start_as_current_span("authenticate_user") as span:
            span.set_attribute("email", email)

            # Detect organization and authentication method
            org_config = await self.detect_organization(email)
            auth_method = AuthenticationMethod(org_config["authentication_method"])

            # Route to appropriate authentication handler
            if auth_method == AuthenticationMethod.AZURE_AD:
                user_info = await self._authenticate_azure_ad(email, auth_code, org_config)
            elif auth_method == AuthenticationMethod.SAML:
                user_info = await self._authenticate_saml(email, token, org_config)
            elif auth_method == AuthenticationMethod.OAUTH2:
                user_info = await self._authenticate_oauth2(email, auth_code, org_config)
            elif auth_method == AuthenticationMethod.LDAP:
                user_info = await self._authenticate_ldap(email, password, org_config)
            else:
                user_info = await self._authenticate_internal(email, password, org_config)

            # Enrich user info with organization context
            user_info["tenant_id"] = org_config["tenant_id"]
            user_info["organization"] = org_config["name"]
            user_info["org_type"] = org_config["type"]

            # Auto-assign roles based on email patterns and job titles
            if "roles" not in user_info:
                user_info["roles"] = await self._auto_assign_roles(email, user_info, org_config)

            # Generate tokens
            tokens = await self._generate_tokens(user_info, org_config)

            # Create session
            await self._create_user_session(user_info, tokens, org_config)

            # Log successful authentication
            await self._log_authentication_event(user_info, "SUCCESS", org_config)

            logger.info(
                "user_authenticated",
                email=email,
                tenant_id=org_config["tenant_id"],
                auth_method=auth_method.value,
            )

            return user_info, tokens

    async def _authenticate_azure_ad(
        self, email: str, auth_code: str, org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using Azure AD"""
        try:
            # Initialize MSAL app if not already done
            if not self.msal_app:
                self.msal_app = ConfidentialClientApplication(
                    client_id=settings.azure.client_id,
                    client_credential=settings.azure.client_secret,
                    authority=f"https://login.microsoftonline.com/{org_config['tenant_id']}",
                )

            # Exchange auth code for tokens
            result = self.msal_app.acquire_token_by_authorization_code(
                code=auth_code,
                scopes=["User.Read", "profile", "email"],
                redirect_uri=settings.azure.redirect_uri,
            )

            if "error" in result:
                raise Exception(f"Azure AD authentication failed: {result['error_description']}")

            # Extract user information from ID token
            id_token = result.get("id_token_claims", {})

            return {
                "id": id_token.get("oid", id_token.get("sub")),
                "email": email,
                "name": id_token.get("name", ""),
                "given_name": id_token.get("given_name", ""),
                "family_name": id_token.get("family_name", ""),
                "job_title": id_token.get("jobTitle", ""),
                "department": id_token.get("department", ""),
                "groups": id_token.get("groups", []),
                "auth_method": AuthenticationMethod.AZURE_AD.value,
            }

        except Exception as e:
            logger.error("azure_ad_auth_failed", email=email, error=str(e))
            raise Exception(f"Azure AD authentication failed: {str(e)}")

    async def _authenticate_internal(
        self, email: str, password: str, org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using internal authentication"""
        # This would typically check against a database
        # For now, return a mock user
        return {
            "id": hashlib.sha256(email.encode()).hexdigest()[:32],
            "email": email,
            "name": email.split("@")[0].replace(".", " ").title(),
            "auth_method": AuthenticationMethod.INTERNAL.value,
        }

    async def _auto_assign_roles(
        self, email: str, user_info: Dict[str, Any], org_config: Dict[str, Any]
    ) -> List[str]:
        """Auto-assign roles based on email patterns and job titles"""
        roles = []

        # Check job title patterns
        job_title = user_info.get("job_title", "").lower()
        job_role_mapping = {
            "ceo": [Role.GLOBAL_ADMIN, Role.EXECUTIVE_VIEWER],
            "cto": [Role.GLOBAL_ADMIN, Role.POLICY_ADMINISTRATOR],
            "ciso": [Role.GLOBAL_ADMIN, Role.COMPLIANCE_OFFICER],
            "compliance": [Role.COMPLIANCE_OFFICER],
            "security": [Role.POLICY_ADMINISTRATOR],
            "risk": [Role.RISK_ANALYST],
            "manager": [Role.DEPARTMENT_MANAGER],
            "director": [Role.DEPARTMENT_MANAGER],
            "analyst": [Role.RISK_ANALYST],
            "executive": [Role.EXECUTIVE_VIEWER],
            "admin": [Role.POLICY_ADMINISTRATOR],
        }

        for pattern, role_list in job_role_mapping.items():
            if pattern in job_title:
                roles.extend([r.value for r in role_list])

        # Check email patterns
        email_local = email.split("@")[0].lower()
        if any(admin_pattern in email_local for admin_pattern in ["admin", "root", "sysadmin"]):
            roles.append(Role.GLOBAL_ADMIN.value)

        # Check department
        department = user_info.get("department", "").lower()
        dept_role_mapping = {
            "it": [Role.POLICY_ADMINISTRATOR],
            "security": [Role.COMPLIANCE_OFFICER],
            "compliance": [Role.COMPLIANCE_OFFICER],
            "risk": [Role.RISK_ANALYST],
            "executive": [Role.EXECUTIVE_VIEWER],
        }

        for dept, role_list in dept_role_mapping.items():
            if dept in department:
                roles.extend([r.value for r in role_list])

        # Default role if none assigned
        if not roles:
            roles = [Role.TEAM_MEMBER.value]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(roles))

    async def _generate_tokens(
        self, user_info: Dict[str, Any], org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate JWT tokens with organization context"""
        jwt_secret = await self._get_jwt_secret()

        now = datetime.utcnow()
        session_timeout = org_config["settings"]["session_timeout_minutes"]

        # Access token payload
        access_payload = {
            "sub": user_info["id"],
            "email": user_info["email"],
            "name": user_info.get("name", ""),
            "tenant_id": user_info["tenant_id"],
            "organization": user_info["organization"],
            "org_type": user_info["org_type"],
            "roles": user_info.get("roles", []),
            "permissions": await self._get_role_permissions(user_info.get("roles", [])),
            "auth_method": user_info.get("auth_method"),
            "iat": now,
            "exp": now + timedelta(minutes=session_timeout),
            "jti": hashlib.sha256(f"{user_info['id']}_{now.timestamp()}".encode()).hexdigest()[:16],
        }

        # Generate access token
        access_token = jose_jwt.encode(
            access_payload, jwt_secret, algorithm=settings.security.jwt_algorithm
        )

        # Refresh token with longer expiration
        refresh_payload = access_payload.copy()
        refresh_payload["exp"] = now + timedelta(days=30)
        refresh_payload["type"] = "refresh"

        refresh_token = jose_jwt.encode(
            refresh_payload, jwt_secret, algorithm=settings.security.jwt_algorithm
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": session_timeout * 60,
        }

    async def _get_role_permissions(self, roles: List[str]) -> List[str]:
        """Get permissions for roles"""
        permission_mapping = {
            Role.GLOBAL_ADMIN.value: ["*"],  # All permissions
            Role.POLICY_ADMINISTRATOR.value: [
                "policies:*",
                "resources:view",
                "audit:view",
                "settings:manage",
            ],
            Role.COMPLIANCE_OFFICER.value: [
                "policies:view",
                "compliance:*",
                "audit:*",
                "reports:*",
            ],
            Role.RISK_ANALYST.value: ["policies:view", "risk:*", "analytics:*", "reports:view"],
            Role.EXECUTIVE_VIEWER.value: ["dashboard:view", "reports:view", "analytics:view"],
            Role.DEPARTMENT_MANAGER.value: [
                "policies:view",
                "resources:view",
                "team:manage",
                "reports:view",
            ],
            Role.TEAM_MEMBER.value: ["policies:view", "resources:view", "dashboard:view"],
            Role.VIEWER.value: ["dashboard:view", "policies:view"],
        }

        permissions = set()
        for role in roles:
            if role in permission_mapping:
                permissions.update(permission_mapping[role])

        return list(permissions)

    async def _create_user_session(
        self, user_info: Dict[str, Any], tokens: Dict[str, Any], org_config: Dict[str, Any]
    ) -> str:
        """Create user session with organization context"""
        redis_client = await self._get_redis_client()

        session_id = hashlib.sha256(
            f"{user_info['id']}_{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:32]

        session_data = {
            "session_id": session_id,
            "user_id": user_info["id"],
            "email": user_info["email"],
            "tenant_id": user_info["tenant_id"],
            "organization": user_info["organization"],
            "org_type": user_info["org_type"],
            "roles": user_info.get("roles", []),
            "auth_method": user_info.get("auth_method"),
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.utcnow()
                + timedelta(minutes=org_config["settings"]["session_timeout_minutes"])
            ).isoformat(),
            "ip_address": None,  # Would be set from request context
            "user_agent": None,  # Would be set from request context
            "mfa_verified": False,
            "revoked": False,
        }

        # Store session with tenant namespace
        session_key = f"session:{user_info['tenant_id']}:{session_id}"
        await redis_client.set(
            session_key,
            json.dumps(session_data),
            ex=org_config["settings"]["session_timeout_minutes"] * 60,
        )

        # Add to user's active sessions
        user_sessions_key = f"user_sessions:{user_info['tenant_id']}:{user_info['id']}"
        await redis_client.sadd(user_sessions_key, session_id)

        # Enforce concurrent session limits
        await self._enforce_session_limits(user_info, org_config)

        return session_id

    async def _enforce_session_limits(
        self, user_info: Dict[str, Any], org_config: Dict[str, Any]
    ) -> None:
        """Enforce concurrent session limits based on organization type"""
        redis_client = await self._get_redis_client()

        max_sessions = 10  # Default max sessions per user
        if org_config["org_type"] == OrganizationType.TRIAL.value:
            max_sessions = 1
        elif org_config["org_type"] == OrganizationType.STARTER.value:
            max_sessions = 3
        elif org_config["org_type"] == OrganizationType.PROFESSIONAL.value:
            max_sessions = 5

        user_sessions_key = f"user_sessions:{user_info['tenant_id']}:{user_info['id']}"
        sessions = await redis_client.smembers(user_sessions_key)

        if len(sessions) > max_sessions:
            # Remove oldest sessions
            sessions_with_time = []
            for session_id in sessions:
                session_key = f"session:{user_info['tenant_id']}:{session_id}"
                session_data = await redis_client.get(session_key)
                if session_data:
                    session = json.loads(session_data)
                    sessions_with_time.append((session_id, session["created_at"]))

            # Sort by creation time and remove oldest
            sessions_with_time.sort(key=lambda x: x[1])
            for session_id, _ in sessions_with_time[:-max_sessions]:
                await self._revoke_session(user_info["tenant_id"], session_id)

    async def _revoke_session(self, tenant_id: str, session_id: str) -> None:
        """Revoke a specific session"""
        redis_client = await self._get_redis_client()

        session_key = f"session:{tenant_id}:{session_id}"
        session_data = await redis_client.get(session_key)

        if session_data:
            session = json.loads(session_data)
            session["revoked"] = True
            session["revoked_at"] = datetime.utcnow().isoformat()

            # Keep revoked session for audit trail
            await redis_client.set(session_key, json.dumps(session), ex=300)  # Keep for 5 minutes

            # Remove from active sessions
            user_sessions_key = f"user_sessions:{tenant_id}:{session['user_id']}"
            await redis_client.srem(user_sessions_key, session_id)

    async def _log_authentication_event(
        self, user_info: Dict[str, Any], status: str, org_config: Dict[str, Any]
    ) -> None:
        """Log authentication event for audit trail"""
        async with async_db_transaction() as session:
            await DatabaseUtils.log_audit_event(
                session=session,
                entity_type="authentication",
                entity_id=user_info["id"],
                action="LOGIN",
                user_id=user_info["id"],
                details=json.dumps(
                    {
                        "status": status,
                        "email": user_info["email"],
                        "tenant_id": user_info["tenant_id"],
                        "organization": org_config["name"],
                        "auth_method": user_info.get("auth_method"),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
            )

        # Send to Azure Monitor if in production
        if settings.is_production():
            logger.info(
                "authentication_event",
                user_id=user_info["id"],
                email=user_info["email"],
                tenant_id=user_info["tenant_id"],
                status=status,
                auth_method=user_info.get("auth_method"),
            )

    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for session management"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                settings.database.redis_url,
                password=settings.database.redis_password,
                ssl=settings.database.redis_ssl,
                decode_responses=True,
            )
        return self.redis_client

    async def _get_jwt_secret(self) -> str:
        """Get JWT secret from Key Vault or configuration"""
        try:
            if settings.is_production():
                if self.key_vault_client is None:
                    if self.azure_credential is None:
                        self.azure_credential = DefaultAzureCredential()
                    self.key_vault_client = SecretClient(
                        vault_url=settings.azure.key_vault_url, credential=self.azure_credential
                    )

                secret = await self.key_vault_client.get_secret("jwt-secret-key")
                return secret.value
            else:
                return settings.security.jwt_secret_key
        except Exception as e:
            logger.warning("failed_to_get_jwt_secret", error=str(e))
            return settings.security.jwt_secret_key

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token with tenant isolation"""
        try:
            jwt_secret = await self._get_jwt_secret()

            # Decode token
            payload = jose_jwt.decode(
                token, jwt_secret, algorithms=[settings.security.jwt_algorithm]
            )

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token has expired")

            # Validate session
            tenant_id = payload.get("tenant_id")
            session_id = payload.get("jti")

            if tenant_id and session_id:
                redis_client = await self._get_redis_client()
                session_key = f"session:{tenant_id}:{session_id}"
                session_data = await redis_client.get(session_key)

                if not session_data:
                    raise Exception("Session not found")

                session = json.loads(session_data)
                if session.get("revoked"):
                    raise Exception("Session has been revoked")

                # Update last activity
                session["last_activity"] = datetime.utcnow().isoformat()
                await redis_client.set(
                    session_key,
                    json.dumps(session),
                    ex=int(
                        (
                            datetime.fromisoformat(session["expires_at"]) - datetime.utcnow()
                        ).total_seconds()
                    ),
                )

            return payload

        except Exception as e:
            logger.error("token_validation_failed", error=str(e))
            raise Exception(f"Token validation failed: {str(e)}")

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        try:
            # Validate refresh token
            payload = await self.validate_token(refresh_token)

            if payload.get("type") != "refresh":
                raise Exception("Invalid token type for refresh")

            # Get organization config
            org_config = await self._lookup_organization_config(payload["email"].split("@")[1])

            # Generate new tokens
            user_info = {
                "id": payload["sub"],
                "email": payload["email"],
                "name": payload.get("name"),
                "tenant_id": payload["tenant_id"],
                "organization": payload["organization"],
                "org_type": payload["org_type"],
                "roles": payload.get("roles", []),
                "auth_method": payload.get("auth_method"),
            }

            tokens = await self._generate_tokens(user_info, org_config)

            logger.info(
                "token_refreshed", user_id=user_info["id"], tenant_id=user_info["tenant_id"]
            )

            return tokens

        except Exception as e:
            logger.error("token_refresh_failed", error=str(e))
            raise Exception(f"Token refresh failed: {str(e)}")

    async def _authenticate_saml(
        self, email: str, saml_response: str, org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using SAML (placeholder for actual implementation)"""
        # This would parse and validate SAML response
        # For now, return mock user
        return {
            "id": hashlib.sha256(email.encode()).hexdigest()[:32],
            "email": email,
            "name": email.split("@")[0].replace(".", " ").title(),
            "auth_method": AuthenticationMethod.SAML.value,
        }

    async def _authenticate_oauth2(
        self, email: str, auth_code: str, org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using OAuth2 (placeholder for actual implementation)"""
        # This would exchange auth code for tokens
        # For now, return mock user
        return {
            "id": hashlib.sha256(email.encode()).hexdigest()[:32],
            "email": email,
            "name": email.split("@")[0].replace(".", " ").title(),
            "auth_method": AuthenticationMethod.OAUTH2.value,
        }

    async def _authenticate_ldap(
        self, email: str, password: str, org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate using LDAP (placeholder for actual implementation)"""
        # This would validate against LDAP server
        # For now, return mock user
        return {
            "id": hashlib.sha256(email.encode()).hexdigest()[:32],
            "email": email,
            "name": email.split("@")[0].replace(".", " ").title(),
            "auth_method": AuthenticationMethod.LDAP.value,
        }
