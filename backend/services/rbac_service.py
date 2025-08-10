"""
Role-Based Access Control (RBAC) Service for PolicyCortex
Provides comprehensive authorization and permission management
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions"""
    # Resource permissions
    RESOURCE_VIEW = "resource:view"
    RESOURCE_CREATE = "resource:create"
    RESOURCE_UPDATE = "resource:update"
    RESOURCE_DELETE = "resource:delete"
    
    # Policy permissions
    POLICY_VIEW = "policy:view"
    POLICY_CREATE = "policy:create"
    POLICY_UPDATE = "policy:update"
    POLICY_DELETE = "policy:delete"
    POLICY_ENFORCE = "policy:enforce"
    
    # Compliance permissions
    COMPLIANCE_VIEW = "compliance:view"
    COMPLIANCE_APPROVE = "compliance:approve"
    COMPLIANCE_EXCEPTION = "compliance:exception"
    
    # Cost permissions
    COST_VIEW = "cost:view"
    COST_ANALYZE = "cost:analyze"
    COST_OPTIMIZE = "cost:optimize"
    
    # Security permissions
    SECURITY_VIEW = "security:view"
    SECURITY_INVESTIGATE = "security:investigate"
    SECURITY_REMEDIATE = "security:remediate"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_AUDIT = "admin:audit"
    
    # Report permissions
    REPORT_VIEW = "report:view"
    REPORT_CREATE = "report:create"
    REPORT_EXPORT = "report:export"
    
    # Action permissions
    ACTION_EXECUTE = "action:execute"
    ACTION_APPROVE = "action:approve"
    ACTION_CANCEL = "action:cancel"

class ResourceScope(Enum):
    """Resource scope levels"""
    GLOBAL = "global"
    ORGANIZATION = "organization"
    SUBSCRIPTION = "subscription"
    RESOURCE_GROUP = "resource_group"
    RESOURCE = "resource"

@dataclass
class Role:
    """Role definition"""
    id: str
    name: str
    description: str
    permissions: Set[Permission]
    scope: ResourceScope
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_system: bool = False
    is_enabled: bool = True

@dataclass
class User:
    """User definition"""
    id: str
    email: str
    name: str
    roles: List[str]  # Role IDs
    direct_permissions: Set[Permission] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    mfa_enabled: bool = False

@dataclass
class RoleAssignment:
    """Role assignment to user"""
    id: str
    user_id: str
    role_id: str
    scope: str  # Specific resource scope (e.g., subscription ID)
    conditions: Dict[str, Any] = field(default_factory=dict)
    assigned_by: str = ""
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class PermissionCheck:
    """Permission check result"""
    allowed: bool
    user_id: str
    permission: Permission
    resource: Optional[str]
    reason: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

class RBACService:
    """Role-Based Access Control service"""
    
    def __init__(self):
        """Initialize RBAC service"""
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.assignments: Dict[str, RoleAssignment] = {}
        self.permission_cache: Dict[str, Dict] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Initialize system roles
        self._initialize_system_roles()
    
    def _initialize_system_roles(self):
        """Initialize default system roles"""
        system_roles = [
            Role(
                id="role-admin",
                name="Administrator",
                description="Full system access",
                permissions=set(Permission),  # All permissions
                scope=ResourceScope.GLOBAL,
                is_system=True
            ),
            Role(
                id="role-operator",
                name="Operator",
                description="Operational management access",
                permissions={
                    Permission.RESOURCE_VIEW,
                    Permission.RESOURCE_CREATE,
                    Permission.RESOURCE_UPDATE,
                    Permission.POLICY_VIEW,
                    Permission.POLICY_ENFORCE,
                    Permission.COMPLIANCE_VIEW,
                    Permission.COST_VIEW,
                    Permission.SECURITY_VIEW,
                    Permission.REPORT_VIEW,
                    Permission.ACTION_EXECUTE
                },
                scope=ResourceScope.SUBSCRIPTION,
                is_system=True
            ),
            Role(
                id="role-analyst",
                name="Analyst",
                description="Read and analyze access",
                permissions={
                    Permission.RESOURCE_VIEW,
                    Permission.POLICY_VIEW,
                    Permission.COMPLIANCE_VIEW,
                    Permission.COST_VIEW,
                    Permission.COST_ANALYZE,
                    Permission.SECURITY_VIEW,
                    Permission.REPORT_VIEW,
                    Permission.REPORT_CREATE,
                    Permission.REPORT_EXPORT
                },
                scope=ResourceScope.SUBSCRIPTION,
                is_system=True
            ),
            Role(
                id="role-auditor",
                name="Auditor",
                description="Audit and compliance access",
                permissions={
                    Permission.RESOURCE_VIEW,
                    Permission.POLICY_VIEW,
                    Permission.COMPLIANCE_VIEW,
                    Permission.COMPLIANCE_APPROVE,
                    Permission.SECURITY_VIEW,
                    Permission.ADMIN_AUDIT,
                    Permission.REPORT_VIEW,
                    Permission.REPORT_CREATE,
                    Permission.REPORT_EXPORT
                },
                scope=ResourceScope.ORGANIZATION,
                is_system=True
            ),
            Role(
                id="role-viewer",
                name="Viewer",
                description="Read-only access",
                permissions={
                    Permission.RESOURCE_VIEW,
                    Permission.POLICY_VIEW,
                    Permission.COMPLIANCE_VIEW,
                    Permission.COST_VIEW,
                    Permission.SECURITY_VIEW,
                    Permission.REPORT_VIEW
                },
                scope=ResourceScope.RESOURCE_GROUP,
                is_system=True
            ),
            Role(
                id="role-security-admin",
                name="Security Administrator",
                description="Security management access",
                permissions={
                    Permission.SECURITY_VIEW,
                    Permission.SECURITY_INVESTIGATE,
                    Permission.SECURITY_REMEDIATE,
                    Permission.POLICY_VIEW,
                    Permission.POLICY_CREATE,
                    Permission.POLICY_UPDATE,
                    Permission.POLICY_ENFORCE,
                    Permission.COMPLIANCE_VIEW,
                    Permission.COMPLIANCE_EXCEPTION,
                    Permission.ACTION_EXECUTE,
                    Permission.ACTION_APPROVE
                },
                scope=ResourceScope.SUBSCRIPTION,
                is_system=True
            ),
            Role(
                id="role-cost-manager",
                name="Cost Manager",
                description="Cost optimization access",
                permissions={
                    Permission.COST_VIEW,
                    Permission.COST_ANALYZE,
                    Permission.COST_OPTIMIZE,
                    Permission.RESOURCE_VIEW,
                    Permission.REPORT_VIEW,
                    Permission.REPORT_CREATE,
                    Permission.ACTION_EXECUTE
                },
                scope=ResourceScope.SUBSCRIPTION,
                is_system=True
            )
        ]
        
        for role in system_roles:
            self.roles[role.id] = role
    
    async def create_user(
        self,
        email: str,
        name: str,
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user"""
        user_id = f"user-{hashlib.sha256(email.encode()).hexdigest()[:12]}"
        
        user = User(
            id=user_id,
            email=email,
            name=name,
            roles=roles or ["role-viewer"],  # Default to viewer role
            attributes=attributes or {}
        )
        
        self.users[user_id] = user
        
        # Log user creation
        await self._audit_log_entry("user_created", user_id, {"email": email, "name": name})
        
        return user
    
    async def create_role(
        self,
        name: str,
        description: str,
        permissions: Set[Permission],
        scope: ResourceScope = ResourceScope.RESOURCE_GROUP,
        conditions: Optional[Dict[str, Any]] = None
    ) -> Role:
        """Create a custom role"""
        role_id = f"role-custom-{secrets.token_hex(6)}"
        
        role = Role(
            id=role_id,
            name=name,
            description=description,
            permissions=permissions,
            scope=scope,
            conditions=conditions or {},
            is_system=False
        )
        
        self.roles[role_id] = role
        
        # Log role creation
        await self._audit_log_entry("role_created", role_id, {
            "name": name,
            "permissions": [p.value for p in permissions]
        })
        
        return role
    
    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        scope: str,
        assigned_by: str,
        expires_in_days: Optional[int] = None
    ) -> RoleAssignment:
        """Assign a role to a user"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        if role_id not in self.roles:
            raise ValueError(f"Role {role_id} not found")
        
        assignment_id = f"assign-{secrets.token_hex(8)}"
        expires_at = None
        
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        assignment = RoleAssignment(
            id=assignment_id,
            user_id=user_id,
            role_id=role_id,
            scope=scope,
            assigned_by=assigned_by,
            expires_at=expires_at
        )
        
        self.assignments[assignment_id] = assignment
        
        # Add role to user
        user = self.users[user_id]
        if role_id not in user.roles:
            user.roles.append(role_id)
        
        # Clear permission cache for user
        if user_id in self.permission_cache:
            del self.permission_cache[user_id]
        
        # Log assignment
        await self._audit_log_entry("role_assigned", user_id, {
            "role_id": role_id,
            "scope": scope,
            "assigned_by": assigned_by,
            "expires_at": expires_at.isoformat() if expires_at else None
        })
        
        return assignment
    
    async def revoke_role(self, user_id: str, role_id: str, revoked_by: str):
        """Revoke a role from a user"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        if role_id in user.roles:
            user.roles.remove(role_id)
        
        # Remove assignments
        assignments_to_remove = []
        for assign_id, assignment in self.assignments.items():
            if assignment.user_id == user_id and assignment.role_id == role_id:
                assignments_to_remove.append(assign_id)
        
        for assign_id in assignments_to_remove:
            del self.assignments[assign_id]
        
        # Clear permission cache
        if user_id in self.permission_cache:
            del self.permission_cache[user_id]
        
        # Log revocation
        await self._audit_log_entry("role_revoked", user_id, {
            "role_id": role_id,
            "revoked_by": revoked_by
        })
    
    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PermissionCheck:
        """Check if user has permission"""
        if user_id not in self.users:
            return PermissionCheck(
                allowed=False,
                user_id=user_id,
                permission=permission,
                resource=resource,
                reason="User not found"
            )
        
        user = self.users[user_id]
        
        # Check if user is active
        if not user.is_active:
            return PermissionCheck(
                allowed=False,
                user_id=user_id,
                permission=permission,
                resource=resource,
                reason="User is inactive"
            )
        
        # Check direct permissions
        if permission in user.direct_permissions:
            return PermissionCheck(
                allowed=True,
                user_id=user_id,
                permission=permission,
                resource=resource,
                reason="Direct permission granted"
            )
        
        # Check role permissions
        for role_id in user.roles:
            if role_id not in self.roles:
                continue
            
            role = self.roles[role_id]
            
            # Check if role is enabled
            if not role.is_enabled:
                continue
            
            # Check if permission is in role
            if permission in role.permissions:
                # Check scope if resource is provided
                if resource and not self._check_resource_scope(role.scope, resource, context):
                    continue
                
                # Check conditions
                if role.conditions and not self._evaluate_conditions(role.conditions, user, context):
                    continue
                
                # Check assignment expiration
                if not self._check_assignment_validity(user_id, role_id):
                    continue
                
                return PermissionCheck(
                    allowed=True,
                    user_id=user_id,
                    permission=permission,
                    resource=resource,
                    reason=f"Permission granted via role: {role.name}"
                )
        
        return PermissionCheck(
            allowed=False,
            user_id=user_id,
            permission=permission,
            resource=resource,
            reason="Permission denied"
        )
    
    def _check_resource_scope(
        self,
        role_scope: ResourceScope,
        resource: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if resource is within role scope"""
        # Simplified scope checking
        scope_hierarchy = {
            ResourceScope.GLOBAL: 5,
            ResourceScope.ORGANIZATION: 4,
            ResourceScope.SUBSCRIPTION: 3,
            ResourceScope.RESOURCE_GROUP: 2,
            ResourceScope.RESOURCE: 1
        }
        
        # If role has global scope, allow all
        if role_scope == ResourceScope.GLOBAL:
            return True
        
        # Parse resource path (simplified)
        # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/{provider}/{type}/{name}
        if context and "resource_scope" in context:
            resource_scope = ResourceScope(context["resource_scope"])
            return scope_hierarchy[role_scope] >= scope_hierarchy[resource_scope]
        
        return True  # Default allow if scope cannot be determined
    
    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        user: User,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate role conditions"""
        for condition_type, condition_value in conditions.items():
            if condition_type == "time_restriction":
                # Check time-based access
                current_hour = datetime.utcnow().hour
                allowed_hours = condition_value.get("allowed_hours", [])
                if allowed_hours and current_hour not in allowed_hours:
                    return False
            
            elif condition_type == "ip_restriction":
                # Check IP-based access
                if context and "client_ip" in context:
                    allowed_ips = condition_value.get("allowed_ips", [])
                    if allowed_ips and context["client_ip"] not in allowed_ips:
                        return False
            
            elif condition_type == "mfa_required":
                # Check MFA requirement
                if condition_value and not user.mfa_enabled:
                    return False
            
            elif condition_type == "attribute_match":
                # Check user attributes
                for attr_name, attr_value in condition_value.items():
                    if user.attributes.get(attr_name) != attr_value:
                        return False
        
        return True
    
    def _check_assignment_validity(self, user_id: str, role_id: str) -> bool:
        """Check if role assignment is still valid"""
        for assignment in self.assignments.values():
            if assignment.user_id == user_id and assignment.role_id == role_id:
                if assignment.expires_at and assignment.expires_at < datetime.utcnow():
                    return False
                return True
        return True
    
    async def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user"""
        if user_id not in self.users:
            return set()
        
        # Check cache
        if user_id in self.permission_cache:
            cache_entry = self.permission_cache[user_id]
            if cache_entry["expires"] > datetime.utcnow():
                return cache_entry["permissions"]
        
        user = self.users[user_id]
        permissions = set(user.direct_permissions)
        
        # Add permissions from roles
        for role_id in user.roles:
            if role_id in self.roles:
                role = self.roles[role_id]
                if role.is_enabled and self._check_assignment_validity(user_id, role_id):
                    permissions.update(role.permissions)
        
        # Cache permissions for 5 minutes
        self.permission_cache[user_id] = {
            "permissions": permissions,
            "expires": datetime.utcnow() + timedelta(minutes=5)
        }
        
        return permissions
    
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles for a user"""
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        roles = []
        
        for role_id in user.roles:
            if role_id in self.roles:
                role = self.roles[role_id]
                if role.is_enabled and self._check_assignment_validity(user_id, role_id):
                    roles.append(role)
        
        return roles
    
    async def _audit_log_entry(self, action: str, user_id: str, details: Dict[str, Any]):
        """Log an audit entry"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user_id": user_id,
            "details": details
        }
        
        self.audit_log.append(entry)
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        logger.info(f"RBAC Audit: {action} for user {user_id}")
    
    def get_audit_log(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        if user_id:
            logs = [log for log in self.audit_log if log["user_id"] == user_id]
        else:
            logs = self.audit_log
        
        return logs[-limit:]
    
    def export_rbac_config(self) -> Dict[str, Any]:
        """Export RBAC configuration"""
        return {
            "roles": {
                role_id: {
                    "name": role.name,
                    "description": role.description,
                    "permissions": [p.value for p in role.permissions],
                    "scope": role.scope.value,
                    "is_system": role.is_system,
                    "is_enabled": role.is_enabled
                }
                for role_id, role in self.roles.items()
            },
            "users": {
                user_id: {
                    "email": user.email,
                    "name": user.name,
                    "roles": user.roles,
                    "is_active": user.is_active,
                    "mfa_enabled": user.mfa_enabled
                }
                for user_id, user in self.users.items()
            },
            "assignments": {
                assign_id: {
                    "user_id": assignment.user_id,
                    "role_id": assignment.role_id,
                    "scope": assignment.scope,
                    "expires_at": assignment.expires_at.isoformat() if assignment.expires_at else None
                }
                for assign_id, assignment in self.assignments.items()
            }
        }
    
    def import_rbac_config(self, config: Dict[str, Any]):
        """Import RBAC configuration"""
        # Import roles
        for role_id, role_data in config.get("roles", {}).items():
            if role_id not in self.roles or not self.roles[role_id].is_system:
                self.roles[role_id] = Role(
                    id=role_id,
                    name=role_data["name"],
                    description=role_data["description"],
                    permissions={Permission(p) for p in role_data["permissions"]},
                    scope=ResourceScope(role_data["scope"]),
                    is_system=role_data.get("is_system", False),
                    is_enabled=role_data.get("is_enabled", True)
                )
        
        # Import users
        for user_id, user_data in config.get("users", {}).items():
            self.users[user_id] = User(
                id=user_id,
                email=user_data["email"],
                name=user_data["name"],
                roles=user_data["roles"],
                is_active=user_data.get("is_active", True),
                mfa_enabled=user_data.get("mfa_enabled", False)
            )
        
        logger.info("RBAC configuration imported")

# Decorator for permission checking
def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or first arg
            user_id = kwargs.get("user_id")
            if not user_id and len(args) > 0:
                user_id = args[0]
            
            if not user_id:
                raise PermissionError("User ID required for permission check")
            
            # Get RBAC service instance
            rbac = rbac_service
            
            # Check permission
            check = await rbac.check_permission(user_id, permission)
            
            if not check.allowed:
                raise PermissionError(f"Permission denied: {check.reason}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Singleton instance
rbac_service = RBACService()