/**
 * Role-Based Access Control (RBAC) Permission System
 * Defines roles, permissions, and authorization checks
 */

// Permission definitions
export enum Permission {
  // Resource permissions
  RESOURCE_READ = 'resource:read',
  RESOURCE_CREATE = 'resource:create',
  RESOURCE_UPDATE = 'resource:update',
  RESOURCE_DELETE = 'resource:delete',
  
  // Policy permissions
  POLICY_READ = 'policy:read',
  POLICY_CREATE = 'policy:create',
  POLICY_UPDATE = 'policy:update',
  POLICY_DELETE = 'policy:delete',
  POLICY_ENFORCE = 'policy:enforce',
  
  // User management permissions
  USER_READ = 'user:read',
  USER_CREATE = 'user:create',
  USER_UPDATE = 'user:update',
  USER_DELETE = 'user:delete',
  USER_ROLE_ASSIGN = 'user:role:assign',
  
  // RBAC permissions
  RBAC_READ = 'rbac:read',
  RBAC_MANAGE = 'rbac:manage',
  RBAC_AUDIT = 'rbac:audit',
  
  // Compliance permissions
  COMPLIANCE_READ = 'compliance:read',
  COMPLIANCE_REPORT = 'compliance:report',
  COMPLIANCE_REMEDIATE = 'compliance:remediate',
  COMPLIANCE_APPROVE = 'compliance:approve',
  
  // Security permissions
  SECURITY_READ = 'security:read',
  SECURITY_ALERT = 'security:alert',
  SECURITY_INVESTIGATE = 'security:investigate',
  SECURITY_REMEDIATE = 'security:remediate',
  
  // Cost management permissions
  COST_READ = 'cost:read',
  COST_OPTIMIZE = 'cost:optimize',
  COST_APPROVE = 'cost:approve',
  
  // AI/ML permissions
  AI_READ = 'ai:read',
  AI_TRAIN = 'ai:train',
  AI_PREDICT = 'ai:predict',
  AI_FEEDBACK = 'ai:feedback',
  
  // Export permissions
  EXPORT_DATA = 'export:data',
  EXPORT_REPORT = 'export:report',
  EXPORT_AUDIT = 'export:audit',
  
  // Admin permissions
  ADMIN_FULL = 'admin:full',
  ADMIN_SETTINGS = 'admin:settings',
  ADMIN_AUDIT = 'admin:audit',
}

// Role definitions with assigned permissions
export enum Role {
  SUPER_ADMIN = 'super_admin',
  ADMIN = 'admin',
  SECURITY_ADMIN = 'security_admin',
  COMPLIANCE_OFFICER = 'compliance_officer',
  COST_ANALYST = 'cost_analyst',
  DEVELOPER = 'developer',
  OPERATOR = 'operator',
  AUDITOR = 'auditor',
  VIEWER = 'viewer',
}

// Role to permissions mapping
export const rolePermissions: Record<Role, Permission[]> = {
  [Role.SUPER_ADMIN]: [
    Permission.ADMIN_FULL,
    // Super admin has all permissions
  ],
  
  [Role.ADMIN]: [
    Permission.RESOURCE_READ,
    Permission.RESOURCE_CREATE,
    Permission.RESOURCE_UPDATE,
    Permission.RESOURCE_DELETE,
    Permission.POLICY_READ,
    Permission.POLICY_CREATE,
    Permission.POLICY_UPDATE,
    Permission.POLICY_DELETE,
    Permission.POLICY_ENFORCE,
    Permission.USER_READ,
    Permission.USER_CREATE,
    Permission.USER_UPDATE,
    Permission.USER_DELETE,
    Permission.USER_ROLE_ASSIGN,
    Permission.RBAC_READ,
    Permission.RBAC_MANAGE,
    Permission.COMPLIANCE_READ,
    Permission.COMPLIANCE_REPORT,
    Permission.COMPLIANCE_REMEDIATE,
    Permission.COMPLIANCE_APPROVE,
    Permission.SECURITY_READ,
    Permission.SECURITY_ALERT,
    Permission.SECURITY_INVESTIGATE,
    Permission.SECURITY_REMEDIATE,
    Permission.COST_READ,
    Permission.COST_OPTIMIZE,
    Permission.COST_APPROVE,
    Permission.AI_READ,
    Permission.AI_PREDICT,
    Permission.EXPORT_DATA,
    Permission.EXPORT_REPORT,
    Permission.ADMIN_SETTINGS,
  ],
  
  [Role.SECURITY_ADMIN]: [
    Permission.RESOURCE_READ,
    Permission.POLICY_READ,
    Permission.POLICY_CREATE,
    Permission.POLICY_UPDATE,
    Permission.POLICY_ENFORCE,
    Permission.USER_READ,
    Permission.RBAC_READ,
    Permission.RBAC_MANAGE,
    Permission.RBAC_AUDIT,
    Permission.SECURITY_READ,
    Permission.SECURITY_ALERT,
    Permission.SECURITY_INVESTIGATE,
    Permission.SECURITY_REMEDIATE,
    Permission.COMPLIANCE_READ,
    Permission.AI_READ,
    Permission.AI_PREDICT,
    Permission.EXPORT_DATA,
    Permission.EXPORT_AUDIT,
  ],
  
  [Role.COMPLIANCE_OFFICER]: [
    Permission.RESOURCE_READ,
    Permission.POLICY_READ,
    Permission.POLICY_CREATE,
    Permission.POLICY_UPDATE,
    Permission.COMPLIANCE_READ,
    Permission.COMPLIANCE_REPORT,
    Permission.COMPLIANCE_REMEDIATE,
    Permission.COMPLIANCE_APPROVE,
    Permission.SECURITY_READ,
    Permission.RBAC_READ,
    Permission.AI_READ,
    Permission.AI_PREDICT,
    Permission.EXPORT_DATA,
    Permission.EXPORT_REPORT,
    Permission.EXPORT_AUDIT,
  ],
  
  [Role.COST_ANALYST]: [
    Permission.RESOURCE_READ,
    Permission.COST_READ,
    Permission.COST_OPTIMIZE,
    Permission.COMPLIANCE_READ,
    Permission.AI_READ,
    Permission.AI_PREDICT,
    Permission.EXPORT_DATA,
    Permission.EXPORT_REPORT,
  ],
  
  [Role.DEVELOPER]: [
    Permission.RESOURCE_READ,
    Permission.RESOURCE_CREATE,
    Permission.RESOURCE_UPDATE,
    Permission.POLICY_READ,
    Permission.COMPLIANCE_READ,
    Permission.SECURITY_READ,
    Permission.COST_READ,
    Permission.AI_READ,
    Permission.AI_PREDICT,
    Permission.AI_FEEDBACK,
    Permission.EXPORT_DATA,
  ],
  
  [Role.OPERATOR]: [
    Permission.RESOURCE_READ,
    Permission.RESOURCE_UPDATE,
    Permission.POLICY_READ,
    Permission.COMPLIANCE_READ,
    Permission.SECURITY_READ,
    Permission.COST_READ,
    Permission.AI_READ,
    Permission.EXPORT_DATA,
  ],
  
  [Role.AUDITOR]: [
    Permission.RESOURCE_READ,
    Permission.POLICY_READ,
    Permission.USER_READ,
    Permission.RBAC_READ,
    Permission.RBAC_AUDIT,
    Permission.COMPLIANCE_READ,
    Permission.COMPLIANCE_REPORT,
    Permission.SECURITY_READ,
    Permission.COST_READ,
    Permission.EXPORT_AUDIT,
    Permission.ADMIN_AUDIT,
  ],
  
  [Role.VIEWER]: [
    Permission.RESOURCE_READ,
    Permission.POLICY_READ,
    Permission.COMPLIANCE_READ,
    Permission.SECURITY_READ,
    Permission.COST_READ,
    Permission.AI_READ,
  ],
};

/**
 * Check if a user has a specific permission
 */
export function hasPermission(
  userRoles: Role[] | string[],
  permission: Permission
): boolean {
  // Super admin has all permissions
  if (userRoles.includes(Role.SUPER_ADMIN)) {
    return true;
  }
  
  // Check if any of the user's roles have the permission
  for (const role of userRoles) {
    const permissions = rolePermissions[role as Role];
    if (permissions && permissions.includes(permission)) {
      return true;
    }
  }
  
  return false;
}

/**
 * Check if a user has any of the specified permissions
 */
export function hasAnyPermission(
  userRoles: Role[] | string[],
  permissions: Permission[]
): boolean {
  return permissions.some(permission => hasPermission(userRoles, permission));
}

/**
 * Check if a user has all of the specified permissions
 */
export function hasAllPermissions(
  userRoles: Role[] | string[],
  permissions: Permission[]
): boolean {
  return permissions.every(permission => hasPermission(userRoles, permission));
}

/**
 * Get all permissions for a set of roles
 */
export function getPermissionsForRoles(roles: Role[] | string[]): Permission[] {
  const permissions = new Set<Permission>();
  
  // Super admin gets all permissions
  if (roles.includes(Role.SUPER_ADMIN)) {
    return Object.values(Permission);
  }
  
  // Accumulate permissions from all roles
  for (const role of roles) {
    const rolePerms = rolePermissions[role as Role];
    if (rolePerms) {
      rolePerms.forEach(perm => permissions.add(perm));
    }
  }
  
  return Array.from(permissions);
}

/**
 * Middleware helper to check permissions in API routes
 */
export function requirePermission(
  userRoles: Role[] | string[],
  permission: Permission
): void {
  if (!hasPermission(userRoles, permission)) {
    throw new Error(`Insufficient permissions. Required: ${permission}`);
  }
}

/**
 * Middleware helper to check any of the permissions in API routes
 */
export function requireAnyPermission(
  userRoles: Role[] | string[],
  permissions: Permission[]
): void {
  if (!hasAnyPermission(userRoles, permissions)) {
    throw new Error(`Insufficient permissions. Required one of: ${permissions.join(', ')}`);
  }
}

/**
 * Format permission for display
 */
export function formatPermission(permission: Permission): string {
  return permission
    .replace(':', ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Format role for display
 */
export function formatRole(role: Role): string {
  return role
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}