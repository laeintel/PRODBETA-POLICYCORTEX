// Development authentication bypass
// Only use in development environment for testing

export const DEV_USER = {
  id: 'dev-user-001',
  name: 'Development User',
  email: 'developer@aeolitech.com',
  roles: ['admin'],
  permissions: ['*'],
  tenantId: '9ef5b184-d371-462a-bc75-5024ce8baff7',
  oid: 'dev-object-id-001'
}

export const isDevelopmentBypass = () => {
  return process.env.NODE_ENV === 'development' && 
         (import.meta.env.VITE_DEV_BYPASS_AUTH === 'true' || 
          window.location.search.includes('bypass=true'))
}

export const mockAuthState = {
  isAuthenticated: true,
  user: DEV_USER,
  account: {
    homeAccountId: 'dev-home-account',
    environment: 'development',
    tenantId: DEV_USER.tenantId,
    username: DEV_USER.email,
    localAccountId: DEV_USER.id,
    name: DEV_USER.name,
    idTokenClaims: {
      name: DEV_USER.name,
      email: DEV_USER.email,
      oid: DEV_USER.oid,
      roles: DEV_USER.roles
    }
  },
  inProgress: 'none' as const
}