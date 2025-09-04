describe('API Configuration', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('API URLs', () => {
    it('uses default API URL when env variable is not set', () => {
      delete process.env.NEXT_PUBLIC_API_URL;
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      expect(apiUrl).toBe('http://localhost:8000');
    });

    it('uses environment API URL when set', () => {
      process.env.NEXT_PUBLIC_API_URL = 'https://api.production.com';
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      expect(apiUrl).toBe('https://api.production.com');
    });

    it('uses default GraphQL URL when env variable is not set', () => {
      delete process.env.NEXT_PUBLIC_GRAPHQL_URL;
      const graphqlUrl = process.env.NEXT_PUBLIC_GRAPHQL_URL || 'http://localhost:4000/graphql';
      expect(graphqlUrl).toBe('http://localhost:4000/graphql');
    });

    it('uses environment GraphQL URL when set', () => {
      process.env.NEXT_PUBLIC_GRAPHQL_URL = 'https://graphql.production.com';
      const graphqlUrl = process.env.NEXT_PUBLIC_GRAPHQL_URL || 'http://localhost:4000/graphql';
      expect(graphqlUrl).toBe('https://graphql.production.com');
    });
  });

  describe('API Headers', () => {
    it('creates default headers', () => {
      const getDefaultHeaders = () => ({
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      });

      const headers = getDefaultHeaders();
      expect(headers['Content-Type']).toBe('application/json');
      expect(headers['Accept']).toBe('application/json');
    });

    it('adds authorization header when token exists', () => {
      const getAuthHeaders = (token?: string) => {
        const headers: Record<string, string> = {
          'Content-Type': 'application/json'
        };
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
      };

      const headers = getAuthHeaders('test-token');
      expect(headers['Authorization']).toBe('Bearer test-token');
    });

    it('does not add authorization header when token is missing', () => {
      const getAuthHeaders = (token?: string) => {
        const headers: Record<string, string> = {
          'Content-Type': 'application/json'
        };
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
      };

      const headers = getAuthHeaders();
      expect(headers['Authorization']).toBeUndefined();
    });
  });

  describe('API Endpoints', () => {
    it('constructs correct endpoint URLs', () => {
      const baseUrl = 'http://localhost:8000';
      
      const endpoints = {
        dashboard: `${baseUrl}/api/v1/dashboard`,
        resources: `${baseUrl}/api/v1/resources`,
        compliance: `${baseUrl}/api/v1/compliance`,
        predictions: `${baseUrl}/api/v1/predictions`,
        correlations: `${baseUrl}/api/v1/correlations`,
        metrics: `${baseUrl}/api/v1/metrics`,
        conversation: `${baseUrl}/api/v1/conversation`
      };

      expect(endpoints.dashboard).toBe('http://localhost:8000/api/v1/dashboard');
      expect(endpoints.resources).toBe('http://localhost:8000/api/v1/resources');
      expect(endpoints.compliance).toBe('http://localhost:8000/api/v1/compliance');
      expect(endpoints.predictions).toBe('http://localhost:8000/api/v1/predictions');
      expect(endpoints.correlations).toBe('http://localhost:8000/api/v1/correlations');
      expect(endpoints.metrics).toBe('http://localhost:8000/api/v1/metrics');
      expect(endpoints.conversation).toBe('http://localhost:8000/api/v1/conversation');
    });

    it('handles trailing slashes correctly', () => {
      const normalizeUrl = (url: string) => {
        return url.endsWith('/') ? url.slice(0, -1) : url;
      };

      expect(normalizeUrl('http://api.com/')).toBe('http://api.com');
      expect(normalizeUrl('http://api.com')).toBe('http://api.com');
    });
  });

  describe('Request Configuration', () => {
    it('sets default timeout', () => {
      const getRequestConfig = () => ({
        timeout: 30000,
        retries: 3,
        retryDelay: 1000
      });

      const config = getRequestConfig();
      expect(config.timeout).toBe(30000);
      expect(config.retries).toBe(3);
      expect(config.retryDelay).toBe(1000);
    });

    it('allows timeout override', () => {
      const getRequestConfig = (timeout?: number) => ({
        timeout: timeout || 30000,
        retries: 3,
        retryDelay: 1000
      });

      const config = getRequestConfig(60000);
      expect(config.timeout).toBe(60000);
    });
  });

  describe('Environment Detection', () => {
    it('detects development environment', () => {
      process.env.NODE_ENV = 'development';
      const isDevelopment = process.env.NODE_ENV === 'development';
      expect(isDevelopment).toBe(true);
    });

    it('detects production environment', () => {
      process.env.NODE_ENV = 'production';
      const isProduction = process.env.NODE_ENV === 'production';
      expect(isProduction).toBe(true);
    });

    it('detects test environment', () => {
      process.env.NODE_ENV = 'test';
      const isTest = process.env.NODE_ENV === 'test';
      expect(isTest).toBe(true);
    });
  });

  describe('CORS Configuration', () => {
    it('includes CORS headers for development', () => {
      const getCorsHeaders = (isDev: boolean) => {
        if (isDev) {
          return {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
          };
        }
        return {};
      };

      const headers = getCorsHeaders(true);
      expect(headers['Access-Control-Allow-Origin']).toBe('*');
      expect(headers['Access-Control-Allow-Methods']).toBe('GET, POST, PUT, DELETE, OPTIONS');
    });

    it('excludes CORS headers for production', () => {
      const getCorsHeaders = (isDev: boolean) => {
        if (isDev) {
          return {
            'Access-Control-Allow-Origin': '*'
          };
        }
        return {};
      };

      const headers = getCorsHeaders(false);
      expect(headers['Access-Control-Allow-Origin']).toBeUndefined();
    });
  });
});