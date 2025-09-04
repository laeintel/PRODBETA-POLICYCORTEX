import { apiClient } from '@/lib/api-client';

// Mock fetch globally
global.fetch = jest.fn();

describe('ApiClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Clear the cache before each test
    (apiClient as any).cache.clear();
    // Reset CSRF token
    (apiClient as any).csrfToken = null;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Initialization', () => {
    it('initializes with correct base URL', () => {
      expect((apiClient as any).baseUrl).toBe('http://localhost:8000');
    });

    it('initializes with empty cache', () => {
      expect((apiClient as any).cache.size).toBe(0);
    });

    it('attempts to initialize CSRF token on construction', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ csrfToken: 'test-csrf-token' })
      } as Response);

      // Wait for async initialization
      await new Promise(resolve => setTimeout(resolve, 100));
      
      expect(mockFetch).toHaveBeenCalledWith('/api/auth/csrf');
    });
  });

  describe('Request Method', () => {
    it('makes GET request with correct headers', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: 'test' })
      } as Response);

      await (apiClient as any).request('/test');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/test',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
    });

    it('includes CSRF token in POST requests', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      (apiClient as any).csrfToken = 'test-csrf-token';
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: 'test' })
      } as Response);

      await (apiClient as any).request('/test', {
        method: 'POST',
        body: { test: 'data' }
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/test',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-CSRF-Token': 'test-csrf-token'
          })
        })
      );
    });

    it('throws error for non-ok responses', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Bad Request',
        text: async () => 'Error message'
      } as Response);

      await expect((apiClient as any).request('/test')).rejects.toThrow('Bad Request');
    });

    it('handles network errors', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect((apiClient as any).request('/test')).rejects.toThrow('Network error');
    });

    it('caches GET requests', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const testData = { data: 'test' };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => testData
      } as Response);

      // First request
      const result1 = await (apiClient as any).request('/test');
      expect(result1).toEqual(testData);
      expect(mockFetch).toHaveBeenCalledTimes(1);

      // Second request should use cache
      const result2 = await (apiClient as any).request('/test');
      expect(result2).toEqual(testData);
      expect(mockFetch).toHaveBeenCalledTimes(1); // Still only 1 call
    });

    it('does not cache non-GET requests', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const testData = { data: 'test' };
      
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => testData
      } as Response);

      // First POST request
      await (apiClient as any).request('/test', { method: 'POST' });
      expect(mockFetch).toHaveBeenCalledTimes(1);

      // Second POST request should not use cache
      await (apiClient as any).request('/test', { method: 'POST' });
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('respects cache timeout', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'test' })
      } as Response);

      // First request
      await (apiClient as any).request('/test');
      expect(mockFetch).toHaveBeenCalledTimes(1);

      // Mock expired cache
      const cache = (apiClient as any).cache;
      const cachedEntry = cache.get('GET:/test');
      if (cachedEntry) {
        cachedEntry.timestamp = Date.now() - 31000; // Expired
      }

      // Second request should fetch again
      await (apiClient as any).request('/test');
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  describe('Dashboard Methods', () => {
    it('fetches dashboard metrics', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockMetrics = {
        totalResources: 100,
        compliance: 95,
        risks: 5,
        costs: 10000
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockMetrics
      } as Response);

      const result = await apiClient.getDashboardMetrics();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/dashboard/metrics'),
        expect.any(Object)
      );
      expect(result).toEqual(mockMetrics);
    });

    it('fetches alerts with pagination', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockAlerts = [
        { id: '1', severity: 'high', title: 'Alert 1' }
      ];
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockAlerts
      } as Response);

      const result = await apiClient.getAlerts(1, 20);
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/alerts?page=1&limit=20'),
        expect.any(Object)
      );
      expect(result).toEqual(mockAlerts);
    });

    it('fetches recent activities', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockActivities = [
        { id: '1', type: 'resource_created', timestamp: '2024-01-01' }
      ];
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockActivities
      } as Response);

      const result = await apiClient.getRecentActivities();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/activities/recent'),
        expect.any(Object)
      );
      expect(result).toEqual(mockActivities);
    });
  });

  describe('Compliance Methods', () => {
    it('fetches compliance status', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockStatus = {
        overallScore: 95,
        policies: 100,
        violations: 5
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockStatus
      } as Response);

      const result = await apiClient.getComplianceStatus();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/compliance/status'),
        expect.any(Object)
      );
      expect(result).toEqual(mockStatus);
    });

    it('fetches policy violations', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockViolations = [
        { id: '1', policyId: 'pol-1', severity: 'high' }
      ];
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockViolations
      } as Response);

      const result = await apiClient.getPolicyViolations();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/compliance/violations'),
        expect.any(Object)
      );
      expect(result).toEqual(mockViolations);
    });

    it('remediates violation', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockResponse = { success: true };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await apiClient.remediateViolation('violation-1');
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/compliance/violations/violation-1/remediate'),
        expect.objectContaining({
          method: 'POST'
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Resource Management', () => {
    it('fetches resources with filters', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockResources = [
        { id: '1', name: 'Resource 1', type: 'vm' }
      ];
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResources
      } as Response);

      const result = await apiClient.getResources(1, 20, { type: 'vm' });
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/resources?page=1&limit=20&type=vm'),
        expect.any(Object)
      );
      expect(result).toEqual(mockResources);
    });

    it('fetches single resource by ID', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockResource = { id: '1', name: 'Resource 1' };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResource
      } as Response);

      const result = await apiClient.getResource('1');
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/resources/1'),
        expect.any(Object)
      );
      expect(result).toEqual(mockResource);
    });

    it('updates resource', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const updateData = { name: 'Updated Resource' };
      const mockResponse = { id: '1', ...updateData };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await apiClient.updateResource('1', updateData);
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/resources/1'),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify(updateData)
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('deletes resource', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockResponse = { success: true };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await apiClient.deleteResource('1');
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/resources/1'),
        expect.objectContaining({
          method: 'DELETE'
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Patent Features', () => {
    it('fetches predictions (Patent #4)', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockPredictions = {
        predictions: [
          { resourceId: '1', riskScore: 0.75, prediction: 'drift' }
        ]
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockPredictions
      } as Response);

      const result = await apiClient.getPredictions();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/predictions'),
        expect.any(Object)
      );
      expect(result).toEqual(mockPredictions);
    });

    it('fetches correlations (Patent #1)', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockCorrelations = {
        correlations: [
          { domain1: 'security', domain2: 'compliance', strength: 0.85 }
        ]
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockCorrelations
      } as Response);

      const result = await apiClient.getCorrelations();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/correlations'),
        expect.any(Object)
      );
      expect(result).toEqual(mockCorrelations);
    });

    it('sends chat message (Patent #2)', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockResponse = {
        response: 'AI response',
        intent: 'query',
        confidence: 0.95
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await apiClient.sendChatMessage('Test message');
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/conversation'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ message: 'Test message' })
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('fetches unified metrics (Patent #3)', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      const mockMetrics = {
        governance_score: 95,
        risk_score: 25,
        compliance_score: 98
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockMetrics
      } as Response);

      const result = await apiClient.getUnifiedMetrics();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/metrics'),
        expect.any(Object)
      );
      expect(result).toEqual(mockMetrics);
    });
  });

  describe('Error Handling', () => {
    it('handles 404 errors', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: async () => 'Resource not found'
      } as Response);

      await expect(apiClient.getResource('nonexistent')).rejects.toThrow('Not Found');
    });

    it('handles 401 unauthorized errors', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        text: async () => 'Unauthorized'
      } as Response);

      await expect(apiClient.getDashboardMetrics()).rejects.toThrow('Unauthorized');
    });

    it('handles 500 server errors', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: async () => 'Server error'
      } as Response);

      await expect(apiClient.getDashboardMetrics()).rejects.toThrow('Internal Server Error');
    });

    it('handles JSON parse errors', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => {
          throw new Error('Invalid JSON');
        }
      } as Response);

      await expect(apiClient.getDashboardMetrics()).rejects.toThrow('Invalid JSON');
    });
  });

  describe('Cache Management', () => {
    it('clears cache on demand', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'test' })
      } as Response);

      // Make a request to populate cache
      await (apiClient as any).request('/test');
      expect((apiClient as any).cache.size).toBe(1);

      // Clear cache
      apiClient.clearCache();
      expect((apiClient as any).cache.size).toBe(0);
    });

    it('uses different cache keys for different endpoints', async () => {
      const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'test' })
      } as Response);

      await (apiClient as any).request('/test1');
      await (apiClient as any).request('/test2');

      expect((apiClient as any).cache.size).toBe(2);
      expect((apiClient as any).cache.has('GET:/test1')).toBe(true);
      expect((apiClient as any).cache.has('GET:/test2')).toBe(true);
    });
  });
});