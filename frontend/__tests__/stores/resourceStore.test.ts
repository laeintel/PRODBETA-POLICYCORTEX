import { renderHook, act, waitFor } from '@testing-library/react';
import { useResourceStore } from '../../stores/resourceStore';
import { apiClient } from '../../lib/api-client';

// Mock the API client
jest.mock('../../lib/api-client', () => ({
  apiClient: {
    getResources: jest.fn(),
    getCorrelations: jest.fn(),
    updateResource: jest.fn(),
    deleteResource: jest.fn(),
    getPredictions: jest.fn(),
    getUnifiedMetrics: jest.fn()
  }
}));

// Mock axios
jest.mock('axios', () => ({
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn()
}));

describe('ResourceStore', () => {
  const mockApiClient = apiClient as jest.Mocked<typeof apiClient>;

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset store state
    useResourceStore.setState({
      resources: [],
      selectedResource: null,
      filters: {},
      loading: false,
      error: null,
      viewMode: 'cards',
      search: '',
      sortBy: 'name',
      sortOrder: 'asc',
      autoRefresh: false,
      refreshInterval: 30000,
      correlations: [],
      predictions: [],
      metrics: null
    });
  });

  describe('Initial State', () => {
    it('initializes with correct default values', () => {
      const { result } = renderHook(() => useResourceStore());
      
      expect(result.current.resources).toEqual([]);
      expect(result.current.selectedResource).toBeNull();
      expect(result.current.filters).toEqual({});
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeNull();
      expect(result.current.viewMode).toBe('cards');
      expect(result.current.search).toBe('');
      expect(result.current.sortBy).toBe('name');
      expect(result.current.sortOrder).toBe('asc');
      expect(result.current.autoRefresh).toBe(false);
      expect(result.current.refreshInterval).toBe(30000);
    });
  });

  describe('Resource Management', () => {
    const mockResources = [
      {
        id: 'res-001',
        name: 'VM-Production-01',
        display_name: 'Production VM 1',
        resource_type: 'Virtual Machine',
        category: 'ComputeStorage' as const,
        location: 'East US',
        tags: { environment: 'production' },
        status: {
          state: 'running',
          provisioning_state: 'succeeded',
          availability: 99.9,
          performance_score: 85
        },
        health: {
          status: 'Healthy' as const,
          issues: [],
          recommendations: []
        },
        compliance_status: {
          is_compliant: true,
          compliance_score: 95,
          violations: [],
          last_scan: '2024-01-01T00:00:00Z'
        },
        metadata: {
          created_by: 'admin',
          created_at: '2024-01-01T00:00:00Z',
          modified_by: 'admin',
          modified_at: '2024-01-01T00:00:00Z',
          subscription_id: 'sub-001',
          resource_group: 'rg-prod'
        }
      },
      {
        id: 'res-002',
        name: 'VM-Dev-01',
        display_name: 'Development VM 1',
        resource_type: 'Virtual Machine',
        category: 'ComputeStorage' as const,
        location: 'West US',
        tags: { environment: 'dev' },
        status: {
          state: 'stopped',
          provisioning_state: 'succeeded',
          availability: 95,
          performance_score: 75
        },
        health: {
          status: 'Degraded' as const,
          issues: [{
            severity: 'Medium' as const,
            title: 'High CPU',
            description: 'CPU usage above 80%',
            affected_components: ['CPU'],
            mitigation: 'Scale up VM size'
          }],
          recommendations: ['Consider scaling up']
        },
        compliance_status: {
          is_compliant: false,
          compliance_score: 70,
          violations: [{
            policy_id: 'pol-001',
            policy_name: 'Security baseline',
            severity: 'High' as const,
            description: 'Missing security patches',
            remediation: 'Apply latest patches',
            affected_resources: ['res-002'],
            detected_at: '2024-01-01T00:00:00Z'
          }],
          last_scan: '2024-01-01T00:00:00Z'
        },
        metadata: {
          created_by: 'developer',
          created_at: '2024-01-01T00:00:00Z',
          modified_by: 'developer',
          modified_at: '2024-01-01T00:00:00Z',
          subscription_id: 'sub-001',
          resource_group: 'rg-dev'
        }
      }
    ];

    it('fetches resources successfully', async () => {
      mockApiClient.getResources.mockResolvedValueOnce(mockResources);
      
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        await result.current.fetchResources();
      });
      
      expect(mockApiClient.getResources).toHaveBeenCalledWith(1, 100, {});
      expect(result.current.resources).toEqual(mockResources);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('handles resource fetch errors', async () => {
      const error = new Error('Failed to fetch resources');
      mockApiClient.getResources.mockRejectedValueOnce(error);
      
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        await result.current.fetchResources();
      });
      
      expect(result.current.resources).toEqual([]);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe('Failed to fetch resources');
    });

    it('selects and deselects resources', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setSelectedResource(mockResources[0]);
      });
      
      expect(result.current.selectedResource).toEqual(mockResources[0]);
      
      act(() => {
        result.current.setSelectedResource(null);
      });
      
      expect(result.current.selectedResource).toBeNull();
    });

    it('updates a resource', async () => {
      const updatedResource = { ...mockResources[0], name: 'Updated-VM' };
      mockApiClient.updateResource.mockResolvedValueOnce(updatedResource);
      
      const { result } = renderHook(() => useResourceStore());
      
      // Set initial resources
      act(() => {
        result.current.setResources(mockResources);
      });
      
      await act(async () => {
        await result.current.updateResource('res-001', { name: 'Updated-VM' });
      });
      
      expect(mockApiClient.updateResource).toHaveBeenCalledWith('res-001', { name: 'Updated-VM' });
      expect(result.current.resources[0].name).toBe('Updated-VM');
    });

    it('deletes a resource', async () => {
      mockApiClient.deleteResource.mockResolvedValueOnce({ success: true });
      
      const { result } = renderHook(() => useResourceStore());
      
      // Set initial resources
      act(() => {
        result.current.setResources(mockResources);
      });
      
      await act(async () => {
        await result.current.deleteResource('res-001');
      });
      
      expect(mockApiClient.deleteResource).toHaveBeenCalledWith('res-001');
      expect(result.current.resources.length).toBe(1);
      expect(result.current.resources[0].id).toBe('res-002');
    });
  });

  describe('Filtering and Sorting', () => {
    it('applies filters correctly', async () => {
      const { result } = renderHook(() => useResourceStore());
      const filters = {
        resource_type: ['Virtual Machine'],
        compliance_status: 'compliant',
        location: ['East US']
      };
      
      mockApiClient.getResources.mockResolvedValueOnce([]);
      
      await act(async () => {
        result.current.setFilters(filters);
        await result.current.fetchResources();
      });
      
      expect(mockApiClient.getResources).toHaveBeenCalledWith(1, 100, filters);
      expect(result.current.filters).toEqual(filters);
    });

    it('clears filters', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setFilters({ resource_type: ['VM'] });
        result.current.clearFilters();
      });
      
      expect(result.current.filters).toEqual({});
    });

    it('sets and clears search term', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setSearch('production');
      });
      
      expect(result.current.search).toBe('production');
      
      act(() => {
        result.current.clearSearch();
      });
      
      expect(result.current.search).toBe('');
    });

    it('changes sort configuration', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setSortBy('compliance_score');
        result.current.setSortOrder('desc');
      });
      
      expect(result.current.sortBy).toBe('compliance_score');
      expect(result.current.sortOrder).toBe('desc');
    });

    it('toggles sort order', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.toggleSortOrder();
      });
      
      expect(result.current.sortOrder).toBe('desc');
      
      act(() => {
        result.current.toggleSortOrder();
      });
      
      expect(result.current.sortOrder).toBe('asc');
    });
  });

  describe('View Management', () => {
    it('changes view mode between cards and visualizations', () => {
      const { result } = renderHook(() => useResourceStore());
      
      expect(result.current.viewMode).toBe('cards');
      
      act(() => {
        result.current.setViewMode('visualizations');
      });
      
      expect(result.current.viewMode).toBe('visualizations');
      
      act(() => {
        result.current.toggleViewMode();
      });
      
      expect(result.current.viewMode).toBe('cards');
    });
  });

  describe('Auto Refresh', () => {
    jest.useFakeTimers();

    it('enables and disables auto-refresh', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setAutoRefresh(true);
      });
      
      expect(result.current.autoRefresh).toBe(true);
      
      act(() => {
        result.current.setAutoRefresh(false);
      });
      
      expect(result.current.autoRefresh).toBe(false);
    });

    it('sets refresh interval', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setRefreshInterval(60000);
      });
      
      expect(result.current.refreshInterval).toBe(60000);
    });

    it('toggles auto-refresh', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.toggleAutoRefresh();
      });
      
      expect(result.current.autoRefresh).toBe(true);
      
      act(() => {
        result.current.toggleAutoRefresh();
      });
      
      expect(result.current.autoRefresh).toBe(false);
    });

    jest.useRealTimers();
  });

  describe('Patent Feature Integration', () => {
    it('fetches correlations (Patent #1)', async () => {
      const mockCorrelations = [
        {
          id: 'corr-001',
          source_domain: 'security',
          target_domain: 'compliance',
          correlation_strength: 0.85,
          pattern_type: 'anomaly',
          confidence: 0.92
        }
      ];
      
      mockApiClient.getCorrelations.mockResolvedValueOnce({ correlations: mockCorrelations });
      
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        await result.current.fetchCorrelations();
      });
      
      expect(mockApiClient.getCorrelations).toHaveBeenCalled();
      expect(result.current.correlations).toEqual(mockCorrelations);
    });

    it('fetches predictions (Patent #4)', async () => {
      const mockPredictions = [
        {
          resource_id: 'res-001',
          prediction_type: 'compliance_drift',
          risk_score: 0.75,
          confidence: 0.88,
          predicted_time: '2024-02-01T00:00:00Z'
        }
      ];
      
      mockApiClient.getPredictions.mockResolvedValueOnce({ predictions: mockPredictions });
      
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        await result.current.fetchPredictions();
      });
      
      expect(mockApiClient.getPredictions).toHaveBeenCalled();
      expect(result.current.predictions).toEqual(mockPredictions);
    });

    it('fetches unified metrics (Patent #3)', async () => {
      const mockMetrics = {
        governance_score: 95,
        risk_score: 25,
        compliance_score: 98,
        cost_optimization: 85,
        security_posture: 92
      };
      
      mockApiClient.getUnifiedMetrics.mockResolvedValueOnce(mockMetrics);
      
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        await result.current.fetchUnifiedMetrics();
      });
      
      expect(mockApiClient.getUnifiedMetrics).toHaveBeenCalled();
      expect(result.current.metrics).toEqual(mockMetrics);
    });
  });

  describe('Error Handling', () => {
    it('sets and clears errors', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setError('Test error message');
      });
      
      expect(result.current.error).toBe('Test error message');
      
      act(() => {
        result.current.clearError();
      });
      
      expect(result.current.error).toBeNull();
    });

    it('handles network errors gracefully', async () => {
      mockApiClient.getResources.mockRejectedValueOnce(new Error('Network error'));
      
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        await result.current.fetchResources();
      });
      
      expect(result.current.error).toBe('Network error');
      expect(result.current.resources).toEqual([]);
    });
  });

  describe('Batch Operations', () => {
    it('sets multiple resources at once', () => {
      const { result } = renderHook(() => useResourceStore());
      const resources = [
        { id: '1', name: 'Resource 1' },
        { id: '2', name: 'Resource 2' }
      ];
      
      act(() => {
        result.current.setResources(resources as any);
      });
      
      expect(result.current.resources).toEqual(resources);
    });

    it('clears all resources', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setResources([{ id: '1', name: 'Resource 1' }] as any);
        result.current.clearResources();
      });
      
      expect(result.current.resources).toEqual([]);
    });
  });

  describe('Loading States', () => {
    it('manages loading state during operations', async () => {
      mockApiClient.getResources.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve([]), 100))
      );
      
      const { result } = renderHook(() => useResourceStore());
      
      const fetchPromise = act(async () => {
        await result.current.fetchResources();
      });
      
      // Check loading is true while fetching
      expect(result.current.loading).toBe(true);
      
      await fetchPromise;
      
      // Check loading is false after fetching
      expect(result.current.loading).toBe(false);
    });

    it('sets loading state manually', () => {
      const { result } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setLoading(true);
      });
      
      expect(result.current.loading).toBe(true);
      
      act(() => {
        result.current.setLoading(false);
      });
      
      expect(result.current.loading).toBe(false);
    });
  });

  describe('Store Persistence', () => {
    it('persists selected filters across renders', () => {
      const { result, rerender } = renderHook(() => useResourceStore());
      
      act(() => {
        result.current.setFilters({ location: ['East US'] });
      });
      
      rerender();
      
      expect(result.current.filters).toEqual({ location: ['East US'] });
    });

    it('maintains state consistency during concurrent updates', async () => {
      const { result } = renderHook(() => useResourceStore());
      
      await act(async () => {
        // Simulate concurrent updates
        result.current.setLoading(true);
        result.current.setSearch('test');
        result.current.setViewMode('visualizations');
      });
      
      expect(result.current.loading).toBe(true);
      expect(result.current.search).toBe('test');
      expect(result.current.viewMode).toBe('visualizations');
    });
  });
});