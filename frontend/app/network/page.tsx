'use client';

import { Card } from '@/components/ui/card';
import { Network, Globe, Shield, Activity, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useEffect, useState } from 'react';

interface NetworkResource {
  id: string;
  name: string;
  type: string;
  status: 'healthy' | 'warning' | 'critical';
  region: string;
  connections: number;
  throughput: string;
  latency: string;
}

export default function NetworkPage() {
  const [resources, setResources] = useState<NetworkResource[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data - replace with real API call
    const mockData: NetworkResource[] = [
      {
        id: 'vnet-001',
        name: 'Production VNet',
        type: 'Virtual Network',
        status: 'healthy',
        region: 'East US',
        connections: 124,
        throughput: '1.2 Gbps',
        latency: '12ms'
      },
      {
        id: 'lb-001',
        name: 'Main Load Balancer',
        type: 'Load Balancer',
        status: 'healthy',
        region: 'East US',
        connections: 892,
        throughput: '450 Mbps',
        latency: '8ms'
      },
      {
        id: 'gw-001',
        name: 'VPN Gateway',
        type: 'VPN Gateway',
        status: 'warning',
        region: 'East US',
        connections: 45,
        throughput: '120 Mbps',
        latency: '45ms'
      },
      {
        id: 'nsg-001',
        name: 'Web NSG',
        type: 'Network Security Group',
        status: 'healthy',
        region: 'East US',  
        connections: 0,
        throughput: 'N/A',
        latency: 'N/A'
      }
    ];

    setTimeout(() => {
      setResources(mockData);
      setLoading(false);
    }, 500);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle2 className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <AlertCircle className="h-5 w-5 text-yellow-500" />;
      case 'critical':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Virtual Network':
        return <Network className="h-5 w-5 text-blue-500" />;
      case 'Load Balancer':
        return <Globe className="h-5 w-5 text-purple-500" />;
      case 'VPN Gateway':
        return <Shield className="h-5 w-5 text-indigo-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Network Resources</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Monitor and manage your cloud network infrastructure
          </p>
        </div>
        <div className="flex gap-4">
          <Card className="px-4 py-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Resources</div>
            <div className="text-2xl font-bold">{resources.length}</div>
          </Card>
          <Card className="px-4 py-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">Active Connections</div>
            <div className="text-2xl font-bold">
              {resources.reduce((sum, r) => sum + r.connections, 0)}
            </div>
          </Card>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4].map(i => (
            <Card key={i} className="p-6 animate-pulse">
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {resources.map(resource => (
            <Card key={resource.id} className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  {getTypeIcon(resource.type)}
                  <div>
                    <h3 className="font-semibold">{resource.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{resource.type}</p>
                  </div>
                </div>
                {getStatusIcon(resource.status)}
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Region:</span>
                  <span className="font-medium">{resource.region}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Connections:</span>
                  <span className="font-medium">{resource.connections}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Throughput:</span>
                  <span className="font-medium">{resource.throughput}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Latency:</span>
                  <span className="font-medium">{resource.latency}</span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between items-center">
                  <span className={`text-sm font-medium ${
                    resource.status === 'healthy' ? 'text-green-600' : 
                    resource.status === 'warning' ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {resource.status.charAt(0).toUpperCase() + resource.status.slice(1)}
                  </span>
                  <button className="text-sm text-blue-600 hover:underline">
                    View Details →
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}