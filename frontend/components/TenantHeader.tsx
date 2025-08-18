'use client';

import React, { useState, useEffect } from 'react';
import { Building2, ChevronDown } from 'lucide-react';

interface Tenant {
  id: string;
  name: string;
  tier: 'free' | 'pro' | 'enterprise';
  region: string;
}

const demoTenants: Tenant[] = [
  { id: 'org-1', name: 'Contoso Corporation', tier: 'enterprise', region: 'East US' },
  { id: 'org-2', name: 'Fabrikam Industries', tier: 'pro', region: 'West Europe' },
  { id: 'org-3', name: 'Adventure Works', tier: 'free', region: 'Southeast Asia' }
];

export const TenantHeader: React.FC = () => {
  const [selectedTenant, setSelectedTenant] = useState<Tenant>(demoTenants[0]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    // Check for tenant in localStorage or URL params
    const savedTenantId = localStorage.getItem('selected-tenant-id');
    if (savedTenantId) {
      const tenant = demoTenants.find(t => t.id === savedTenantId);
      if (tenant) setSelectedTenant(tenant);
    }
  }, []);

  const handleTenantSwitch = (tenant: Tenant) => {
    setSelectedTenant(tenant);
    localStorage.setItem('selected-tenant-id', tenant.id);
    setIsDropdownOpen(false);
    
    // Trigger a custom event for other components to react to tenant change
    window.dispatchEvent(new CustomEvent('tenant-switched', { detail: tenant }));
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'enterprise':
        return 'bg-gradient-to-r from-yellow-400 to-orange-400';
      case 'pro':
        return 'bg-gradient-to-r from-blue-400 to-purple-400';
      default:
        return 'bg-gradient-to-r from-gray-400 to-gray-500';
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        className="flex items-center gap-3 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-all border border-white/20"
      >
        <Building2 className="w-4 h-4 text-white" />
        <div className="text-left">
          <div className="text-sm font-semibold text-white">{selectedTenant.name}</div>
          <div className="text-xs text-gray-300 flex items-center gap-2">
            <span>ID: {selectedTenant.id}</span>
            <span className={`px-2 py-0.5 rounded-full text-xs font-semibold text-white ${getTierColor(selectedTenant.tier)}`}>
              {selectedTenant.tier.toUpperCase()}
            </span>
          </div>
        </div>
        <ChevronDown className={`w-4 h-4 text-white transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
      </button>

      {isDropdownOpen && (
        <div className="absolute top-full mt-2 w-full min-w-[300px] bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50">
          <div className="p-2">
            <div className="text-xs text-gray-400 px-3 py-2 uppercase tracking-wider">
              Switch Tenant
            </div>
            {demoTenants.map((tenant) => (
              <button
                key={tenant.id}
                onClick={() => handleTenantSwitch(tenant)}
                className={`w-full text-left px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors ${
                  selectedTenant.id === tenant.id ? 'bg-gray-800' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-white">{tenant.name}</div>
                    <div className="text-xs text-gray-400">
                      {tenant.id} • {tenant.region}
                    </div>
                  </div>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-semibold text-white ${getTierColor(tenant.tier)}`}>
                    {tenant.tier.toUpperCase()}
                  </span>
                </div>
              </button>
            ))}
          </div>
          <div className="border-t border-gray-700 p-3">
            <div className="text-xs text-gray-400">
              Demo Mode: Showing seeded tenants. In production, this would reflect your actual Azure tenants.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Hook to use current tenant in other components
export const useCurrentTenant = () => {
  const [tenant, setTenant] = useState<Tenant>(demoTenants[0]);

  useEffect(() => {
    const savedTenantId = localStorage.getItem('selected-tenant-id');
    if (savedTenantId) {
      const foundTenant = demoTenants.find(t => t.id === savedTenantId);
      if (foundTenant) setTenant(foundTenant);
    }

    const handleTenantSwitch = (e: CustomEvent) => {
      setTenant(e.detail);
    };

    window.addEventListener('tenant-switched', handleTenantSwitch as any);
    return () => window.removeEventListener('tenant-switched', handleTenantSwitch as any);
  }, []);

  return tenant;
};