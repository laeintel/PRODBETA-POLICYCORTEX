'use client'

import { useEffect } from 'react';
import { usePCGStore } from '@/stores/resourceStore';
import { CheckCircle, Shield, FileText, Clock } from 'lucide-react';

export default function ProvePage() {
  const { evidence, isLoading, error, fetchEvidence, verifyEvidence } = usePCGStore();

  useEffect(() => {
    fetchEvidence();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Error: {error}</p>
      </div>
    );
  }

  const verifiedCount = evidence.filter(e => e.status === 'verified').length;
  const pendingCount = evidence.filter(e => e.status === 'pending').length;
  const verificationRate = evidence.length > 0 
    ? Math.round((verifiedCount / evidence.length) * 100)
    : 0;

  const handleVerify = async (id: string) => {
    await verifyEvidence(id);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Prove</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Immutable evidence chain and audit trail verification
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Evidence</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                {evidence.length}
              </p>
            </div>
            <FileText className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Verified</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">
                {verifiedCount}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Pending</p>
              <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400 mt-1">
                {pendingCount}
              </p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Verification Rate</p>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">
                {verificationRate}%
              </p>
            </div>
            <Shield className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Evidence Chain */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Evidence Chain</h2>
        </div>
        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {evidence.slice(0, 10).map((item) => (
            <div key={item.id} className="px-6 py-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      item.status === 'verified' 
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : item.status === 'pending'
                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {item.status.toUpperCase()}
                    </span>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      {item.type}
                    </span>
                  </div>
                  <p className="mt-2 text-gray-900 dark:text-white">{item.description}</p>
                  <div className="mt-2 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                    <span>Hash: {item.hash.substring(0, 16)}...</span>
                    <span>By: {item.verifiedBy}</span>
                    <span>{new Date(item.timestamp).toLocaleDateString()}</span>
                  </div>
                </div>
                {item.status === 'pending' && (
                  <button
                    onClick={() => handleVerify(item.id)}
                    className="ml-4 px-3 py-1 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Verify
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
        {evidence.length === 0 && (
          <div className="px-6 py-8 text-center text-gray-500 dark:text-gray-400">
            No evidence items available
          </div>
        )}
      </div>
    </div>
  );
}