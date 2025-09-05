import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import apiClient, { PredictionData, EvidenceItem, ROIMetrics } from '../lib/api-client';

interface PCGStore {
  // State
  predictions: PredictionData[];
  evidence: EvidenceItem[];
  roiMetrics: ROIMetrics | null;
  isLoading: boolean;
  error: string | null;

  // Actions - Predictions (Prevent)
  fetchPredictions: () => Promise<void>;
  addPrediction: (prediction: Partial<PredictionData>) => Promise<void>;
  
  // Actions - Evidence (Prove)
  fetchEvidence: () => Promise<void>;
  verifyEvidence: (id: string) => Promise<void>;
  addEvidence: (evidence: Partial<EvidenceItem>) => Promise<void>;
  
  // Actions - ROI (Payback)
  fetchROIMetrics: () => Promise<void>;
  calculateROI: (startDate: string, endDate: string) => Promise<void>;
  
  // Utility Actions
  clearError: () => void;
  reset: () => void;
}

const initialState = {
  predictions: [],
  evidence: [],
  roiMetrics: null,
  isLoading: false,
  error: null,
};

export const usePCGStore = create<PCGStore>()(
  persist(
    immer((set, get) => ({
      ...initialState,

      // Predictions (Prevent)
      fetchPredictions: async () => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const predictions = await apiClient.getPredictions();
          set((state) => {
            state.predictions = Array.isArray(predictions) ? predictions : [];
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to fetch predictions';
            state.predictions = [];
            state.isLoading = false;
          });
        }
      },

      addPrediction: async (prediction) => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const newPrediction = await apiClient.createPrediction(prediction);
          set((state) => {
            state.predictions.push(newPrediction);
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to create prediction';
            state.isLoading = false;
          });
        }
      },

      // Evidence (Prove)
      fetchEvidence: async () => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const evidence = await apiClient.getEvidence();
          set((state) => {
            state.evidence = Array.isArray(evidence) ? evidence : [];
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to fetch evidence';
            state.evidence = [];
            state.isLoading = false;
          });
        }
      },

      verifyEvidence: async (id) => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const verifiedEvidence = await apiClient.verifyEvidence(id);
          set((state) => {
            const index = state.evidence.findIndex(e => e.id === id);
            if (index !== -1) {
              state.evidence[index] = verifiedEvidence;
            }
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to verify evidence';
            state.isLoading = false;
          });
        }
      },

      addEvidence: async (evidence) => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const newEvidence = await apiClient.createEvidence(evidence);
          set((state) => {
            state.evidence.push(newEvidence);
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to create evidence';
            state.isLoading = false;
          });
        }
      },

      // ROI (Payback)
      fetchROIMetrics: async () => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const metrics = await apiClient.getROIMetrics();
          set((state) => {
            state.roiMetrics = metrics;
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to fetch ROI metrics';
            state.isLoading = false;
          });
        }
      },

      calculateROI: async (startDate, endDate) => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        try {
          const metrics = await apiClient.calculateROI({ startDate, endDate });
          set((state) => {
            state.roiMetrics = metrics;
            state.isLoading = false;
          });
        } catch (error) {
          set((state) => {
            state.error = error instanceof Error ? error.message : 'Failed to calculate ROI';
            state.isLoading = false;
          });
        }
      },

      // Utility
      clearError: () => {
        set((state) => {
          state.error = null;
        });
      },

      reset: () => {
        set(initialState);
      },
    })),
    {
      name: 'pcg-store',
      partialize: (state) => ({
        predictions: state.predictions,
        evidence: state.evidence,
        roiMetrics: state.roiMetrics,
      }),
    }
  )
);