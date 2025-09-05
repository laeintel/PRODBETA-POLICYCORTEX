import type { Hash } from './artifacts';

export type PolicyEvaluated = {
  type: 'PolicyEvaluated';
  policyId: string;
  timestamp: string;
};

export type PredictionIssued = {
  type: 'PredictionIssued';
  ruleId: string;
  etaDays: number;
  confidence: number;
  repo: string;
  fixBranch: string;
  explanations?: [string, number][];
  timestamp: string;
};

export type FixPrOpened = {
  type: 'FixPrOpened';
  ruleId: string;
  prUrl: string;
  timestamp: string;
};

export type ChainVerified = {
  type: 'ChainVerified';
  hash: Hash;
  verified: boolean;
  merkleRoot: Hash;
  timestamp: string;
};

export type PnlForecasted = {
  type: 'PnlForecasted';
  items: { policy: string; savingsMTD: number; forecast90d: number }[];
  timestamp: string;
};

export type Event =
  | PolicyEvaluated | PredictionIssued | FixPrOpened | ChainVerified | PnlForecasted;