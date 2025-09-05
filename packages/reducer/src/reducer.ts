import type { Event } from '../../types/src/events';
import type { Hash } from '../../types/src/artifacts';

export interface ExecutiveState {
  predictions: Record<string, { etaDays: number; confidence: number; repo: string; fixBranch: string; explanations?: [string, number][] }>;
  prByRule: Record<string, string>; // ruleId -> prUrl
  chain: Record<Hash, { verified: boolean; merkleRoot: Hash }>;
  pnl: Record<string, { savingsMTD: number; forecast90d: number }>;
}

export const initialState: ExecutiveState = {
  predictions: {},
  prByRule: {},
  chain: {},
  pnl: {},
};

function clone<T>(x: T): T {
  // deterministic deep clone without relying on structuredClone availability
  return JSON.parse(JSON.stringify(x));
}

export function reduce(state: ExecutiveState, event: Event): ExecutiveState {
  const s = clone(state);
  switch (event.type) {
    case 'PredictionIssued':
      s.predictions[event.ruleId] = {
        etaDays: event.etaDays,
        confidence: event.confidence,
        repo: event.repo,
        fixBranch: event.fixBranch,
        explanations: event.explanations,
      };
      return s;
    case 'FixPrOpened':
      s.prByRule[event.ruleId] = event.prUrl;
      return s;
    case 'ChainVerified':
      s.chain[event.hash] = { verified: event.verified, merkleRoot: event.merkleRoot };
      return s;
    case 'PnlForecasted':
      for (const item of event.items) {
        s.pnl[item.policy] = { savingsMTD: item.savingsMTD, forecast90d: item.forecast90d };
      }
      return s;
    case 'PolicyEvaluated':
    default:
      return s;
  }
}

export function replay(events: Event[], start: ExecutiveState = initialState): ExecutiveState {
  let s = start;
  for (const e of events) s = reduce(s, e);
  return s;
}