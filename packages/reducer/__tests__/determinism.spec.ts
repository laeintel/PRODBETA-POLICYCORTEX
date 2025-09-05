import { describe, expect, it } from 'vitest';
import { initialState, replay } from '../src/reducer';
import type { Event } from '../../types/src/events';

const sample: Event[] = [
  {
    type: 'PredictionIssued',
    ruleId: 'AZ-NSG-OPEN-22',
    etaDays: 7,
    confidence: 0.82,
    repo: 'org/infrastructure',
    fixBranch: 'pcx/autofix/AZ-NSG-OPEN-22',
    explanations: [['nsg_open_ports', 0.61], ['subnet_public', 0.21]],
    timestamp: '2025-09-05T00:00:00Z',
  },
  {
    type: 'FixPrOpened',
    ruleId: 'AZ-NSG-OPEN-22',
    prUrl: 'https://github.com/org/infrastructure/pull/123',
    timestamp: '2025-09-05T00:05:00Z',
  },
  {
    type: 'ChainVerified',
    hash: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    verified: true,
    merkleRoot: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    timestamp: '2025-09-05T00:10:00Z',
  },
  {
    type: 'PnlForecasted',
    items: [{ policy: 'AZ-NET-INGRESS', savingsMTD: 1234.56, forecast90d: 9876.54 }],
    timestamp: '2025-09-05T00:15:00Z',
  },
];

describe('Reducer determinism', () => {
  it('replay is deterministic (same input → same output)', () => {
    const s1 = replay(sample, initialState);
    const s2 = replay(sample, initialState);
    expect(s1).toStrictEqual(s2);
  });
});