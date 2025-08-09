# 16. Digital Twin & Dry‑Run Simulation

## 16.1 Purpose
Safely preview changes before execution by computing diffs against a cached digital twin of the tenant environment, with blast‑radius scoring and validation checks.

## 16.2 Twin Sources
- Cached resource inventory (normalized across clouds)
- Effective policy set (including inheritance/assignments/exceptions)
- Cost and utilization snapshots

## 16.3 Diff Envelope (JSON)
```json
{
  "plan_id": "ae6f...",
  "tenant_id": "...",
  "target": {"action_type": "enable_encryption", "resource_id": "/subscriptions/.../storageAccounts/st1"},
  "changes": [
    {
      "resource_id": "/subscriptions/.../storageAccounts/st1",
      "provider": "azure",
      "type": "Microsoft.Storage/storageAccounts",
      "before": {"encryption": {"enabled": false, "mode": null}},
      "after":  {"encryption": {"enabled": true,  "mode": "platformManaged"}},
      "validation": {"policy_conflicts": [], "dependencies": []}
    }
  ],
  "blast_radius": {
    "resource_count": 1,
    "estimated_cost_delta": 0.0,
    "risk_tier": "low"
  },
  "guards": {
    "requires_approval": false,
    "blocked_reasons": []
  },
  "evidence": {
    "generated_at": "2025-01-10T12:00:00Z",
    "artifacts": [
      {"type": "plan-json", "path": "s3://.../plan.json", "checksum": "sha256:..."}
    ]
  }
}
```

## 16.4 Blast‑Radius Scoring
- Resource count touched (including dependencies)
- Environment sensitivity (prod/test) and tags
- Cost delta and rollback difficulty
- Control gaps (e.g., missing backup/encryption)

## 16.5 Validation Checks
- Policy conflicts; required tags; location constraints
- Access guardrails (PIM/JIT required?)
- Dependency resolution (e.g., key vault availability)

## 16.6 Approval Matrix
- Example tiers
  - Low: auto‑approve
  - Medium: operator + approver
  - High/Critical: multi‑party with break‑glass reason

## 16.7 Evidence Artifacts
- Plan JSON, human summary (markdown)
- Pre/post screenshots/exports where applicable
- Signed manifest with checksums

## 16.8 Failure Handling
- If guard denies: return `blocked_reasons` and suggested mitigations
- If validation fails: propose corrective steps before retry
