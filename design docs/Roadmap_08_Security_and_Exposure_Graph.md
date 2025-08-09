# 8. Security Exposure Graph

## 8.1 Goal
Correlate RBAC privileges, network reachability, and data assets to surface attack paths and propose control bundles to break them.

## 8.2 Graph Schema
- Nodes: `Identity`, `Role`, `Resource`, `NetworkEndpoint`, `DataStore`
- Edges: `ASSUMES_ROLE(identity→role)`, `HAS_PERMISSION(role→resource, action)`, `REACHABLE(src→dst, port)`, `CONTAINS_DATA(resource→dataStore)`

## 8.3 Signals
- Public exposure (RDP/SSH/HTTP), excessive privileges (Owner/*), missing encryption/backup, lateral paths via flat networks

## 8.4 Path Scoring
- `score = w1*privilege_weight + w2*reachability_weight + w3*data_sensitivity + w4*control_gaps`
- Weights from knowledge base; normalize 0–1; highlight >0.7 as critical

## 8.5 Mitigation Bundles
- NSG/SG deny rules + private endpoints + shrink role scopes + enable encryption/backup

## 8.6 Actions
- “Fix path” via orchestrator: preflight plan lists all changes with blast‑radius estimates; approvals required for critical

## 8.7 KPIs
- Reduction in reachable critical data assets; time to neutralize path; number of stale admin roles

## 8.8 Example Path
- Identity: `svc-deploy` assumes `Contributor` at subscription
- Network: NSG allows `0.0.0.0/0 -> 3389` to VM `vm-app`
- Data: VM mounts storage account `stappdata` without firewall

Score: privilege 0.6 + reachability 0.8 + data sensitivity 0.7 + gaps 0.5 → 0.65 (critical threshold 0.7; borderline)

Mitigation bundle:
- NSG: deny inbound 3389 from internet; enable Azure Bastion
- RBAC: reduce `svc-deploy` to `Virtual Machine Contributor` scoped to RG
- Storage: enable firewall + private endpoint; enforce encryption with CMK

Estimated residual risk: < 0.3
