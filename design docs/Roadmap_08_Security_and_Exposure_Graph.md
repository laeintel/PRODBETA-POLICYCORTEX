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
