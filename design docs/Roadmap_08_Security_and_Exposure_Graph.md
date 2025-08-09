# 8. Security Exposure Graph

## 8.1 Goal
Correlate RBAC privileges, network reachability, and data assets to surface attack paths and propose control bundles to break them.

## 8.2 Graph Model
- Nodes: identities, roles, resources, NSGs/SGs, endpoints, data stores
- Edges: permission grants, network flows, data access

## 8.3 Signals
- Public exposure (RDP/SSH/HTTP), excessive privileges, missing encryption/backup, lateral paths

## 8.4 Outputs
- Ranked attack paths with mitigation bundles (NSG rules, identity tightening, policy blocks)

## 8.5 Actions
- “Fix path” one‑click execution via orchestrator with preflight analysis

## 8.6 KPIs
- Reduction in reachable critical data assets; time to neutralize path
