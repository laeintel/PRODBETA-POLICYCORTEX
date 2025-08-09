# 3. Data Model & Storage Schema

Primary stores: Postgres (relational state/audit), Redis (queues/cache), ClickHouse/Timescale (metrics), Object store (evidence).

## 3.1 Entities (Postgres)
- tenants(id, name, created_at)
- users(id, tenant_id, upn, role, created_at)
- resources(id, tenant_id, provider, type, name, group_name, location, tags jsonb, status, created_at, updated_at)
- policies(id, tenant_id, name, provider, type, effect, scope, status, metadata jsonb, created_at, updated_at)
- violations(id, tenant_id, policy_id, resource_id, severity, reason, first_seen, last_seen, status)
- recommendations(id, tenant_id, kind, severity, title, description, savings numeric, risk_reduction numeric, confidence numeric, created_at)
- actions(id, tenant_id, type, resource_id, requested_by, status, params jsonb, result jsonb, created_at, updated_at)
- action_events(id serial, action_id, ts timestamptz, message)
- exceptions(id, tenant_id, policy_id, resource_id, reason, expires_at, created_by, status, created_at)
- audits(id, tenant_id, actor, action, target_id, target_type, payload jsonb, created_at)

Indexes: (tenant_id, updated_at) on hot tables; GIN on tags/metadata; partial on status.

## 3.2 Metrics (ClickHouse/Timescale)
- cost_timeseries(tenant_id, service, resource_id, day, amount)
- compliance_timeseries(tenant_id, control, ts, score)
- security_events(tenant_id, ts, kind, severity, attributes)

## 3.3 Evidence Store (Object Storage)
- /tenants/{tenant}/evidence/{action_id}/{artifact}
- /tenants/{tenant}/reports/{framework}/{period}.zip

## 3.4 Queues (Redis / NATS subjects)
- actions.execute
- scans.policy
- scans.finops
- scans.security

## 3.5 Migrations & Versioning
- Liquibase/Flyway/SQLx migrations; semantic versioned schema; backward compatible API projections.
