# 19. Postgres Migrations & Seed Data

## 19.1 Migration Strategy
- Tooling: SQLx/Flyway; semantic versioning; forward‑only preferred
- Environments: dev/staging/prod with change approvals

## 19.2 Example Migration Files
```
V1_0_0__init.sql
  -- creates tenants, users, resources, policies, violations, recommendations

V1_1_0__actions_and_events.sql
  -- creates actions, action_events; adds indexes

V1_2_0__exceptions_and_audit.sql
  -- creates exceptions, audits; RLS policies per tenant
```

## 19.3 Seed Data (Dev)
```sql
insert into tenants(id,name) values ('00000000-0000-0000-0000-000000000001','Demo Tenant');
insert into users(id,tenant_id,upn,role) values (
  '00000000-0000-0000-0000-0000000000aa','00000000-0000-0000-0000-000000000001','admin@demo.com','admin'
);
insert into resources(id,tenant_id,provider,type,name,location,tags,status) values (
  '/subscriptions/demo/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stprod',
  '00000000-0000-0000-0000-000000000001','azure','Microsoft.Storage/storageAccounts','stprod','eastus','{"Environment":"Production"}','Compliant'
);
insert into policies(id,tenant_id,name,provider,type,effect,scope,status) values (
  '/providers/Microsoft.Authorization/policyDefinitions/require-encryption',
  '00000000-0000-0000-0000-000000000001','Require Encryption','azure','BuiltIn','Deny','/subscriptions/demo','Active'
);
```

## 19.4 Developer Convenience
- Makefile tasks: `db-up`, `db-migrate`, `db-seed`, `db-reset`
- Docker compose service for Postgres with mounted migrations

## 19.5 RLS (Sketch)
```sql
alter table resources enable row level security;
create policy tenant_isolation on resources
  using (tenant_id = current_setting('app.tenant_id')::uuid);
```
