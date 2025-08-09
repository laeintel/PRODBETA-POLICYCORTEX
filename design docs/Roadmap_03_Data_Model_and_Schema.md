# 3. Data Model & Storage Schema

Primary stores: Postgres (relational state/audit), Redis (queues/cache), ClickHouse/Timescale (metrics), Object store (evidence).

## 3.1 Entities (Postgres)

### 3.1.1 Tables (DDL)
```sql
-- Tenancy
create table tenants (
  id uuid primary key,
  name text not null,
  created_at timestamptz not null default now()
);

create table users (
  id uuid primary key,
  tenant_id uuid not null references tenants(id) on delete cascade,
  upn text not null,
  role text not null check (role in ('viewer','operator','approver','admin')),
  created_at timestamptz not null default now(),
  unique(tenant_id, upn)
);

-- Catalogs
create table resources (
  id text primary key, -- cloud resource id
  tenant_id uuid not null references tenants(id) on delete cascade,
  provider text not null check (provider in ('azure','aws','gcp','ibm')),
  type text not null,
  name text not null,
  group_name text,
  location text,
  tags jsonb not null default '{}',
  status text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index on resources(tenant_id, updated_at desc);
create index on resources(tenant_id, provider, type);
create index resources_tags_gin on resources using gin(tags);

create table policies (
  id text primary key, -- provider policy id or generated id
  tenant_id uuid not null references tenants(id) on delete cascade,
  name text not null,
  provider text not null,
  type text not null,
  effect text,
  scope text,
  status text not null,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index on policies(tenant_id, status);

-- Findings / Recommendations
create table violations (
  id uuid primary key,
  tenant_id uuid not null references tenants(id) on delete cascade,
  policy_id text references policies(id) on delete set null,
  resource_id text references resources(id) on delete cascade,
  severity text not null check (severity in ('low','medium','high','critical')),
  reason text not null,
  first_seen timestamptz not null default now(),
  last_seen timestamptz not null default now(),
  status text not null default 'open' check (status in ('open','suppressed','fixed'))
);
create index on violations(tenant_id, status, severity);
create index on violations(tenant_id, resource_id);

create table recommendations (
  id uuid primary key,
  tenant_id uuid not null references tenants(id) on delete cascade,
  kind text not null, -- cost_optimization, compliance, security, rightsizing
  severity text not null,
  title text not null,
  description text,
  savings numeric,
  risk_reduction numeric,
  confidence numeric,
  created_at timestamptz not null default now()
);
create index on recommendations(tenant_id, kind, severity);

-- Actions & Orchestration
create table actions (
  id uuid primary key,
  tenant_id uuid not null references tenants(id) on delete cascade,
  type text not null,
  resource_id text,
  requested_by uuid references users(id),
  status text not null check (status in ('queued','preflight','in_progress','verifying','completed','failed','rolled_back')),
  params jsonb not null default '{}',
  result jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index on actions(tenant_id, status, updated_at desc);

create table action_events (
  id bigserial primary key,
  action_id uuid not null references actions(id) on delete cascade,
  ts timestamptz not null default now(),
  message text not null
);
create index on action_events(action_id, ts);

create table exceptions (
  id uuid primary key,
  tenant_id uuid not null references tenants(id) on delete cascade,
  policy_id text references policies(id) on delete cascade,
  resource_id text references resources(id) on delete cascade,
  reason text not null,
  expires_at timestamptz,
  created_by uuid references users(id),
  status text not null default 'approved' check (status in ('pending','approved','rejected','expired')),
  created_at timestamptz not null default now()
);
create index on exceptions(tenant_id, status, expires_at);

-- Audit trail
create table audits (
  id uuid primary key,
  tenant_id uuid not null references tenants(id) on delete cascade,
  actor text not null,
  action text not null,
  target_id text,
  target_type text,
  payload jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create index on audits(tenant_id, created_at desc);
```

### 3.1.2 Constraints & Policies
- Row‑level security (RLS) by `tenant_id` for all tenant tables
- Foreign keys cascade appropriately; soft delete optional via status columns

## 3.2 Metrics (ClickHouse/Timescale)
```sql
-- Timescale example
create table cost_timeseries (
  tenant_id uuid not null,
  service text not null,
  resource_id text,
  day date not null,
  amount numeric not null,
  primary key (tenant_id, service, day)
);
select create_hypertable('cost_timeseries','day');
create index on cost_timeseries(tenant_id, service, day desc);
```

## 3.3 Evidence Store (Object Storage)
- Path conventions: `/tenants/{tenant}/evidence/{action_id}/{artifact}`
- Artifact manifest stored in `actions.result` with checksums

## 3.4 Queues (Redis / NATS subjects)
- `actions.execute`, `scans.policy`, `scans.finops`, `scans.security`
- Payload envelope: `{ id, tenant_id, type, params, idempotency_key, created_at }`

## 3.5 Sample Queries
```sql
-- Latest open violations per resource
select v.* from violations v
join (
  select resource_id, max(last_seen) as last_seen
  from violations where status='open' and tenant_id=$1
  group by resource_id
) x on x.resource_id = v.resource_id and x.last_seen = v.last_seen
where v.tenant_id = $1
order by v.severity desc;

-- Actions needing approval
select * from actions where tenant_id=$1 and status='preflight' and (params->>'requires_approval')='true';
```

## 3.6 Migrations & Versioning
- Use SQL migrations (e.g., `sqlx migrate` / Flyway)
- Semantic version for schema (`schema_version` table)
```sql
create table schema_version (version text primary key, applied_at timestamptz default now());
insert into schema_version(version) values ('1.0.0');
```

## 3.7 Retention & Archival
- Hot data (30–90 days) in primary tables; archive closed violations/events to partitioned history tables quarterly
- Evidence artifacts retained per‑framework policy (e.g., 1–3 years)
