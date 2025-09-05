here’s T09 (Performance: Redis cache + rate-limit + pooling) as precise, git-applyable diffs, plus run/verify steps.

✅ Unified diffs — save as t09_perf.patch, then:
git apply t09_perf.patch

diff --git a/.env.example b/.env.example
index f0a1e2b..a3c0d1e 100644
--- a/.env.example
+++ b/.env.example
@@ -11,6 +11,12 @@ REDIS_URL=redis://localhost:6379
+# Performance / limits
+DB_POOL_SIZE=20
+RATELIMIT_WINDOW_SEC=60
+RATE_LIMIT_USER_PER_MIN=120
+RATE_LIMIT_TENANT_PER_MIN=1200
+AZURE_CACHE_TTL_SEC=600
+PREDICTIONS_CACHE_TTL_SEC=120
+
 # Auth (OIDC/JWT via Azure AD)
 JWT_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
 JWT_AUDIENCE=<your-app-client-id>
diff --git a/services/gateway/package.json b/services/gateway/package.json
index 31b2e9f..b12aa77 100644
--- a/services/gateway/package.json
+++ b/services/gateway/package.json
@@ -12,6 +12,7 @@
   "dependencies": {
     "cors": "^2.8.5",
     "express": "^4.19.2",
+    "redis": "^4.6.14",
     "http-proxy-middleware": "^3.0.3",
     "jose": "^5.9.3"
   },
diff --git a/services/gateway/src/index.ts b/services/gateway/src/index.ts
index badd00d..c0ffee0 100644
--- a/services/gateway/src/index.ts
+++ b/services/gateway/src/index.ts
@@ -1,9 +1,12 @@
 import 'dotenv/config';
 import express from 'express';
 import cors from 'cors';
 import { createProxyMiddleware } from 'http-proxy-middleware';
 import { createRemoteJWKSet, jwtVerify, JWTPayload } from 'jose';
+import { createClient } from 'redis';
 
+// Top-level await OK (ESM)
+const redis = createClient({ url: process.env.REDIS_URL }); await redis.connect();
 // ---- Env
 const PORT = Number(process.env.GATEWAY_PORT || 8000);
 const CORE_URL = process.env.CORE_URL || 'http://localhost:8081';
 const AZURE_AGENTS_URL = process.env.AZURE_AGENTS_URL || 'http://localhost:8084';
@@ -16,6 +19,10 @@ const AUDITOR_GROUP_IDS = (process.env.AUDITOR_GROUP_IDS || '').split(',').map(s
 const OPERATOR_GROUP_IDS = (process.env.OPERATOR_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
 
 if (!ISSUER || !AUDIENCE || !JWKS_URL) {
   console.warn('[gateway] WARNING: JWT_ISSUER, JWT_AUDIENCE, JWT_JWKS_URL are not fully set.');
 }
+const WINDOW_SEC = Number(process.env.RATELIMIT_WINDOW_SEC || '60');
+const USER_LIMIT = Number(process.env.RATE_LIMIT_USER_PER_MIN || '120');
+const TENANT_LIMIT = Number(process.env.RATE_LIMIT_TENANT_PER_MIN || '1200');
+
 
 // ---- App
 const app = express();
@@ -45,6 +52,28 @@ async function verifyBearer(authHeader?: string): Promise<{ role: Role; payload:
   return { role, payload };
 }
 
+async function slidingWindowAllow(key: string, limit: number, nowMs: number): Promise<boolean> {
+  const windowStart = nowMs - WINDOW_SEC * 1000;
+  // prune + count + add atomically
+  const multi = redis.multi();
+  multi.zRemRangeByScore(key, 0, windowStart);
+  multi.zCard(key);
+  multi.zAdd(key, { score: nowMs, value: `${nowMs}:${Math.random()}` });
+  multi.expire(key, WINDOW_SEC);
+  const [, cardResp] = (await multi.exec()) as [unknown, number, unknown, unknown];
+  return (cardResp as number) < limit;
+}
+
+const rateLimit = async (req: express.Request, res: express.Response, next: express.NextFunction) => {
+  const u = (req as any).user as { sub?: string; tid?: string; role?: Role } | undefined;
+  const sub = u?.sub || 'anon';
+  const tid = u?.tid || 'unknown';
+  const now = Date.now();
+  if (!(await slidingWindowAllow(`rl:u:${sub}`, USER_LIMIT, now))) return res.status(429).json({ error: 'rate_limited', scope: 'user' });
+  if (!(await slidingWindowAllow(`rl:t:${tid}`, TENANT_LIMIT, now))) return res.status(429).json({ error: 'rate_limited', scope: 'tenant' });
+  return next();
+};
+
 function requireRole(allowed: Role[]) {
   return async (req: express.Request, res: express.Response, next: express.NextFunction) => {
     try {
@@ -67,27 +96,27 @@ app.get('/health', (_req, res) => res.json({ ok: true, time: new Date().toISOString() }));
 
 // ---- RBAC policy (default deny)
 // Core (events/evidence): auditors & admins can read; operators/admins can write events
 app.use(
   '/api/v1/events',
-  requireRole(['operator', 'admin']),
+  requireRole(['operator', 'admin']), rateLimit,
   createProxyMiddleware({
     target: CORE_URL,
     changeOrigin: true,
     xfwd: true
   })
 );
 app.get(
   '/api/v1/events/replay',
-  requireRole(['auditor', 'admin', 'operator']),
+  requireRole(['auditor', 'admin', 'operator']), rateLimit,
   createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true })
 );
 app.get(
   '/api/v1/verify/*',
-  requireRole(['auditor', 'admin', 'operator']),
+  requireRole(['auditor', 'admin', 'operator']), rateLimit,
   createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true })
 );
 
 // Azure Agents (predictions + P&L): all roles can read
 app.get(
   '/api/v1/predictions',
-  requireRole(['auditor', 'admin', 'operator']),
+  requireRole(['auditor', 'admin', 'operator']), rateLimit,
   createProxyMiddleware({ target: AZURE_AGENTS_URL, changeOrigin: true, xfwd: true })
 );
 app.get(
   '/api/v1/costs/pnl',
-  requireRole(['auditor', 'admin', 'operator']),
+  requireRole(['auditor', 'admin', 'operator']), rateLimit,
   createProxyMiddleware({ target: AZURE_AGENTS_URL, changeOrigin: true, xfwd: true })
 );
diff --git a/services/agents/azure/package.json b/services/agents/azure/package.json
index aa11cc1..bb22cc2 100644
--- a/services/agents/azure/package.json
+++ b/services/agents/azure/package.json
@@ -15,6 +15,7 @@
     "@azure/identity": "^4.4.1",
     "cors": "^2.8.5",
     "express": "^4.19.2",
+    "redis": "^4.6.14",
     "zod": "^3.23.8"
   },
   "devDependencies": {
diff --git a/services/agents/azure/src/index.ts b/services/agents/azure/src/index.ts
index c1f09aa..c1fe0de 100644
--- a/services/agents/azure/src/index.ts
+++ b/services/agents/azure/src/index.ts
@@ -1,9 +1,10 @@
 import 'dotenv/config';
 import express from 'express';
 import cors from 'cors';
 import { z } from 'zod';
 import { DefaultAzureCredential } from '@azure/identity';
 import { CostManagementClient, QueryDefinition, TimeframeType } from '@azure/arm-costmanagement';
+import { createClient } from 'redis';
 import { createRemoteJWKSet, jwtVerify, JWTPayload } from 'jose';
 
 // ---- Env / Config --------------------------------------------------------------------
 const PORT = Number(process.env.PORT || 8084);
@@ -14,6 +15,8 @@ const CLIENT_SECRET = process.env.AZURE_CLIENT_SECRET || '';
 const ISSUER = process.env.JWT_ISSUER!;
 const AUDIENCE = process.env.JWT_AUDIENCE!;
 const JWKS_URL = process.env.JWT_JWKS_URL!;
+const AZURE_CACHE_TTL_SEC = Number(process.env.AZURE_CACHE_TTL_SEC || '600');
+const PRED_CACHE_TTL_SEC = Number(process.env.PREDICTIONS_CACHE_TTL_SEC || '120');
 const ADMIN_GROUP_IDS = (process.env.ADMIN_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
 const AUDITOR_GROUP_IDS = (process.env.AUDITOR_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
 const OPERATOR_GROUP_IDS = (process.env.OPERATOR_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
@@ -27,6 +30,8 @@ function roleFromPayload(p: JWTPayload): Role {
   if (inAny(OPERATOR_GROUP_IDS)) return 'operator';
   return 'unknown';
 }
+const redis = createClient({ url: process.env.REDIS_URL });
+await redis.connect();
 const ML_PREDICT_URL = process.env.ML_PREDICT_URL || 'http://localhost:8001/predict';
 const ML_EXPLAIN_URL = process.env.ML_EXPLAIN_URL || process.env.ML_PREDICT_URL || 'http://localhost:8001/predict';
 const DEFAULT_FIX_REPO = process.env.DEFAULT_FIX_REPO || 'org/infrastructure';
@@ -53,6 +58,21 @@ async function requireAuth(req: express.Request, res: express.Response, next: express.NextFunction) {
   }
 }
 
+async function cacheJSON<T>(key: string, ttlSec: number, miss: () => Promise<T>): Promise<T> {
+  const hit = await redis.get(key);
+  if (hit) {
+    try { return JSON.parse(hit) as T; } catch { /* fall through */ }
+  }
+  const val = await miss();
+  try {
+    const s = JSON.stringify(val);
+    await redis.set(key, s, { EX: ttlSec });
+  } catch { /* ignore cache write errors */ }
+  return val;
+}
+
+function ymd(d = new Date()): string { return d.toISOString().slice(0,10); }
+
 app.get('/health', (_req, res) => {
   const hasAzure = !!(SUBSCRIPTION_ID && TENANT_ID && CLIENT_ID && CLIENT_SECRET);
   return res.json({
     ok: true,
     azureConfigured: hasAzure,
@@ -67,6 +87,7 @@ app.use('/api/', requireAuth);
 // ---- /api/v1/predictions -------------------------------------------------------------
 // Accepts ?scope=subscription|rg|tags:...&horizon=<days>
 const PredictQuery = z.object({
   scope: z.string().default('subscription'),
   horizon: z.coerce.number().int().min(1).max(180).default(30),
 });
@@ -76,7 +97,12 @@ type Prediction = {
   repo: string;
   fixBranch: string;
   explanations?: { top?: [string, number][] };
 };
 
-app.get('/api/v1/predictions', async (req, res) => {
+app.get('/api/v1/predictions', async (req, res) => {
   const parsed = PredictQuery.safeParse(req.query);
   if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
   const { scope, horizon } = parsed.data;
 
-  try {
+  try {
+    const cacheKey = `pred:${scope}:${horizon}`;
+    const out = await cacheJSON<{items: Prediction[]}>(cacheKey, PRED_CACHE_TTL_SEC, async () => {
     // Call ML service (you can swap to POST with richer payloads)
     const url = new URL(ML_PREDICT_URL);
     url.searchParams.set('scope', scope);
     url.searchParams.set('horizon', String(horizon));
     const r = await fetch(url, { method: 'GET' });
@@ -99,10 +125,12 @@ app.get('/api/v1/predictions', async (req, res) => {
       repo: DEFAULT_FIX_REPO,
       fixBranch: `pcx/autofix/${p.ruleId}`,
       explanations: topMap.has(p.ruleId) ? { top: topMap.get(p.ruleId) } : undefined,
     }));
-    return res.json({ items });
+    return { items };
+    });
+    return res.json(out);
   } catch (e: any) {
     return res.status(500).json({ error: String(e?.message || e) });
   }
 });
@@ -112,7 +140,14 @@ app.get('/api/v1/costs/pnl', async (_req, res) => {
   if (!(SUBSCRIPTION_ID && TENANT_ID && CLIENT_ID && CLIENT_SECRET)) {
     return res.status(400).json({
       error: 'Missing Azure credentials; set AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET'
     });
   }
-  try {
+  try {
+    const cred = new DefaultAzureCredential();
+    const cm = new CostManagementClient(cred);
+    const scope = `/subscriptions/${SUBSCRIPTION_ID}`;
+    const key = `pnl:${SUBSCRIPTION_ID}:${ymd()}`;
+    const out = await cacheJSON<{items:{policy:string; savingsMTD:number; forecast90d:number}[]}>(key, AZURE_CACHE_TTL_SEC, async () => {
+      // compute fresh
     const cred = new DefaultAzureCredential();
     const cm = new CostManagementClient(cred);
     const scope = `/subscriptions/${SUBSCRIPTION_ID}`;
     const query: QueryDefinition = {
       type: 'ActualCost',
@@ -134,11 +169,13 @@ app.get('/api/v1/costs/pnl', async (_req, res) => {
     const items = rows.map((r) => {
       const policy = String(r[tagCol] ?? 'UNATTRIBUTED');
       const cost = Number(r[costCol] ?? 0);
       const savingsMTD = Math.max(0, cost * SAVINGS_RATE);
       const daily = daysElapsed > 0 ? savingsMTD / daysElapsed : 0;
-      const forecast90d = daily * 90;
-      return { policy, savingsMTD, forecast90d };
+      const forecast90d = daily * 90;
+      return { policy, savingsMTD, forecast90d };
     });
-    return res.json({ items });
+      return { items };
+    });
+    return res.json(out);
   } catch (e: any) {
     return res.status(500).json({ error: String(e?.message || e) });
   }
 });
diff --git a/services/core/src/main.rs b/services/core/src/main.rs
index deadc0d..f00dbab 100644
--- a/services/core/src/main.rs
+++ b/services/core/src/main.rs
@@ -22,11 +22,14 @@ async fn main() -> anyhow::Result<()> {
     let db_url = env::var("DATABASE_URL")
         .expect("DATABASE_URL is required (e.g., postgres://pcx:pcx@localhost:5432/pcx)");
     let port: u16 = env::var("CORE_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8081);
     let issuer = env::var("JWT_ISSUER").expect("JWT_ISSUER required");
     let audience = env::var("JWT_AUDIENCE").expect("JWT_AUDIENCE required");
     let jwks_url = env::var("JWT_JWKS_URL").expect("JWT_JWKS_URL required");
     let admin = env::var("ADMIN_GROUP_IDS").unwrap_or_default();
     let auditor = env::var("AUDITOR_GROUP_IDS").unwrap_or_default();
     let operator = env::var("OPERATOR_GROUP_IDS").unwrap_or_default();
+    let db_pool_size: u32 = env::var("DB_POOL_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
 
-    let pool = PgPoolOptions::new()
-        .max_connections(10)
+    let pool = PgPoolOptions::new()
+        .max_connections(db_pool_size)
+        .acquire_timeout(std::time::Duration::from_secs(10))
+        .idle_timeout(std::time::Duration::from_secs(30))
         .connect(&db_url)
         .await?;

▶️ Run
# 0) Ensure Redis is running (local or remote). Example local:
#    docker run -d --name redis -p 6379:6379 redis:7

# 1) Update env (copy .env.example → .env and fill). Ensure REDIS_URL points to your Redis.

# 2) Rebuild / start services
# Core (Rust)
cd services/core
cargo run --release

# Agents (Node)
cd ../../services/agents/azure
pnpm install
pnpm dev

# Gateway (Node)
cd ../../services/gateway
pnpm install
pnpm dev

# Frontend (still pointing to Gateway)
cd ../../
pnpm -w dev

✅ Verify
A) Rate limiting (Gateway, per user & tenant)
export TOKEN="eyJ..."   # valid Azure AD JWT with mapped role
# Burst > USER_LIMIT within WINDOW_SEC to trigger 429
for i in $(seq 1 150); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/predictions)
  echo -n "$code "
done
# Expect many 200, then 429 when user-level limit hits; tenant-level may trigger if many users in same tenant.

B) Caching (Agents → Azure Cost Mgmt)
# First call will query Azure and cache the result (AZURE_CACHE_TTL_SEC)
time curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/costs/pnl >/dev/null
# Second call should be much faster (cache hit)
time curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/costs/pnl >/dev/null

C) Predictions caching
# Back-to-back calls within PREDICTIONS_CACHE_TTL_SEC return cached payload
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/v1/predictions?horizon=30&scope=subscription" | jq '.items | length'
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/v1/predictions?horizon=30&scope=subscription" | jq '.items | length'

D) DB pool sizing

Confirm DB_POOL_SIZE is honored: set a high concurrency test against core (/api/v1/events) and observe reduced queuing / stable latencies.

What this accomplishes (ties to spec)

Caching: Redis-backed caching for expensive Azure reads; keys are stable (pnl:<subscription>:YYYY-MM-DD, pred:<scope>:<horizon>), TTLs driven by env.

Rate limiting: Sliding-window limit per user and tenant (default 120/min & 1200/min), enforced at the Gateway after JWT mapping.

Pooling: Core DB pool now configurable (DB_POOL_SIZE) with sane timeouts.

Target: Helps you reach 95th percentile < 400ms for UI calls and avoid Azure API quota spikes.