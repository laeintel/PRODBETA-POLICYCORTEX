Awesome — shipping T12 (CI/CD: build → demo env → smoke + acceptance) as exact, git-applyable diffs. This adds Dockerfiles, a demo docker-compose, Playwright e2e tests, a tiny ML mock, a GitHub Actions workflow, and two tiny tweaks (safe for prod) so CI can run without real Azure:

frontend/lib/real.ts: optional Authorization header from NEXT_PUBLIC_TEST_BEARER (test-only env).

agents/azure: CI_ALLOW_EMPTY_PNL=true returns {items:[]} when Azure creds are absent (200, not 500).

✅ Apply these patches (save as t12_ci_cd.patch, then run git apply t12_ci_cd.patch)
diff --git a/frontend/lib/real.ts b/frontend/lib/real.ts
index 6b04a26..b77cc10 100644
--- a/frontend/lib/real.ts
+++ b/frontend/lib/real.ts
@@ -1,6 +1,7 @@
 export const REAL_API_BASE =
   process.env.NEXT_PUBLIC_REAL_API_BASE || 'http://localhost:8084';
+const TEST_BEARER = process.env.NEXT_PUBLIC_TEST_BEARER; // used only in CI/demo
 
 type JSONValue =
   | string
@@ -15,9 +16,15 @@ export async function real<T = any>(
 ): Promise<T> {
   const url = `${REAL_API_BASE}${path}`;
-  const res = await fetch(url, { cache: 'no-store', ...init });
+  const headers: HeadersInit = {
+    ...(init.headers || {}),
+    ...(TEST_BEARER ? { Authorization: `Bearer ${TEST_BEARER}` } : {}),
+  };
+  const res = await fetch(url, { cache: 'no-store', ...init, headers });
   if (!res.ok) {
     const text = await res.text().catch(() => '');
     throw new Error(
       `Real API ${res.status} ${res.statusText} at ${url}${text ? ': ' + text : ''}`
     );
   }
   return res.json() as Promise<T>;
 }
 
 export type { JSONValue };
diff --git a/services/agents/azure/src/index.ts b/services/agents/azure/src/index.ts
index f0e0e0f..a0e0e0f 100644
--- a/services/agents/azure/src/index.ts
+++ b/services/agents/azure/src/index.ts
@@ -140,7 +140,12 @@ app.get('/api/v1/costs/pnl', async (_req, res) => {
-  if (!(SUBSCRIPTION_ID && TENANT_ID && CLIENT_ID && CLIENT_SECRET)) {
-    return res.status(400).json({
-      error: 'Missing Azure credentials; set AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET'
-    });
-  }
+  if (!(SUBSCRIPTION_ID && TENANT_ID && CLIENT_ID && CLIENT_SECRET)) {
+    if ((process.env.CI_ALLOW_EMPTY_PNL || 'false') === 'true') {
+      return res.json({ items: [] }); // CI/demo mode: allow empty but valid response
+    }
+    return res.status(400).json({
+      error: 'Missing Azure credentials; set AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET'
+    });
+  }
   try {
diff --git a/services/core/Dockerfile b/services/core/Dockerfile
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/services/core/Dockerfile
@@ -0,0 +1,31 @@
+FROM rust:1.78-slim as build
+WORKDIR /app
+RUN apt-get update && apt-get install -y pkg-config libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
+COPY services/core/Cargo.toml services/core/Cargo.lock ./ 
+RUN mkdir -p src && echo "fn main(){}" > src/main.rs && cargo build --release || true
+COPY services/core/src ./src
+RUN cargo build --release
+
+FROM debian:12-slim
+RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
+WORKDIR /srv
+COPY --from=build /app/target/release/pcx-core /usr/local/bin/pcx-core
+ENV CORE_PORT=8081
+EXPOSE 8081
+CMD ["/usr/local/bin/pcx-core"]
diff --git a/services/agents/azure/Dockerfile b/services/agents/azure/Dockerfile
new file mode 100644
index 0000000..2222222
--- /dev/null
+++ b/services/agents/azure/Dockerfile
@@ -0,0 +1,18 @@
+FROM node:20-alpine
+WORKDIR /app
+COPY services/agents/azure/package.json services/agents/azure/tsconfig.json ./
+RUN npm ci
+COPY services/agents/azure/src ./src
+RUN npm run build
+ENV PORT=8084
+EXPOSE 8084
+CMD ["node","dist/index.js"]
diff --git a/services/gateway/Dockerfile b/services/gateway/Dockerfile
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/services/gateway/Dockerfile
@@ -0,0 +1,18 @@
+FROM node:20-alpine
+WORKDIR /app
+COPY services/gateway/package.json services/gateway/tsconfig.json ./
+RUN npm ci
+COPY services/gateway/src ./src
+RUN npm run build
+ENV GATEWAY_PORT=8000
+EXPOSE 8000
+CMD ["node","dist/index.js"]
diff --git a/services/integrations/slack/Dockerfile b/services/integrations/slack/Dockerfile
new file mode 100644
index 0000000..4444444
--- /dev/null
+++ b/services/integrations/slack/Dockerfile
@@ -0,0 +1,16 @@
+FROM node:20-alpine
+WORKDIR /app
+COPY services/integrations/slack/package.json services/integrations/slack/tsconfig.json ./
+RUN npm ci
+COPY services/integrations/slack/src ./src
+RUN npm run build
+EXPOSE 8090
+CMD ["node","dist/index.js"]
diff --git a/services/integrations/itsm/Dockerfile b/services/integrations/itsm/Dockerfile
new file mode 100644
index 0000000..5555555
--- /dev/null
+++ b/services/integrations/itsm/Dockerfile
@@ -0,0 +1,16 @@
+FROM node:20-alpine
+WORKDIR /app
+COPY services/integrations/itsm/package.json services/integrations/itsm/tsconfig.json ./
+RUN npm ci
+COPY services/integrations/itsm/src ./src
+RUN npm run build
+EXPOSE 8091
+CMD ["node","dist/index.js"]
diff --git a/services/ml/mock/package.json b/services/ml/mock/package.json
new file mode 100644
index 0000000..6666666
--- /dev/null
+++ b/services/ml/mock/package.json
@@ -0,0 +1,21 @@
+{
+  "name": "pcx-ml-mock",
+  "private": true,
+  "type": "module",
+  "version": "0.1.0",
+  "scripts": {
+    "start": "node index.mjs"
+  },
+  "dependencies": {
+    "express": "^4.19.2"
+  }
+}
diff --git a/services/ml/mock/index.mjs b/services/ml/mock/index.mjs
new file mode 100644
index 0000000..7777777
--- /dev/null
+++ b/services/ml/mock/index.mjs
@@ -0,0 +1,33 @@
+import express from 'express';
+const app = express();
+const PORT = process.env.PORT || 8001;
+app.get('/predict', (_req,res) => {
+  res.json([
+    { ruleId: 'AZ-NSG-OPEN-22', etaDays: 7, confidence: 0.82 },
+    { ruleId: 'AZ-VM-DISK-UNENCRYPTED', etaDays: 14, confidence: 0.74 }
+  ]);
+});
+app.get('/explain', (_req,res) => {
+  res.json([
+    { ruleId: 'AZ-NSG-OPEN-22', top: [['nsg_open_ports',0.61],['subnet_public',0.21]] },
+    { ruleId: 'AZ-VM-DISK-UNENCRYPTED', top: [['disk_unencrypted',0.72],['rg_inheritance',0.11]] }
+  ]);
+});
+app.get('/health', (_req,res)=>res.json({ok:true,time:new Date().toISOString()}));
+app.listen(PORT, ()=>console.log(`[pcx-ml-mock] http://0.0.0.0:${PORT}`));
diff --git a/services/ml/mock/Dockerfile b/services/ml/mock/Dockerfile
new file mode 100644
index 0000000..8888888
--- /dev/null
+++ b/services/ml/mock/Dockerfile
@@ -0,0 +1,10 @@
+FROM node:20-alpine
+WORKDIR /app
+COPY services/ml/mock/package.json .
+RUN npm ci
+COPY services/ml/mock/index.mjs .
+EXPOSE 8001
+CMD ["npm","start"]
diff --git a/frontend/Dockerfile b/frontend/Dockerfile
new file mode 100644
index 0000000..9999999
--- /dev/null
+++ b/frontend/Dockerfile
@@ -0,0 +1,29 @@
+FROM node:20-alpine AS deps
+WORKDIR /app
+COPY frontend/package.json ./
+RUN npm ci
+
+FROM node:20-alpine AS build
+WORKDIR /app
+COPY --from=deps /app/node_modules ./node_modules
+COPY frontend ./
+ARG NEXT_PUBLIC_REAL_API_BASE
+ARG NEXT_PUBLIC_TEST_BEARER
+ENV NEXT_PUBLIC_REAL_API_BASE=$NEXT_PUBLIC_REAL_API_BASE
+ENV NEXT_PUBLIC_TEST_BEARER=$NEXT_PUBLIC_TEST_BEARER
+RUN npm run build
+
+FROM node:20-alpine AS run
+WORKDIR /app
+ENV NODE_ENV=production
+COPY --from=build /app/.next ./.next
+COPY --from=build /app/public ./public
+COPY --from=build /app/package.json ./package.json
+COPY --from=build /app/node_modules ./node_modules
+EXPOSE 3000
+CMD ["npm","start","--","-p","3000"]
diff --git a/docker-compose.demo.yml b/docker-compose.demo.yml
new file mode 100644
index 0000000..aaaaaaa
--- /dev/null
+++ b/docker-compose.demo.yml
@@ -0,0 +1,151 @@
+version: "3.8"
+services:
+  db:
+    image: postgres:15-alpine
+    environment:
+      POSTGRES_USER: pcx
+      POSTGRES_PASSWORD: pcx
+      POSTGRES_DB: pcx
+    ports: ["5432:5432"]
+    healthcheck: { test: ["CMD-SHELL","pg_isready -U pcx"], interval: 5s, timeout: 3s, retries: 10 }
+
+  redis:
+    image: redis:7-alpine
+    ports: ["6379:6379"]
+    healthcheck: { test: ["CMD","redis-cli","ping"], interval: 5s, timeout: 3s, retries: 10 }
+
+  jwks:
+    image: nginx:alpine
+    volumes:
+      - ./ops/jwks:/usr/share/nginx/html:ro
+    ports: ["8088:80"]
+
+  core:
+    build:
+      context: .
+      dockerfile: services/core/Dockerfile
+    environment:
+      DATABASE_URL: postgres://pcx:pcx@db:5432/pcx
+      CORE_PORT: 8081
+      JWT_ISSUER: pcx-ci
+      JWT_AUDIENCE: pcx
+      JWT_JWKS_URL: http://jwks/jwks.json
+      ADMIN_GROUP_IDS: "00000000-0000-0000-0000-000000000001"
+      AUDITOR_GROUP_IDS: "00000000-0000-0000-0000-000000000002"
+      OPERATOR_GROUP_IDS: "00000000-0000-0000-0000-000000000003"
+      DB_POOL_SIZE: 20
+    depends_on:
+      db: { condition: service_healthy }
+    ports: ["8081:8081"]
+
+  ml:
+    build:
+      context: .
+      dockerfile: services/ml/mock/Dockerfile
+    environment:
+      PORT: 8001
+    ports: ["8001:8001"]
+
+  agents:
+    build:
+      context: .
+      dockerfile: services/agents/azure/Dockerfile
+    environment:
+      PORT: 8084
+      REDIS_URL: redis://redis:6379
+      ML_PREDICT_URL: http://ml:8001/predict
+      ML_EXPLAIN_URL: http://ml:8001/explain
+      JWT_ISSUER: pcx-ci
+      JWT_AUDIENCE: pcx
+      JWT_JWKS_URL: http://jwks/jwks.json
+      ADMIN_GROUP_IDS: "00000000-0000-0000-0000-000000000001"
+      AUDITOR_GROUP_IDS: "00000000-0000-0000-0000-000000000002"
+      OPERATOR_GROUP_IDS: "00000000-0000-0000-0000-000000000003"
+      CI_ALLOW_EMPTY_PNL: "true"
+    depends_on:
+      redis: { condition: service_healthy }
+      ml: { condition: service_started }
+    ports: ["8084:8084"]
+
+  gateway:
+    build:
+      context: .
+      dockerfile: services/gateway/Dockerfile
+    environment:
+      GATEWAY_PORT: 8000
+      CORE_URL: http://core:8081
+      AZURE_AGENTS_URL: http://agents:8084
+      REDIS_URL: redis://redis:6379
+      JWT_ISSUER: pcx-ci
+      JWT_AUDIENCE: pcx
+      JWT_JWKS_URL: http://jwks/jwks.json
+      ADMIN_GROUP_IDS: "00000000-0000-0000-0000-000000000001"
+      AUDITOR_GROUP_IDS: "00000000-0000-0000-0000-000000000002"
+      OPERATOR_GROUP_IDS: "00000000-0000-0000-0000-000000000003"
+    depends_on:
+      redis: { condition: service_healthy }
+      core: { condition: service_started }
+      agents: { condition: service_started }
+    ports: ["8000:8000"]
+
+  frontend:
+    build:
+      context: .
+      dockerfile: frontend/Dockerfile
+      args:
+        NEXT_PUBLIC_REAL_API_BASE: http://gateway:8000
+        NEXT_PUBLIC_TEST_BEARER: ${PCX_TEST_TOKEN}
+    environment:
+      NEXT_PUBLIC_REAL_API_BASE: http://gateway:8000
+      NEXT_PUBLIC_TEST_BEARER: ${PCX_TEST_TOKEN}
+    depends_on:
+      gateway: { condition: service_started }
+    ports: ["3000:3000"]
diff --git a/tests/e2e/flows.spec.ts b/tests/e2e/flows.spec.ts
new file mode 100644
index 0000000..bbbbbbb
--- /dev/null
+++ b/tests/e2e/flows.spec.ts
@@ -0,0 +1,67 @@
+import { test, expect, request } from '@playwright/test';
+
+const BASE = process.env.BASE_URL || 'http://localhost:3000';
+const API = process.env.API_URL || 'http://localhost:8000';
+const TOKEN = process.env.PCX_TEST_TOKEN || '';
+
+test('Predictions page shows cards and PR link', async ({ page }) => {
+  await page.goto(`${BASE}/ai/predictions`);
+  await expect(page.getByText('AI Predictions')).toBeVisible();
+  await expect(page.getByRole('link', { name: /Create Fix PR/i })).toBeVisible();
+});
+
+test('P&L page renders table (may be empty)', async ({ page }) => {
+  await page.goto(`${BASE}/finops/pnl`);
+  await expect(page.getByText(/Governance P&L/i)).toBeVisible();
+  await expect(page.locator('table')).toBeVisible();
+});
+
+test('Verify evidence hash via API', async ({ }) => {
+  test.skip(!TOKEN, 'PCX_TEST_TOKEN not set');
+  const ctx = await request.newContext({
+    baseURL: API,
+    extraHTTPHeaders: TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}
+  });
+  const hash = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
+  const r = await ctx.get(`/api/v1/verify/${hash}`);
+  expect(r.status()).toBe(200);
+  const j = await r.json();
+  expect(j).toHaveProperty('verified');
+});
diff --git a/tests/playwright.config.ts b/tests/playwright.config.ts
new file mode 100644
index 0000000..ccccccc
--- /dev/null
+++ b/tests/playwright.config.ts
@@ -0,0 +1,17 @@
+import { defineConfig } from '@playwright/test';
+export default defineConfig({
+  timeout: 60_000,
+  use: {
+    baseURL: process.env.BASE_URL || 'http://localhost:3000',
+    trace: 'retain-on-failure',
+  },
+  reporter: [['list']],
+  workers: 2,
+});
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
new file mode 100644
index 0000000..ddddddd
--- /dev/null
+++ b/.github/workflows/ci.yml
@@ -0,0 +1,146 @@
+name: pcx-ci
+on:
+  push:
+  pull_request:
+jobs:
+  e2e:
+    runs-on: ubuntu-latest
+    permissions:
+      contents: read
+    steps:
+      - name: Checkout
+        uses: actions/checkout@v4
+
+      - name: Setup Node
+        uses: actions/setup-node@v4
+        with:
+          node-version: '20'
+
+      - name: Generate JWKS and mint admin JWT
+        id: mint
+        run: |
+          cat > ops/jwks/mint.mjs <<'JS'
+          import { generateKeyPair, exportJWK, SignJWT } from 'jose';
+          import { writeFileSync, mkdirSync } from 'node:fs';
+          const { privateKey, publicKey } = await generateKeyPair('RS256', { modulusLength: 2048 });
+          const kid = 'pcx-ci-1';
+          const jwk = await exportJWK(publicKey); jwk.kid = kid; jwk.alg='RS256'; jwk.use='sig';
+          mkdirSync('ops/jwks', { recursive: true });
+          writeFileSync('ops/jwks/jwks.json', JSON.stringify({ keys:[jwk] }, null, 2));
+          const now = Math.floor(Date.now()/1000);
+          const token = await new SignJWT({
+            groups:['00000000-0000-0000-0000-000000000001'], // admin
+            tid:'pcx-ci-tenant'
+          })
+            .setProtectedHeader({ alg:'RS256', kid })
+            .setIssuer('pcx-ci')
+            .setAudience('pcx')
+            .setSubject('tester@pcx')
+            .setIssuedAt(now)
+            .setExpirationTime(now + 60*60)
+            .sign(privateKey);
+          console.log('TOKEN='+token);
+          writeFileSync('ops/jwks/token.txt', token);
+          JS
+          node ops/jwks/mint.mjs | tee /tmp/token.out
+          echo "token=$(grep TOKEN /tmp/token.out | cut -d= -f2)" >> $GITHUB_OUTPUT
+
+      - name: Build & start demo stack
+        env:
+          PCX_TEST_TOKEN: ${{ steps.mint.outputs.token }}
+        run: |
+          docker compose -f docker-compose.demo.yml up -d --build
+          # Apply DB schema
+          until docker exec $(docker ps -qf "name=_db_") pg_isready -U pcx; do sleep 2; done
+          docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
+          docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
+          # Seed one evidence hash
+          docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','ci') on conflict do nothing;"
+          # Wait for services
+          for u in http://localhost:8081/health http://localhost:8084/health http://localhost:8000/health http://localhost:3000; do
+            echo "Waiting for $u"; for i in {1..60}; do curl -fsS "$u" >/dev/null && break || sleep 2; done
+          done
+
+      - name: Install Playwright
+        run: |
+          npx --yes playwright@1.46.0 install --with-deps
+
+      - name: Run e2e tests
+        env:
+          BASE_URL: http://localhost:3000
+          API_URL: http://localhost:8000
+          PCX_TEST_TOKEN: ${{ steps.mint.outputs.token }}
+        run: |
+          npx --yes playwright@1.46.0 test tests/e2e --config=tests/playwright.config.ts
+
+      - name: Dump docker logs on failure
+        if: failure()
+        run: |
+          docker ps
+          docker compose -f docker-compose.demo.yml logs --no-color | tail -n 5000

▶️ Run (locally or on a branch)
# 1) Apply patch
git apply t12_ci_cd.patch

# 2) Commit
git add .
git commit -m "ci: demo compose, Dockerfiles, Playwright e2e, GH Actions (T12)"

# 3) (Optional) Local dry-run of demo stack
export PCX_TEST_TOKEN="dev-token"  # any string; frontend only uses it in CI
docker compose -f docker-compose.demo.yml up -d --build

# 4) Seed DB and test endpoints (as in workflow)
docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','ci') on conflict do nothing;"
curl -s http://localhost:3000 | head -n 5

✅ Verify (acceptance tied to baseline)

Build & deploy demo env: docker-compose.demo.yml brings up db, redis, jwks, core, agents, gateway, ml, frontend.

Smoke passes: Frontend loads, Predictions shows Create Fix PR, P&L page renders (empty allowed), /verify/<hash> succeeds against seeded hash.

CI: .github/workflows/ci.yml mints a local JWKS + admin JWT, starts the stack, seeds evidence, installs Playwright, and runs e2e tests.

No prod drift: Auth is still enforced. Only CI-specific allowances are:

Frontend may send Authorization: Bearer $NEXT_PUBLIC_TEST_BEARER if present.

Agents return 200 {items:[]} for P&L when CI_ALLOW_EMPTY_PNL=true.