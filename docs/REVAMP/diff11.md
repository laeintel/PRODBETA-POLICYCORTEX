here’s T11 (Omnichannel triggers: Slack + ITSM → tool calls) as precise, git-applyable diffs plus run/verify steps. This adds two tiny integration services that translate commands/webhooks into your Gateway’s typed endpoints (no free-form prompts), matching the baseline spec.

✅ Unified diffs — save as t11_omnichannel.patch, then:
git apply t11_omnichannel.patch

diff --git a/.env.example b/.env.example
index e1f0aa9..a9ce002 100644
--- a/.env.example
+++ b/.env.example
@@ -6,6 +6,14 @@ NEXT_PUBLIC_REAL_API_BASE=http://localhost:8000
 USE_REAL_DATA=true
 
+# Omnichannel (Slack + ITSM adapters call the Gateway with this token)
+GATEWAY_BASE=http://localhost:8000
+SERVICE_BEARER_TOKEN= # Azure AD JWT for this service principal (aud = JWT_AUDIENCE)
+
+# Slack slash commands
+SLACK_SIGNING_SECRET=
+SLACK_PORT=8090
+SLACK_SKIP_VERIFY=false
+
+# ITSM webhook
+ITSM_SHARED_SECRET=
+ITSM_PORT=8091
+
 # Azure (Entra ID)
 AZURE_TENANT_ID=your-tenant-id
 AZURE_SUBSCRIPTION_ID=your-subscription-id
diff --git a/services/integrations/slack/package.json b/services/integrations/slack/package.json
new file mode 100644
index 0000000..11aa22b
--- /dev/null
+++ b/services/integrations/slack/package.json
@@ -0,0 +1,27 @@
+{
+  "name": "pcx-slack",
+  "private": true,
+  "type": "module",
+  "version": "0.1.0",
+  "description": "Slack slash command adapter (maps /pcx commands to Gateway tool calls)",
+  "scripts": {
+    "dev": "tsx src/index.ts",
+    "start": "node --enable-source-maps dist/index.js",
+    "build": "tsc -p tsconfig.json"
+  },
+  "dependencies": {
+    "cors": "^2.8.5",
+    "express": "^4.19.2"
+  },
+  "devDependencies": {
+    "tsx": "^4.16.2",
+    "typescript": "^5.5.4"
+  }
+}
diff --git a/services/integrations/slack/tsconfig.json b/services/integrations/slack/tsconfig.json
new file mode 100644
index 0000000..5c5c5c5
--- /dev/null
+++ b/services/integrations/slack/tsconfig.json
@@ -0,0 +1,18 @@
+{
+  "compilerOptions": {
+    "target": "ES2022",
+    "module": "ES2022",
+    "moduleResolution": "Bundler",
+    "strict": true,
+    "esModuleInterop": true,
+    "skipLibCheck": true,
+    "outDir": "dist"
+  },
+  "include": ["src/**/*.ts"]
+}
diff --git a/services/integrations/slack/src/index.ts b/services/integrations/slack/src/index.ts
new file mode 100644
index 0000000..cafe1a1
--- /dev/null
+++ b/services/integrations/slack/src/index.ts
@@ -0,0 +1,198 @@
+import 'dotenv/config';
+import express from 'express';
+import cors from 'cors';
+import crypto from 'node:crypto';
+
+const app = express();
+const PORT = Number(process.env.SLACK_PORT || 8090);
+const SIGNING_SECRET = process.env.SLACK_SIGNING_SECRET || '';
+const SKIP_VERIFY = String(process.env.SLACK_SKIP_VERIFY || 'false') === 'true';
+const GATEWAY_BASE = process.env.GATEWAY_BASE || 'http://localhost:8000';
+const SERVICE_BEARER_TOKEN = process.env.SERVICE_BEARER_TOKEN || '';
+
+// Capture raw body for Slack signature
+app.use(
+  '/slack/commands',
+  express.urlencoded({
+    extended: false,
+    verify: (req: any, _res, buf) => {
+      req.rawBody = buf.toString('utf8');
+    },
+  })
+);
+app.use(cors());
+app.get('/health', (_req, res) => res.json({ ok: true, time: new Date().toISOString() }));
+app.post('/dev-null', (_req, res) => res.json({ ok: true })); // local test sink for response_url
+
+function verifySlack(req: express.Request): boolean {
+  if (SKIP_VERIFY) return true;
+  const ts = req.get('x-slack-request-timestamp') || '';
+  const sig = req.get('x-slack-signature') || '';
+  if (!ts || !sig || !SIGNING_SECRET) return false;
+  const fiveMin = 60 * 5;
+  if (Math.abs(Date.now() / 1000 - Number(ts)) > fiveMin) return false;
+  const base = `v0:${ts}:${(req as any).rawBody || ''}`;
+  const hmac = crypto.createHmac('sha256', SIGNING_SECRET).update(base).digest('hex');
+  const expected = `v0=${hmac}`;
+  const a = Buffer.from(sig);
+  const b = Buffer.from(expected);
+  return a.length === b.length && crypto.timingSafeEqual(a, b);
+}
+
+function parseKV(text: string): Record<string, string> {
+  const out: Record<string, string> = {};
+  for (const tok of text.split(/\s+/)) {
+    const m = /^([^=\s:]+)\s*[:=]\s*(.+)$/.exec(tok);
+    if (m) out[m[1]] = m[2];
+  }
+  return out;
+}
+
+async function postToResponseUrl(url: string, payload: any) {
+  try {
+    await fetch(url, {
+      method: 'POST',
+      headers: { 'content-type': 'application/json' },
+      body: JSON.stringify({ response_type: 'ephemeral', ...payload }),
+    });
+  } catch {
+    // swallow
+  }
+}
+
+app.post('/slack/commands', async (req, res) => {
+  if (!verifySlack(req)) return res.status(401).send('bad signature');
+  const params = new URLSearchParams((req as any).rawBody || '');
+  const command = params.get('command') || '';
+  const text = (params.get('text') || '').trim();
+  const response_url = params.get('response_url') || '';
+
+  // Immediate 200 to avoid Slack timeouts; we’ll respond via response_url
+  res.status(200).send(); 
+
+  if (command !== '/pcx') {
+    if (response_url) await postToResponseUrl(response_url, { text: `Unsupported command: ${command}` });
+    return;
+  }
+
+  const [sub, ...rest] = text.split(/\s+/);
+  const argline = rest.join(' ');
+  const kv = parseKV(argline);
+  const auth = SERVICE_BEARER_TOKEN ? { Authorization: `Bearer ${SERVICE_BEARER_TOKEN}` } : {};
+
+  try {
+    if (sub === 'predict') {
+      const scope = kv.scope || 'subscription';
+      const horizon = kv.horizon || '30';
+      const url = `${GATEWAY_BASE}/api/v1/predictions?scope=${encodeURIComponent(scope)}&horizon=${encodeURIComponent(horizon)}`;
+      const r = await fetch(url, { headers: { ...auth } });
+      const j = await r.json();
+      if (!r.ok) throw new Error(JSON.stringify(j));
+      const items = (j.items || []).slice(0, 5);
+      const lines = items.map((p: any) => `• *${p.ruleId}* — ETA ${p.etaDays}d, conf ${(p.confidence*100).toFixed(0)}%  |  PR: https://github.com/${p.repo}/compare/${p.fixBranch}?quick_pull=1&title=Auto-fix:${encodeURIComponent(p.ruleId)}`);
+      const uiHint = `Open: ${GATEWAY_BASE.replace(':8000','')}/ai/predictions`;
+      await postToResponseUrl(response_url, { text: lines.length ? lines.join('\n')+`\n_${uiHint}_` : 'No predictions.' });
+      return;
+    }
+    if (sub === 'verify') {
+      const hash = kv.hash || rest[0];
+      if (!hash) return postToResponseUrl(response_url, { text: 'Usage: /pcx verify hash=<sha256>' });
+      const url = `${GATEWAY_BASE}/api/v1/verify/${encodeURIComponent(hash)}`;
+      const r = await fetch(url, { headers: { ...auth } });
+      const j = await r.json();
+      if (!r.ok) throw new Error(JSON.stringify(j));
+      await postToResponseUrl(response_url, { text: j.verified ? `✔ Verified\nroot: \`${j.merkleRoot}\`\nproof: ${j.proof.length} hops` : '✖ Not found' });
+      return;
+    }
+    if (sub === 'pnl') {
+      const url = `${GATEWAY_BASE}/api/v1/costs/pnl`;
+      const r = await fetch(url, { headers: { ...auth } });
+      const j = await r.json();
+      if (!r.ok) throw new Error(JSON.stringify(j));
+      const items: any[] = j.items || [];
+      const top = items.sort((a,b)=>b.savingsMTD-a.savingsMTD).slice(0,5);
+      const lines = top.map(i => `• *${i.policy}* — MTD $${i.savingsMTD.toFixed(2)} | 90d $${i.forecast90d.toFixed(2)}`);
+      const uiHint = `Open: ${GATEWAY_BASE.replace(':8000','')}/finops/pnl`;
+      await postToResponseUrl(response_url, { text: lines.length ? lines.join('\n')+`\n_${uiHint}_` : 'No P&L data.' });
+      return;
+    }
+    await postToResponseUrl(response_url, { text: 'Usage:\n• `/pcx predict scope=<subscription|rg|tags:...> horizon=<days>`\n• `/pcx verify hash=<sha256>`\n• `/pcx pnl`' });
+  } catch (e: any) {
+    await postToResponseUrl(response_url, { text: `Error: ${String(e?.message || e)}` });
+  }
+});
+
+app.listen(PORT, () => {
+  console.log(`[pcx-slack] listening on http://0.0.0.0:${PORT}`);
+});
diff --git a/services/integrations/itsm/package.json b/services/integrations/itsm/package.json
new file mode 100644
index 0000000..33cc77a
--- /dev/null
+++ b/services/integrations/itsm/package.json
@@ -0,0 +1,25 @@
+{
+  "name": "pcx-itsm",
+  "private": true,
+  "type": "module",
+  "version": "0.1.0",
+  "description": "ITSM webhook adapter (maps incidents/changes to Gateway runs/events)",
+  "scripts": {
+    "dev": "tsx src/index.ts",
+    "start": "node --enable-source-maps dist/index.js",
+    "build": "tsc -p tsconfig.json"
+  },
+  "dependencies": {
+    "cors": "^2.8.5",
+    "express": "^4.19.2"
+  },
+  "devDependencies": {
+    "tsx": "^4.16.2",
+    "typescript": "^5.5.4"
+  }
+}
diff --git a/services/integrations/itsm/tsconfig.json b/services/integrations/itsm/tsconfig.json
new file mode 100644
index 0000000..5c5c5c5
--- /dev/null
+++ b/services/integrations/itsm/tsconfig.json
@@ -0,0 +1,18 @@
+{
+  "compilerOptions": {
+    "target": "ES2022",
+    "module": "ES2022",
+    "moduleResolution": "Bundler",
+    "strict": true,
+    "esModuleInterop": true,
+    "skipLibCheck": true,
+    "outDir": "dist"
+  },
+  "include": ["src/**/*.ts"]
+}
diff --git a/services/integrations/itsm/src/index.ts b/services/integrations/itsm/src/index.ts
new file mode 100644
index 0000000..b0b0b0b
--- /dev/null
+++ b/services/integrations/itsm/src/index.ts
@@ -0,0 +1,120 @@
+import 'dotenv/config';
+import express from 'express';
+import cors from 'cors';
+
+const app = express();
+const PORT = Number(process.env.ITSM_PORT || 8091);
+const SHARED = process.env.ITSM_SHARED_SECRET || '';
+const GATEWAY_BASE = process.env.GATEWAY_BASE || 'http://localhost:8000';
+const SERVICE_BEARER_TOKEN = process.env.SERVICE_BEARER_TOKEN || '';
+
+app.use(cors());
+app.use(express.json({ limit: '1mb' }));
+app.get('/health', (_req, res) => res.json({ ok: true, time: new Date().toISOString() }));
+
+function authOk(req: express.Request) {
+  const h = req.get('x-pcx-itsm-secret') || '';
+  return SHARED && h && h === SHARED;
+}
+
+async function appendEvent(payload: any) {
+  const r = await fetch(`${GATEWAY_BASE}/api/v1/events`, {
+    method: 'POST',
+    headers: {
+      'content-type': 'application/json',
+      ...(SERVICE_BEARER_TOKEN ? { Authorization: `Bearer ${SERVICE_BEARER_TOKEN}` } : {})
+    },
+    body: JSON.stringify({ payload }),
+  });
+  if (!r.ok) {
+    const t = await r.text().catch(()=>'');
+    throw new Error(`append_event: ${r.status} ${t}`);
+  }
+  return r.json();
+}
+
+// Generic ITSM webhook: accept incident/change payloads from ServiceNow/Jira/etc.
+app.post('/webhook/itsm', async (req, res) => {
+  if (!authOk(req)) return res.status(401).json({ error: 'unauthorized' });
+  const body = req.body || {};
+  const type = body.type || body.event_type || 'ITSMEvent';
+  const normalized = {
+    type,
+    source: body.source || 'itsm',
+    externalId: body.id || body.key || body.number || null,
+    severity: body.severity || body.priority || null,
+    status: body.status || null,
+    title: body.title || body.summary || body.short_description || null,
+    description: body.description || null,
+    labels: body.labels || body.tags || null,
+    timestamp: new Date().toISOString(),
+  };
+  try {
+    const out = await appendEvent(normalized);
+    res.status(202).json({ ok: true, id: out.id, ts: out.ts });
+  } catch (e: any) {
+    res.status(500).json({ error: String(e?.message || e) });
+  }
+});
+
+app.listen(PORT, () => {
+  console.log(`[pcx-itsm] listening on http://0.0.0.0:${PORT}`);
+});

▶️ Run
# 1) Fill .env with:
#    GATEWAY_BASE, SERVICE_BEARER_TOKEN (AAD JWT for this adapter), SLACK_SIGNING_SECRET, ITSM_SHARED_SECRET
#    (Gateway must already be running on :8000 with Auth/RBAC from T08.)
#
# 2) Start Slack adapter
cd services/integrations/slack
pnpm install
pnpm dev
# → http://localhost:8090/health

# 3) Start ITSM adapter
cd ../itsm
pnpm install
pnpm dev
# → http://localhost:8091/health

✅ Verify
A) Slack commands (local simulation)

For local testing without Slack, you can bypass signature verification:

export SLACK_SKIP_VERIFY=true  # set in env before starting pcx-slack (or restart)


Now simulate a /pcx predict:

# NOTE: substitute your SERVICE_BEARER_TOKEN if Gateway enforces JWT (recommended)
curl -s -X POST \
  -H 'content-type: application/x-www-form-urlencoded' \
  --data 'command=/pcx&text=predict scope=subscription horizon=30&response_url=http://localhost:8090/dev-null' \
  http://localhost:8090/slack/commands
# Expect 200 immediately; pcx-slack will call Gateway /api/v1/predictions and POST a summary to /dev-null (logged).


Verify /pcx verify:

curl -s -X POST \
  -H 'content-type: application/x-www-form-urlencoded' \
  --data 'command=/pcx&text=verify hash=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&response_url=http://localhost:8090/dev-null' \
  http://localhost:8090/slack/commands


Verify /pcx pnl:

curl -s -X POST \
  -H 'content-type: application/x-www-form-urlencoded' \
  --data 'command=/pcx&text=pnl&response_url=http://localhost:8090/dev-null' \
  http://localhost:8090/slack/commands


When you wire a real Slack app, set the Slash Command URL to:

https://<public-url>/slack/commands


and keep SLACK_SKIP_VERIFY=false in prod.

B) ITSM webhook
export ITSM_SHARED_SECRET=changeme   # ensure it matches your .env
curl -s -X POST http://localhost:8091/webhook/itsm \
  -H "x-pcx-itsm-secret: $ITSM_SHARED_SECRET" \
  -H 'content-type: application/json' \
  -d '{"type":"incident_created","id":"INC-1234","severity":"high","title":"API latency spike","description":"p95 > 400ms"}' | jq
# Expect { ok: true, id, ts } and see the event in Core via Gateway.


List events (authorized token required):

export TOKEN="$SERVICE_BEARER_TOKEN"
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/events/replay | jq '.[-1]'

What this lands (ties to spec)

Omnichannel: Slack /pcx and ITSM webhooks directly map to Gateway tool calls (/predictions, /verify/:hash, /costs/pnl, /events) — no prompt variance, no free-form RPC.

Security: Both adapters call Gateway with a service bearer token (AAD JWT). Slack request signing is verified (toggleable for local). ITSM uses a shared secret header.

Determinism: Adapters don’t alter control-flow—only pass parameters; all policy remains in Gateway/RBAC.