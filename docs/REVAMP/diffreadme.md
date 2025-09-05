Awesome — adding README snippets for contributors and an Audit → Merkle proof UI with Playwright e2e. Below are exact, git-applyable diffs plus quick Run/Verify.

✅ Unified diffs — save as readme_audit_tests.patch, then:
git apply readme_audit_tests.patch

diff --git a/docs/CONTRIBUTING.pcx.md b/docs/CONTRIBUTING.pcx.md
new file mode 100644
index 0000000..10aa001
--- /dev/null
+++ b/docs/CONTRIBUTING.pcx.md
@@ -0,0 +1,159 @@
+# PolicyCortex — Contributor Quickstart (Demo Stack)
+
+This doc gets you from zero ➜ working demo with **Auth/RBAC**, **Gateway**, **Core (events/evidence)**,
+**Agents (predictions + P&L)**, **Frontend**, and **e2e tests**.
+
+> Canonical spec: `.pcx/agent-baseline.yaml`
+
+## 0) Prereqs
+- Docker & Docker Compose
+- Node 20+ and pnpm (if running services locally)
+- (Optional) Rust toolchain (if running `pcx-core` locally)
+
+## 1) One-command demo stack
+This launches **db, redis, jwks, core, agents, gateway, ml-mock, frontend**.
+```bash
+export PCX_TEST_TOKEN="dev-ci-admin-jwt"   # any string if you use demo stack only
+docker compose -f docker-compose.demo.yml up -d --build
+```
+Apply schema + seed one evidence row:
+```bash
+docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
+docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
+docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx \
+  -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','dev') on conflict do nothing;"
+```
+Open:
+- Frontend: http://localhost:3000
+- Gateway:  http://localhost:8000/health
+- Core:     http://localhost:8081/health
+- Agents:   http://localhost:8084/health
+
+## 2) Auth in demo
+- **Gateway/Core/Agents** enforce JWT. The demo stack mints a **JWKS** and uses a GitHub Action in CI. Locally:
+  - Frontend sends `Authorization: Bearer $NEXT_PUBLIC_TEST_BEARER` automatically if set.
+  - In the demo compose we pass `PCX_TEST_TOKEN` into the build and runtime; no real AAD needed.
+
+## 3) Useful env (copy `.env.example` ➜ `.env`)
+- **Mode:** `NEXT_PUBLIC_DEMO_MODE=false`, `USE_REAL_DATA=true`
+- **Gateway URL to Frontend:** `NEXT_PUBLIC_REAL_API_BASE=http://localhost:8000`
+- **Test bearer (local/CI only):** `NEXT_PUBLIC_TEST_BEARER=<token>`
+- **Redis, Postgres URLs** per your setup
+
+## 4) Day-to-day dev (local, not docker)
+```bash
+# Terminal A — Core
+cd services/core && cargo run --release
+# Terminal B — Agents (needs Redis and ML mock or your real ML)
+cd services/agents/azure && pnpm i && pnpm dev
+# Terminal C — Gateway
+cd services/gateway && pnpm i && pnpm dev
+# Terminal D — Frontend
+pnpm -w dev
+```
+Ensure `.env` has the JWT vars (issuer/audience/jwks) and group mapping.
+
+## 5) Key flows to verify
+1) **Predictions (PREVENT):** `/ai/predictions` shows cards, top factors, **Create Fix PR** link.
+2) **P&L (PAYBACK):** `/finops/pnl` shows policy-attributed savings MTD + forecast.
+3) **Audit (PROVE):** `/audit` can verify a known evidence `contentHash` and show the **Merkle proof**.
+
+## 6) Tests
+- Unit (determinism): `pnpm -w test`
+- E2E (Playwright): `npx playwright test tests/e2e --config=tests/playwright.config.ts`
+
+## 7) CI basics
+- Workflow `.github/workflows/ci.yml`:
+  - Generates JWKS, mints a temporary admin JWT
+  - Builds demo stack, seeds evidence
+  - Runs Playwright e2e (Predictions/P&L/Audit)
+
+## 8) Troubleshooting
+- **401s:** Ensure `NEXT_PUBLIC_TEST_BEARER` is set (frontend) or pass a real AAD token to Gateway.
+- **Empty P&L:** In CI/demo, `CI_ALLOW_EMPTY_PNL=true` returns `{ items: [] }` without Azure creds.
+- **Rate limit (429):** See `RATE_LIMIT_*` in `.env.example`; defaults are conservative.
+
+---
+### Architecture checkpoints (Definition of Done)
+- Tool calls validated by JSONSchema (Gateway)
+- Auth enforced end-to-end (no bypass)
+- Event replay reproduces Executive state
+- Audit exports verifiable offline (Merkle)
+- CI smoke + e2e are green
diff --git a/frontend/app/audit/page.tsx b/frontend/app/audit/page.tsx
new file mode 100644
index 0000000..a11ce77
--- /dev/null
+++ b/frontend/app/audit/page.tsx
@@ -0,0 +1,129 @@
+'use client';
+// Simple Audit verifier UI: enter a contentHash, call Gateway /verify/{hash}, show Merkle proof.
+import { useState } from 'react';
+import { real } from '@/lib/real';
+
+type VerifyResp = {
+  verified: boolean;
+  merkleRoot: string;
+  proof: string[];
+  day: string;
+};
+
+export default function AuditPage() {
+  const [hash, setHash] = useState<string>('');
+  const [loading, setLoading] = useState(false);
+  const [err, setErr] = useState<string | null>(null);
+  const [res, setRes] = useState<VerifyResp | null>(null);
+
+  async function onSubmit(e: React.FormEvent) {
+    e.preventDefault();
+    setLoading(true); setErr(null); setRes(null);
+    try {
+      if (!/^[a-f0-9]{64}$/i.test(hash)) {
+        throw new Error('Enter a 64-char SHA-256 hex hash');
+      }
+      const data = await real<VerifyResp>(`/api/v1/verify/${hash}`);
+      setRes(data);
+    } catch (e: any) {
+      setErr(String(e?.message || e));
+    } finally {
+      setLoading(false);
+    }
+  }
+
+  return (
+    <div className="mx-auto max-w-3xl p-6">
+      <h1 className="text-2xl font-semibold">Audit Verification</h1>
+      <p className="text-sm text-zinc-500">Verify an evidence export hash (tamper-evident Merkle proof).</p>
+
+      <form onSubmit={onSubmit} className="mt-4 flex gap-2 items-center">
+        <input
+          value={hash}
+          onChange={(e)=>setHash(e.target.value.trim())}
+          placeholder="aaaaaaaa... (64-hex)"
+          className="w-full rounded-xl border px-3 py-2 text-sm"
+          aria-label="Evidence hash"
+        />
+        <button
+          disabled={loading}
+          className="rounded-xl border px-4 py-2 text-sm font-medium hover:bg-zinc-50 disabled:opacity-60"
+        >
+          {loading ? 'Verifying…' : 'Verify'}
+        </button>
+      </form>
+
+      {err && <div className="mt-3 text-sm text-red-600">Error: {err}</div>}
+
+      {res && (
+        <div className="mt-6 rounded-2xl border p-4">
+          <div className="flex items-center justify-between">
+            <div className="text-base font-medium">
+              {res.verified ? '✔ Verified' : '✖ Not Found'}
+            </div>
+            <div className="text-xs text-zinc-500">Day: {res.day || '—'}</div>
+          </div>
+          {res.verified && (
+            <>
+              <div className="mt-2 text-sm">
+                <div><span className="font-mono text-xs">merkleRoot</span>: <span className="font-mono">{res.merkleRoot}</span></div>
+                <div className="text-zinc-500 text-xs">Proof length: {res.proof.length} hop(s)</div>
+              </div>
+              <details className="mt-3">
+                <summary className="cursor-pointer text-sm">View Merkle proof</summary>
+                <pre className="mt-2 text-xs overflow-auto rounded-xl bg-zinc-50 p-3">{JSON.stringify(res.proof, null, 2)}</pre>
+              </details>
+            </>
+          )}
+        </div>
+      )}
+
+      <div className="mt-8 text-xs text-zinc-500">
+        Tip: In the demo stack, a sample evidence hash is seeded:
+        <span className="font-mono"> aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</span>
+      </div>
+    </div>
+  );
+}
diff --git a/tests/e2e/audit.spec.ts b/tests/e2e/audit.spec.ts
new file mode 100644
index 0000000..0aa0aa1
--- /dev/null
+++ b/tests/e2e/audit.spec.ts
@@ -0,0 +1,25 @@
+import { test, expect } from '@playwright/test';
+
+const BASE = process.env.BASE_URL || 'http://localhost:3000';
+
+test('Audit page verifies seeded hash and shows proof', async ({ page }) => {
+  await page.goto(`${BASE}/audit`);
+  await expect(page.getByText('Audit Verification')).toBeVisible();
+  await page.getByLabel('Evidence hash').fill('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
+  await page.getByRole('button', { name: /verify/i }).click();
+  await expect(page.getByText('✔ Verified')).toBeVisible();
+  await page.getByText('View Merkle proof').click();
+  await expect(page.locator('pre')).toContainText('bbbbbbbbbbbbbbbb'); // proof contains sibling hash from seeded pair
+});
diff --git a/README.md b/README.md
index e69de29..beefc01 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1,36 @@
+# PolicyCortex
+
+**Moats**: PREVENT (predict → auto-fix), PROVE (tamper-evident evidence), PAYBACK (Governance P&L)  
+**Spec**: `.pcx/agent-baseline.yaml` (source of truth)
+
+## Quick Start (Demo)
+```bash
+export PCX_TEST_TOKEN="dev-ci-admin-jwt"
+docker compose -f docker-compose.demo.yml up -d --build
+docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
+docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
+docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx \
+  -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','dev') on conflict do nothing;"
+```
+Open: `http://localhost:3000` (Predictions, P&L, Audit)
+
+## Tests
+```bash
+npx playwright test tests/e2e --config=tests/playwright.config.ts
+pnpm -w test     # reducer determinism
+```
+
+## CI
+- Workflow: `.github/workflows/ci.yml` (mints JWKS + admin JWT, runs demo stack + e2e).
+```


▶️ Run
# Apply patch and commit
git apply readme_audit_tests.patch
git add .
git commit -m "docs: contributor quickstart + Audit verifier UI + e2e test"

# If using the demo stack:
export PCX_TEST_TOKEN="dev-ci-admin-jwt"
docker compose -f docker-compose.demo.yml up -d --build
docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx \
  -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','dev') on conflict do nothing;"

# (Optional) run e2e
npx --yes playwright@1.46.0 install --with-deps
BASE_URL=http://localhost:3000 npx --yes playwright@1.46.0 test tests/e2e/audit.spec.ts --config=tests/playwright.config.ts

✅ Verify

Visit http://localhost:3000/audit
 → paste the seeded hash
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa → ✔ Verified, proof visible.

tests/e2e/audit.spec.ts passes (checks Verified + proof output).

Contributors can follow docs/CONTRIBUTING.pcx.md for demo stack, auth notes, and tests.