Awesome — here’s the “Export Evidence” endpoint + Audit download button with an e2e test. Apply the patch, rebuild, and you’ll be able to download a JSON artifact (pcx-evidence-<hash>.json) that the offline verifier (T05) already validates.

✅ Git-applyable diffs — save as export_evidence.patch, then run: git apply export_evidence.patch
diff --git a/services/core/src/main.rs b/services/core/src/main.rs
index abcdded..b1c0bee 100644
--- a/services/core/src/main.rs
+++ b/services/core/src/main.rs
@@ -1,10 +1,11 @@
-use axum::{extract::{Path, State}, http::Request, middleware::from_fn_with_state, routing::{get, post}, Json, Router};
+use axum::{extract::{Path, State, Query}, http::{Request, StatusCode, header}, middleware::from_fn_with_state, response::Response, routing::{get, post}, Json, Router};
 use serde::{Deserialize, Serialize};
 use serde_json::Value as JsonValue;
 use sqlx::{postgres::PgPoolOptions, PgPool};
 use std::{env, net::SocketAddr};
 use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};
 use chrono::{Utc, NaiveDate};
 use headers::HeaderMapExt;
+use axum::body::Body;
 use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
 
 mod merkle;
@@ -29,6 +30,12 @@ use metrics::{REGISTRY, HTTP_REQS, HTTP_DUR, EVENTS_TOTAL};
 #[derive(Clone)]
 struct AppState {
     pool: PgPool,
     auth: AuthState,
 }
+
+#[derive(Serialize)]
+struct EvidenceArtifact {
+    contentHash: String, signer: Option<String>, timestamp: String,
+    merkleRoot: String, proof: Vec<String>, version: String,
+}
 
 #[tokio::main]
 async fn main() -> anyhow::Result<()> {
@@ -83,6 +90,8 @@ async fn main() -> anyhow::Result<()> {
         .route("/api/v1/events/replay", get(replay_events))
         // Evidence verify (T05)
         .route("/api/v1/verify/:hash", get(verify_hash))
+        // Evidence export (NEW)
+        .route("/api/v1/evidence/export", get(export_evidence))
         // Protect all /api routes (health is public)
         .layer(from_fn_with_state(state.clone(), auth_middleware))
         .with_state(state)
@@ -168,6 +177,71 @@ async fn verify_hash(
     }))
 }
 
+// ---------- Evidence Export (artifact JSON download) ----------
+#[derive(Deserialize)]
+struct ExportQ { hash: String }
+
+async fn export_evidence(
+    State(state): State<AppState>,
+    Query(q): Query<ExportQ>,
+) -> Result<Response, (StatusCode, String)> {
+    let hash_hex = q.hash.to_lowercase();
+    if hash_hex.len() != 64 || !hash_hex.chars().all(|c| c.is_ascii_hexdigit()) {
+        return Err((StatusCode::BAD_REQUEST, "invalid hash".into()));
+    }
+    // find evidence row
+    let row = sqlx::query!(
+        r#"select ts, signer from evidence where content_hash = $1 limit 1"#,
+        hash_hex
+    ).fetch_optional(&state.pool).await.map_err(db)?;
+    if row.is_none() {
+        return Err((StatusCode::NOT_FOUND, "hash not found".into()));
+    }
+    let ts = row.as_ref().unwrap().ts;
+    let signer = row.as_ref().unwrap().signer.clone();
+    let day: NaiveDate = ts.date_naive();
+
+    // pull all hashes for that UTC day, in deterministic order
+    let rows = sqlx::query!(
+        r#"
+          select content_hash
+          from evidence
+          where date_trunc('day', ts) = date_trunc('day', $1::timestamptz)
+          order by content_hash asc
+        "#,
+        ts
+    ).fetch_all(&state.pool).await.map_err(db)?;
+
+    let mut leaves: Vec<Hash32> = Vec::with_capacity(rows.len());
+    let mut idx: Option<usize> = None;
+    for (i, r) in rows.iter().enumerate() {
+        let h = hex_to_hash(&r.content_hash).map_err(internal)?;
+        if r.content_hash.eq_ignore_ascii_case(&hash_hex) { idx = Some(i); }
+        leaves.push(h);
+    }
+    let target_idx = idx.ok_or_else(|| internal_str("hash disappeared during export"))?;
+    let (root, proof) = merkle_root_and_proof(&leaves, target_idx);
+    let root_hex = hex::encode(root);
+    let proof_hex: Vec<String> = proof.into_iter().map(|p| hex::encode(p)).collect();
+
+    let artifact = EvidenceArtifact {
+        contentHash: hash_hex.clone(),
+        signer,
+        timestamp: ts.to_rfc3339(),
+        merkleRoot: root_hex,
+        proof: proof_hex,
+        version: "pcx-evidence/1".into(),
+    };
+    let body = serde_json::to_vec(&artifact).map_err(internal)?;
+    let filename = format!("pcx-evidence-{}.json", hash_hex);
+    let resp = axum::http::Response::builder()
+        .status(StatusCode::OK)
+        .header(header::CONTENT_TYPE, "application/json")
+        .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{}\"", filename))
+        .body(Body::from(body))
+        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("response build: {e}")))?;
+    Ok(resp)
+}
+
 // ---------- Helpers ----------
 fn db<E: std::fmt::Display>(e: E) -> (axum::http::StatusCode, String) {
     (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}"))
 }
diff --git a/services/gateway/src/index.ts b/services/gateway/src/index.ts
index feadbee..ad0beef 100644
--- a/services/gateway/src/index.ts
+++ b/services/gateway/src/index.ts
@@ -187,6 +187,12 @@ app.get(
   createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true, ...withTrace() })
 );
 
+// Evidence export (download JSON artifact)
+app.get(
+  '/api/v1/evidence/export',
+  requireRole(['auditor', 'admin', 'operator']), rateLimit,
+  createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true, ...withTrace() })
+);
 // ---- Fallback
 app.use((_req, res) => res.status(404).json({ error: 'not_found' }));
 
diff --git a/frontend/app/audit/page.tsx b/frontend/app/audit/page.tsx
index a11ce77..0b7cafe 100644
--- a/frontend/app/audit/page.tsx
+++ b/frontend/app/audit/page.tsx
@@ -1,7 +1,9 @@
 'use client';
 // Simple Audit verifier UI: enter a contentHash, call Gateway /verify/{hash}, show Merkle proof.
 import { useState } from 'react';
 import { real } from '@/lib/real';
+const API_BASE = process.env.NEXT_PUBLIC_REAL_API_BASE || '';
+const TEST_BEARER = process.env.NEXT_PUBLIC_TEST_BEARER;
 
 type VerifyResp = {
   verified: boolean;
   merkleRoot: string;
@@ -19,6 +21,24 @@ export default function AuditPage() {
   const [err, setErr] = useState<string | null>(null);
   const [res, setRes] = useState<VerifyResp | null>(null);
 
+  async function exportEvidence() {
+    try {
+      if (!res?.verified) throw new Error('Verify a hash first');
+      const url = `${API_BASE}/api/v1/evidence/export?hash=${encodeURIComponent(hash)}`;
+      const headers: Record<string, string> = {};
+      if (TEST_BEARER) headers['Authorization'] = `Bearer ${TEST_BEARER}`;
+      const r = await fetch(url, { headers });
+      if (!r.ok) throw new Error(`Export failed: ${r.status}`);
+      const blob = await r.blob();
+      const a = document.createElement('a');
+      a.href = URL.createObjectURL(blob);
+      a.download = `pcx-evidence-${hash}.json`;
+      document.body.appendChild(a); a.click(); a.remove();
+      setTimeout(() => URL.revokeObjectURL(a.href), 0);
+    } catch (e: any) {
+      setErr(String(e?.message || e));
+    }
+  }
   async function onSubmit(e: React.FormEvent) {
     e.preventDefault();
     setLoading(true); setErr(null); setRes(null);
@@ -65,6 +85,11 @@ export default function AuditPage() {
               </details>
             </>
           )}
+          {res.verified && (
+            <div className="mt-4">
+              <button className="rounded-xl border px-4 py-2 text-sm font-medium hover:bg-zinc-50" onClick={exportEvidence}>Export Evidence</button>
+            </div>
+          )}
         </div>
       )}
 
diff --git a/tests/e2e/export.spec.ts b/tests/e2e/export.spec.ts
new file mode 100644
index 0000000..badd00d
--- /dev/null
+++ b/tests/e2e/export.spec.ts
@@ -0,0 +1,33 @@
+import { test, expect } from '@playwright/test';
+import fs from 'node:fs/promises';
+
+const BASE = process.env.BASE_URL || 'http://localhost:3000';
+
+test('Export Evidence downloads artifact JSON and contains hash/root/proof', async ({ page }) => {
+  await page.goto(`${BASE}/audit`);
+  await page.getByLabel('Evidence hash').fill('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
+  await page.getByRole('button', { name: /verify/i }).click();
+  await expect(page.getByText('✔ Verified')).toBeVisible();
+  const [download] = await Promise.all([
+    page.waitForEvent('download'),
+    page.getByRole('button', { name: /Export Evidence/i }).click(),
+  ]);
+  const file = await download.path();
+  expect(file).toBeTruthy();
+  const txt = await fs.readFile(file!, 'utf8');
+  const j = JSON.parse(txt);
+  expect(j).toHaveProperty('contentHash', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
+  expect(j).toHaveProperty('merkleRoot');
+  expect(Array.isArray(j.proof)).toBeTruthy();
+});

▶️ Run
# Rebuild / restart Core & Gateway (and Frontend if you want SSR rebuild)
# Core (Rust)
cd services/core && cargo run --release
# Gateway (Node)
cd ../../services/gateway && pnpm install && pnpm dev
# Frontend (Next.js)
cd ../../ && pnpm -w dev


If you use the demo stack:

# Rebuild just the changed services
docker compose -f docker-compose.demo.yml build core gateway frontend
docker compose -f docker-compose.demo.yml up -d

✅ Verify

Visit /audit, paste the seeded hash
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa → Verify → Export Evidence → a JSON file downloads.

Open it; it should contain:

{ "contentHash":"…", "timestamp":"…", "signer":"…", "merkleRoot":"…", "proof":[ "…" ], "version":"pcx-evidence/1" }


Playwright:

npx --yes playwright@1.46.0 install --with-deps
BASE_URL=http://localhost:3000 npx --yes playwright@1.46.0 test tests/e2e/export.spec.ts --config=tests/playwright.config.ts


This cleanly completes the PROVE loop: UI → /verify → Export Evidence → offline verifier (from T05) can