# PolicyCortex — Contributor Quickstart (Demo Stack)

This doc gets you from zero ➜ working demo with **Auth/RBAC**, **Gateway**, **Core (events/evidence)**,
**Agents (predictions + P&L)**, **Frontend**, and **e2e tests**.

> Canonical spec: `.pcx/agent-baseline.yaml`

## 0) Prereqs
- Docker & Docker Compose
- Node 20+ and pnpm (if running services locally)
- (Optional) Rust toolchain (if running `pcx-core` locally)

## 1) One-command demo stack
This launches **db, redis, jwks, core, agents, gateway, ml-mock, frontend**.
```bash
export PCX_TEST_TOKEN="dev-ci-admin-jwt"   # any string if you use demo stack only
docker compose -f docker-compose.demo.yml up -d --build
```
Apply schema + seed one evidence row:
```bash
docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx \
  -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','dev') on conflict do nothing;"
```
Open:
- Frontend: http://localhost:3000
- Gateway:  http://localhost:8000/health
- Core:     http://localhost:8081/health
- Agents:   http://localhost:8084/health

## 2) Auth in demo
- **Gateway/Core/Agents** enforce JWT. The demo stack mints a **JWKS** and uses a GitHub Action in CI. Locally:
  - Frontend sends `Authorization: Bearer $NEXT_PUBLIC_TEST_BEARER` automatically if set.
  - In the demo compose we pass `PCX_TEST_TOKEN` into the build and runtime; no real AAD needed.

## 3) Useful env (copy `.env.example` ➜ `.env`)
- **Mode:** `NEXT_PUBLIC_DEMO_MODE=false`, `USE_REAL_DATA=true`
- **Gateway URL to Frontend:** `NEXT_PUBLIC_REAL_API_BASE=http://localhost:8000`
- **Test bearer (local/CI only):** `NEXT_PUBLIC_TEST_BEARER=<token>`
- **Redis, Postgres URLs** per your setup

## 4) Day-to-day dev (local, not docker)
```bash
# Terminal A — Core
cd services/core && cargo run --release
# Terminal B — Agents (needs Redis and ML mock or your real ML)
cd services/agents/azure && pnpm i && pnpm dev
# Terminal C — Gateway
cd services/gateway && pnpm i && pnpm dev
# Terminal D — Frontend
pnpm -w dev
```
Ensure `.env` has the JWT vars (issuer/audience/jwks) and group mapping.

## 5) Key flows to verify
1) **Predictions (PREVENT):** `/ai/predictions` shows cards, top factors, **Create Fix PR** link.
2) **P&L (PAYBACK):** `/finops/pnl` shows policy-attributed savings MTD + forecast.
3) **Audit (PROVE):** `/audit` can verify a known evidence `contentHash` and show the **Merkle proof**.

## 6) Tests
- Unit (determinism): `pnpm -w test`
- E2E (Playwright): `npx playwright test tests/e2e --config=tests/playwright.config.ts`

## 7) CI basics
- Workflow `.github/workflows/ci.yml`:
  - Generates JWKS, mints a temporary admin JWT
  - Builds demo stack, seeds evidence
  - Runs Playwright e2e (Predictions/P&L/Audit)

## 8) Troubleshooting
- **401s:** Ensure `NEXT_PUBLIC_TEST_BEARER` is set (frontend) or pass a real AAD token to Gateway.
- **Empty P&L:** In CI/demo, `CI_ALLOW_EMPTY_PNL=true` returns `{ items: [] }` without Azure creds.
- **Rate limit (429):** See `RATE_LIMIT_*` in `.env.example`; defaults are conservative.

---
### Architecture checkpoints (Definition of Done)
- Tool calls validated by JSONSchema (Gateway)
- Auth enforced end-to-end (no bypass)
- Event replay reproduces Executive state
- Audit exports verifiable offline (Merkle)
- CI smoke + e2e are green