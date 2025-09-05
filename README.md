# PolicyCortex

**Moats**: PREVENT (predict → auto-fix), PROVE (tamper-evident evidence), PAYBACK (Governance P&L)  
**Spec**: `.pcx/agent-baseline.yaml` (source of truth)

## Quick Start (Demo)
```bash
export PCX_TEST_TOKEN="dev-ci-admin-jwt"
docker compose -f docker-compose.demo.yml up -d --build
docker cp services/core/migrations/0001_init.sql $(docker ps -qf "name=_db_"):/tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx -f /tmp/m.sql
docker exec $(docker ps -qf "name=_db_") psql -U pcx -d pcx \
  -c "insert into evidence(content_hash, signer) values ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','dev') on conflict do nothing;"
```
Open: `http://localhost:3000` (Predictions, P&L, Audit)

## Tests
```bash
npx playwright test tests/e2e --config=tests/playwright.config.ts
pnpm -w test     # reducer determinism
```

## CI
- Workflow: `.github/workflows/ci.yml` (mints JWKS + admin JWT, runs demo stack + e2e).