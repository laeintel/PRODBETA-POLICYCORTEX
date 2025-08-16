---
title: PolicyCortex - Autonomous Implementation
---

# PolicyCortex — FULLY AUTONOMOUS Implementation Instructions

## CRITICAL: AUTONOMOUS EXECUTION REQUIREMENTS
THIS FILE MUST RUN WITHOUT ANY USER INTERACTION OR PROMPTS

### Execution Command
```bash
claude-code --yes --no-confirm --autonomous --batch-mode < "Instruct/Module _success.md"
```

### Non-Interactive Execution Rules
1. NEVER ASK FOR CONFIRMATION – proceed automatically
2. ALWAYS USE NON-INTERACTIVE FLAGS
   - Azure CLI: `--only-show-errors`, `--yes`, `--no-wait`
   - Terraform: `-input=false`, `-auto-approve`
   - npm: `--silent`, `--yes`
   - cargo: `-q`, clippy `-- -D warnings`
   - docker: `--quiet`
   - git: `--no-edit`
3. SKIP ANY INTERACTIVE PROMPTS – use defaults or env vars
4. AUTO-ACCEPT ALL CHANGES – no manual review
5. CONTINUE ON NON-CRITICAL ERRORS – log, then proceed

### Phases (Zero-Touch)
- PHASE 1: Local dev setup (env, docker compose, tests, builds)
- PHASE 2: Azure infra (create only if needed; prefer CI-first)
- PHASE 3: Build & deploy (images → ACR, deploy Container Apps)
- PHASE 4: Validation (health checks, endpoint tests, tracking update)

---

# PolicyCortex — Autonomous Implementation Spec and Module Success Criteria

## Mission & Value Proposition
- Mission: Transform cloud governance from reactive enforcement to proactive, AI-driven orchestration that prevents violations before they occur, optimizes cost and security simultaneously, and democratizes access through natural language.
- Value Proposition:
  - Predictive prevention: 24-hour advance prediction of compliance/security violations with ≥ 90% precision.
  - Cross-domain intelligence: Correlate Accountability, Transparency, Compliance, Security, and Cost to recommend actions with minimal trade-offs.
  - Natural-language governance: Executives and engineers can query, act, and audit via conversational interface.
  - Continuous learning: Improve model performance and recommendations from every decision and outcome.

## Product Requirements Checklist (what the app must be able to do)
1) Governance Domains (Functional)
   - Accountability: Analyze RBAC/Entra ID activity to detect responsibility gaps; recommend role changes.
   - Transparency: Generate adaptive, stakeholder-specific reports and dashboards with anomaly explanations.
   - Compliance: Predict policy violations 24 hours ahead; monitor frameworks; propose remediation.
   - Security: Detect behavioral anomalies, correlate threats, predict attack paths, orchestrate response.
   - Cost Optimization: Forecast spend; recommend rightsizing; detect/prevent waste with ML.

2) AI Capabilities
   - Conversational interface with intent classification, entity extraction, tool-use calls to backend APIs.
   - Predictive compliance and cost models with explainability (SHAP/Captum) and confidence scores.
   - Cross-domain correlation engine with a persistent knowledge graph and What-If simulation.
   - Continuous learning loop: drift detection, auto-retrain, A/B deployment.

3) Automation & Orchestration
   - Auto-remediation with approval workflow, rollback checkpoints, and impact assessment.
   - Event-driven pipelines for policy changes, security alerts, and cost anomalies.

4) Platform & Ops
   - Azure-first deployment (ACR, Container Apps/AKS, Key Vault, Log Analytics), private networking, mTLS.
   - Multi-tenant isolation (per-tenant namespaces, secrets, and access boundaries).
   - Full observability (metrics, logs, traces, model governance dashboards).
   - CI/CD with canary, SLO gates, and automatic rollback.

5) Security & Compliance
   - Integrate Defender for Cloud; adhere to Azure Security Benchmark controls.
   - SOC 2 evidence collection automation; audit trails for decisions and model actions.
   - Field-level encryption for sensitive data; least-privilege RBAC across services.

6) Data & Integration
   - Live data ingestion from Azure Resource Graph, Policy, Cost Management, Activity Logs, and Security Center.
   - Robust schema/version handling; nightly auto-refresh of Azure API/Policy SDKs.

7) UX & Reporting
   - Clear dashboards per domain; transparency reports with NL summaries; export to PDF/PowerBI feed.
   - Graceful degradation when services are unavailable (empty but 200 with banners).

8) Cost & Scale
   - Usage metering & plan tiering (Free/Pro/Enterprise) with quotas and exportable billing events.
   - Performance targets: API p95 < 100 ms, inference p99 < 200 ms; horizontally scalable.

## System-wide Acceptance Criteria (DoD gates)
- Precision/quality: predictive compliance precision ≥ 0.90; cost forecast MAPE ≤ 0.05.
- Explainability: ≥ 80% of predictions accompanied by attributions and model-card references.
- Drift: detection MTTA < 2 weeks; auto-retrain pipelines executed when drift persists > 48h.
- Security: all critical/high vulns = 0; mTLS enforced; multi-tenant isolation verified.
- Reliability: 99.9% SLO; canary pass with automated rollback on SLO breach.
- Observability: traces, logs, and governance dashboards show non-zero, recent data.
- Compliance: SOC 2 evidence artifacts generated for all relevant controls; Defender ingestion active.
- Usability: conversational queries execute tool calls with ≥ 85% intent accuracy on test set.
- Cost: metering events emitted per request; quotas enforced per plan.

---

# PolicyCortex — Autonomous Implementation Spec and Module Success Criteria

This document is the single source of truth for an autonomous AI coder to set up, configure, deploy, and validate the entire PolicyCortex platform end-to-end on Azure without any prompts. If followed precisely, the app will run locally and in Azure with live data, update needed secrets/config automatically, and pass all health and smoke tests.

## Golden Rules for Autonomous Execution
- Operate non-interactively. Always pass flags like `--yes`, `--only-show-errors`, or equivalents.
- Be idempotent. Re-running any step must not break the system.
- Prefer environment variables over editing code. Only edit code/config when absolutely required and commit changes with a descriptive message.
- Use retries with exponential backoff for flaky cloud operations.
- Log every external change you make (Azure, GitHub, secrets) to `docs/PROJECT_TRACKING.MD`.

### Zero-Touch Non-Interactive Policy and Flags (MUST USE)
- Global env for non-interactive tools:
  - `export DEBIAN_FRONTEND=noninteractive`
  - `export CI=true`
  - `export AUTOMATED_MODE=true`
  - `export PIP_DISABLE_PIP_VERSION_CHECK=1`
  - `export npm_config_yes=true`
- Command flags to suppress prompts:
  - Azure CLI: `--only-show-errors`, `--yes`, `--no-wait`
  - Terraform: `-input=false`, `-auto-approve`, `-lock-timeout=5m`
  - Docker: `--quiet`, `--pull=missing`
  - Git: `--no-edit`, `-f` (force) where safe, pre-set user.name/email
  - npm: `--silent`, `--yes`
  - cargo: `-q` (quiet), clippy `-- -D warnings`
  - gh: non-interactive usage only (no `gh auth login` prompts in this flow)
- Never ask for confirmation; default to “proceed”. If a tool lacks a non-interactive flag, prefer environment variables or add a fallback timeout wrapper.

### Prefer CI-First, Provision Only If Needed
- Primary path: trigger and watch `.github/workflows/entry.yml` to deploy infra and apps. If CI completes successfully, skip local provisioning.
- Fallback path: if CI fails or times out, provision missing Azure resources locally using the idempotent `ensure_*` helpers below, then re-trigger CI.

### Idempotent Ensure-Helpers (Create-If-Needed)
```bash
# Common flags
AZ_FLAGS=(--only-show-errors)

ensure_rg() {
  local rg="$1" loc="$2"
  az group show -n "$rg" "${AZ_FLAGS[@]}" >/dev/null 2>&1 || \
  az group create -n "$rg" -l "$loc" "${AZ_FLAGS[@]}" >/dev/null
}

ensure_sa() {
  local rg="$1" sa="$2" loc="$3"
  az storage account show -g "$rg" -n "$sa" "${AZ_FLAGS[@]}" >/dev/null 2>&1 || \
  az storage account create -g "$rg" -n "$sa" -l "$loc" --sku Standard_LRS "${AZ_FLAGS[@]}" >/dev/null
}

ensure_container() {
  local rg="$1" sa="$2" container="$3"
  local conn; conn=$(az storage account show-connection-string -g "$rg" -n "$sa" -o tsv 2>/dev/null || echo)
  az storage container show --name "$container" --connection-string "$conn" >/dev/null 2>&1 || \
  az storage container create --name "$container" --connection-string "$conn" >/dev/null
}

ensure_logws() {
  local rg="$1" name="$2" loc="$3"
  az monitor log-analytics workspace show -g "$rg" -n "$name" "${AZ_FLAGS[@]}" >/dev/null 2>&1 || \
  az monitor log-analytics workspace create -g "$rg" -n "$name" -l "$loc" "${AZ_FLAGS[@]}" >/div/null
}

ensure_acr() {
  local rg="$1" name="$2" loc="$3"
  az acr show -g "$rg" -n "$name" "${AZ_FLAGS[@]}" >/dev/null 2>&1 || \
  az acr create -g "$rg" -n "$name" --sku Basic -l "$loc" "${AZ_FLAGS[@]}" >/dev/null
}

ensure_cae() {
  local rg="$1" env="$2" loc="$3" wsid="$4" wskey="$5"
  az containerapp env show -g "$rg" -n "$env" "${AZ_FLAGS[@]}" >/dev/null 2>&1 || \
  az containerapp env create -g "$rg" -n "$env" -l "$loc" \
    --logs-workspace-id "$wsid" --logs-workspace-key "$wskey" "${AZ_FLAGS[@]}" >/dev/null
}

ensure_ca() {
  local rg="$1" name="$2" env="$3" image="$4" port="$5" registry="$6"
  az containerapp show -g "$rg" -n "$name" "${AZ_FLAGS[@]}" >/dev/null 2>&1 || \
  az containerapp create -g "$rg" -n "$name" --environment "$env" \
    --image "$image" --ingress external --target-port "$port" \
    --registry-server "$registry" "${AZ_FLAGS[@]}" >/dev/null
}
```

### CI-First Flow (no prompts)
```bash
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
# Trigger and watch entry.yml; if success, skip local provisioning
( gh workflow run entry.yml -r "$BRANCH" >/dev/null || true )
RUN_ID="$(gh run list --workflow entry.yml --branch "$BRANCH" --limit 1 --json databaseId -q '.[0].databaseId')"
[ -n "$RUN_ID" ] && gh run watch "$RUN_ID" --exit-status && CI_OK=true || CI_OK=false

if [ "$CI_OK" != true ]; then
  echo "CI failed or not available; falling back to local ensure-provisioning"
  # call ensure_* helpers here with required variables
fi
```

## Repository Module Map (What connects to what)
- `core/` (Rust): primary API with endpoints `/health`, `/api/v1/{metrics,predictions,recommendations,correlations}`.
  - Depends on: Azure SDK creds (for live data), optional data stores, OpenTelemetry exporter.
- `frontend/` (Next.js 14): UI that consumes Core API and optional GraphQL gateway.
  - Env: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_GRAPHQL_ENDPOINT`, `USE_REAL_DATA`, `REQUIRE_AUTH`.
- `graphql/` (Node): Gateway facade (optional for demo). Proxies/aggregates Core + future services.
- `backend/services/` (Python): AI utilities (e.g., RLHF, explainability stubs, cost optimizer). Optional sidecars.
- `infrastructure/terraform/` and `infrastructure/kubernetes/`: Azure infra (ACR, Container Apps/AKS, Log Analytics, Key Vault, state).
- `observability/`: OpenTelemetry collector configuration.
- `scripts/`: Automation (OIDC setup, seeding, DR, etc.).

## Tooling Prerequisites (install if missing)
- Azure CLI ≥ 2.57 (`az version`)
- Terraform ≥ 1.6 (`terraform -v`)
- Docker + Docker Compose
- Node 18 LTS + npm 9
- Rust stable + cargo + clippy
- Python 3.10+ + pip
- `jq`, `yq`, `curl`, `git`, `gh` (GitHub CLI, authenticated)

If any tool is absent, install silently and continue.

## Global Environment
Set once per run (export in shell):
```bash
export ORG="policycortex"
export ENV="dev"
export REGION="eastus"
export PREFIX="pcx${RANDOM}${RANDOM}"
export RESOURCE_GROUP="${PREFIX}-rg"
export STATE_RG="${PREFIX}-state-rg"
export STATE_SA="${PREFIX}state"
export STATE_CONTAINER="tfstate"
export ACR_NAME="${PREFIX}acr"
export LOGWS_NAME="${PREFIX}-log"
export CAE_NAME="${PREFIX}-cae"
export SUBSCRIPTION_ID="$(az account show --query id -o tsv 2>/dev/null || echo)"
```

If `SUBSCRIPTION_ID` is empty, perform service principal login:
```bash
# Create SP with least privilege for demo (Owner can be used if necessary; prefer scoped Reader + specific roles)
az ad sp create-for-rbac --name "${PREFIX}-sp" --role Owner --scopes "/subscriptions/${SUBSCRIPTION_ID}" --sdk-auth \
  | tee AZURE_CREDENTIALS.json
export AZURE_TENANT_ID="$(jq -r .tenantId AZURE_CREDENTIALS.json)"
export AZURE_CLIENT_ID="$(jq -r .clientId AZURE_CREDENTIALS.json)"
export AZURE_CLIENT_SECRET="$(jq -r .clientSecret AZURE_CREDENTIALS.json)"
az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" --tenant "$AZURE_TENANT_ID" --only-show-errors
az account set --subscription "$SUBSCRIPTION_ID"
```

## Azure Foundation (Resource Group, State, Registry, Logging, Container Apps)
```bash
set -euo pipefail
# Resource groups
az group create --name "$RESOURCE_GROUP" --location "$REGION" --only-show-errors
az group create --name "$STATE_RG" --location "$REGION" --only-show-errors

# Remote state (Azure Storage)
az storage account create -g "$STATE_RG" -n "$STATE_SA" -l "$REGION" --sku Standard_LRS --kind StorageV2 --only-show-errors
export STATE_KEY="terraform-${ENV}.tfstate"
export STATE_CONN_STRING="$(az storage account show-connection-string -g "$STATE_RG" -n "$STATE_SA" -o tsv)"
az storage container create --name "$STATE_CONTAINER" --connection-string "$STATE_CONN_STRING" >/dev/null

# Log Analytics
az monitor log-analytics workspace create -g "$RESOURCE_GROUP" -n "$LOGWS_NAME" -l "$REGION" --only-show-errors
export LOGWS_ID="$(az monitor log-analytics workspace show -g "$RESOURCE_GROUP" -n "$LOGWS_NAME" --query id -o tsv)"
export LOGWS_CUSTOMER_ID="$(az monitor log-analytics workspace show -g "$RESOURCE_GROUP" -n "$LOGWS_NAME" --query customerId -o tsv)"
export LOGWS_PRIMARY_KEY="$(az monitor log-analytics workspace get-shared-keys -g "$RESOURCE_GROUP" -n "$LOGWS_NAME" --query primarySharedKey -o tsv)"

# ACR
az acr create -g "$RESOURCE_GROUP" -n "$ACR_NAME" --sku Basic --only-show-errors
export ACR_LOGIN_SERVER="$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)"
az acr login -n "$ACR_NAME" --only-show-errors

# Container Apps Environment
az extension add --name containerapp --upgrade --only-show-errors || true
az containerapp env create -g "$RESOURCE_GROUP" -n "$CAE_NAME" -l "$REGION" --logs-workspace-id "$LOGWS_CUSTOMER_ID" --logs-workspace-key "$LOGWS_PRIMARY_KEY" --only-show-errors
```

## GitHub Secrets (gh CLI)
```bash
# Requires gh auth login done beforehand (token with repo/admin:repo_hook)
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

echo "$(cat AZURE_CREDENTIALS.json)" | gh secret set AZURE_CREDENTIALS --repo "$REPO"
echo "$SUBSCRIPTION_ID" | gh secret set AZURE_SUBSCRIPTION_ID --repo "$REPO"
echo "$AZURE_TENANT_ID" | gh secret set AZURE_TENANT_ID --repo "$REPO"
echo "$AZURE_CLIENT_ID" | gh secret set AZURE_CLIENT_ID --repo "$REPO"
echo "$AZURE_CLIENT_SECRET" | gh secret set AZURE_CLIENT_SECRET --repo "$REPO"
echo "$ACR_LOGIN_SERVER" | gh secret set ACR_LOGIN_SERVER --repo "$REPO"
```

## Terraform Remote State Init and Apply
```bash
pushd infrastructure/terraform >/dev/null
terraform init \
  -backend-config="resource_group_name=$STATE_RG" \
  -backend-config="storage_account_name=$STATE_SA" \
  -backend-config="container_name=$STATE_CONTAINER" \
  -backend-config="key=$STATE_KEY"
terraform apply -auto-approve \
  -var "prefix=$PREFIX" -var "location=$REGION" -var "acr_name=$ACR_NAME" -var "log_analytics_workspace_id=$LOGWS_ID"
popd >/dev/null
```

## Build and Push Images
```bash
# Core (Rust)
docker build -t "$ACR_LOGIN_SERVER/core:${ENV}" -f core/Dockerfile core
# Frontend
docker build -t "$ACR_LOGIN_SERVER/frontend:${ENV}" -f frontend/Dockerfile frontend
# GraphQL
docker build -t "$ACR_LOGIN_SERVER/graphql:${ENV}" -f graphql/Dockerfile graphql

docker push "$ACR_LOGIN_SERVER/core:${ENV}"
docker push "$ACR_LOGIN_SERVER/frontend:${ENV}"
docker push "$ACR_LOGIN_SERVER/graphql:${ENV}"
```

## Deploy Services (Azure Container Apps)
```bash
# Core API
CORE_NAME="${PREFIX}-core"
az containerapp create -g "$RESOURCE_GROUP" -n "$CORE_NAME" --environment "$CAE_NAME" \
  --image "$ACR_LOGIN_SERVER/core:${ENV}" \
  --ingress external --target-port 8080 \
  --registry-server "$ACR_LOGIN_SERVER" --query systemData.createdAt -o tsv >/dev/null 2>&1 || \
az containerapp update -g "$RESOURCE_GROUP" -n "$CORE_NAME" --image "$ACR_LOGIN_SERVER/core:${ENV}" >/dev/null
export CORE_URL="$(az containerapp show -g "$RESOURCE_GROUP" -n "$CORE_NAME" --query properties.configuration.ingress.fqdn -o tsv)"

# GraphQL (optional)
GQL_NAME="${PREFIX}-graphql"
az containerapp create -g "$RESOURCE_GROUP" -n "$GQL_NAME" --environment "$CAE_NAME" \
  --image "$ACR_LOGIN_SERVER/graphql:${ENV}" \
  --ingress external --target-port 4000 \
  --registry-server "$ACR_LOGIN_SERVER" >/dev/null 2>&1 || \
az containerapp update -g "$RESOURCE_GROUP" -n "$GQL_NAME" --image "$ACR_LOGIN_SERVER/graphql:${ENV}" >/dev/null
export GRAPHQL_URL="$(az containerapp show -g "$RESOURCE_GROUP" -n "$GQL_NAME" --query properties.configuration.ingress.fqdn -o tsv)"

# Frontend
FE_NAME="${PREFIX}-frontend"
az containerapp create -g "$RESOURCE_GROUP" -n "$FE_NAME" --environment "$CAE_NAME" \
  --image "$ACR_LOGIN_SERVER/frontend:${ENV}" \
  --ingress external --target-port 3000 \
  --registry-server "$ACR_LOGIN_SERVER" \
  --env-vars NEXT_PUBLIC_API_URL="https://$CORE_URL" NEXT_PUBLIC_GRAPHQL_ENDPOINT="https://$GRAPHQL_URL" USE_REAL_DATA="false" REQUIRE_AUTH="false" >/dev/null 2>&1 || \
az containerapp update -g "$RESOURCE_GROUP" -n "$FE_NAME" \
  --image "$ACR_LOGIN_SERVER/frontend:${ENV}" \
  --set-env-vars NEXT_PUBLIC_API_URL="https://$CORE_URL" NEXT_PUBLIC_GRAPHQL_ENDPOINT="https://$GRAPHQL_URL" USE_REAL_DATA="false" REQUIRE_AUTH="false" >/dev/null
export FRONTEND_URL="$(az containerapp show -g "$RESOURCE_GROUP" -n "$FE_NAME" --query properties.configuration.ingress.fqdn -o tsv)"
```

## Local Development (optional)
```bash
# Simulated mode
export NEXT_PUBLIC_API_URL="http://localhost:8080"
docker compose -f docker-compose.local.yml up -d core graphql frontend
```

## Live Data Enablement
Set runtime secrets for Azure SDK access in Core container app:
```bash
az containerapp secret set -g "$RESOURCE_GROUP" -n "$CORE_NAME" \
  --secrets AZURE_TENANT_ID="$AZURE_TENANT_ID" AZURE_CLIENT_ID="$AZURE_CLIENT_ID" AZURE_CLIENT_SECRET="$AZURE_CLIENT_SECRET" AZURE_SUBSCRIPTION_ID="$SUBSCRIPTION_ID"
az containerapp update -g "$RESOURCE_GROUP" -n "$CORE_NAME" \
  --set-env-vars USE_REAL_DATA="true" AZURE_TENANT_ID="secretref:AZURE_TENANT_ID" AZURE_CLIENT_ID="secretref:AZURE_CLIENT_ID" AZURE_CLIENT_SECRET="secretref:AZURE_CLIENT_SECRET" AZURE_SUBSCRIPTION_ID="secretref:AZURE_SUBSCRIPTION_ID"
```

If OpenAI/Azure OpenAI is required:
```bash
export AZURE_OPENAI_ENDPOINT="$(./set-azure-openai-env.bat 2>/dev/null || echo)" # fallback if script echoes
# Or set explicitly and store in secrets
echo "$AZURE_OPENAI_ENDPOINT" | gh secret set AZURE_OPENAI_ENDPOINT --repo "$REPO"
```

## Module Success Criteria (per component)
- Core API (`core/`)
  - Build: `cargo build --release` returns 0.
  - Test: `cargo test` returns 0.
  - Lint: `cargo clippy -- -D warnings` returns 0.
  - Health: `curl -fsS https://$CORE_URL/health` returns 200 with JSON `{ "status": "ok" }`.
  - Metrics: `curl -fsS https://$CORE_URL/api/v1/metrics` returns 200 and non-empty array.
  - Live Data: if `USE_REAL_DATA=true`, returned metrics must include Azure subscription-scoped items (non-simulated flag present).

- Frontend (`frontend/`)
  - Build: `npm ci && npm run build` returns 0.
  - E2E: `npm test` (or Playwright) returns 0.
  - Load: `curl -fsS https://$FRONTEND_URL/` returns 200 and includes `window.__PCX__` bootstrap or landing HTML.

- GraphQL (`graphql/`)
  - Health: `curl -fsS https://$GRAPHQL_URL/health` returns 200 or `/graphql` responds.
  - Queries that aggregate Core resolve without 5xx; on failure return empty arrays (graceful).

- Infrastructure (`infrastructure/terraform/`)
  - `terraform plan` clean; `terraform apply` idempotent.
  - Remote state exists in Azure Storage.

- Observability (`observability/`)
  - Logs visible in Log Analytics for Container Apps.
  - If OTEL collector deployed, traces reach workspace (non-zero count in last 5 minutes).

- Security/Auth
  - When `REQUIRE_AUTH=true`, Azure AD login flow returns tokens and frontend gates content.
  - With `false`, app displays demo without auth.

## Autonomous CI/CD & Secrets Sync
```bash
# Ensure branch exists and push changes
BRANCH="AUTONOMOUS-BOOTSTRAP-${PREFIX}" 
git checkout -B "$BRANCH"
# Update docs/PROJECT_TRACKING.MD with outputs and URLs
sed -i.bak "/^## Deployments/a\n- Core: https://$CORE_URL\n- Frontend: https://$FRONTEND_URL\n- GraphQL: https://$GRAPHQL_URL\n- Resource Group: $RESOURCE_GROUP\n" docs/PROJECT_TRACKING.MD || true

git add -A && git commit -m "Autonomous: provisioned Azure infra, deployed services, set secrets, updated tracking" || true
git push -u origin "$BRANCH" || true
```

## Autonomous CI Monitoring & Troubleshooting (GitHub Actions)
- Monitor the monorepo entry workflow and fail-fast locally if CI fails. Treat CI errors as first-class signals to fix and re-run automatically.

### Watch the Monorepo CI Entry (entry.yml)
```bash
# Identify repo and branch
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

# Trigger (if not already running) or ensure latest workflow is executing
if ! gh run list --workflow entry.yml --branch "$BRANCH" --limit 1 | grep -q ".*"; then
  gh workflow run entry.yml -r "$BRANCH" >/dev/null || true
fi

# Watch to completion (non-interactive)
RUN_ID="$(gh run list --workflow entry.yml --branch "$BRANCH" --limit 1 --json databaseId -q '.[0].databaseId')"
[ -n "$RUN_ID" ] && gh run watch "$RUN_ID" --exit-status || true
```

### Pull Logs, Parse Errors, and Auto-Retry
```bash
# Download logs and artifacts for latest entry.yml run
RUN_ID="$(gh run list --workflow entry.yml --branch "$BRANCH" --limit 1 --json databaseId,conclusion -q '.[0].databaseId')"
[ -n "$RUN_ID" ] && gh run download "$RUN_ID" -D ci_logs || true

# Extract top failures (grep common error patterns)
if [ -d ci_logs ]; then
  echo "--- CI FAILURE SUMMARY (if any) ---"
  rg -n "(error:|FAIL|AssertionError|panic!|TypeError|ReferenceError|E[0-9]{3}|CLIPPY|lint error)" ci_logs || true
fi

# If failed, attempt targeted rerun of failed jobs only
CONCLUSION="$(gh run view "$RUN_ID" --json conclusion -q .conclusion 2>/dev/null || echo)"
if [ "$CONCLUSION" != "success" ] && [ -n "$RUN_ID" ]; then
  gh run rerun "$RUN_ID" --failed || true
  gh run watch "$RUN_ID" --exit-status || true
fi
```

### Inject CI Errors into Local Troubleshooting Loop
```bash
# If CI still fails, surface failing job names and logs to guide local fixes
if [ -n "$RUN_ID" ]; then
  gh run view "$RUN_ID" --json jobs -q '.jobs[] | {name: .name, conclusion: .conclusion}'
  # Example: fetch a specific job log for deeper analysis
  # gh run view "$RUN_ID" --log --job "build-core"
fi
```

### CI Success Gate (Hard Requirement)
- The workflow `.github/workflows/entry.yml` must run from start to finish without failures ("A to Z"), including:
  - Linting (TS, Rust clippy), unit tests, integration tests
  - Security scans (SAST/deps), container build/push
  - Terraform plan (and apply where appropriate), deploy-to-dev
  - Smoke/integration checks
- Any non-optional job marked `required` must complete with conclusion `success`. Skipped jobs are allowed only if conditionally disabled by design.

## Health & Smoke Test Suite
```bash
set -e
curl -fsS https://$CORE_URL/health | jq .
curl -fsS https://$CORE_URL/api/v1/metrics | jq '.[0]' > /dev/null
[ -n "$FRONTEND_URL" ] && curl -fsS https://$FRONTEND_URL/ | head -n 1
```

## Error Handling & Retries (template)
```bash
retry() { n=0; until "$@"; do n=$((n+1)); [ $n -ge 5 ] && return 1; sleep $((2**n)); done }
retry az group show -n "$RESOURCE_GROUP" -o none
```

## Rollback
```bash
# Keep infra but scale to zero to cut cost
az containerapp revision set-mode -g "$RESOURCE_GROUP" -n "$CORE_NAME" --mode single
az containerapp update -g "$RESOURCE_GROUP" -n "$CORE_NAME" --min-replicas 0
az containerapp update -g "$RESOURCE_GROUP" -n "$FE_NAME" --min-replicas 0
az containerapp update -g "$RESOURCE_GROUP" -n "$GQL_NAME" --min-replicas 0
```

## Completion Criteria (Definition of Done)
- All module success criteria above pass.
- Live data path enabled and visible in metrics (unless explicitly disabled for a demo run).
- URLs for Core, Frontend (and GraphQL if used) recorded in `docs/PROJECT_TRACKING.MD`.
- CI build is green on branch with secrets present in repository settings.
- Monorepo entry workflow `.github/workflows/entry.yml` completed successfully (A→Z) on the current branch; logs archived in `ci_logs/`.
- Rerunning this playbook is idempotent and keeps services healthy.

## Claude-Code Pre-Approvals (Zero-Prompt Mode)

Use these commands to pre-accept tools and write the local settings so no confirmations are shown. Each line uses --yes.

```bash
# Accept trust/onboarding and set broad tool permissions
claude-code --yes config set --global hasTrustDialogAccepted true
claude-code --yes config set --global hasCompletedProjectOnboarding true
# Optionally clear ignore patterns
claude-code --yes config set --global ignorePatterns '[]'

# Ensure the local settings file exists with full allow-list
mkdir -p .claude
cat > .claude/settings.local.json <<'JSON'
{
  "permissions": {
    "allow": [
      "Bash(cargo:*)",
      "Bash(SUBSCRIPTION_ID=\"205b477d-17e7-4b3b-92c1-32cf02626b78\")",
      "Bash(export AZURE_SUBSCRIPTION_ID=\"205b477d-17e7-4b3b-92c1-32cf02626b78\")",
      "Bash(-e AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78 )",
      "Bash(-e AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 )",
      "Bash(-e AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c )",
      "Bash(-e REDIS_URL=redis://host.docker.internal:6379 )",
      "Bash(-e RUST_LOG=debug )",
      "Bash(policycortex-core)",
      "Bash(-e AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78 )",
      "Bash(-e AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 )",
      "Bash(-e AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c )",
      "Bash(-e REDIS_URL=redis://host.docker.internal:6379 )",
      "Bash(-e RUST_LOG=debug )",
      "Bash(-v C:/Users/leona/.azure:/root/.azure:ro )",
      "Bash(./start-dev.bat)",
      "Bash(dir:*)",
      "Bash(./scripts/test-workflow.sh:*)",
      "Bash(scripts\\test-workflow.sh:*)",
      "Bash(powershell:*)",
      "Bash(start-training.bat --mode local)",
      "Bash(cmd /c:*)",
      "Bash(python:*)",
      "Bash(pip install:*)",
      "Bash(curl:*)",
      "Bash(docker ps:*)",
      "Bash(npm install)",
      "Bash(npm run dev:*)",
      "Bash(az account show:*)",
      "Bash(git checkout:*)",
      "Bash(git add:*)",
      "Bash(git push:*)",
      "Bash(git pull:*)",
      "Bash(start-api.bat)",
      "Bash(./start-local.bat)",
      "Bash(npm run build:*)",
      "Bash(npm run type-check:*)",
      "Bash(npx tsc:*)",
      "Bash(rm:*)",
      "Bash(gh project create:*)",
      "Bash(gh issue create:*)",
      "Bash(gh auth:*)",
      "Bash(gh label create:*)",
      "Bash(gh issue list:*)",
      "Bash(gh api:*)",
      "Bash(gh issue edit:*)",
      "Bash(gh project list:*)",
      "Bash(for:*)",
      "Bash(do)",
      "Bash(echo:*)",
      "Bash(done)",
      "Bash(gh issue comment:*)",
      "Bash(gh label:*)",
      "Bash(where cargo)",
      "Bash(do gh issue edit $issue --add-label \"in-review\")",
      "Bash(gh repo view:*)",
      "Bash(mkdir:*)",
      "Bash(wsl:*)",
      "Bash(C:Usersleonacursor-agent.bat --version)",
      "Bash(cursor-agent.bat --version)",
      "Bash(C:\\Users\\leona\\cursor-agent.bat --version)",
      "Bash(call C:Usersleonacursor-agent.bat --version)",
      "Bash(set USE_REAL_DATA=true)",
      "Bash(set AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78)",
      "Bash(set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7)",
      "Bash(sed:*)",
      "Bash(.start-backend.bat)",
      "Bash(docker-compose:*)",
      "Bash(docker-build-run.bat)",
      "Bash(az ad app federated-credential list:*)",
      "Bash(docker:*)",
      "Bash(az ad app federated-credential create:*)",
      "Bash(az role assignment create:*)",
      "Bash(az ad sp show:*)",
      "Bash(az role assignment list:*)",
      "Bash(az account set:*)",
      "Bash(az account list:*)",
      "Bash(terraform:*)",
      "Bash(az storage account create:*)",
      "Bash(gh secret:*)",
      "Bash(az storage account show:*)",
      "Bash(az storage account list:*)",
      "Bash(git branch:*)",
      "Bash(az login:*)",
      "Bash(git check-ignore:*)",
      "Bash(.restart-services.bat)",
      "Bash(bash:*)",
      "Bash(az:*)",
      "Bash(.start-dev.bat)",
      "Bash(start-dev.bat)",
      "Bash(./restart-services.bat)",
      "Bash(taskkill:*)",
      "Bash(npm run start:*)",
      "Bash(uvicorn:*)",
      "Bash(npx next:*)",
      "Bash(npm install:*)",
      "Bash(git commit:*)",
      "Bash(psql:*)",
      "Bash(./start-api-only.bat)",
      "Bash(git archive:*)",
      "Bash(./create-backup.bat)",
      "Bash(set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex)",
      "Bash(git rm:*)",
      "Bash(set SQLX_OFFLINE=true)",
      "Bash(set SQLX_OFFLINE=)",
      "Bash(git fetch:*)",
      "Bash(git restore:*)",
      "Bash(grep:*)",
      "Bash(gh workflow:*)",
      "Bash(gh run list:*)",
      "Bash(gh run view:*)",
      "Bash(gh run:*)",
      "Bash(git tag:*)",
      "Bash(git rev-parse:*)",
      "Bash(do echo -n \"$endpoint: \")",
      "Bash(git reset:*)",
      "Bash(npm test)",
      "Bash(npx playwright test:*)",
      "Bash(del nul)",
      "Bash(chmod:*)",
      "Bash(gh pr merge:*)",
      "Bash(gh pr checkout:*)",
      "Bash(./test-local.bat)",
      "Bash(timeout 120 cargo build)",
      "Bash(does not have a Release file.\"\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\nEOF\n)")",
      "Bash(export AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78)",
      "Bash(export AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7)",
      "Bash(export AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c)",
      "Bash(./ci-deploy.sh:*)",
      "Bash(find:*)",
      "Bash(timeout 120 cargo build --release)",
      "Bash(move:*)",
      "Bash(gh repo create:*)",
      "Bash(git remote add:*)",
      "Bash(cp:*)",
      "Bash(xcopy:*)",
      "Bash(timeout 60 terraform plan:*)",
      "Bash(git config:*)",
      "Bash(cat:*)",
      "Bash(./set-env.bat)",
      "Bash(timeout:*)",
      "Bash(gh pr view:*)",
      "Bash(git stash:*)",
      "Bash(rustup component:*)",
      "Bash(git rebase:*)",
      "Bash(./start-simple.bat)",
      "Bash(start-production.bat)",
      "Bash(./start-production.bat)",
      "Bash(set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c)",
      "Bash(export RUST_LOG=info)",
      "Bash(npm start)",
      "Bash(set RUST_LOG=info)",
      "Bash(start http://localhost:3000)",
      "Bash(set REDIS_URL=redis://localhost:6379)",
      "Bash(start-backend-live.bat)",
      "Bash(./start-backend-live.bat)",
      "Bash(start-local-with-openai.bat)",
      "Bash(mv:*)",
      "Bash(gh issue close:*)",
      "Bash(git merge:*)",
      "Bash(git cherry-pick:*)",
      "Bash(where npm)",
      "Bash(true)",
      "Bash(rustc --version)",
      "Bash(del start_automation.sh)",
      "Bash(set CI=true)",
      "Bash(set AUTOMATED_MODE=true)",
      "Bash(set DEBIAN_FRONTEND=noninteractive)",
      "Bash(set PIP_DISABLE_PIP_VERSION_CHECK=1)",
      "Bash(set npm_config_yes=true)",
      "Bash(set PREFIX=pcx42178531)",
      "Bash(set RESOURCE_GROUP=pcx42178531-rg)",
      "Bash(set STATE_RG=pcx42178531-state-rg)",
      "Bash(set STATE_SA=pcx42178531state)",
      "Bash(set STATE_CONTAINER=tfstate)",
      "Bash(set ACR_NAME=pcx42178531acr)",
      "Bash(set LOGWS_NAME=pcx42178531-log)",
      "Bash(set CAE_NAME=pcx42178531-cae)",
      "Bash(set REGION=eastus)",
      "Bash(set ENV=dev)"
    ],
    "deny": [],
    "defaultMode": "acceptEdits",
    "additionalDirectories": [
      "C:\\c\\Users\\leona\\Documents\\AeoliTech\\policycortex",
      "C:\\Users\\leona\\Documents\\AeoliTech\\milestone_backup",
      "C:\\c\\Users\\leona\\Documents\\AeoliTech",
      "C:\\mnt\\c\\Users\\leona\\Documents\\AeoliTech\\policycortex"
    ]
  }
}
JSON
```

This ensures claude-code operates non-interactively with the pre-approved commands above. If the CLI ignores local settings, proceed with direct PowerShell/Bash runners to maintain zero-touch behavior.
