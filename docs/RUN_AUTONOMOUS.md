# AUTONOMOUS EXECUTION INSTRUCTIONS

## CRITICAL DIRECTIVES
- **AUTONOMOUS MODE**: Execute all tasks without prompting
- **AUTO-APPROVE**: Assume YES to all confirmations
- **NO INTERACTION**: Make all decisions independently
- **CONTINUOUS EXECUTION**: Don't stop for user input

## EXECUTION FLAGS
- `--yes`: Auto-approve all operations
- `--no-confirm`: Skip all confirmations
- `--autonomous`: Full autonomous mode
- `--batch-mode`: Process all tasks in sequence

## TASK: Close Azure GA Critical Backlog (Immediate High-Impact Defenses)

### Objective
Ship the five defense clusters required for Azure GA and competitive resilience. Execute in order; fix builds/tests as you go; update tracking.

### Clusters & Work Items

1) Service Mesh + Multi-Tenant Isolation (AKS)
- Add Istio (or Linkerd) manifests under `infrastructure/kubernetes/istio/`
- Enable mTLS, PeerAuthentication, AuthorizationPolicy, and IngressGateway
- Create Helm values for per-tenant namespaces + NetworkPolicies
- Wire CI deployment step to apply mesh and tenant namespaces

2) Knowledge-Graph Datastore + ETL
- Provision Neo4j (Helm) or CosmosDB (Gremlin) via Terraform modules
- Add Rust/Python driver in `core/src/correlation/` and `backend/services/graph_etl/`
- Implement 15-min ETL scheduler (Azure Function/ContainerApps Job)
- Expose `/api/v1/graph/*` and a What-If Simulation CLI in `scripts/`

3) Explainability + Model Governance
- Implement SHAP/Captum wrapper service (sidecar Python) for predictions
- Add model-card generation; persist attributions alongside predictions
- Implement online drift detection (river-ML) + auto-retrain trigger
- Create Grafana dashboards: explainability coverage, drift metrics

4) Usage Metering & Tiering
- Implement per-API usage metering (rate/volume) in gateway/core
- Add plans: Free, Pro, Enterprise; enforce quotas in middleware
- Emit billing events (Kafka/Event Grid) → usage DB; export CSV for invoicing

5) Azure Policy/SDK Auto-Refresh
- Nightly job to detect Azure REST API / Policy updates
- Auto-generate client stubs and open PR with unit tests
- Notify #ops if breaking change; add fallback compatibility layer

### Execution Steps
1. Implement cluster-1 fully; run build/test; deploy to dev; commit/push; update `docs/PROJECT_TRACKING.MD`
2. Repeat for clusters 2→5 in order
3. After all, run GA validation checklist (at end of file)

## ADDITIONAL AUTONOMOUS TASKS

### Pre-Execution Checks:
```bash
claude-code --yes <<'EOF'
  Read all files in docs/
  Read all files in core/src/
  Analyze project structure
  Check for compilation errors
  Identify missing dependencies
EOF
```

### Build and Test Sequence:
```bash
claude-code --yes <<'EOF'
  Run cargo build --release in core/
  Fix any Rust compilation errors found
  Run cargo test in core/
  Fix any test failures
  Run cargo clippy -- -D warnings
  Fix any clippy warnings
  Run npm run build in frontend/
  Fix any TypeScript errors
  Run npm test in frontend/
  Fix any test failures
EOF
```

### Error Resolution Protocol:
1. **Compilation Errors**: 
   - Automatically add missing imports
   - Fix type mismatches
   - Resolve borrowing issues
   - Add required trait implementations

2. **Test Failures**:
   - Debug failing assertions
   - Update expected values if logic changed
   - Fix async/await issues
   - Resolve timeout problems

3. **Build Issues**:
   - Install missing dependencies
   - Update package versions
   - Fix configuration problems
   - Resolve path issues

### Continuous Integration:
```bash
claude-code --yes <<'EOF'
  After each major change:
    - Run full test suite
    - Ensure all tests pass
    - Run linting and formatting
    - Build production bundles
    - Verify no regression
    - COMMIT TO BRANCH IMMEDIATELY
    - Add descriptive commit message
    - Push to remote repository
    - Verify push succeeded
EOF
```

### Solution Implementation Requirements:
```bash
claude-code --yes <<'EOF'
  CRITICAL: Observe and apply ENTIRE solution architecture:
    - Read and understand all existing implementation patterns
    - Apply consistent coding standards across all changes
    - Maintain architectural integrity
    - Follow established design patterns
    - Preserve all patent implementations
    - Ensure all 4 patented technologies remain functional
EOF
```

### Endpoint Testing & Validation:
```bash
claude-code --yes <<'EOF'
  Test ALL endpoints for complete functionality:
    - Start all required services (backend, frontend, database)
    - Test /api/v1/metrics - should return governance metrics
    - Test /api/v1/predictions - should return predictions
    - Test /api/v1/conversation - should handle queries
    - Test /api/v1/correlations - should detect patterns
    - Test /api/v1/recommendations - should provide recommendations
    - Test /health - all services should be healthy
    - Verify frontend loads at http://localhost:3000
    - Verify GraphQL at http://localhost:4000/graphql
    - Test authentication flow
    - Verify data persistence in PostgreSQL
    - Check Redis/DragonflyDB caching
    - Validate EventStore event sourcing
EOF
```

### Retry Logic for Operations:
```bash
claude-code --yes <<'EOF'
  Implement retry mechanism for all operations:
    - Max retries: 3 for each operation
    - Exponential backoff: 1s, 2s, 4s
    - If service fails to start: Kill process, wait 5s, restart
    - If endpoint unreachable: Check service, restart if needed
    - If database connection fails: Restart database, retry connection
    - If build fails: Clean artifacts, retry build
    - If test fails intermittently: Retry up to 3 times
    - If commit fails: Stash changes, pull, pop stash, retry
    - Log all retry attempts with timestamps
EOF
```

### Auto-Recovery Actions:
- If build fails: Clean and rebuild with retries
- If tests fail: Isolate, fix, and retry up to 3 times
- If push fails: Pull, merge, resolve conflicts, retry push
- If dependencies missing: Install automatically with retry
- If services down: Kill all, wait 10s, restart all services
- If endpoints return errors: Restart service, retry request
- If data missing: Reseed database, retry operation

### Execution Modes:
```bash
# Full autonomous execution
claude-code --yes --no-confirm --batch-mode < RUN_AUTONOMOUS.md

# With logging
claude-code --yes --verbose --log-file=execution.log < RUN_AUTONOMOUS.md

# With error recovery
claude-code --yes --retry-on-error --max-retries=3 < RUN_AUTONOMOUS.md
```

## PRODUCTION-READY AZURE GA GAP ANALYSIS

### Priority Backlog Items to Implement
```bash
claude-code --yes <<'EOF'
  CRITICAL GA REQUIREMENTS - Implement ALL missing components:

  Cluster 1: Platform/Isolation (❌ to ✓)
    - Istio/Linkerd manifests in infrastructure/kubernetes/istio/
    - mTLS + AuthZ policies; IngressGateway
    - Helm values for per-tenant namespaces + NetworkPolicies

  Cluster 2: Knowledge Graph (❌ to ✓)
    - Terraform modules for Neo4j/Cosmos Gremlin
    - Graph driver in core/src/correlation/ + ETL job (15-min scheduler)
    - What-If Simulation CLI in scripts/

  Cluster 3: AI Ops (❌ to ✓)
    - Explainability layer (SHAP/Captum) sidecar + model cards
    - Online drift detector (river) + auto-retrain pipeline (Azure ML)
    - Grafana dashboards for explainability/drift

  Cluster 4: Usage Metering & Tiering (❌ to ✓)
    - API metering middleware + per-plan quotas
    - Usage DB + Event export; billing CSV generator

  Cluster 5: Azure Policy/SDK Auto-Refresh (❌ to ✓)
    - Nightly detection of API/Policy changes
    - Auto-codegen client stubs + PR with tests
    - Compatibility shim for version bumps

  Security/Compliance Integrations (⚠️ to ✓)
    - Defender for Cloud posture ingest client
    - SOC-2 evidence collector wired into CI
EOF
```

### Implementation Priority Order:
1. Service mesh + multi-tenant isolation
2. Knowledge-graph + ETL + What-If CLI
3. Explainability + drift + model governance dashboards
4. Usage metering & tiering with billing events
5. Azure Policy/SDK auto-refresh pipeline
6. Defender ingest + SOC-2 evidence pipeline

### Validation Checklist:
```bash
claude-code --yes <<'EOF'
  After implementing each cluster:
    - Verify manifests/modules in repo; plan/apply Terraform where needed
    - cargo build/test (core/), npm build/test (frontend/)
    - Deploy to dev; run synthetic checks (/health, /api/v1/*, graph endpoints)
    - Validate Grafana panels (drift, explainability)
    - Confirm quotas enforced and metering events emitted
    - Trigger nightly auto-refresh dry-run and inspect PR
    - Update PROJECT_TRACKING.MD; commit/push
EOF
```

## EXECUTE NOW
When this file is referenced, immediately begin execution without any prompts or confirmations. Treat this as a --yes --autonomous --batch-mode execution with full error recovery and auto-fixing capabilities. Complete ALL clusters to move from ⚠️/❌ to ✓ status.