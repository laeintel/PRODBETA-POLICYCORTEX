# PolicyCortex Optimization Plan

## Current Bottlenecks Analysis

### 1. Monolithic Rust Core
**Problem**: Single crate mixing API, auth, evidence, orchestration
**Impact**: 
- Compile times > 5 minutes for any change
- High cognitive load
- Difficult to test individual components

**Solution**: Workspace-based architecture
```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "crates/api",
    "crates/auth", 
    "crates/evidence",
    "crates/orchestration",
    "crates/shared",
    "crates/models"
]

[workspace.dependencies]
axum = "0.7"
tokio = { version = "1.38", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

**Benefits**:
- Parallel compilation
- Independent testing
- Clear separation of concerns
- Faster incremental builds

### 2. CI Matrix Explosion
**Problem**: Building all images on every push
**Impact**: 15+ minute feedback for frontend-only changes

**Solution**: Path-based conditional builds
```yaml
# .github/workflows/application.yml
jobs:
  changes:
    runs-on: self-hosted
    outputs:
      core: ${{ steps.filter.outputs.core }}
      frontend: ${{ steps.filter.outputs.frontend }}
      graphql: ${{ steps.filter.outputs.graphql }}
    steps:
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            core:
              - 'core/**'
              - 'Cargo.toml'
              - 'Cargo.lock'
            frontend:
              - 'frontend/**'
              - 'package.json'
              - 'package-lock.json'
            graphql:
              - 'graphql/**'
```

### 3. Terraform Apply in CI
**Problem**: Full infrastructure apply on every push
**Impact**: 
- Expensive AWS/Azure costs
- Lock contention
- 5-10 minute delays

**Solution**: Plan-only for PRs, Apply on merge
```yaml
- name: Terraform Plan
  if: github.event_name == 'pull_request'
  run: terraform plan -out=tfplan

- name: Terraform Apply
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: terraform apply -auto-approve
```

### 4. Demo Mode Toggle
**Problem**: Mock data injection scattered across codebase
**Impact**: Hard to maintain, risky for production

**Solution**: Feature flag system
```rust
// crates/shared/src/features.rs
pub struct Features {
    pub demo_mode: bool,
    pub mock_azure: bool,
    pub synthetic_metrics: bool,
}

impl Features {
    pub fn from_env() -> Self {
        Self {
            demo_mode: env::var("DEMO_MODE").is_ok(),
            mock_azure: env::var("MOCK_AZURE").is_ok(),
            synthetic_metrics: env::var("USE_SYNTHETIC_METRICS").is_ok(),
        }
    }
}
```

### 5. Security Scans (SBOM+SLSA+Trivy)
**Problem**: Sequential execution, single point of failure
**Impact**: 10+ minutes, fails on network issues

**Solution**: Parallel execution with soft failures
```yaml
security-scans:
  runs-on: self-hosted
  strategy:
    matrix:
      scan: [sbom, slsa, trivy]
    fail-fast: false
  steps:
    - name: Run ${{ matrix.scan }}
      continue-on-error: true
      run: |
        case "${{ matrix.scan }}" in
          sbom) ./scripts/generate-sbom.sh ;;
          slsa) ./scripts/slsa-provenance.sh ;;
          trivy) trivy fs . ;;
        esac
```

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Path-based CI filtering** - Immediate 50% reduction in build times
2. **Parallel security scans** - 70% faster security checks
3. **Terraform plan-only mode** - Eliminate unnecessary applies

### Phase 2: Structural (3-5 days)
1. **Rust workspace migration** - Long-term maintainability
2. **Feature flag system** - Clean demo/prod separation
3. **Build caching optimization** - Docker layer caching, Rust target caching

### Phase 3: Advanced (1 week)
1. **Incremental deployments** - Deploy only changed services
2. **Canary releases** - Gradual rollout with automatic rollback
3. **Performance monitoring** - Track and alert on degradation

## Metrics for Success

| Metric | Current | Target | Impact |
|--------|---------|--------|--------|
| Average CI time | 15-20 min | 5-7 min | 65% reduction |
| Rust compile time | 5+ min | 2 min | 60% reduction |
| Terraform apply frequency | Every push | On merge only | 90% reduction |
| Security scan time | 10+ min | 3 min | 70% reduction |
| Deployment rollback time | Manual | Automated < 2 min | 95% reduction |

## Quick Start Commands

```bash
# Implement path filtering
./scripts/optimize-ci.sh --path-filter

# Convert to Rust workspace
./scripts/refactor-to-workspace.sh

# Enable parallel scans
./scripts/parallelize-security.sh

# Setup feature flags
./scripts/setup-feature-flags.sh
```

## Risk Mitigation

1. **Backup current state** before major changes
2. **Gradual migration** - one component at a time
3. **Feature flags** for all optimizations
4. **Monitoring** before and after changes
5. **Rollback plan** for each optimization

## Next Steps

1. Create optimization branch
2. Implement Phase 1 quick wins
3. Measure impact with benchmarks
4. Document learnings
5. Proceed to Phase 2 if metrics improve