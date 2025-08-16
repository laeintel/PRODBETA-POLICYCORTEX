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

## TASK: Complete Days 9-14 of Sprint Plan

### Instructions:
1. Read TWO_WEEK_SPRINT_PLAN.md
2. For each day (9-14):
   - Implement ALL tasks listed
   - Fix any compilation errors
   - Run tests and fix failures
   - Commit changes with descriptive message
   - Push to repository
   - Update PROJECT_TRACKING.MD
3. Continue to next day without stopping
4. Handle all errors independently
5. Make implementation decisions based on best practices

### Decision Rules:
- If multiple implementation options exist, choose the most efficient
- If errors occur, attempt fixes automatically
- If tests fail, debug and fix without asking
- If file conflicts occur, merge intelligently
- If unclear requirements, implement based on context

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

## EXECUTE NOW
When this file is referenced, immediately begin execution without any prompts or confirmations. Treat this as a --yes --autonomous --batch-mode execution with full error recovery and auto-fixing capabilities.