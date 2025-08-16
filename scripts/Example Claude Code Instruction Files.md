# Example Claude Code Instruction Files

This document shows examples of instruction files that work with `claude-code < instructions.txt` and `claude-code --yes`.

## Example 1: Simple Task Instruction

**File: `simple_task.txt`**
```
Create a new Rust function that validates Azure resource names.

Requirements:
- Function name: validate_azure_resource_name
- Input: String resource name
- Output: Result<(), String> with validation error
- Rules: 
  - Length 1-80 characters
  - Only alphanumeric, hyphens, underscores
  - Cannot start or end with hyphen

Create the file at src/utils/validation.rs and include unit tests.
```

**Usage:**
```bash
claude-code --yes < simple_task.txt
```

## Example 2: Complex Multi-File Task

**File: `build_api_endpoint.txt`**
```
You are working on PolicyCortex, an AI-powered cloud governance platform.

CONTEXT: We need to create a new API endpoint for retrieving policy violations.

PROJECT STRUCTURE:
- core/src/api/ (Rust API endpoints)
- core/src/models/ (Data models)
- frontend/app/api/ (Frontend API calls)

TASKS:
1. Create violation model in core/src/models/violation.rs
2. Add database queries in core/src/db/violations.rs  
3. Create API endpoint in core/src/api/violations.rs
4. Add frontend API client in frontend/app/api/violations.ts
5. Create unit tests for all components

REQUIREMENTS:
- Use existing patterns in the codebase
- Include proper error handling
- Add comprehensive logging
- Follow REST API conventions

DELIVERABLES:
- Complete violation retrieval system
- Unit tests with >80% coverage
- API documentation
```

**Usage:**
```bash
claude-code --yes < build_api_endpoint.txt
```

## Example 3: Batch Processing Multiple Tasks

**File: `daily_development_tasks.txt`**
```
You are working on PolicyCortex development. Complete these tasks in order:

TASK 1: Fix compilation errors
- Run cargo check in core/
- Fix any compilation errors found
- Ensure all tests pass

TASK 2: Update dependencies
- Update Cargo.toml dependencies to latest versions
- Run cargo update
- Fix any breaking changes

TASK 3: Add new feature
- Implement cost prediction API endpoint
- Add corresponding frontend component
- Include integration tests

TASK 4: Documentation
- Update README.md with new features
- Add API documentation
- Update deployment guide

Complete each task before moving to the next. Report progress after each task.
```

**Usage:**
```bash
claude-code --yes < daily_development_tasks.txt
```

## Example 4: Automated Testing and Deployment

**File: `test_and_deploy.txt`**
```
Run complete testing and deployment pipeline for PolicyCortex:

PHASE 1: Testing
- Run all unit tests: cargo test
- Run integration tests: cargo test --test integration
- Run frontend tests: npm test
- Check code coverage and ensure >80%

PHASE 2: Quality Checks
- Run cargo clippy for linting
- Run cargo fmt for formatting
- Check for security vulnerabilities
- Validate API documentation

PHASE 3: Build
- Build release version: cargo build --release
- Build frontend: npm run build
- Create deployment artifacts

PHASE 4: Deploy (if all tests pass)
- Deploy to staging environment
- Run smoke tests
- If successful, deploy to production

Stop at any phase if errors occur and report the issue.
```

**Usage:**
```bash
claude-code --yes < test_and_deploy.txt
```

## Example 5: AI Model Training Task

**File: `train_ml_model.txt`**
```
Train and deploy the PolicyCortex violation prediction model:

SETUP:
- Working directory: ml/
- Model type: LSTM for time-series prediction
- Target: 90% accuracy on violation prediction

TASKS:
1. Data Preparation
   - Load historical violation data from data/violations.csv
   - Clean and preprocess the data
   - Create train/validation/test splits (70/15/15)

2. Model Training
   - Implement LSTM model in ml/models/violation_predictor.py
   - Train with early stopping and validation monitoring
   - Save best model to ml/models/violation_predictor.pkl

3. Evaluation
   - Test on holdout set
   - Generate performance metrics
   - Create confusion matrix and ROC curve

4. Deployment
   - Create inference API in ml/api/prediction_service.py
   - Add model loading and prediction endpoints
   - Test API with sample data

5. Integration
   - Update core Rust service to call ML API
   - Add prediction results to dashboard
   - Include confidence scores

REQUIREMENTS:
- Achieve >90% accuracy
- Inference time <100ms
- Include comprehensive logging
- Add model monitoring

Report metrics and performance after each step.
```

**Usage:**
```bash
claude-code --yes < train_ml_model.txt
```

## Example 6: Here Document Style (Alternative)

**File: `here_doc_example.sh`**
```bash
#!/bin/bash

# Using here document style
claude-code --yes <<'EOF'
Create a complete authentication system for PolicyCortex:

1. Implement JWT token generation and validation
2. Add login/logout endpoints
3. Create middleware for protected routes
4. Add user session management
5. Include password hashing and validation

Use existing project structure and follow security best practices.
EOF
```

**Usage:**
```bash
./here_doc_example.sh
```

## Best Practices for Instruction Files

### 1. Be Specific and Clear
- Provide exact file paths and names
- Specify requirements and constraints
- Include expected outcomes

### 2. Provide Context
- Explain the project structure
- Reference existing patterns
- Clarify the purpose and goals

### 3. Break Down Complex Tasks
- List tasks in logical order
- Specify dependencies between tasks
- Include validation steps

### 4. Include Quality Requirements
- Specify testing requirements
- Include performance criteria
- Add documentation expectations

### 5. Handle Errors Gracefully
- Include error handling requirements
- Specify what to do if tasks fail
- Add logging and monitoring

## Automation Patterns

### Pattern 1: Sequential Task Execution
```
Task 1: Setup
Task 2: Implementation (depends on Task 1)
Task 3: Testing (depends on Task 2)
Task 4: Deployment (depends on Task 3)
```

### Pattern 2: Parallel Task Groups
```
Group A: Backend Development
Group B: Frontend Development
Group C: Testing (depends on A and B)
```

### Pattern 3: Iterative Development
```
Iteration 1: Core functionality
Iteration 2: Enhanced features
Iteration 3: Optimization
Iteration 4: Production readiness
```

## Integration with PolicyCortex Roadmap

The automation system maps your technical roadmap to instruction files:

- **Day 1-3**: Remediation system → `day1-3_remediation.txt`
- **Day 4-6**: ML predictions → `day4-6_ml_predictions.txt`
- **Day 7-9**: Correlation engine → `day7-9_correlation.txt`
- **Day 10-12**: NLP interface → `day10-12_nlp_interface.txt`
- **Day 13-14**: Integration → `day13-14_integration.txt`

Each file contains detailed instructions for implementing the specific features outlined in your roadmap, ensuring consistent progress toward your goals.

