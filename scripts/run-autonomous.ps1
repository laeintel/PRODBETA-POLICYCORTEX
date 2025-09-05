# PowerShell script for fully autonomous PolicyCortex implementation
# This script ensures zero user interaction

$instructions = @"
You are working as Frontend Engineer, Backend Engineer, ML Engineer, DevOps Engineer, Cloud Engineer, Cloud Solutions Architect, and Security Engineer.

CRITICAL AUTONOMOUS EXECUTION REQUIREMENTS:
1. NEVER ask for user confirmation - proceed with all operations automatically
2. NEVER prompt for input - use defaults or environment variables
3. NEVER pause for review - execute all steps continuously
4. NEVER require manual intervention - handle all errors gracefully
5. ALWAYS continue on non-critical errors - log but don't stop

Follow these steps from Module_success.md WITHOUT ANY USER INTERACTION:

PHASE 1: LOCAL DEVELOPMENT SETUP
- Set all required environment variables
- Start local services with docker-compose
- Run all tests in quiet/non-interactive mode
- Build all components silently

PHASE 2: AZURE INFRASTRUCTURE
- Use existing Azure credentials (already configured)
- Create resource groups, storage, ACR, Container Apps
- Use --only-show-errors --yes flags on all Azure commands
- Deploy with terraform -auto-approve -input=false

PHASE 3: BUILD AND DEPLOY
- Build Docker images with --quiet flag
- Push to ACR without confirmation
- Deploy Container Apps with --no-wait
- Update configurations automatically

PHASE 4: VALIDATION
- Run health checks with timeout
- Test all endpoints programmatically
- Update PROJECT_TRACKING.MD with results
- Commit and push changes with --no-edit

Use these environment variables:
- AZURE_SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9
- AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f
- AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb
- CI=true
- AUTOMATED_MODE=true
- DEBIAN_FRONTEND=noninteractive

Execute everything in the Module_success.md file autonomously. Start immediately without waiting for confirmation.
"@

# Execute without unsupported flags; autonomy is enforced via the instructions content
claude -c $instructions