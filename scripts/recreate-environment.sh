#!/bin/bash
# Script to completely recreate the environment from scratch
# This proves Infrastructure as Code is working correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="${1:-dev}"
SUBSCRIPTION_ID="${2:-205b477d-17e7-4b3b-92c1-32cf02626b78}"
DESTROY_FIRST="${3:-false}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PolicyCortex Environment Recreation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Environment: ${GREEN}$ENVIRONMENT${NC}"
echo -e "Subscription: ${GREEN}$SUBSCRIPTION_ID${NC}"
echo -e "Destroy First: ${YELLOW}$DESTROY_FIRST${NC}"
echo ""

# Set subscription
echo -e "${YELLOW}Setting Azure subscription...${NC}"
az account set --subscription "$SUBSCRIPTION_ID"

# Get resource names using consistent pattern
PROJECT="cortex"
HASH=$(echo -n "$SUBSCRIPTION_ID-$PROJECT" | md5sum | cut -c1-6)

# Resource names
RG="rg-${PROJECT}-${ENVIRONMENT}"
CR="cr${PROJECT}${ENVIRONMENT}${HASH}"
CAE="cae-${PROJECT}-${ENVIRONMENT}"

if [[ "$DESTROY_FIRST" == "true" ]]; then
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}DESTROYING EXISTING ENVIRONMENT${NC}"
    echo -e "${RED}========================================${NC}"
    
    read -p "Are you SURE you want to destroy the $ENVIRONMENT environment? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo -e "${YELLOW}Destruction cancelled.${NC}"
        exit 0
    fi
    
    # Delete resource group (this deletes everything in it)
    echo -e "${YELLOW}Deleting resource group $RG...${NC}"
    if az group exists -n "$RG" 2>/dev/null; then
        az group delete -n "$RG" --yes --no-wait
        echo -e "${GREEN}Resource group deletion initiated.${NC}"
        
        # Wait for deletion to complete
        echo -e "${YELLOW}Waiting for deletion to complete...${NC}"
        while az group exists -n "$RG" 2>/dev/null; do
            echo -n "."
            sleep 10
        done
        echo ""
        echo -e "${GREEN}Resource group deleted successfully.${NC}"
    else
        echo -e "${YELLOW}Resource group does not exist.${NC}"
    fi
    
    # Clean Terraform state
    echo -e "${YELLOW}Cleaning Terraform state...${NC}"
    cd infrastructure/terraform
    rm -rf .terraform terraform.tfstate* .terraform.lock.hcl
    echo -e "${GREEN}Terraform state cleaned.${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CREATING ENVIRONMENT FROM SCRATCH${NC}"
echo -e "${GREEN}========================================${NC}"

# Navigate to Terraform directory
cd infrastructure/terraform

# Initialize Terraform
echo -e "${YELLOW}Initializing Terraform...${NC}"
terraform init \
    -backend-config="resource_group_name=rg-tfstate-${PROJECT}-${ENVIRONMENT}" \
    -backend-config="storage_account_name=sttf${PROJECT}${ENVIRONMENT}${HASH}" \
    -backend-config="container_name=tfstate" \
    -backend-config="key=${ENVIRONMENT}.tfstate"

# Create terraform.tfvars
echo -e "${YELLOW}Creating terraform.tfvars...${NC}"
cat > terraform.tfvars <<EOF
environment = "${ENVIRONMENT}"
location    = "eastus"
project_name = "${PROJECT}"
EOF

# Plan Terraform
echo -e "${YELLOW}Planning Terraform deployment...${NC}"
terraform plan -out=tfplan

# Apply Terraform
echo -e "${YELLOW}Applying Terraform configuration...${NC}"
terraform apply tfplan

# Get outputs
echo -e "${YELLOW}Getting Terraform outputs...${NC}"
terraform output -json > outputs.json

# Extract key values
REGISTRY_NAME=$(terraform output -raw container_registry_name)
REGISTRY_URL=$(terraform output -raw container_registry_url)
CORE_APP_URL=$(terraform output -raw core_app_url)
FRONTEND_APP_URL=$(terraform output -raw frontend_app_url)
GRAPHQL_APP_URL=$(terraform output -raw graphql_app_url)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ENVIRONMENT CREATED SUCCESSFULLY!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Resource Details:${NC}"
echo -e "  Resource Group: ${GREEN}$RG${NC}"
echo -e "  Container Registry: ${GREEN}$REGISTRY_NAME${NC}"
echo -e "  Container Environment: ${GREEN}$CAE${NC}"
echo ""
echo -e "${BLUE}Application URLs:${NC}"
echo -e "  Frontend: ${GREEN}$FRONTEND_APP_URL${NC}"
echo -e "  Core API: ${GREEN}$CORE_APP_URL${NC}"
echo -e "  GraphQL: ${GREEN}$GRAPHQL_APP_URL${NC}"
echo ""

# Trigger CI/CD pipeline to deploy applications
echo -e "${YELLOW}Triggering CI/CD pipeline to deploy applications...${NC}"
gh workflow run entry.yml \
    --ref $(git branch --show-current) \
    -f full_run=true \
    -f target_env="${ENVIRONMENT}"

echo -e "${GREEN}Pipeline triggered. Check GitHub Actions for progress.${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}RECREATION COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"