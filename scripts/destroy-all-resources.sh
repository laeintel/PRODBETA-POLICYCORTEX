#!/bin/bash
# Script to completely destroy ALL resources in the environment
# WARNING: This will delete EVERYTHING!

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

echo -e "${RED}========================================${NC}"
echo -e "${RED}⚠️  COMPLETE RESOURCE DESTRUCTION  ⚠️${NC}"
echo -e "${RED}========================================${NC}"
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Subscription: ${YELLOW}$SUBSCRIPTION_ID${NC}"
echo ""

# Confirm destruction
read -p "Are you ABSOLUTELY SURE you want to destroy ALL resources? Type 'DESTROY' to confirm: " confirm
if [[ "$confirm" != "DESTROY" ]]; then
    echo -e "${GREEN}Destruction cancelled.${NC}"
    exit 0
fi

# Set subscription
echo -e "${YELLOW}Setting Azure subscription...${NC}"
az account set --subscription "$SUBSCRIPTION_ID"

# List of resource groups to delete
RESOURCE_GROUPS=(
    "rg-cortex-${ENVIRONMENT}"
    "rg-tfstate-cortex-${ENVIRONMENT}"
    "rg-datafactory-demo"
    "rg-hklaw-datasource-prod"
    "policycortex-gpt4o-resource"
)

echo -e "${BLUE}Resource groups to be deleted:${NC}"
for rg in "${RESOURCE_GROUPS[@]}"; do
    echo "  - $rg"
done
echo ""

# Delete each resource group
for rg in "${RESOURCE_GROUPS[@]}"; do
    echo -e "${YELLOW}Checking resource group: $rg${NC}"
    
    if az group exists -n "$rg" 2>/dev/null; then
        echo -e "${RED}Deleting resource group: $rg${NC}"
        
        # List resources in the group first
        echo "Resources in this group:"
        az resource list -g "$rg" --query "[].{Name:name, Type:type}" -o table || true
        
        # Delete the resource group
        az group delete -n "$rg" --yes --no-wait
        echo -e "${GREEN}Deletion initiated for: $rg${NC}"
    else
        echo -e "${YELLOW}Resource group does not exist: $rg${NC}"
    fi
done

echo ""
echo -e "${YELLOW}Waiting for all deletions to complete...${NC}"

# Wait for all resource groups to be deleted
for rg in "${RESOURCE_GROUPS[@]}"; do
    while az group exists -n "$rg" 2>/dev/null; do
        echo -n "."
        sleep 10
    done
done

echo ""
echo -e "${GREEN}All resource groups have been deleted!${NC}"

# Clean up any orphaned resources
echo -e "${YELLOW}Checking for orphaned resources...${NC}"

# Check for any remaining Container Apps
CONTAINER_APPS=$(az containerapp list --query "[?contains(name, 'cortex')].{Name:name, RG:resourceGroup}" -o json)
if [[ "$CONTAINER_APPS" != "[]" ]]; then
    echo -e "${YELLOW}Found orphaned Container Apps:${NC}"
    echo "$CONTAINER_APPS"
    echo "Please delete these manually if needed."
fi

# Check for any remaining Container Registries
REGISTRIES=$(az acr list --query "[?contains(name, 'cortex')].{Name:name, RG:resourceGroup}" -o json)
if [[ "$REGISTRIES" != "[]" ]]; then
    echo -e "${YELLOW}Found orphaned Container Registries:${NC}"
    echo "$REGISTRIES"
    echo "Please delete these manually if needed."
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ DESTRUCTION COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run the Terraform pipeline to recreate everything"
echo "2. Or use: terraform apply -var='environment=${ENVIRONMENT}'"
echo ""