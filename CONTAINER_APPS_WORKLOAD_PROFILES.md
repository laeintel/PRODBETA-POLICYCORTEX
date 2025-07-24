# Container Apps Workload Profile Migration

## Overview

This document outlines the migration of Azure Container Apps from the default Consumption workload profile to dedicated workload profiles for better performance, predictable costs, and improved isolation.

## Problem Statement

Container Apps were initially deployed using the default Consumption workload profile, which can lead to:
- Unpredictable performance due to shared resources
- Variable costs based on actual usage
- Potential resource contention during peak loads
- Less isolation between applications

## Solution

Migrate all Container Apps to use dedicated workload profiles:
- **Dedicated-D4**: For most applications (API Gateway, Azure Integration, Data Processing, Conversation, Notification, Frontend)
- **Dedicated-D8**: For high-performance applications (AI Engine)

## Migration Scripts

### Immediate Migration (Azure CLI)

1. **Single Environment Migration**:
   ```powershell
   .\migrate-container-apps-to-dedicated-profile.ps1 -Environment "dev"
   ```

2. **All Environments Migration**:
   ```powershell
   # Preview changes
   .\migrate-all-environments.ps1 -WhatIf
   
   # Perform migration
   .\migrate-all-environments.ps1
   ```

3. **Bicep-Specific Migration**:
   ```powershell
   # Preview changes
   .\migrate-bicep-container-apps.ps1 -Environment "dev" -WhatIf
   
   # Perform migration
   .\migrate-bicep-container-apps.ps1 -Environment "dev"
   ```

4. **Bash Script** (for Linux/macOS):
   ```bash
   ./migrate-container-apps-to-dedicated-profile.sh dev
   ```

### Infrastructure Code Updates

#### Terraform Configuration

Updated `infrastructure/terraform/container-apps.tf` to include:

```hcl
resource "azurerm_container_app" "api_gateway" {
  # ... other configuration ...
  workload_profile_name = "Dedicated-D4"
}

resource "azurerm_container_app" "ai_engine" {
  # ... other configuration ...
  workload_profile_name = "Dedicated-D8"  # High-performance profile
}
```

#### Bicep Configuration

Updated `infrastructure/bicep/modules/container-apps.bicep` to include:

```bicep
// Service configurations with workload profile assignments
var services = [
  {
    name: 'api-gateway'
    // ... other properties
    workloadProfile: 'GeneralPurpose'  // Dedicated-D4 equivalent
  }
  {
    name: 'ai-engine'
    // ... other properties
    workloadProfile: 'HighPerformance'  // Dedicated-D8 equivalent
  }
  // ... other services
]

// In template configuration
template: {
  workloadProfileName: service.workloadProfile
  containers: [
    // ... container configuration
  ]
}
```

Updated `infrastructure/bicep/modules/container-apps-environment.bicep` to include:

```bicep
workloadProfiles: [
  {
    name: 'Consumption'
    workloadProfileType: 'Consumption'
  }
  {
    name: 'GeneralPurpose'
    workloadProfileType: 'D4'
    minimumCount: 0
    maximumCount: 10
  }
  {
    name: 'HighPerformance'
    workloadProfileType: 'D8'
    minimumCount: 0
    maximumCount: 5
  }
]
```

## Workload Profile Definitions

### Environment Configuration

Each Container Apps Environment includes these workload profiles:

```hcl
# Terraform (main.tf)
workload_profile {
  name                  = "Consumption"
  workload_profile_type = "Consumption"
}

workload_profile {
  name                  = "Dedicated-D4"
  workload_profile_type = "D4"
  minimum_count         = 1
  maximum_count         = 3
}

workload_profile {
  name                  = "Dedicated-D8"
  workload_profile_type = "D8"
  minimum_count         = 0
  maximum_count         = 2
}
```

```bicep
// Bicep (container-apps-environment.bicep)
workloadProfiles: [
  {
    name: 'Consumption'
    workloadProfileType: 'Consumption'
  }
  {
    name: 'GeneralPurpose'
    workloadProfileType: 'D4'
    minimumCount: 0
    maximumCount: 10
  }
  {
    name: 'HighPerformance'
    workloadProfileType: 'D8'
    minimumCount: 0
    maximumCount: 5
  }
]
```

## Application-Specific Profile Assignments

| Application | Workload Profile | Reasoning |
|-------------|------------------|-----------|
| API Gateway | Dedicated-D4 | Public-facing, needs consistent performance |
| Azure Integration | Dedicated-D4 | Core service requiring reliability |
| AI Engine | HighPerformance (D8) | High CPU/memory requirements for ML workloads |
| Data Processing | Dedicated-D4 | Steady workload requiring consistent resources |
| Conversation | Dedicated-D4 | User-facing, needs responsive performance |
| Notification | Dedicated-D4 | Critical for user notifications |
| Frontend | Dedicated-D4 | User interface requiring fast response times |

## Verification Commands

### Check Current Workload Profile Assignments

```bash
# Single environment
az containerapp list \
  --resource-group rg-policycortex-dev \
  --query "[].{Name:name,WorkloadProfile:properties.template.workloadProfileName}" \
  -o table

# All environments
for env in dev staging prod; do
  echo "=== $env Environment ==="
  az containerapp list \
    --resource-group rg-policycortex-$env \
    --query "[].{Name:name,WorkloadProfile:properties.template.workloadProfileName}" \
    -o table
  echo
done
```

### Check Workload Profile Utilization

```bash
az containerapp env show \
  --name cae-policycortex-dev \
  --resource-group rg-policycortex-dev \
  --query "properties.workloadProfiles" \
  -o table
```

## Cost Impact

### Before Migration
- Pay per execution time and memory consumption
- Variable costs based on actual usage
- Potential for cost spikes during high traffic

### After Migration
- Fixed cost per workload profile instance
- More predictable monthly costs
- Better cost planning and budgeting

### Estimated Monthly Costs (USD)
- **Dedicated-D4**: ~$140-200/month per profile
- **Dedicated-D8**: ~$280-400/month per profile

> Note: Actual costs depend on region, scaling configuration, and usage patterns.

## Monitoring and Optimization

### Key Metrics to Monitor
1. **CPU Utilization**: Should be 50-80% for optimal efficiency
2. **Memory Usage**: Monitor for memory leaks or inefficient allocation
3. **Response Times**: Should improve with dedicated resources
4. **Scaling Events**: Monitor auto-scaling behavior
5. **Cost Trends**: Track monthly spend vs. previous consumption-based billing

### Optimization Recommendations
1. **Right-size Profiles**: Start with D4, scale to D8 only if needed
2. **Auto-scaling**: Configure appropriate min/max replica counts
3. **Resource Limits**: Set CPU/memory limits to prevent resource hogging
4. **Regular Review**: Monthly review of utilization and costs

## Rollback Plan

If issues occur, rollback steps:

1. **Immediate Rollback** (Azure CLI):
   ```bash
   az containerapp update \
     --name ca-app-name-dev \
     --resource-group rg-policycortex-dev \
     --workload-profile-name Consumption
   ```

2. **Infrastructure Rollback**:
   - Remove `workload_profile_name` from Terraform resources
   - Apply Terraform changes
   - Redeploy with infrastructure as code

## Deployment Integration

### CI/CD Pipeline Updates

Ensure deployment pipelines use updated infrastructure code:

1. **Terraform Deployments**: Use updated `.tf` files
2. **Bicep Deployments**: Use updated `.bicep` files
3. **Testing**: Verify apps start correctly on dedicated profiles
4. **Monitoring**: Add alerts for workload profile resource usage

### Environment Promotion

When promoting between environments:
1. Ensure workload profiles exist in target environment
2. Verify profile capacity meets application needs
3. Test application performance post-migration
4. Monitor for 24-48 hours after promotion

## Troubleshooting

### Common Issues

1. **Profile Not Found Error**
   - Verify workload profile exists in environment
   - Check profile name spelling/case sensitivity

2. **Insufficient Capacity**
   - Increase profile maximum count
   - Consider using larger profile type (D8 instead of D4)

3. **Application Won't Start**
   - Check resource limits (CPU/memory)
   - Verify profile has sufficient capacity
   - Review application logs for specific errors

### Support Contacts

- **Infrastructure Team**: For workload profile configuration
- **Application Teams**: For application-specific performance issues
- **Azure Support**: For platform-level issues

## References

- [Azure Container Apps Workload Profiles](https://docs.microsoft.com/en-us/azure/container-apps/workload-profiles-overview)
- [Container Apps Pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/)
- [Performance Best Practices](https://docs.microsoft.com/en-us/azure/container-apps/best-practices) 