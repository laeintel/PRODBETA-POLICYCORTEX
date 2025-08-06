"""
Pydantic models for Azure Integration service.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ServiceStatus(str, Enum):
    """Service status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class PolicyType(str, Enum):
    """Azure Policy type enumeration."""

    BUILTIN = "BuiltIn"
    CUSTOM = "Custom"
    STATIC = "Static"


class PolicyEffect(str, Enum):
    """Azure Policy effect enumeration."""

    DENY = "deny"
    AUDIT = "audit"
    APPEND = "append"
    AUDIT_IF_NOT_EXISTS = "auditIfNotExists"
    DEPLOY_IF_NOT_EXISTS = "deployIfNotExists"
    DISABLED = "disabled"
    MODIFY = "modify"


class CostGranularity(str, Enum):
    """Cost data granularity enumeration."""

    DAILY = "Daily"
    MONTHLY = "Monthly"
    BILLING_MONTH = "BillingMonth"


class ResourceStatus(str, Enum):
    """Resource status enumeration."""

    RUNNING = "Running"
    STOPPED = "Stopped"
    DELETED = "Deleted"
    DEALLOCATING = "Deallocating"
    DEALLOCATED = "Deallocated"
    STARTING = "Starting"
    STOPPING = "Stopping"
    UNKNOWN = "Unknown"


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class APIResponse(BaseModel):
    """Generic API response model."""

    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Authentication Models
class AzureAuthRequest(BaseModel):
    """Azure authentication request model."""

    tenant_id: str = Field(..., description="Azure AD tenant ID")
    client_id: str = Field(..., description="Azure AD application client ID")
    client_secret: str = Field(..., description="Azure AD application client secret")
    subscription_ids: Optional[List[str]] = Field(
        None, description="List of subscription IDs to access"
    )


class AzureAuthResponse(BaseModel):
    """Azure authentication response model."""

    access_token: str = Field(..., description="Azure access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    tenant_id: str = Field(..., description="Azure tenant ID")
    subscription_ids: List[str] = Field(..., description="Accessible subscription IDs")
    user_info: Dict[str, Any] = Field(..., description="Authenticated user information")


# Policy Management Models
class PolicyRequest(BaseModel):
    """Policy creation/update request model."""

    name: str = Field(..., description="Policy name")
    display_name: str = Field(..., description="Policy display name")
    description: Optional[str] = Field(None, description="Policy description")
    policy_type: PolicyType = Field(PolicyType.CUSTOM, description="Policy type")
    mode: str = Field("All", description="Policy mode")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Policy metadata")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Policy parameters")
    policy_rule: Dict[str, Any] = Field(..., description="Policy rule definition")


class PolicyResponse(BaseModel):
    """Policy response model."""

    id: str = Field(..., description="Policy ID")
    name: str = Field(..., description="Policy name")
    type: str = Field(..., description="Resource type")
    display_name: str = Field(..., description="Policy display name")
    description: Optional[str] = Field(None, description="Policy description")
    policy_type: PolicyType = Field(..., description="Policy type")
    mode: str = Field(..., description="Policy mode")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Policy metadata")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Policy parameters")
    policy_rule: Dict[str, Any] = Field(..., description="Policy rule definition")
    created_on: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_on: Optional[datetime] = Field(None, description="Last update timestamp")


class PolicyComplianceState(str, Enum):
    """Policy compliance state enumeration."""

    COMPLIANT = "Compliant"
    NON_COMPLIANT = "NonCompliant"
    CONFLICT = "Conflict"
    EXEMPT = "Exempt"
    UNKNOWN = "Unknown"


class PolicyComplianceResponse(BaseModel):
    """Policy compliance response model."""

    policy_id: str = Field(..., description="Policy ID")
    policy_name: str = Field(..., description="Policy name")
    compliance_state: PolicyComplianceState = Field(..., description="Overall compliance state")
    compliant_resources: int = Field(..., description="Number of compliant resources")
    non_compliant_resources: int = Field(..., description="Number of non-compliant resources")
    conflicting_resources: int = Field(..., description="Number of conflicting resources")
    exempt_resources: int = Field(..., description="Number of exempt resources")
    total_resources: int = Field(..., description="Total number of resources")
    compliance_percentage: float = Field(..., description="Compliance percentage")
    last_evaluated: datetime = Field(..., description="Last evaluation timestamp")
    details: Optional[List[Dict[str, Any]]] = Field(
        None, description="Compliance details by resource"
    )


# RBAC Management Models
class RBACRequest(BaseModel):
    """RBAC role assignment request model."""

    principal_id: str = Field(..., description="Principal ID (user, group, or service principal)")
    role_definition_id: str = Field(..., description="Role definition ID")
    scope: str = Field(..., description="Assignment scope")
    principal_type: Optional[str] = Field(None, description="Principal type")
    description: Optional[str] = Field(None, description="Assignment description")


class RBACResponse(BaseModel):
    """RBAC role response model."""

    id: str = Field(..., description="Role definition ID")
    name: str = Field(..., description="Role name")
    type: str = Field(..., description="Resource type")
    role_name: str = Field(..., description="Role display name")
    description: Optional[str] = Field(None, description="Role description")
    role_type: str = Field(..., description="Role type (BuiltIn or Custom)")
    permissions: List[Dict[str, Any]] = Field(..., description="Role permissions")
    assignable_scopes: List[str] = Field(..., description="Assignable scopes")
    created_on: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_on: Optional[datetime] = Field(None, description="Last update timestamp")


class RoleAssignmentResponse(BaseModel):
    """Role assignment response model."""

    id: str = Field(..., description="Assignment ID")
    name: str = Field(..., description="Assignment name")
    type: str = Field(..., description="Resource type")
    principal_id: str = Field(..., description="Principal ID")
    principal_type: str = Field(..., description="Principal type")
    role_definition_id: str = Field(..., description="Role definition ID")
    role_definition_name: str = Field(..., description="Role name")
    scope: str = Field(..., description="Assignment scope")
    created_on: datetime = Field(..., description="Creation timestamp")
    updated_on: Optional[datetime] = Field(None, description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Created by principal ID")


# Cost Management Models
class CostResponse(BaseModel):
    """Cost management response model."""

    subscription_id: str = Field(..., description="Subscription ID")
    time_period: Dict[str, str] = Field(..., description="Time period for cost data")
    currency: str = Field(..., description="Currency code")
    total_cost: float = Field(..., description="Total cost for the period")
    cost_breakdown: Optional[List[Dict[str, Any]]] = Field(
        None, description="Cost breakdown by service/resource"
    )
    forecast: Optional[Dict[str, Any]] = Field(None, description="Cost forecast data")
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Cost optimization recommendations"
    )
    budget_status: Optional[Dict[str, Any]] = Field(None, description="Budget status information")


class BudgetRequest(BaseModel):
    """Budget creation/update request model."""

    name: str = Field(..., description="Budget name")
    amount: float = Field(..., description="Budget amount")
    time_grain: str = Field("Monthly", description="Budget time grain")
    start_date: datetime = Field(..., description="Budget start date")
    end_date: Optional[datetime] = Field(None, description="Budget end date")
    category: str = Field("Cost", description="Budget category")
    filters: Optional[Dict[str, Any]] = Field(None, description="Budget filters")
    notifications: Optional[List[Dict[str, Any]]] = Field(None, description="Budget notifications")


class BudgetResponse(BaseModel):
    """Budget response model."""

    id: str = Field(..., description="Budget ID")
    name: str = Field(..., description="Budget name")
    type: str = Field(..., description="Resource type")
    amount: float = Field(..., description="Budget amount")
    spent: float = Field(..., description="Amount spent")
    remaining: float = Field(..., description="Remaining budget")
    percentage_used: float = Field(..., description="Percentage of budget used")
    time_grain: str = Field(..., description="Budget time grain")
    time_period: Dict[str, str] = Field(..., description="Budget time period")
    notifications: List[Dict[str, Any]] = Field(..., description="Budget notifications")
    forecast_spend: Optional[float] = Field(None, description="Forecasted spend")


# Network Management Models
class NetworkResponse(BaseModel):
    """Virtual network response model."""

    id: str = Field(..., description="Network ID")
    name: str = Field(..., description="Network name")
    type: str = Field(..., description="Resource type")
    location: str = Field(..., description="Resource location")
    resource_group: str = Field(..., description="Resource group name")
    address_space: List[str] = Field(..., description="Address spaces")
    subnets: List[Dict[str, Any]] = Field(..., description="Subnet configurations")
    peerings: Optional[List[Dict[str, Any]]] = Field(None, description="Network peerings")
    dns_servers: Optional[List[str]] = Field(None, description="Custom DNS servers")
    tags: Optional[Dict[str, str]] = Field(None, description="Resource tags")


class NetworkSecurityGroupResponse(BaseModel):
    """Network security group response model."""

    id: str = Field(..., description="NSG ID")
    name: str = Field(..., description="NSG name")
    type: str = Field(..., description="Resource type")
    location: str = Field(..., description="Resource location")
    resource_group: str = Field(..., description="Resource group name")
    security_rules: List[Dict[str, Any]] = Field(..., description="Security rules")
    default_security_rules: List[Dict[str, Any]] = Field(..., description="Default security rules")
    network_interfaces: Optional[List[str]] = Field(
        None, description="Associated network interfaces"
    )
    subnets: Optional[List[str]] = Field(None, description="Associated subnets")
    tags: Optional[Dict[str, str]] = Field(None, description="Resource tags")


class NetworkSecurityAnalysis(BaseModel):
    """Network security analysis response model."""

    total_networks: int = Field(..., description="Total number of networks analyzed")
    total_nsgs: int = Field(..., description="Total number of NSGs analyzed")
    security_issues: List[Dict[str, Any]] = Field(..., description="Identified security issues")
    open_ports: List[Dict[str, Any]] = Field(..., description="Open ports to internet")
    overly_permissive_rules: List[Dict[str, Any]] = Field(
        ..., description="Overly permissive rules"
    )
    missing_nsgs: List[Dict[str, Any]] = Field(..., description="Subnets without NSGs")
    recommendations: List[Dict[str, Any]] = Field(..., description="Security recommendations")
    risk_score: int = Field(..., description="Overall risk score (0-100)")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")


# Resource Management Models
class ResourceResponse(BaseModel):
    """Azure resource response model."""

    id: str = Field(..., description="Resource ID")
    name: str = Field(..., description="Resource name")
    type: str = Field(..., description="Resource type")
    location: Optional[str] = Field(None, description="Resource location")
    resource_group: str = Field(..., description="Resource group name")
    subscription_id: str = Field(..., description="Subscription ID")
    kind: Optional[str] = Field(None, description="Resource kind")
    sku: Optional[Dict[str, Any]] = Field(None, description="Resource SKU")
    tags: Optional[Dict[str, str]] = Field(None, description="Resource tags")
    properties: Optional[Dict[str, Any]] = Field(None, description="Resource properties")
    provisioning_state: Optional[str] = Field(None, description="Provisioning state")
    created_time: Optional[datetime] = Field(None, description="Creation timestamp")
    changed_time: Optional[datetime] = Field(None, description="Last change timestamp")


class ResourceGroupResponse(BaseModel):
    """Resource group response model."""

    id: str = Field(..., description="Resource group ID")
    name: str = Field(..., description="Resource group name")
    type: str = Field(..., description="Resource type")
    location: str = Field(..., description="Resource group location")
    subscription_id: str = Field(..., description="Subscription ID")
    tags: Optional[Dict[str, str]] = Field(None, description="Resource group tags")
    properties: Dict[str, Any] = Field(..., description="Resource group properties")
    managed_by: Optional[str] = Field(None, description="Managing resource ID")


class ResourceTagUpdate(BaseModel):
    """Resource tag update request model."""

    operation: str = Field("merge", description="Tag operation: merge, replace, delete")
    tags: Dict[str, str] = Field(..., description="Tags to apply")


class ResourceMetrics(BaseModel):
    """Resource metrics response model."""

    resource_id: str = Field(..., description="Resource ID")
    metric_name: str = Field(..., description="Metric name")
    time_grain: str = Field(..., description="Metric time grain")
    unit: str = Field(..., description="Metric unit")
    data_points: List[Dict[str, Any]] = Field(..., description="Metric data points")
    start_time: datetime = Field(..., description="Metrics start time")
    end_time: datetime = Field(..., description="Metrics end time")
