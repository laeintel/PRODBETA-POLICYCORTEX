"""
Subscription Manager Module
Manages customer subscriptions and plans
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    TRIAL = "trial"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"


class BillingCycle(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class PlanTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class SubscriptionPlan:
    """Represents a subscription plan"""

    plan_id: str
    name: str
    tier: PlanTier
    description: str
    features: List[str]
    limits: Dict[str, Any]
    pricing: Dict[str, float]
    trial_days: int = 0
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """Represents a customer subscription"""

    subscription_id: str
    tenant_id: str
    plan_id: str
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    created_at: datetime
    activated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageRecord:
    """Represents usage tracking"""

    tenant_id: str
    metric: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubscriptionManager:
    """
    Manages customer subscriptions and billing
    """

    def __init__(self):
        self.plans = self._initialize_plans()
        self.subscriptions = {}
        self.usage_records = []
        self.upgrade_paths = self._define_upgrade_paths()

    def _initialize_plans(self) -> Dict[str, SubscriptionPlan]:
        """Initialize available subscription plans"""

        plans = {
            "free": SubscriptionPlan(
                plan_id="free",
                name="Free",
                tier=PlanTier.FREE,
                description="Perfect for trying out PolicyCortex",
                features=[
                    "basic_compliance_monitoring",
                    "up_to_10_resources",
                    "email_support",
                    "community_access",
                ],
                limits={
                    "resources": 10,
                    "users": 1,
                    "api_calls_per_month": 1000,
                    "data_retention_days": 7,
                    "scan_frequency": "daily",
                },
                pricing={"monthly": 0, "annual": 0},
                trial_days=0,
            ),
            "starter": SubscriptionPlan(
                plan_id="starter",
                name="Starter",
                tier=PlanTier.STARTER,
                description="Great for small teams and growing businesses",
                features=[
                    "full_compliance_monitoring",
                    "cost_optimization",
                    "basic_analytics",
                    "email_and_chat_support",
                    "api_access",
                    "custom_policies",
                ],
                limits={
                    "resources": 100,
                    "users": 5,
                    "api_calls_per_month": 10000,
                    "data_retention_days": 30,
                    "scan_frequency": "hourly",
                },
                pricing={
                    "monthly": 99,
                    "quarterly": 267,  # 10% discount
                    "annual": 990,  # 17% discount
                },
                trial_days=14,
            ),
            "professional": SubscriptionPlan(
                plan_id="professional",
                name="Professional",
                tier=PlanTier.PROFESSIONAL,
                description="Comprehensive governance for professional teams",
                features=[
                    "advanced_compliance_monitoring",
                    "ai_powered_analytics",
                    "cost_optimization",
                    "security_posture_management",
                    "auto_remediation",
                    "custom_dashboards",
                    "priority_support",
                    "sla_guarantee",
                ],
                limits={
                    "resources": 1000,
                    "users": 25,
                    "api_calls_per_month": 100000,
                    "data_retention_days": 90,
                    "scan_frequency": "real_time",
                    "custom_policies": 50,
                },
                pricing={
                    "monthly": 499,
                    "quarterly": 1347,  # 10% discount
                    "annual": 4990,  # 17% discount
                },
                trial_days=30,
            ),
            "enterprise": SubscriptionPlan(
                plan_id="enterprise",
                name="Enterprise",
                tier=PlanTier.ENTERPRISE,
                description="Enterprise-grade governance platform",
                features=[
                    "unlimited_compliance_monitoring",
                    "advanced_ai_analytics",
                    "multi_cloud_support",
                    "full_automation",
                    "custom_integrations",
                    "dedicated_support",
                    "custom_sla",
                    "onboarding_assistance",
                    "quarterly_business_reviews",
                ],
                limits={
                    "resources": None,  # Unlimited
                    "users": None,  # Unlimited
                    "api_calls_per_month": None,  # Unlimited
                    "data_retention_days": 365,
                    "scan_frequency": "real_time",
                    "custom_policies": None,  # Unlimited
                },
                pricing={
                    "monthly": 2499,
                    "quarterly": 6747,  # 10% discount
                    "annual": 24990,  # 17% discount
                },
                trial_days=30,
            ),
        }

        return plans

    def _define_upgrade_paths(self) -> Dict[str, List[str]]:
        """Define allowed upgrade paths between plans"""
        return {
            "free": ["starter", "professional", "enterprise"],
            "starter": ["professional", "enterprise"],
            "professional": ["enterprise"],
            "enterprise": [],
        }

    async def create_subscription(
        self, tenant_id: str, plan_id: str, billing_cycle: BillingCycle = BillingCycle.MONTHLY
    ) -> Subscription:
        """
        Create a new subscription

        Args:
            tenant_id: Tenant identifier
            plan_id: Plan to subscribe to
            billing_cycle: Billing cycle preference

        Returns:
            Created subscription
        """

        if plan_id not in self.plans:
            raise ValueError(f"Invalid plan ID: {plan_id}")

        plan = self.plans[plan_id]

        subscription_id = str(uuid.uuid4())

        # Determine initial status
        if plan.trial_days > 0:
            status = SubscriptionStatus.TRIAL
            expires_at = datetime.utcnow() + timedelta(days=plan.trial_days)
        elif plan.tier == PlanTier.FREE:
            status = SubscriptionStatus.ACTIVE
            expires_at = None
        else:
            status = SubscriptionStatus.PENDING
            expires_at = None

        # Calculate billing period
        current_period_start = datetime.utcnow()
        if billing_cycle == BillingCycle.MONTHLY:
            current_period_end = current_period_start + timedelta(days=30)
        elif billing_cycle == BillingCycle.QUARTERLY:
            current_period_end = current_period_start + timedelta(days=90)
        else:  # Annual
            current_period_end = current_period_start + timedelta(days=365)

        subscription = Subscription(
            subscription_id=subscription_id,
            tenant_id=tenant_id,
            plan_id=plan_id,
            status=status,
            billing_cycle=billing_cycle,
            created_at=datetime.utcnow(),
            activated_at=datetime.utcnow() if status == SubscriptionStatus.ACTIVE else None,
            expires_at=expires_at,
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            usage={"resources": 0, "users": 0, "api_calls": 0},
        )

        self.subscriptions[subscription_id] = subscription

        logger.info(
            f"Created subscription {subscription_id} for tenant {tenant_id} on plan {plan_id}"
        )

        return subscription

    async def activate_subscription(self, subscription_id: str) -> bool:
        """
        Activate a pending subscription

        Args:
            subscription_id: Subscription to activate

        Returns:
            Success status
        """

        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]

        if subscription.status != SubscriptionStatus.PENDING:
            logger.warning(
                f"Cannot activate subscription {subscription_id} with status {subscription.status}"
            )
            return False

        subscription.status = SubscriptionStatus.ACTIVE
        subscription.activated_at = datetime.utcnow()

        logger.info(f"Activated subscription {subscription_id}")

        return True

    async def upgrade_subscription(
        self, subscription_id: str, new_plan_id: str, immediate: bool = True
    ) -> bool:
        """
        Upgrade a subscription to a higher plan

        Args:
            subscription_id: Subscription to upgrade
            new_plan_id: Target plan
            immediate: Apply upgrade immediately vs at end of billing period

        Returns:
            Success status
        """

        if subscription_id not in self.subscriptions:
            return False

        if new_plan_id not in self.plans:
            return False

        subscription = self.subscriptions[subscription_id]
        current_plan_id = subscription.plan_id

        # Check upgrade path
        allowed_upgrades = self.upgrade_paths.get(current_plan_id, [])
        if new_plan_id not in allowed_upgrades:
            logger.warning(f"Invalid upgrade path from {current_plan_id} to {new_plan_id}")
            return False

        # Calculate prorated amount if immediate
        if immediate:
            days_remaining = (subscription.current_period_end - datetime.utcnow()).days
            proration_factor = days_remaining / 30  # Simplified calculation

            old_plan_price = self.plans[current_plan_id].pricing.get(
                subscription.billing_cycle.value, 0
            )
            new_plan_price = self.plans[new_plan_id].pricing.get(
                subscription.billing_cycle.value, 0
            )

            proration_amount = (new_plan_price - old_plan_price) * proration_factor

            subscription.metadata["proration"] = {
                "amount": proration_amount,
                "days_remaining": days_remaining,
                "upgraded_at": datetime.utcnow().isoformat(),
            }

            # Apply upgrade immediately
            subscription.plan_id = new_plan_id
        else:
            # Schedule upgrade for next billing period
            subscription.metadata["scheduled_upgrade"] = {
                "new_plan_id": new_plan_id,
                "effective_date": subscription.current_period_end.isoformat(),
            }

        logger.info(
            f"Upgraded subscription {subscription_id} from {current_plan_id} to {new_plan_id}"
        )

        return True

    async def downgrade_subscription(self, subscription_id: str, new_plan_id: str) -> bool:
        """
        Downgrade a subscription to a lower plan

        Args:
            subscription_id: Subscription to downgrade
            new_plan_id: Target plan

        Returns:
            Success status
        """

        if subscription_id not in self.subscriptions:
            return False

        if new_plan_id not in self.plans:
            return False

        subscription = self.subscriptions[subscription_id]

        # Downgrades typically happen at end of billing period
        subscription.metadata["scheduled_downgrade"] = {
            "new_plan_id": new_plan_id,
            "effective_date": subscription.current_period_end.isoformat(),
        }

        logger.info(f"Scheduled downgrade for subscription {subscription_id} to {new_plan_id}")

        return True

    async def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> bool:
        """
        Cancel a subscription

        Args:
            subscription_id: Subscription to cancel
            immediate: Cancel immediately vs at end of billing period

        Returns:
            Success status
        """

        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]

        if immediate:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.cancelled_at = datetime.utcnow()
        else:
            # Cancel at end of billing period
            subscription.metadata["scheduled_cancellation"] = {
                "effective_date": subscription.current_period_end.isoformat()
            }

        logger.info(f"Cancelled subscription {subscription_id}")

        return True

    async def suspend_subscription(self, subscription_id: str, reason: str) -> bool:
        """
        Suspend a subscription

        Args:
            subscription_id: Subscription to suspend
            reason: Reason for suspension

        Returns:
            Success status
        """

        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        subscription.status = SubscriptionStatus.SUSPENDED
        subscription.metadata["suspension"] = {
            "reason": reason,
            "suspended_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"Suspended subscription {subscription_id}: {reason}")

        return True

    async def reactivate_subscription(self, subscription_id: str) -> bool:
        """
        Reactivate a suspended subscription

        Args:
            subscription_id: Subscription to reactivate

        Returns:
            Success status
        """

        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]

        if subscription.status != SubscriptionStatus.SUSPENDED:
            return False

        subscription.status = SubscriptionStatus.ACTIVE
        subscription.metadata["reactivation"] = {"reactivated_at": datetime.utcnow().isoformat()}

        logger.info(f"Reactivated subscription {subscription_id}")

        return True

    async def record_usage(
        self, tenant_id: str, metric: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record usage for a tenant

        Args:
            tenant_id: Tenant identifier
            metric: Usage metric name
            value: Usage value
            metadata: Additional metadata
        """

        usage_record = UsageRecord(
            tenant_id=tenant_id,
            metric=metric,
            value=value,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.usage_records.append(usage_record)

        # Update subscription usage
        subscription = self.get_subscription_by_tenant(tenant_id)
        if subscription:
            if metric not in subscription.usage:
                subscription.usage[metric] = 0
            subscription.usage[metric] += value

    def get_subscription_by_tenant(self, tenant_id: str) -> Optional[Subscription]:
        """Get subscription for a tenant"""

        for subscription in self.subscriptions.values():
            if subscription.tenant_id == tenant_id:
                return subscription
        return None

    async def check_usage_limits(self, tenant_id: str) -> Dict[str, Any]:
        """
        Check if tenant is within usage limits

        Args:
            tenant_id: Tenant to check

        Returns:
            Usage status and warnings
        """

        subscription = self.get_subscription_by_tenant(tenant_id)
        if not subscription:
            return {"status": "no_subscription"}

        plan = self.plans[subscription.plan_id]
        warnings = []
        exceeded = []

        for metric, limit in plan.limits.items():
            if limit is None:  # Unlimited
                continue

            current_usage = subscription.usage.get(metric, 0)

            if current_usage >= limit:
                exceeded.append({"metric": metric, "limit": limit, "current": current_usage})
            elif current_usage >= limit * 0.8:  # 80% warning threshold
                warnings.append(
                    {
                        "metric": metric,
                        "limit": limit,
                        "current": current_usage,
                        "percentage": (current_usage / limit) * 100,
                    }
                )

        return {
            "status": "exceeded" if exceeded else "warning" if warnings else "ok",
            "exceeded": exceeded,
            "warnings": warnings,
            "usage": subscription.usage,
            "limits": plan.limits,
        }

    def calculate_invoice(self, subscription_id: str) -> Dict[str, Any]:
        """
        Calculate invoice for a subscription

        Args:
            subscription_id: Subscription to invoice

        Returns:
            Invoice details
        """

        if subscription_id not in self.subscriptions:
            return {}

        subscription = self.subscriptions[subscription_id]
        plan = self.plans[subscription.plan_id]

        # Base price
        base_price = plan.pricing.get(subscription.billing_cycle.value, 0)

        # Apply discounts
        discount = 0
        if subscription.billing_cycle == BillingCycle.QUARTERLY:
            discount = 0.10  # 10% discount
        elif subscription.billing_cycle == BillingCycle.ANNUAL:
            discount = 0.17  # 17% discount

        discounted_price = base_price * (1 - discount)

        # Check for proration
        proration = subscription.metadata.get("proration", {}).get("amount", 0)

        # Calculate overages
        overages = []
        overage_total = 0

        for metric, limit in plan.limits.items():
            if limit is None:
                continue

            current_usage = subscription.usage.get(metric, 0)
            if current_usage > limit:
                overage = current_usage - limit
                overage_cost = overage * 0.10  # $0.10 per unit overage (simplified)
                overages.append({"metric": metric, "overage": overage, "cost": overage_cost})
                overage_total += overage_cost

        # Calculate total
        subtotal = discounted_price + proration + overage_total
        tax_rate = 0.08  # 8% tax (simplified)
        tax = subtotal * tax_rate
        total = subtotal + tax

        return {
            "subscription_id": subscription_id,
            "tenant_id": subscription.tenant_id,
            "plan": plan.name,
            "billing_cycle": subscription.billing_cycle.value,
            "period": {
                "start": (
                    subscription.current_period_start.isoformat()
                    if subscription.current_period_start
                    else None
                ),
                "end": (
                    subscription.current_period_end.isoformat()
                    if subscription.current_period_end
                    else None
                ),
            },
            "charges": {
                "base_price": base_price,
                "discount_percentage": discount * 100,
                "discount_amount": base_price * discount,
                "discounted_price": discounted_price,
                "proration": proration,
                "overages": overages,
                "overage_total": overage_total,
            },
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
        }

    def get_subscription_status(self, subscription_id: str) -> Dict[str, Any]:
        """Get detailed subscription status"""

        if subscription_id not in self.subscriptions:
            return {"status": "not_found"}

        subscription = self.subscriptions[subscription_id]
        plan = self.plans[subscription.plan_id]

        # Check for trial expiration
        is_trial_expired = False
        if subscription.status == SubscriptionStatus.TRIAL and subscription.expires_at:
            is_trial_expired = datetime.utcnow() > subscription.expires_at

        return {
            "subscription_id": subscription_id,
            "tenant_id": subscription.tenant_id,
            "plan": {
                "id": plan.plan_id,
                "name": plan.name,
                "tier": plan.tier.value,
                "features": plan.features,
            },
            "status": subscription.status.value,
            "billing_cycle": subscription.billing_cycle.value,
            "created_at": subscription.created_at.isoformat(),
            "activated_at": (
                subscription.activated_at.isoformat() if subscription.activated_at else None
            ),
            "expires_at": subscription.expires_at.isoformat() if subscription.expires_at else None,
            "trial_expired": is_trial_expired,
            "current_period": {
                "start": (
                    subscription.current_period_start.isoformat()
                    if subscription.current_period_start
                    else None
                ),
                "end": (
                    subscription.current_period_end.isoformat()
                    if subscription.current_period_end
                    else None
                ),
            },
            "usage": subscription.usage,
            "scheduled_changes": {
                "upgrade": subscription.metadata.get("scheduled_upgrade"),
                "downgrade": subscription.metadata.get("scheduled_downgrade"),
                "cancellation": subscription.metadata.get("scheduled_cancellation"),
            },
        }

    def get_available_plans(self) -> List[Dict[str, Any]]:
        """Get list of available plans"""

        return [
            {
                "plan_id": plan.plan_id,
                "name": plan.name,
                "tier": plan.tier.value,
                "description": plan.description,
                "features": plan.features,
                "limits": plan.limits,
                "pricing": plan.pricing,
                "trial_days": plan.trial_days,
            }
            for plan in self.plans.values()
        ]

    def get_usage_summary(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage summary for a tenant"""

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        tenant_usage = [
            record
            for record in self.usage_records
            if record.tenant_id == tenant_id and record.timestamp > cutoff_date
        ]

        # Aggregate by metric
        usage_by_metric = {}
        for record in tenant_usage:
            if record.metric not in usage_by_metric:
                usage_by_metric[record.metric] = {
                    "total": 0,
                    "count": 0,
                    "min": float("inf"),
                    "max": 0,
                }

            usage_by_metric[record.metric]["total"] += record.value
            usage_by_metric[record.metric]["count"] += 1
            usage_by_metric[record.metric]["min"] = min(
                usage_by_metric[record.metric]["min"], record.value
            )
            usage_by_metric[record.metric]["max"] = max(
                usage_by_metric[record.metric]["max"], record.value
            )

        # Calculate averages
        for metric in usage_by_metric:
            count = usage_by_metric[metric]["count"]
            if count > 0:
                usage_by_metric[metric]["average"] = usage_by_metric[metric]["total"] / count

        return {
            "tenant_id": tenant_id,
            "period_days": days,
            "from_date": cutoff_date.isoformat(),
            "to_date": datetime.utcnow().isoformat(),
            "metrics": usage_by_metric,
        }
