"""
Trial Manager Module
Manages free trial periods and conversions
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class TrialStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CONVERTED = "converted"
    CANCELLED = "cancelled"
    EXTENDED = "extended"


class TrialFeature(str, Enum):
    FULL_ACCESS = "full_access"
    LIMITED_ACCESS = "limited_access"
    DEMO_DATA = "demo_data"
    SUPPORT = "support"
    TRAINING = "training"


@dataclass
class Trial:
    """Represents a trial period"""

    trial_id: str
    tenant_id: str
    plan_id: str
    status: TrialStatus
    started_at: datetime
    expires_at: datetime
    features: List[TrialFeature]
    usage_limits: Dict[str, Any]
    conversion_data: Optional[Dict[str, Any]] = None
    extension_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialActivity:
    """Tracks trial user activity"""

    tenant_id: str
    activity_type: str
    timestamp: datetime
    details: Dict[str, Any]


class TrialManager:
    """
    Manages free trial periods and conversions
    """

    def __init__(self):
        self.trials = {}
        self.trial_activities = []
        self.conversion_strategies = self._load_conversion_strategies()

    def _load_conversion_strategies(self) -> Dict[str, Any]:
        """Load trial conversion strategies"""
        return {
            "email_campaigns": [
                {"day": 1, "template": "welcome", "subject": "Welcome to PolicyCortex!"},
                {
                    "day": 3,
                    "template": "getting_started",
                    "subject": "Get the most from your trial",
                },
                {
                    "day": 7,
                    "template": "feature_highlight",
                    "subject": "Discover AI-powered compliance",
                },
                {
                    "day": -3,  # 3 days before expiration
                    "template": "expiring_soon",
                    "subject": "Your trial expires soon",
                },
                {
                    "day": -1,
                    "template": "last_chance",
                    "subject": "Last day to convert with discount",
                },
            ],
            "in_app_messages": [
                {
                    "trigger": "first_login",
                    "message": "Welcome! Need help getting started?",
                    "action": "start_tour",
                },
                {
                    "trigger": "feature_usage_low",
                    "message": "Explore more features to maximize value",
                    "action": "show_features",
                },
                {
                    "trigger": "approaching_limit",
                    "message": "Approaching usage limit - upgrade for unlimited",
                    "action": "show_pricing",
                },
            ],
            "conversion_offers": [
                {"timing": "day_7", "discount": 20, "duration": "first_3_months"},
                {"timing": "last_day", "discount": 30, "duration": "first_month"},
            ],
        }

    async def start_trial(self, tenant_id: str, plan_id: str, trial_days: int = 14) -> Trial:
        """
        Start a new trial

        Args:
            tenant_id: Tenant identifier
            plan_id: Plan to trial
            trial_days: Duration of trial

        Returns:
            Created trial
        """

        trial_id = str(uuid.uuid4())

        # Determine trial features based on plan
        features = self._get_trial_features(plan_id)
        usage_limits = self._get_trial_limits(plan_id)

        trial = Trial(
            trial_id=trial_id,
            tenant_id=tenant_id,
            plan_id=plan_id,
            status=TrialStatus.ACTIVE,
            started_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=trial_days),
            features=features,
            usage_limits=usage_limits,
            metadata={"original_duration": trial_days, "source": "website"},
        )

        self.trials[trial_id] = trial

        # Record trial start activity
        await self.record_activity(
            tenant_id, "trial_started", {"plan_id": plan_id, "duration_days": trial_days}
        )

        # Schedule conversion campaigns
        await self._schedule_conversion_campaigns(trial)

        logger.info(f"Started trial {trial_id} for tenant {tenant_id}")

        return trial

    def _get_trial_features(self, plan_id: str) -> List[TrialFeature]:
        """Get features available during trial"""

        if plan_id in ["professional", "enterprise"]:
            return [TrialFeature.FULL_ACCESS, TrialFeature.SUPPORT, TrialFeature.TRAINING]
        else:
            return [TrialFeature.LIMITED_ACCESS, TrialFeature.DEMO_DATA, TrialFeature.SUPPORT]

    def _get_trial_limits(self, plan_id: str) -> Dict[str, Any]:
        """Get usage limits during trial"""

        limits = {
            "starter": {"resources": 50, "users": 3, "api_calls": 5000},
            "professional": {"resources": 500, "users": 10, "api_calls": 50000},
            "enterprise": {"resources": 1000, "users": 25, "api_calls": 100000},
        }

        return limits.get(plan_id, limits["starter"])

    async def extend_trial(self, trial_id: str, additional_days: int, reason: str) -> bool:
        """
        Extend a trial period

        Args:
            trial_id: Trial to extend
            additional_days: Days to add
            reason: Reason for extension

        Returns:
            Success status
        """

        if trial_id not in self.trials:
            return False

        trial = self.trials[trial_id]

        if trial.status != TrialStatus.ACTIVE:
            logger.warning(f"Cannot extend trial {trial_id} with status {trial.status}")
            return False

        # Limit extensions
        if trial.extension_count >= 2:
            logger.warning(f"Trial {trial_id} has reached maximum extensions")
            return False

        trial.expires_at += timedelta(days=additional_days)
        trial.extension_count += 1
        trial.status = TrialStatus.EXTENDED
        trial.metadata["extensions"] = trial.metadata.get("extensions", [])
        trial.metadata["extensions"].append(
            {
                "days": additional_days,
                "reason": reason,
                "extended_at": datetime.utcnow().isoformat(),
            }
        )

        await self.record_activity(
            trial.tenant_id, "trial_extended", {"days": additional_days, "reason": reason}
        )

        logger.info(f"Extended trial {trial_id} by {additional_days} days")

        return True

    async def convert_trial(self, trial_id: str, subscription_id: str) -> bool:
        """
        Convert trial to paid subscription

        Args:
            trial_id: Trial to convert
            subscription_id: Created subscription ID

        Returns:
            Success status
        """

        if trial_id not in self.trials:
            return False

        trial = self.trials[trial_id]

        if trial.status == TrialStatus.CONVERTED:
            logger.warning(f"Trial {trial_id} already converted")
            return False

        trial.status = TrialStatus.CONVERTED
        trial.conversion_data = {
            "subscription_id": subscription_id,
            "converted_at": datetime.utcnow().isoformat(),
            "days_used": (datetime.utcnow() - trial.started_at).days,
            "conversion_rate": self._calculate_conversion_score(trial),
        }

        await self.record_activity(
            trial.tenant_id, "trial_converted", {"subscription_id": subscription_id}
        )

        logger.info(f"Converted trial {trial_id} to subscription {subscription_id}")

        return True

    async def cancel_trial(self, trial_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a trial

        Args:
            trial_id: Trial to cancel
            reason: Cancellation reason

        Returns:
            Success status
        """

        if trial_id not in self.trials:
            return False

        trial = self.trials[trial_id]

        if trial.status in [TrialStatus.CONVERTED, TrialStatus.CANCELLED]:
            return False

        trial.status = TrialStatus.CANCELLED
        trial.metadata["cancellation"] = {
            "reason": reason,
            "cancelled_at": datetime.utcnow().isoformat(),
        }

        await self.record_activity(trial.tenant_id, "trial_cancelled", {"reason": reason})

        logger.info(f"Cancelled trial {trial_id}")

        return True

    async def check_trial_status(self, trial_id: str) -> Dict[str, Any]:
        """
        Check trial status and remaining time

        Args:
            trial_id: Trial to check

        Returns:
            Trial status information
        """

        if trial_id not in self.trials:
            return {"status": "not_found"}

        trial = self.trials[trial_id]

        now = datetime.utcnow()
        is_expired = now > trial.expires_at and trial.status == TrialStatus.ACTIVE

        if is_expired:
            trial.status = TrialStatus.EXPIRED

        days_remaining = max(0, (trial.expires_at - now).days)
        days_used = (now - trial.started_at).days

        return {
            "trial_id": trial_id,
            "status": trial.status.value,
            "started_at": trial.started_at.isoformat(),
            "expires_at": trial.expires_at.isoformat(),
            "days_remaining": days_remaining,
            "days_used": days_used,
            "is_expired": is_expired,
            "extension_count": trial.extension_count,
            "can_extend": trial.extension_count < 2,
            "features": [f.value for f in trial.features],
            "usage_limits": trial.usage_limits,
            "conversion_status": trial.conversion_data if trial.conversion_data else None,
        }

    async def record_activity(
        self, tenant_id: str, activity_type: str, details: Dict[str, Any]
    ) -> None:
        """
        Record trial activity

        Args:
            tenant_id: Tenant performing activity
            activity_type: Type of activity
            details: Activity details
        """

        activity = TrialActivity(
            tenant_id=tenant_id,
            activity_type=activity_type,
            timestamp=datetime.utcnow(),
            details=details,
        )

        self.trial_activities.append(activity)

        # Check for conversion triggers
        await self._check_conversion_triggers(tenant_id, activity)

    async def _check_conversion_triggers(self, tenant_id: str, activity: TrialActivity) -> None:
        """Check if activity triggers conversion campaigns"""

        trial = self.get_trial_by_tenant(tenant_id)
        if not trial or trial.status != TrialStatus.ACTIVE:
            return

        # Check for low feature usage
        if activity.activity_type == "feature_usage":
            usage_count = sum(
                1
                for a in self.trial_activities
                if a.tenant_id == tenant_id and a.activity_type == "feature_usage"
            )

            if usage_count < 5 and (datetime.utcnow() - trial.started_at).days >= 3:
                # Trigger engagement campaign
                await self._trigger_engagement_campaign(trial, "low_usage")

        # Check for approaching limits
        if activity.activity_type == "resource_created":
            resource_count = activity.details.get("total_resources", 0)
            limit = trial.usage_limits.get("resources", 0)

            if resource_count >= limit * 0.8:
                # Trigger upgrade campaign
                await self._trigger_upgrade_campaign(trial, "approaching_limit")

    async def _schedule_conversion_campaigns(self, trial: Trial) -> None:
        """Schedule automated conversion campaigns"""

        campaigns = self.conversion_strategies["email_campaigns"]

        for campaign in campaigns:
            if campaign["day"] > 0:
                # Schedule from start
                send_date = trial.started_at + timedelta(days=campaign["day"])
            else:
                # Schedule from end
                send_date = trial.expires_at + timedelta(days=campaign["day"])

            trial.metadata.setdefault("scheduled_campaigns", []).append(
                {
                    "template": campaign["template"],
                    "send_date": send_date.isoformat(),
                    "subject": campaign["subject"],
                }
            )

    async def _trigger_engagement_campaign(self, trial: Trial, trigger: str) -> None:
        """Trigger engagement campaign"""

        logger.info(f"Triggering engagement campaign for trial {trial.trial_id}: {trigger}")

        # In production, would send actual campaign
        trial.metadata.setdefault("triggered_campaigns", []).append(
            {
                "type": "engagement",
                "trigger": trigger,
                "triggered_at": datetime.utcnow().isoformat(),
            }
        )

    async def _trigger_upgrade_campaign(self, trial: Trial, trigger: str) -> None:
        """Trigger upgrade campaign"""

        logger.info(f"Triggering upgrade campaign for trial {trial.trial_id}: {trigger}")

        # In production, would send actual campaign
        trial.metadata.setdefault("triggered_campaigns", []).append(
            {"type": "upgrade", "trigger": trigger, "triggered_at": datetime.utcnow().isoformat()}
        )

    def _calculate_conversion_score(self, trial: Trial) -> float:
        """Calculate conversion likelihood score"""

        score = 0.0

        # Factor 1: Days used (max 30 points)
        days_used = (datetime.utcnow() - trial.started_at).days
        total_days = (trial.expires_at - trial.started_at).days
        usage_ratio = days_used / total_days if total_days > 0 else 0
        score += usage_ratio * 30

        # Factor 2: Feature usage (max 40 points)
        feature_activities = sum(
            1
            for a in self.trial_activities
            if a.tenant_id == trial.tenant_id and a.activity_type == "feature_usage"
        )
        score += min(feature_activities * 2, 40)

        # Factor 3: User engagement (max 30 points)
        total_activities = sum(1 for a in self.trial_activities if a.tenant_id == trial.tenant_id)
        score += min(total_activities, 30)

        return min(score, 100)  # Cap at 100

    def get_trial_by_tenant(self, tenant_id: str) -> Optional[Trial]:
        """Get trial for a tenant"""

        for trial in self.trials.values():
            if trial.tenant_id == tenant_id:
                return trial
        return None

    def get_conversion_analytics(self) -> Dict[str, Any]:
        """Get trial conversion analytics"""

        total_trials = len(self.trials)

        if total_trials == 0:
            return {"total_trials": 0, "conversion_rate": 0}

        converted = sum(1 for t in self.trials.values() if t.status == TrialStatus.CONVERTED)
        expired = sum(1 for t in self.trials.values() if t.status == TrialStatus.EXPIRED)
        active = sum(1 for t in self.trials.values() if t.status == TrialStatus.ACTIVE)
        cancelled = sum(1 for t in self.trials.values() if t.status == TrialStatus.CANCELLED)

        # Calculate average conversion time
        conversion_times = []
        for trial in self.trials.values():
            if trial.conversion_data:
                days_to_convert = trial.conversion_data.get("days_used", 0)
                conversion_times.append(days_to_convert)

        avg_conversion_time = (
            sum(conversion_times) / len(conversion_times) if conversion_times else 0
        )

        # Activity engagement metrics
        activities_per_trial = {}
        for activity in self.trial_activities:
            if activity.tenant_id not in activities_per_trial:
                activities_per_trial[activity.tenant_id] = 0
            activities_per_trial[activity.tenant_id] += 1

        avg_activities = (
            sum(activities_per_trial.values()) / len(activities_per_trial)
            if activities_per_trial
            else 0
        )

        return {
            "total_trials": total_trials,
            "active_trials": active,
            "converted_trials": converted,
            "expired_trials": expired,
            "cancelled_trials": cancelled,
            "conversion_rate": (converted / total_trials * 100),
            "average_conversion_time_days": avg_conversion_time,
            "average_activities_per_trial": avg_activities,
            "top_conversion_triggers": self._get_top_conversion_triggers(),
        }

    def _get_top_conversion_triggers(self) -> List[Dict[str, Any]]:
        """Get top conversion triggers"""

        trigger_counts = {}

        for trial in self.trials.values():
            if trial.conversion_data:
                triggered_campaigns = trial.metadata.get("triggered_campaigns", [])
                for campaign in triggered_campaigns:
                    trigger = campaign["trigger"]
                    trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)

        return [{"trigger": trigger, "count": count} for trigger, count in sorted_triggers[:5]]

    async def get_expiring_trials(self, days_ahead: int = 3) -> List[Trial]:
        """Get trials expiring soon"""

        expiry_threshold = datetime.utcnow() + timedelta(days=days_ahead)

        expiring = []
        for trial in self.trials.values():
            if trial.status == TrialStatus.ACTIVE and trial.expires_at <= expiry_threshold:
                expiring.append(trial)

        return expiring
