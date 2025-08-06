"""
Onboarding Wizard Module
Guides customers through the setup process
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class OnboardingStep(str, Enum):
    REGISTRATION = "registration"
    ORGANIZATION = "organization"
    AZURE_CONNECTION = "azure_connection"
    FEATURE_SELECTION = "feature_selection"
    INITIAL_SCAN = "initial_scan"
    USER_INVITES = "user_invites"
    CONFIGURATION = "configuration"
    COMPLETION = "completion"


class OnboardingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class OnboardingSession:
    """Represents a customer onboarding session"""

    session_id: str
    tenant_id: str
    user_id: str
    status: OnboardingStatus
    current_step: OnboardingStep
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps_completed: List[OnboardingStep] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepValidation:
    """Validation result for an onboarding step"""

    step: OnboardingStep
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OnboardingWizard:
    """
    Manages the customer onboarding wizard flow
    """

    def __init__(self):
        self.sessions = {}
        self.step_validators = self._initialize_validators()
        self.step_handlers = self._initialize_handlers()
        self.templates = self._load_templates()

    def _initialize_validators(self) -> Dict:
        """Initialize step validators"""
        return {
            OnboardingStep.REGISTRATION: self._validate_registration,
            OnboardingStep.ORGANIZATION: self._validate_organization,
            OnboardingStep.AZURE_CONNECTION: self._validate_azure_connection,
            OnboardingStep.FEATURE_SELECTION: self._validate_feature_selection,
            OnboardingStep.INITIAL_SCAN: self._validate_initial_scan,
            OnboardingStep.USER_INVITES: self._validate_user_invites,
            OnboardingStep.CONFIGURATION: self._validate_configuration,
            OnboardingStep.COMPLETION: self._validate_completion,
        }

    def _initialize_handlers(self) -> Dict:
        """Initialize step handlers"""
        return {
            OnboardingStep.REGISTRATION: self._handle_registration,
            OnboardingStep.ORGANIZATION: self._handle_organization,
            OnboardingStep.AZURE_CONNECTION: self._handle_azure_connection,
            OnboardingStep.FEATURE_SELECTION: self._handle_feature_selection,
            OnboardingStep.INITIAL_SCAN: self._handle_initial_scan,
            OnboardingStep.USER_INVITES: self._handle_user_invites,
            OnboardingStep.CONFIGURATION: self._handle_configuration,
            OnboardingStep.COMPLETION: self._handle_completion,
        }

    def _load_templates(self) -> Dict[str, Any]:
        """Load onboarding templates"""
        return {
            "small_business": {
                "name": "Small Business",
                "description": "Perfect for small teams getting started with cloud governance",
                "features": ["basic_compliance", "cost_monitoring", "alerts"],
                "user_limit": 10,
                "resource_limit": 100,
                "default_config": {
                    "scan_frequency": "daily",
                    "alert_threshold": "medium",
                    "retention_days": 30,
                },
            },
            "enterprise": {
                "name": "Enterprise",
                "description": "Comprehensive governance for large organizations",
                "features": [
                    "advanced_compliance",
                    "ai_analytics",
                    "custom_policies",
                    "multi_cloud",
                    "api_access",
                ],
                "user_limit": None,
                "resource_limit": None,
                "default_config": {
                    "scan_frequency": "real_time",
                    "alert_threshold": "low",
                    "retention_days": 365,
                    "enable_ml": True,
                },
            },
            "startup": {
                "name": "Startup",
                "description": "Scalable solution for growing companies",
                "features": ["basic_compliance", "cost_optimization", "auto_remediation"],
                "user_limit": 25,
                "resource_limit": 500,
                "default_config": {
                    "scan_frequency": "hourly",
                    "alert_threshold": "high",
                    "retention_days": 90,
                },
            },
        }

    async def start_onboarding(
        self, user_id: str, user_email: str, company_name: str, template: Optional[str] = None
    ) -> OnboardingSession:
        """
        Start a new onboarding session

        Args:
            user_id: User initiating onboarding
            user_email: User's email address
            company_name: Name of the organization
            template: Optional template to use

        Returns:
            New onboarding session
        """

        session_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())

        session = OnboardingSession(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status=OnboardingStatus.IN_PROGRESS,
            current_step=OnboardingStep.REGISTRATION,
            started_at=datetime.utcnow(),
            configuration={
                "user_email": user_email,
                "company_name": company_name,
                "template": template,
            },
        )

        # Apply template defaults if specified
        if template and template in self.templates:
            template_config = self.templates[template]
            session.configuration.update(
                {
                    "selected_features": template_config["features"],
                    "user_limit": template_config["user_limit"],
                    "resource_limit": template_config["resource_limit"],
                    "default_settings": template_config["default_config"],
                }
            )

        self.sessions[session_id] = session

        logger.info(f"Started onboarding session {session_id} for {company_name}")

        return session

    async def process_step(self, session_id: str, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a step in the onboarding wizard

        Args:
            session_id: Onboarding session ID
            step_data: Data for the current step

        Returns:
            Step processing result
        """

        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        if session.status != OnboardingStatus.IN_PROGRESS:
            raise ValueError(f"Session {session_id} is not in progress")

        current_step = session.current_step

        # Validate step data
        validation = await self.step_validators[current_step](step_data, session)

        if not validation.is_valid:
            session.validation_errors = validation.errors
            return {
                "success": False,
                "step": current_step.value,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }

        # Process the step
        try:
            result = await self.step_handlers[current_step](step_data, session)

            # Update session
            session.steps_completed.append(current_step)
            session.configuration.update(result.get("config", {}))

            # Move to next step
            next_step = self._get_next_step(current_step, session)
            if next_step:
                session.current_step = next_step
            else:
                session.status = OnboardingStatus.COMPLETED
                session.completed_at = datetime.utcnow()

            return {
                "success": True,
                "step": current_step.value,
                "next_step": next_step.value if next_step else None,
                "data": result,
                "warnings": validation.warnings,
                "progress": self._calculate_progress(session),
            }

        except Exception as e:
            logger.error(f"Failed to process step {current_step} for session {session_id}: {e}")
            session.status = OnboardingStatus.FAILED
            return {"success": False, "step": current_step.value, "errors": [str(e)]}

    async def _validate_registration(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate registration step"""

        validation = StepValidation(step=OnboardingStep.REGISTRATION, is_valid=True)

        # Check required fields
        required_fields = ["first_name", "last_name", "email", "password", "company_name", "phone"]

        for field in required_fields:
            if field not in step_data or not step_data[field]:
                validation.errors.append(f"Missing required field: {field}")
                validation.is_valid = False

        # Validate email format
        if "email" in step_data:
            email = step_data["email"]
            if "@" not in email or "." not in email.split("@")[1]:
                validation.errors.append("Invalid email format")
                validation.is_valid = False

        # Validate password strength
        if "password" in step_data:
            password = step_data["password"]
            if len(password) < 8:
                validation.errors.append("Password must be at least 8 characters")
                validation.is_valid = False
            if not any(c.isupper() for c in password):
                validation.warnings.append("Password should contain uppercase letters")
            if not any(c.isdigit() for c in password):
                validation.warnings.append("Password should contain numbers")

        return validation

    async def _validate_organization(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate organization setup step"""

        validation = StepValidation(step=OnboardingStep.ORGANIZATION, is_valid=True)

        # Check required fields
        if "organization_type" not in step_data:
            validation.errors.append("Organization type is required")
            validation.is_valid = False

        if "industry" not in step_data:
            validation.errors.append("Industry is required")
            validation.is_valid = False

        # Validate employee count
        if "employee_count" in step_data:
            count = step_data["employee_count"]
            if not isinstance(count, (int, str)) or (
                isinstance(count, str) and not count.isdigit()
            ):
                validation.errors.append("Invalid employee count")
                validation.is_valid = False

        # Validate cloud environments
        if "cloud_environments" in step_data:
            valid_clouds = ["azure", "aws", "gcp", "hybrid", "private"]
            for cloud in step_data["cloud_environments"]:
                if cloud not in valid_clouds:
                    validation.warnings.append(f"Unknown cloud environment: {cloud}")

        return validation

    async def _validate_azure_connection(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate Azure connection step"""

        validation = StepValidation(step=OnboardingStep.AZURE_CONNECTION, is_valid=True)

        connection_type = step_data.get("connection_type")

        if not connection_type:
            validation.errors.append("Connection type is required")
            validation.is_valid = False
            return validation

        if connection_type == "service_principal":
            required = ["tenant_id", "client_id", "client_secret"]
            for field in required:
                if field not in step_data or not step_data[field]:
                    validation.errors.append(f"Missing required field: {field}")
                    validation.is_valid = False

        elif connection_type == "managed_identity":
            if "subscription_id" not in step_data:
                validation.errors.append("Subscription ID is required for managed identity")
                validation.is_valid = False

        elif connection_type == "interactive":
            if "consent_given" not in step_data or not step_data["consent_given"]:
                validation.errors.append("User consent is required for interactive authentication")
                validation.is_valid = False

        # Validate subscription selection
        if "selected_subscriptions" in step_data:
            if not step_data["selected_subscriptions"]:
                validation.warnings.append(
                    "No subscriptions selected - all accessible subscriptions will be monitored"
                )

        return validation

    async def _validate_feature_selection(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate feature selection step"""

        validation = StepValidation(step=OnboardingStep.FEATURE_SELECTION, is_valid=True)

        if "selected_features" not in step_data or not step_data["selected_features"]:
            validation.errors.append("At least one feature must be selected")
            validation.is_valid = False
            return validation

        available_features = [
            "compliance_monitoring",
            "cost_optimization",
            "security_posture",
            "policy_management",
            "ai_analytics",
            "auto_remediation",
            "custom_dashboards",
            "api_access",
            "multi_cloud",
        ]

        for feature in step_data["selected_features"]:
            if feature not in available_features:
                validation.warnings.append(f"Unknown feature: {feature}")

        # Check feature dependencies
        if "ai_analytics" in step_data["selected_features"]:
            if "compliance_monitoring" not in step_data["selected_features"]:
                validation.warnings.append(
                    "AI Analytics works best with Compliance Monitoring enabled"
                )

        if "auto_remediation" in step_data["selected_features"]:
            if "policy_management" not in step_data["selected_features"]:
                validation.errors.append("Auto-remediation requires Policy Management")
                validation.is_valid = False

        return validation

    async def _validate_initial_scan(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate initial scan configuration"""

        validation = StepValidation(step=OnboardingStep.INITIAL_SCAN, is_valid=True)

        scan_type = step_data.get("scan_type", "quick")

        if scan_type not in ["quick", "standard", "comprehensive", "skip"]:
            validation.errors.append("Invalid scan type")
            validation.is_valid = False

        if scan_type != "skip":
            if "resource_types" in step_data:
                valid_types = ["compute", "storage", "network", "database", "identity", "all"]
                for rtype in step_data["resource_types"]:
                    if rtype not in valid_types:
                        validation.warnings.append(f"Unknown resource type: {rtype}")

        if step_data.get("enable_remediation", False):
            if not step_data.get("remediation_confirmed", False):
                validation.errors.append("Remediation must be explicitly confirmed")
                validation.is_valid = False

        return validation

    async def _validate_user_invites(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate user invitation step"""

        validation = StepValidation(step=OnboardingStep.USER_INVITES, is_valid=True)

        if "invitations" in step_data:
            for invite in step_data["invitations"]:
                if "email" not in invite or not invite["email"]:
                    validation.errors.append("Email is required for each invitation")
                    validation.is_valid = False

                if "role" not in invite:
                    validation.errors.append("Role is required for each invitation")
                    validation.is_valid = False
                else:
                    valid_roles = ["admin", "operator", "viewer", "compliance_officer"]
                    if invite["role"] not in valid_roles:
                        validation.errors.append(f"Invalid role: {invite['role']}")
                        validation.is_valid = False

        # Check user limits based on template
        template = session.configuration.get("template")
        if template and template in self.templates:
            user_limit = self.templates[template].get("user_limit")
            if user_limit:
                total_users = len(step_data.get("invitations", [])) + 1  # +1 for admin
                if total_users > user_limit:
                    validation.errors.append(
                        f"User limit exceeded. Maximum {user_limit} users allowed"
                    )
                    validation.is_valid = False

        return validation

    async def _validate_configuration(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate configuration settings"""

        validation = StepValidation(step=OnboardingStep.CONFIGURATION, is_valid=True)

        # Validate notification settings
        if "notifications" in step_data:
            notif = step_data["notifications"]

            if notif.get("enabled", False):
                if "channels" not in notif or not notif["channels"]:
                    validation.warnings.append("No notification channels configured")

                for channel in notif.get("channels", []):
                    if channel["type"] == "email":
                        if "recipients" not in channel or not channel["recipients"]:
                            validation.errors.append("Email channel requires recipients")
                            validation.is_valid = False
                    elif channel["type"] == "slack":
                        if "webhook_url" not in channel:
                            validation.errors.append("Slack channel requires webhook URL")
                            validation.is_valid = False
                    elif channel["type"] == "teams":
                        if "webhook_url" not in channel:
                            validation.errors.append("Teams channel requires webhook URL")
                            validation.is_valid = False

        # Validate compliance settings
        if "compliance" in step_data:
            compliance = step_data["compliance"]

            if "scan_schedule" in compliance:
                valid_schedules = ["real_time", "hourly", "daily", "weekly"]
                if compliance["scan_schedule"] not in valid_schedules:
                    validation.errors.append("Invalid scan schedule")
                    validation.is_valid = False

            if "frameworks" in compliance:
                valid_frameworks = ["cis", "nist", "iso27001", "pci_dss", "hipaa", "gdpr"]
                for framework in compliance["frameworks"]:
                    if framework not in valid_frameworks:
                        validation.warnings.append(f"Unknown compliance framework: {framework}")

        return validation

    async def _validate_completion(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> StepValidation:
        """Validate completion step"""

        validation = StepValidation(step=OnboardingStep.COMPLETION, is_valid=True)

        # Check all required steps are completed
        required_steps = [
            OnboardingStep.REGISTRATION,
            OnboardingStep.ORGANIZATION,
            OnboardingStep.AZURE_CONNECTION,
            OnboardingStep.FEATURE_SELECTION,
        ]

        for step in required_steps:
            if step not in session.steps_completed:
                validation.errors.append(f"Required step not completed: {step.value}")
                validation.is_valid = False

        # Verify terms acceptance
        if not step_data.get("accept_terms", False):
            validation.errors.append("Terms and conditions must be accepted")
            validation.is_valid = False

        if not step_data.get("accept_privacy", False):
            validation.errors.append("Privacy policy must be accepted")
            validation.is_valid = False

        return validation

    async def _handle_registration(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle registration step"""

        # Store registration data
        session.configuration.update(
            {
                "user_details": {
                    "first_name": step_data["first_name"],
                    "last_name": step_data["last_name"],
                    "email": step_data["email"],
                    "phone": step_data.get("phone"),
                    "job_title": step_data.get("job_title"),
                },
                "company_details": {
                    "name": step_data["company_name"],
                    "website": step_data.get("website"),
                    "address": step_data.get("address"),
                },
            }
        )

        # Create user account (would integrate with auth system)
        user_id = str(uuid.uuid4())

        return {"user_id": user_id, "config": {"user_id": user_id, "registration_completed": True}}

    async def _handle_organization(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle organization setup"""

        org_config = {
            "organization_type": step_data["organization_type"],
            "industry": step_data["industry"],
            "employee_count": step_data.get("employee_count"),
            "cloud_environments": step_data.get("cloud_environments", ["azure"]),
            "compliance_requirements": step_data.get("compliance_requirements", []),
            "primary_use_cases": step_data.get("primary_use_cases", []),
        }

        # Determine recommended features based on organization profile
        recommendations = self._get_feature_recommendations(org_config)

        return {"config": {"organization": org_config, "recommended_features": recommendations}}

    async def _handle_azure_connection(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle Azure connection setup"""

        connection_config = {
            "connection_type": step_data["connection_type"],
            "tenant_id": step_data.get("tenant_id"),
            "subscriptions": step_data.get("selected_subscriptions", []),
        }

        # Test connection (would actually connect to Azure)
        connection_test = await self._test_azure_connection(step_data)

        if connection_test["success"]:
            # Discover resources
            discovered_resources = await self._discover_azure_resources(step_data)

            return {
                "connection_status": "connected",
                "discovered_resources": discovered_resources,
                "config": {
                    "azure_connection": connection_config,
                    "resource_summary": discovered_resources["summary"],
                },
            }
        else:
            raise Exception(f"Azure connection failed: {connection_test['error']}")

    async def _handle_feature_selection(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle feature selection"""

        selected_features = step_data["selected_features"]
        feature_config = step_data.get("feature_configuration", {})

        # Calculate pricing based on features
        pricing = self._calculate_pricing(selected_features, session.configuration)

        return {
            "config": {
                "selected_features": selected_features,
                "feature_configuration": feature_config,
                "pricing": pricing,
            }
        }

    async def _handle_initial_scan(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle initial environment scan"""

        scan_type = step_data.get("scan_type", "quick")

        if scan_type == "skip":
            return {"scan_status": "skipped", "config": {"initial_scan": "skipped"}}

        # Start scan (would actually trigger scanning)
        scan_id = str(uuid.uuid4())

        scan_config = {
            "scan_type": scan_type,
            "resource_types": step_data.get("resource_types", ["all"]),
            "enable_remediation": step_data.get("enable_remediation", False),
            "scan_id": scan_id,
        }

        # Simulate scan results
        scan_results = {
            "resources_scanned": 127,
            "compliance_score": 78,
            "critical_issues": 3,
            "recommendations": 15,
        }

        return {
            "scan_id": scan_id,
            "scan_status": "in_progress",
            "initial_results": scan_results,
            "config": {"initial_scan": scan_config, "scan_results": scan_results},
        }

    async def _handle_user_invites(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle user invitation"""

        invitations = step_data.get("invitations", [])

        invited_users = []
        for invite in invitations:
            invitation_id = str(uuid.uuid4())
            invited_users.append(
                {
                    "invitation_id": invitation_id,
                    "email": invite["email"],
                    "role": invite["role"],
                    "status": "pending",
                }
            )

        return {"invitations_sent": len(invited_users), "config": {"invited_users": invited_users}}

    async def _handle_configuration(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle configuration settings"""

        config = {
            "notifications": step_data.get("notifications", {}),
            "compliance": step_data.get("compliance", {}),
            "security": step_data.get("security", {}),
            "integrations": step_data.get("integrations", {}),
            "customization": step_data.get("customization", {}),
        }

        return {"config": {"settings": config}}

    async def _handle_completion(
        self, step_data: Dict[str, Any], session: OnboardingSession
    ) -> Dict[str, Any]:
        """Handle onboarding completion"""

        # Provision tenant (would actually create resources)
        provisioning_result = await self._provision_tenant(session)

        return {
            "status": "completed",
            "tenant_id": session.tenant_id,
            "provisioning": provisioning_result,
            "config": {"onboarding_completed": True, "completed_at": datetime.utcnow().isoformat()},
        }

    def _get_next_step(
        self, current_step: OnboardingStep, session: OnboardingSession
    ) -> Optional[OnboardingStep]:
        """Get the next step in the onboarding flow"""

        step_order = [
            OnboardingStep.REGISTRATION,
            OnboardingStep.ORGANIZATION,
            OnboardingStep.AZURE_CONNECTION,
            OnboardingStep.FEATURE_SELECTION,
            OnboardingStep.INITIAL_SCAN,
            OnboardingStep.USER_INVITES,
            OnboardingStep.CONFIGURATION,
            OnboardingStep.COMPLETION,
        ]

        current_index = step_order.index(current_step)

        # Skip optional steps based on configuration
        while current_index < len(step_order) - 1:
            current_index += 1
            next_step = step_order[current_index]

            # Skip user invites if single user
            if next_step == OnboardingStep.USER_INVITES:
                if session.configuration.get("skip_user_invites", False):
                    continue

            # Skip initial scan if opted out
            if next_step == OnboardingStep.INITIAL_SCAN:
                if session.configuration.get("skip_initial_scan", False):
                    continue

            return next_step

        return None

    def _calculate_progress(self, session: OnboardingSession) -> float:
        """Calculate onboarding progress percentage"""

        total_steps = 8
        completed_steps = len(session.steps_completed)

        return (completed_steps / total_steps) * 100

    def _get_feature_recommendations(self, org_config: Dict[str, Any]) -> List[str]:
        """Get feature recommendations based on organization profile"""

        recommendations = ["compliance_monitoring"]  # Always recommended

        # Industry-specific recommendations
        industry = org_config.get("industry", "").lower()
        if industry in ["healthcare", "finance", "government"]:
            recommendations.extend(["security_posture", "policy_management", "auto_remediation"])

        # Size-based recommendations
        employee_count = org_config.get("employee_count", 0)
        if isinstance(employee_count, str):
            employee_count = int(employee_count) if employee_count.isdigit() else 0

        if employee_count > 100:
            recommendations.extend(["ai_analytics", "custom_dashboards"])

        if employee_count > 500:
            recommendations.extend(["api_access", "multi_cloud"])

        # Cloud environment recommendations
        cloud_envs = org_config.get("cloud_environments", [])
        if len(cloud_envs) > 1:
            recommendations.append("multi_cloud")

        # Compliance recommendations
        compliance_reqs = org_config.get("compliance_requirements", [])
        if compliance_reqs:
            recommendations.extend(["policy_management", "auto_remediation"])

        return list(set(recommendations))  # Remove duplicates

    async def _test_azure_connection(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Azure connection"""

        # Simulate connection test
        await asyncio.sleep(1)

        # In production, would actually test the connection
        return {"success": True, "subscriptions_found": 3, "permissions_verified": True}

    async def _discover_azure_resources(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover Azure resources"""

        # Simulate resource discovery
        await asyncio.sleep(2)

        # In production, would actually discover resources
        return {
            "summary": {
                "total_resources": 247,
                "resource_groups": 12,
                "virtual_machines": 34,
                "storage_accounts": 18,
                "databases": 8,
                "network_resources": 45,
            },
            "compliance_preview": {"compliant": 189, "non_compliant": 58, "score": 76},
        }

    def _calculate_pricing(
        self, features: List[str], configuration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate pricing based on selected features"""

        base_price = 99  # Base monthly price
        feature_prices = {
            "compliance_monitoring": 0,  # Included in base
            "cost_optimization": 49,
            "security_posture": 79,
            "policy_management": 99,
            "ai_analytics": 199,
            "auto_remediation": 149,
            "custom_dashboards": 49,
            "api_access": 99,
            "multi_cloud": 299,
        }

        monthly_total = base_price
        for feature in features:
            monthly_total += feature_prices.get(feature, 0)

        # Apply discounts
        discount = 0
        if len(features) >= 5:
            discount = 0.1  # 10% discount for 5+ features
        elif len(features) >= 3:
            discount = 0.05  # 5% discount for 3+ features

        discounted_price = monthly_total * (1 - discount)

        return {
            "base_price": base_price,
            "feature_costs": {f: feature_prices.get(f, 0) for f in features},
            "monthly_total": monthly_total,
            "discount_percentage": discount * 100,
            "discounted_monthly": discounted_price,
            "annual_total": discounted_price * 12,
            "annual_savings": discounted_price * 12 * 0.15,  # 15% annual discount
        }

    async def _provision_tenant(self, session: OnboardingSession) -> Dict[str, Any]:
        """Provision tenant resources"""

        # Simulate provisioning
        await asyncio.sleep(3)

        # In production, would actually provision resources
        return {
            "tenant_id": session.tenant_id,
            "resources_created": [
                "database_schema",
                "storage_containers",
                "api_keys",
                "default_policies",
                "dashboards",
            ],
            "status": "provisioned",
            "access_url": f"https://app.policycortex.com/{session.tenant_id}",
        }

    def get_session(self, session_id: str) -> Optional[OnboardingSession]:
        """Get onboarding session by ID"""
        return self.sessions.get(session_id)

    def get_active_sessions(self) -> List[OnboardingSession]:
        """Get all active onboarding sessions"""
        return [
            session
            for session in self.sessions.values()
            if session.status == OnboardingStatus.IN_PROGRESS
        ]

    async def abandon_session(self, session_id: str) -> bool:
        """Mark a session as abandoned"""

        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        session.status = OnboardingStatus.ABANDONED

        logger.info(f"Onboarding session {session_id} abandoned")

        return True

    def get_onboarding_analytics(self) -> Dict[str, Any]:
        """Get onboarding analytics"""

        total_sessions = len(self.sessions)
        completed = sum(1 for s in self.sessions.values() if s.status == OnboardingStatus.COMPLETED)
        in_progress = sum(
            1 for s in self.sessions.values() if s.status == OnboardingStatus.IN_PROGRESS
        )
        abandoned = sum(1 for s in self.sessions.values() if s.status == OnboardingStatus.ABANDONED)
        failed = sum(1 for s in self.sessions.values() if s.status == OnboardingStatus.FAILED)

        # Calculate average completion time
        completion_times = []
        for session in self.sessions.values():
            if session.status == OnboardingStatus.COMPLETED and session.completed_at:
                duration = (session.completed_at - session.started_at).total_seconds() / 60
                completion_times.append(duration)

        avg_completion_time = (
            sum(completion_times) / len(completion_times) if completion_times else 0
        )

        # Step completion rates
        step_completions = {}
        for step in OnboardingStep:
            completed_count = sum(1 for s in self.sessions.values() if step in s.steps_completed)
            step_completions[step.value] = (
                (completed_count / total_sessions * 100) if total_sessions > 0 else 0
            )

        return {
            "total_sessions": total_sessions,
            "completed": completed,
            "in_progress": in_progress,
            "abandoned": abandoned,
            "failed": failed,
            "completion_rate": (completed / total_sessions * 100) if total_sessions > 0 else 0,
            "average_completion_time_minutes": avg_completion_time,
            "step_completion_rates": step_completions,
        }
