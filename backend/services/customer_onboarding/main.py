"""
Customer Onboarding Service Main Entry Point
Phase 5: Self-Service Customer Onboarding System
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import structlog
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.shared.config import get_settings
from backend.shared.database import get_async_db

from .onboarding_wizard import OnboardingWizard, OnboardingStep
from .organization_provisioner import OrganizationProvisioner
from .subscription_manager import SubscriptionManager, BillingCycle, PlanTier
from .trial_manager import TrialManager
from .billing_integration import BillingIntegration, PaymentProvider, PaymentMethod

settings = get_settings()
logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PolicyCortex Customer Onboarding Service",
    description="Self-Service Customer Onboarding and Subscription Management",
    version="5.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize onboarding components
onboarding_wizard = OnboardingWizard()
organization_provisioner = OrganizationProvisioner()
subscription_manager = SubscriptionManager()
trial_manager = TrialManager()
billing_integration = BillingIntegration()

# Pydantic models
class StartOnboardingRequest(BaseModel):
    user_email: str
    company_name: str
    template: Optional[str] = None

class ProcessStepRequest(BaseModel):
    step_data: Dict[str, Any]

class CreateSubscriptionRequest(BaseModel):
    plan_id: str
    billing_cycle: BillingCycle = BillingCycle.MONTHLY

class PaymentMethodRequest(BaseModel):
    provider: PaymentProvider
    method_type: PaymentMethod
    details: Dict[str, Any]

class ProcessPaymentRequest(BaseModel):
    amount: float
    currency: str
    provider: PaymentProvider
    method: PaymentMethod
    payment_method_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Customer Onboarding Service starting up...")
    logger.info("Customer Onboarding Service initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "customer-onboarding",
        "timestamp": datetime.utcnow().isoformat()
    }

# Onboarding Wizard Endpoints
@app.post("/api/v1/onboarding/start")
async def start_onboarding(request: StartOnboardingRequest):
    """Start a new onboarding session"""
    try:
        session = await onboarding_wizard.start_onboarding(
            user_id="temp_user_id",  # Would come from auth
            user_email=request.user_email,
            company_name=request.company_name,
            template=request.template
        )
        
        return {
            "session_id": session.session_id,
            "tenant_id": session.tenant_id,
            "current_step": session.current_step.value,
            "progress": 0
        }
        
    except Exception as e:
        logger.error(f"Failed to start onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/onboarding/{session_id}/step")
async def process_onboarding_step(session_id: str, request: ProcessStepRequest):
    """Process a step in the onboarding wizard"""
    try:
        result = await onboarding_wizard.process_step(session_id, request.step_data)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process step: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/onboarding/{session_id}")
async def get_onboarding_session(session_id: str):
    """Get onboarding session details"""
    session = onboarding_wizard.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return {
        "session_id": session.session_id,
        "tenant_id": session.tenant_id,
        "status": session.status.value,
        "current_step": session.current_step.value,
        "steps_completed": [s.value for s in session.steps_completed],
        "configuration": session.configuration,
        "validation_errors": session.validation_errors,
        "started_at": session.started_at.isoformat(),
        "completed_at": session.completed_at.isoformat() if session.completed_at else None
    }

@app.delete("/api/v1/onboarding/{session_id}")
async def abandon_onboarding(session_id: str):
    """Abandon an onboarding session"""
    success = await onboarding_wizard.abandon_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return {"message": "Onboarding session abandoned"}

@app.get("/api/v1/onboarding/templates")
async def get_onboarding_templates():
    """Get available onboarding templates"""
    return onboarding_wizard.templates

@app.get("/api/v1/onboarding/analytics")
async def get_onboarding_analytics():
    """Get onboarding analytics"""
    return onboarding_wizard.get_onboarding_analytics()

# Organization Provisioning Endpoints
@app.post("/api/v1/organizations/{tenant_id}/provision")
async def provision_organization(tenant_id: str,
                               organization_name: str,
                               template: str,
                               configuration: Dict[str, Any],
                               background_tasks: BackgroundTasks):
    """Provision resources for an organization"""
    try:
        # Start provisioning in background
        background_tasks.add_task(
            provision_organization_task,
            tenant_id,
            organization_name,
            template,
            configuration
        )
        
        return {"message": "Provisioning started", "tenant_id": tenant_id}
        
    except Exception as e:
        logger.error(f"Failed to start provisioning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def provision_organization_task(tenant_id: str,
                                    organization_name: str,
                                    template: str,
                                    configuration: Dict[str, Any]):
    """Background task for organization provisioning"""
    try:
        resources = await organization_provisioner.provision_organization(
            tenant_id,
            organization_name,
            template,
            configuration
        )
        logger.info(f"Provisioning completed for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Provisioning failed for tenant {tenant_id}: {e}")

@app.get("/api/v1/organizations/{tenant_id}/provisioning-status")
async def get_provisioning_status(tenant_id: str):
    """Get provisioning status for an organization"""
    return organization_provisioner.get_provisioning_status(tenant_id)

@app.get("/api/v1/organizations/{tenant_id}/resources")
async def get_organization_resources(tenant_id: str):
    """Get provisioned resources for an organization"""
    resources = organization_provisioner.get_organization_resources(tenant_id)
    
    if not resources:
        raise HTTPException(status_code=404, detail="Organization not found")
        
    return {
        "tenant_id": resources.tenant_id,
        "database_url": resources.database_url,
        "storage_account": resources.storage_account,
        "key_vault_url": resources.key_vault_url,
        "api_keys": resources.api_keys,
        "resource_group": resources.resource_group
    }

@app.delete("/api/v1/organizations/{tenant_id}")
async def deprovision_organization(tenant_id: str, background_tasks: BackgroundTasks):
    """Deprovision an organization"""
    background_tasks.add_task(organization_provisioner.deprovision_organization, tenant_id)
    return {"message": "Deprovisioning started"}

# Subscription Management Endpoints
@app.post("/api/v1/subscriptions")
async def create_subscription(tenant_id: str, request: CreateSubscriptionRequest):
    """Create a new subscription"""
    try:
        subscription = await subscription_manager.create_subscription(
            tenant_id,
            request.plan_id,
            request.billing_cycle
        )
        
        return {
            "subscription_id": subscription.subscription_id,
            "status": subscription.status.value,
            "plan_id": subscription.plan_id,
            "billing_cycle": subscription.billing_cycle.value
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/subscriptions/tenant/{tenant_id}")
async def get_tenant_subscription(tenant_id: str):
    """Get subscription for a tenant"""
    subscription = subscription_manager.get_subscription_by_tenant(tenant_id)
    
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
        
    return subscription_manager.get_subscription_status(subscription.subscription_id)

@app.post("/api/v1/subscriptions/{subscription_id}/upgrade")
async def upgrade_subscription(subscription_id: str,
                             new_plan_id: str,
                             immediate: bool = True):
    """Upgrade a subscription"""
    success = await subscription_manager.upgrade_subscription(
        subscription_id,
        new_plan_id,
        immediate
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Upgrade failed")
        
    return {"message": "Subscription upgraded"}

@app.post("/api/v1/subscriptions/{subscription_id}/cancel")
async def cancel_subscription(subscription_id: str, immediate: bool = False):
    """Cancel a subscription"""
    success = await subscription_manager.cancel_subscription(subscription_id, immediate)
    
    if not success:
        raise HTTPException(status_code=404, detail="Subscription not found")
        
    return {"message": "Subscription cancelled"}

@app.get("/api/v1/subscriptions/{subscription_id}/usage")
async def check_subscription_usage(subscription_id: str):
    """Check subscription usage limits"""
    # Get tenant from subscription
    if subscription_id not in subscription_manager.subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
        
    subscription = subscription_manager.subscriptions[subscription_id]
    return await subscription_manager.check_usage_limits(subscription.tenant_id)

@app.get("/api/v1/plans")
async def get_available_plans():
    """Get available subscription plans"""
    return subscription_manager.get_available_plans()

# Trial Management Endpoints
@app.post("/api/v1/trials/{tenant_id}/start")
async def start_trial(tenant_id: str, plan_id: str, trial_days: int = 14):
    """Start a trial period"""
    try:
        trial = await trial_manager.start_trial(tenant_id, plan_id, trial_days)
        
        return {
            "trial_id": trial.trial_id,
            "status": trial.status.value,
            "expires_at": trial.expires_at.isoformat(),
            "features": [f.value for f in trial.features]
        }
        
    except Exception as e:
        logger.error(f"Failed to start trial: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trials/{trial_id}")
async def get_trial_status(trial_id: str):
    """Get trial status"""
    return await trial_manager.check_trial_status(trial_id)

@app.post("/api/v1/trials/{trial_id}/extend")
async def extend_trial(trial_id: str, additional_days: int, reason: str):
    """Extend a trial period"""
    success = await trial_manager.extend_trial(trial_id, additional_days, reason)
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot extend trial")
        
    return {"message": "Trial extended"}

@app.post("/api/v1/trials/{trial_id}/convert")
async def convert_trial(trial_id: str, subscription_id: str):
    """Convert trial to paid subscription"""
    success = await trial_manager.convert_trial(trial_id, subscription_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot convert trial")
        
    return {"message": "Trial converted"}

@app.get("/api/v1/trials/analytics")
async def get_trial_analytics():
    """Get trial conversion analytics"""
    return trial_manager.get_conversion_analytics()

@app.get("/api/v1/trials/expiring")
async def get_expiring_trials(days_ahead: int = 3):
    """Get trials expiring soon"""
    trials = await trial_manager.get_expiring_trials(days_ahead)
    
    return [
        {
            "trial_id": trial.trial_id,
            "tenant_id": trial.tenant_id,
            "expires_at": trial.expires_at.isoformat(),
            "days_remaining": (trial.expires_at - datetime.utcnow()).days
        }
        for trial in trials
    ]

# Billing Integration Endpoints
@app.post("/api/v1/billing/payment-methods")
async def add_payment_method(tenant_id: str, request: PaymentMethodRequest):
    """Add a payment method"""
    try:
        method = await billing_integration.add_payment_method(
            tenant_id,
            request.provider,
            request.method_type,
            request.details
        )
        
        return {
            "method_id": method.method_id,
            "method_type": method.method_type.value,
            "is_default": method.is_default,
            "last_four": method.last_four,
            "brand": method.brand
        }
        
    except Exception as e:
        logger.error(f"Failed to add payment method: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/billing/payment-methods/{tenant_id}")
async def get_payment_methods(tenant_id: str):
    """Get payment methods for a tenant"""
    return billing_integration.get_payment_methods(tenant_id)

@app.delete("/api/v1/billing/payment-methods/{tenant_id}/{method_id}")
async def remove_payment_method(tenant_id: str, method_id: str):
    """Remove a payment method"""
    success = await billing_integration.remove_payment_method(tenant_id, method_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Payment method not found")
        
    return {"message": "Payment method removed"}

@app.post("/api/v1/billing/payments")
async def process_payment(tenant_id: str, request: ProcessPaymentRequest):
    """Process a payment"""
    try:
        payment = await billing_integration.process_payment(
            tenant_id,
            request.amount,
            request.currency,
            request.provider,
            request.method,
            request.payment_method_id
        )
        
        return {
            "payment_id": payment.payment_id,
            "status": payment.status.value,
            "amount": payment.amount,
            "currency": payment.currency,
            "reference_id": payment.reference_id
        }
        
    except Exception as e:
        logger.error(f"Failed to process payment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/billing/payments/{tenant_id}")
async def get_payment_history(tenant_id: str, limit: int = Query(100, le=1000)):
    """Get payment history for a tenant"""
    return billing_integration.get_payment_history(tenant_id, limit)

@app.get("/api/v1/billing/invoices/{tenant_id}")
async def get_invoices(tenant_id: str):
    """Get invoices for a tenant"""
    return billing_integration.get_invoices(tenant_id)

@app.post("/api/v1/billing/webhooks/{provider}")
async def handle_billing_webhook(provider: str, payload: Dict[str, Any], signature: str):
    """Handle billing provider webhooks"""
    try:
        provider_enum = PaymentProvider(provider)
        result = await billing_integration.process_webhook(
            provider_enum,
            payload,
            signature
        )
        
        return result
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid provider")
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Usage Recording Endpoints
@app.post("/api/v1/usage/record")
async def record_usage(tenant_id: str, metric: str, value: float, metadata: Optional[Dict[str, Any]] = None):
    """Record usage for a tenant"""
    await subscription_manager.record_usage(tenant_id, metric, value, metadata)
    return {"message": "Usage recorded"}

@app.get("/api/v1/usage/{tenant_id}/summary")
async def get_usage_summary(tenant_id: str, days: int = Query(30, le=365)):
    """Get usage summary for a tenant"""
    return subscription_manager.get_usage_summary(tenant_id, days)

# Invoice Generation
@app.post("/api/v1/billing/invoices/{subscription_id}/generate")
async def generate_invoice(subscription_id: str):
    """Generate invoice for a subscription"""
    invoice = subscription_manager.calculate_invoice(subscription_id)
    
    if not invoice:
        raise HTTPException(status_code=404, detail="Subscription not found")
        
    return invoice

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)