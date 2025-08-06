"""
Billing Integration Module
Integrates with payment providers and manages billing
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import uuid
import hashlib
import hmac

logger = structlog.get_logger(__name__)

class PaymentProvider(str, Enum):
    STRIPE = "stripe"
    PAYPAL = "paypal"
    AZURE_MARKETPLACE = "azure_marketplace"
    INVOICE = "invoice"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    INVOICE = "invoice"

@dataclass
class PaymentDetails:
    """Payment details"""
    payment_id: str
    tenant_id: str
    amount: float
    currency: str
    provider: PaymentProvider
    method: PaymentMethod
    status: PaymentStatus
    created_at: datetime
    processed_at: Optional[datetime] = None
    reference_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PaymentMethodInfo:
    """Stored payment method information"""
    method_id: str
    tenant_id: str
    method_type: PaymentMethod
    provider: PaymentProvider
    is_default: bool
    last_four: Optional[str] = None
    brand: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Invoice:
    """Invoice details"""
    invoice_id: str
    tenant_id: str
    invoice_number: str
    amount: float
    currency: str
    status: str
    due_date: datetime
    created_at: datetime
    paid_at: Optional[datetime] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BillingIntegration:
    """
    Manages billing and payment processing
    """
    
    def __init__(self):
        self.payments = {}
        self.payment_methods = {}
        self.invoices = {}
        self.webhook_secrets = {}
        self.provider_configs = self._load_provider_configs()
        
    def _load_provider_configs(self) -> Dict[str, Any]:
        """Load payment provider configurations"""
        return {
            PaymentProvider.STRIPE: {
                'api_key': 'sk_test_dummy',  # Would be from environment
                'webhook_secret': 'whsec_dummy',
                'supported_currencies': ['USD', 'EUR', 'GBP'],
                'supported_methods': [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD]
            },
            PaymentProvider.PAYPAL: {
                'client_id': 'paypal_client_dummy',
                'client_secret': 'paypal_secret_dummy',
                'supported_currencies': ['USD', 'EUR', 'GBP', 'CAD'],
                'supported_methods': [PaymentMethod.PAYPAL, PaymentMethod.CREDIT_CARD]
            },
            PaymentProvider.AZURE_MARKETPLACE: {
                'publisher_id': 'policycortex',
                'offer_id': 'policycortex-governance',
                'plan_ids': ['starter', 'professional', 'enterprise'],
                'supported_currencies': ['USD'],
                'supported_methods': [PaymentMethod.INVOICE]
            },
            PaymentProvider.INVOICE: {
                'payment_terms': 30,  # Net 30 days
                'supported_currencies': ['USD', 'EUR'],
                'supported_methods': [PaymentMethod.INVOICE, PaymentMethod.BANK_TRANSFER]
            }
        }
        
    async def process_payment(self,
                            tenant_id: str,
                            amount: float,
                            currency: str,
                            provider: PaymentProvider,
                            method: PaymentMethod,
                            payment_method_id: Optional[str] = None) -> PaymentDetails:
        """
        Process a payment
        
        Args:
            tenant_id: Tenant making payment
            amount: Payment amount
            currency: Currency code
            provider: Payment provider
            method: Payment method
            payment_method_id: Stored payment method ID
            
        Returns:
            Payment details
        """
        
        payment_id = str(uuid.uuid4())
        
        payment = PaymentDetails(
            payment_id=payment_id,
            tenant_id=tenant_id,
            amount=amount,
            currency=currency,
            provider=provider,
            method=method,
            status=PaymentStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        self.payments[payment_id] = payment
        
        try:
            # Process payment based on provider
            if provider == PaymentProvider.STRIPE:
                result = await self._process_stripe_payment(payment, payment_method_id)
            elif provider == PaymentProvider.PAYPAL:
                result = await self._process_paypal_payment(payment)
            elif provider == PaymentProvider.AZURE_MARKETPLACE:
                result = await self._process_azure_marketplace_payment(payment)
            elif provider == PaymentProvider.INVOICE:
                result = await self._process_invoice_payment(payment)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            payment.status = PaymentStatus.SUCCEEDED if result['success'] else PaymentStatus.FAILED
            payment.processed_at = datetime.utcnow()
            payment.reference_id = result.get('reference_id')
            
            if not result['success']:
                payment.error_message = result.get('error')
                
        except Exception as e:
            payment.status = PaymentStatus.FAILED
            payment.error_message = str(e)
            logger.error(f"Payment processing failed: {e}")
            
        logger.info(f"Processed payment {payment_id} with status {payment.status}")
        
        return payment
        
    async def _process_stripe_payment(self,
                                    payment: PaymentDetails,
                                    payment_method_id: Optional[str]) -> Dict[str, Any]:
        """Process payment through Stripe"""
        
        # Simulate Stripe payment processing
        import asyncio
        await asyncio.sleep(1)
        
        # In production, would use actual Stripe SDK
        success = payment.amount < 10000  # Simulate success for amounts under $10k
        
        return {
            'success': success,
            'reference_id': f'ch_{uuid.uuid4().hex[:24]}',
            'error': None if success else 'Payment declined'
        }
        
    async def _process_paypal_payment(self, payment: PaymentDetails) -> Dict[str, Any]:
        """Process payment through PayPal"""
        
        # Simulate PayPal payment processing
        import asyncio
        await asyncio.sleep(1.5)
        
        # In production, would use actual PayPal SDK
        return {
            'success': True,
            'reference_id': f'PAYPAL-{uuid.uuid4().hex[:16].upper()}',
            'error': None
        }
        
    async def _process_azure_marketplace_payment(self, payment: PaymentDetails) -> Dict[str, Any]:
        """Process payment through Azure Marketplace"""
        
        # Simulate Azure Marketplace processing
        import asyncio
        await asyncio.sleep(0.5)
        
        # In production, would integrate with Azure Marketplace API
        return {
            'success': True,
            'reference_id': f'AZ-{uuid.uuid4().hex[:12].upper()}',
            'error': None
        }
        
    async def _process_invoice_payment(self, payment: PaymentDetails) -> Dict[str, Any]:
        """Process invoice payment"""
        
        # Create invoice
        invoice = await self.create_invoice(
            payment.tenant_id,
            payment.amount,
            payment.currency,
            due_days=30
        )
        
        payment.metadata['invoice_id'] = invoice.invoice_id
        
        return {
            'success': True,
            'reference_id': invoice.invoice_number,
            'error': None
        }
        
    async def add_payment_method(self,
                               tenant_id: str,
                               provider: PaymentProvider,
                               method_type: PaymentMethod,
                               details: Dict[str, Any]) -> PaymentMethodInfo:
        """
        Add a payment method for a tenant
        
        Args:
            tenant_id: Tenant ID
            provider: Payment provider
            method_type: Type of payment method
            details: Method details (card info, etc.)
            
        Returns:
            Payment method information
        """
        
        method_id = str(uuid.uuid4())
        
        # Extract display information
        last_four = None
        brand = None
        expiry_month = None
        expiry_year = None
        
        if method_type in [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD]:
            card_number = details.get('card_number', '')
            last_four = card_number[-4:] if len(card_number) >= 4 else None
            brand = self._detect_card_brand(card_number)
            expiry_month = details.get('expiry_month')
            expiry_year = details.get('expiry_year')
            
        # Check if this should be default
        existing_methods = [m for m in self.payment_methods.values() if m.tenant_id == tenant_id]
        is_default = len(existing_methods) == 0
        
        payment_method = PaymentMethodInfo(
            method_id=method_id,
            tenant_id=tenant_id,
            method_type=method_type,
            provider=provider,
            is_default=is_default,
            last_four=last_four,
            brand=brand,
            expiry_month=expiry_month,
            expiry_year=expiry_year
        )
        
        self.payment_methods[method_id] = payment_method
        
        # Tokenize with provider (in production)
        if provider == PaymentProvider.STRIPE:
            # Would create Stripe payment method
            payment_method.metadata['stripe_pm_id'] = f'pm_{uuid.uuid4().hex[:24]}'
            
        logger.info(f"Added payment method {method_id} for tenant {tenant_id}")
        
        return payment_method
        
    def _detect_card_brand(self, card_number: str) -> str:
        """Detect card brand from number"""
        
        if not card_number:
            return 'unknown'
            
        # Simplified brand detection
        if card_number.startswith('4'):
            return 'visa'
        elif card_number.startswith(('51', '52', '53', '54', '55')):
            return 'mastercard'
        elif card_number.startswith(('34', '37')):
            return 'amex'
        elif card_number.startswith('6011'):
            return 'discover'
        else:
            return 'unknown'
            
    async def remove_payment_method(self,
                                  tenant_id: str,
                                  method_id: str) -> bool:
        """
        Remove a payment method
        
        Args:
            tenant_id: Tenant ID
            method_id: Method to remove
            
        Returns:
            Success status
        """
        
        if method_id not in self.payment_methods:
            return False
            
        method = self.payment_methods[method_id]
        
        if method.tenant_id != tenant_id:
            logger.warning(f"Tenant {tenant_id} attempted to remove method owned by {method.tenant_id}")
            return False
            
        # If removing default, make another default
        if method.is_default:
            other_methods = [
                m for m in self.payment_methods.values()
                if m.tenant_id == tenant_id and m.method_id != method_id
            ]
            if other_methods:
                other_methods[0].is_default = True
                
        del self.payment_methods[method_id]
        
        logger.info(f"Removed payment method {method_id}")
        
        return True
        
    async def set_default_payment_method(self,
                                       tenant_id: str,
                                       method_id: str) -> bool:
        """
        Set default payment method
        
        Args:
            tenant_id: Tenant ID
            method_id: Method to make default
            
        Returns:
            Success status
        """
        
        if method_id not in self.payment_methods:
            return False
            
        method = self.payment_methods[method_id]
        
        if method.tenant_id != tenant_id:
            return False
            
        # Remove default from others
        for m in self.payment_methods.values():
            if m.tenant_id == tenant_id:
                m.is_default = False
                
        method.is_default = True
        
        logger.info(f"Set payment method {method_id} as default for tenant {tenant_id}")
        
        return True
        
    async def create_invoice(self,
                          tenant_id: str,
                          amount: float,
                          currency: str,
                          due_days: int = 30,
                          line_items: Optional[List[Dict[str, Any]]] = None) -> Invoice:
        """
        Create an invoice
        
        Args:
            tenant_id: Tenant ID
            amount: Invoice amount
            currency: Currency code
            due_days: Days until due
            line_items: Invoice line items
            
        Returns:
            Created invoice
        """
        
        invoice_id = str(uuid.uuid4())
        invoice_number = f"INV-{datetime.utcnow().strftime('%Y%m')}-{len(self.invoices) + 1:04d}"
        
        invoice = Invoice(
            invoice_id=invoice_id,
            tenant_id=tenant_id,
            invoice_number=invoice_number,
            amount=amount,
            currency=currency,
            status='pending',
            due_date=datetime.utcnow() + timedelta(days=due_days),
            created_at=datetime.utcnow(),
            line_items=line_items or []
        )
        
        self.invoices[invoice_id] = invoice
        
        logger.info(f"Created invoice {invoice_number} for tenant {tenant_id}")
        
        return invoice
        
    async def process_webhook(self,
                            provider: PaymentProvider,
                            payload: Dict[str, Any],
                            signature: str) -> Dict[str, Any]:
        """
        Process payment provider webhook
        
        Args:
            provider: Payment provider
            payload: Webhook payload
            signature: Webhook signature
            
        Returns:
            Processing result
        """
        
        # Verify webhook signature
        if not self._verify_webhook_signature(provider, payload, signature):
            logger.warning(f"Invalid webhook signature from {provider}")
            return {'status': 'invalid_signature'}
            
        # Process based on provider and event type
        if provider == PaymentProvider.STRIPE:
            return await self._process_stripe_webhook(payload)
        elif provider == PaymentProvider.PAYPAL:
            return await self._process_paypal_webhook(payload)
        else:
            return {'status': 'unsupported_provider'}
            
    def _verify_webhook_signature(self,
                                provider: PaymentProvider,
                                payload: Dict[str, Any],
                                signature: str) -> bool:
        """Verify webhook signature"""
        
        if provider not in self.provider_configs:
            return False
            
        config = self.provider_configs[provider]
        secret = config.get('webhook_secret', '')
        
        # Simplified signature verification
        import json
        payload_str = json.dumps(payload, sort_keys=True)
        expected_signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
        
    async def _process_stripe_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process Stripe webhook"""
        
        event_type = payload.get('type')
        
        if event_type == 'payment_intent.succeeded':
            # Update payment status
            reference_id = payload.get('data', {}).get('object', {}).get('id')
            for payment in self.payments.values():
                if payment.reference_id == reference_id:
                    payment.status = PaymentStatus.SUCCEEDED
                    payment.processed_at = datetime.utcnow()
                    break
                    
        elif event_type == 'payment_intent.payment_failed':
            # Handle payment failure
            reference_id = payload.get('data', {}).get('object', {}).get('id')
            for payment in self.payments.values():
                if payment.reference_id == reference_id:
                    payment.status = PaymentStatus.FAILED
                    payment.error_message = 'Payment failed'
                    break
                    
        return {'status': 'processed'}
        
    async def _process_paypal_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process PayPal webhook"""
        
        event_type = payload.get('event_type')
        
        if event_type == 'PAYMENT.CAPTURE.COMPLETED':
            # Update payment status
            reference_id = payload.get('resource', {}).get('id')
            for payment in self.payments.values():
                if payment.reference_id == reference_id:
                    payment.status = PaymentStatus.SUCCEEDED
                    payment.processed_at = datetime.utcnow()
                    break
                    
        return {'status': 'processed'}
        
    async def refund_payment(self,
                           payment_id: str,
                           amount: Optional[float] = None,
                           reason: Optional[str] = None) -> bool:
        """
        Refund a payment
        
        Args:
            payment_id: Payment to refund
            amount: Refund amount (None for full refund)
            reason: Refund reason
            
        Returns:
            Success status
        """
        
        if payment_id not in self.payments:
            return False
            
        payment = self.payments[payment_id]
        
        if payment.status != PaymentStatus.SUCCEEDED:
            logger.warning(f"Cannot refund payment {payment_id} with status {payment.status}")
            return False
            
        refund_amount = amount or payment.amount
        
        # Process refund with provider
        if payment.provider == PaymentProvider.STRIPE:
            # Would process Stripe refund
            pass
        elif payment.provider == PaymentProvider.PAYPAL:
            # Would process PayPal refund
            pass
            
        payment.status = PaymentStatus.REFUNDED
        payment.metadata['refund'] = {
            'amount': refund_amount,
            'reason': reason,
            'refunded_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Refunded payment {payment_id}: ${refund_amount}")
        
        return True
        
    def get_payment_history(self,
                          tenant_id: str,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get payment history for a tenant"""
        
        tenant_payments = [
            p for p in self.payments.values()
            if p.tenant_id == tenant_id
        ]
        
        # Sort by creation date descending
        tenant_payments.sort(key=lambda x: x.created_at, reverse=True)
        
        return [
            {
                'payment_id': p.payment_id,
                'amount': p.amount,
                'currency': p.currency,
                'status': p.status.value,
                'provider': p.provider.value,
                'method': p.method.value,
                'created_at': p.created_at.isoformat(),
                'processed_at': p.processed_at.isoformat() if p.processed_at else None,
                'reference_id': p.reference_id
            }
            for p in tenant_payments[:limit]
        ]
        
    def get_payment_methods(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get payment methods for a tenant"""
        
        methods = [
            m for m in self.payment_methods.values()
            if m.tenant_id == tenant_id
        ]
        
        return [
            {
                'method_id': m.method_id,
                'method_type': m.method_type.value,
                'provider': m.provider.value,
                'is_default': m.is_default,
                'last_four': m.last_four,
                'brand': m.brand,
                'expiry': f"{m.expiry_month}/{m.expiry_year}" if m.expiry_month else None,
                'created_at': m.created_at.isoformat()
            }
            for m in methods
        ]
        
    def get_invoices(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get invoices for a tenant"""
        
        tenant_invoices = [
            i for i in self.invoices.values()
            if i.tenant_id == tenant_id
        ]
        
        return [
            {
                'invoice_id': i.invoice_id,
                'invoice_number': i.invoice_number,
                'amount': i.amount,
                'currency': i.currency,
                'status': i.status,
                'due_date': i.due_date.isoformat(),
                'created_at': i.created_at.isoformat(),
                'paid_at': i.paid_at.isoformat() if i.paid_at else None
            }
            for i in tenant_invoices
        ]