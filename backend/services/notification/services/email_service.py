"""
Email service for sending email notifications with template support.
"""

import asyncio
import base64
import json
import smtplib
import ssl
import uuid
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiosmtplib
import jinja2
import redis.asyncio as redis
import structlog
from shared.config import get_settings

    EmailRequest,
    NotificationResponse,
    NotificationTemplate,
    DeliveryStatusEnum,
    EmailProviderConfig
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class EmailService:
    """Service for sending email notifications with template engine support."""

    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.jinja_env = None
        self.smtp_configs = {}
        self.template_cache = {}

    async def initialize(self) -> None:
        """Initialize the email service."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )

            # Initialize Jinja2 environment
            template_dir = Path(__file__).parent.parent / "templates" / "email"
            template_dir.mkdir(parents=True, exist_ok=True)

            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(template_dir)),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )

            # Load SMTP configurations
            await self._load_smtp_configs()

            # Load templates from cache
            await self._load_templates()

            logger.info("email_service_initialized")

        except Exception as e:
            logger.error("email_service_initialization_failed", error=str(e))
            raise

    async def _load_smtp_configs(self) -> None:
        """Load SMTP configurations from environment/config."""
        try:
            # Default SMTP configuration
            default_config = EmailProviderConfig(
                provider="smtp",
                smtp_host=getattr(self.settings, "smtp_host", "localhost"),
                smtp_port=getattr(self.settings, "smtp_port", 587),
                smtp_username=getattr(self.settings, "smtp_username", ""),
                smtp_password=getattr(self.settings, "smtp_password", ""),
                from_email=getattr(self.settings, "from_email", "noreply@policycortex.com"),
                from_name=getattr(self.settings, "from_name", "PolicyCortex"),
                settings={
                    "use_tls": getattr(self.settings, "smtp_use_tls", True),
                    "use_ssl": getattr(self.settings, "smtp_use_ssl", False)
                }
            )

            self.smtp_configs["default"] = default_config

            # Add more providers as needed (SendGrid, Mailgun, etc.)
            # These would typically be loaded from database or config

            logger.info("smtp_configs_loaded", count=len(self.smtp_configs))

        except Exception as e:
            logger.error("smtp_configs_loading_failed", error=str(e))
            raise

    async def _load_templates(self) -> None:
        """Load email templates from cache."""
        try:
            # Load templates from Redis cache
            template_keys = await self.redis_client.keys("email_template:*")

            for template_key in template_keys:
                template_data = await self.redis_client.get(template_key)
                if template_data:
                    template = json.loads(template_data)
                    template_id = template_key.split(":")[-1]
                    self.template_cache[template_id] = template

            logger.info("email_templates_loaded", count=len(self.template_cache))

        except Exception as e:
            logger.error("email_templates_loading_failed", error=str(e))

    async def send_email(self, request: EmailRequest) -> NotificationResponse:
        """Send email notification."""
        try:
            notification_id = request.id or str(uuid.uuid4())

            # Get SMTP configuration
            smtp_config = self.smtp_configs.get("default")
            if not smtp_config:
                raise Exception("No SMTP configuration available")

            # Prepare email content
            subject, html_content, text_content = await self._prepare_email_content(request)

            # Send to all recipients
            delivery_details = []
            delivered_count = 0
            failed_count = 0

            for recipient in request.recipients:
                try:
                    if not recipient.email:
                        logger.warning("recipient_missing_email", recipient_id=recipient.id)
                        failed_count += 1
                        continue

                    # Send email
                    await self._send_single_email(
                        smtp_config,
                        recipient.email,
                        subject,
                        html_content,
                        text_content,
                        request
                    )

                    delivered_count += 1
                    delivery_details.append({
                        "recipient": recipient.email,
                        "status": "delivered",
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    logger.info(
                        "email_sent",
                        notification_id=notification_id,
                        recipient=recipient.email
                    )

                except Exception as e:
                    failed_count += 1
                    delivery_details.append({
                        "recipient": recipient.email,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    logger.error(
                        "email_send_failed",
                        notification_id=notification_id,
                        recipient=recipient.email,
                        error=str(e)
                    )

            # Store delivery status
            await self._store_delivery_status(
                notification_id,
                request,
                delivery_details,
                delivered_count,
                failed_count
            )

            status = (
                DeliveryStatusEnum.DELIVERED if delivered_count > 0 else DeliveryStatusEnum.FAILED
            )

            return NotificationResponse(
                notification_id=notification_id,
                status=status,
                message=f"Email sent to {delivered_count} recipients, {failed_count} failed",
                sent_at=datetime.utcnow(),
                delivered_count=delivered_count,
                failed_count=failed_count,
                delivery_details=delivery_details
            )

        except Exception as e:
            logger.error("email_service_send_failed", error=str(e))

            return NotificationResponse(
                notification_id=notification_id,
                status=DeliveryStatusEnum.FAILED,
                message=f"Email sending failed: {str(e)}",
                sent_at=datetime.utcnow(),
                delivered_count=0,
                failed_count=len(request.recipients),
                delivery_details=[]
            )

    async def _prepare_email_content(self, request: EmailRequest) -> tuple:
        """Prepare email content with template processing."""
        try:
            # Use template if specified
            if request.content.template_id:
                template = await self._get_template(request.content.template_id)
                if template:
                    subject = self._render_template(
                        template.get("subject", ""),
                        request.content.template_variables or {}
                    )
                    html_content = self._render_template(
                        template.get("html_body", ""),
                        request.content.template_variables or {}
                    )
                    text_content = self._render_template(
                        template.get("text_body", ""),
                        request.content.template_variables or {}
                    )
                else:
                    # Fallback to request content
                    subject = request.content.subject or "Notification"
                    html_content = request.content.html_body or request.content.body
                    text_content = request.content.body
            else:
                # Use direct content
                subject = request.content.subject or "Notification"
                html_content = request.content.html_body or request.content.body
                text_content = request.content.body

            return subject, html_content, text_content

        except Exception as e:
            logger.error("email_content_preparation_failed", error=str(e))
            # Return basic content on error
            return (
                request.content.subject or "Notification",
                request.content.html_body or request.content.body,
                request.content.body
            )

    def _render_template(self, template_content: str, variables: Dict[str, Any]) -> str:
        """Render Jinja2 template with variables."""
        try:
            template = self.jinja_env.from_string(template_content)
            return template.render(**variables)
        except Exception as e:
            logger.error("template_rendering_failed", error=str(e))
            return template_content

    async def _send_single_email(
        self,
        smtp_config: EmailProviderConfig,
        recipient_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        request: EmailRequest
    ) -> None:
        """Send single email using SMTP."""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{smtp_config.from_name} <{smtp_config.from_email}>"
            message["To"] = recipient_email

            if request.reply_to:
                message["Reply-To"] = request.reply_to

            # Add custom headers
            if request.headers:
                for header_name, header_value in request.headers.items():
                    message[header_name] = header_value

            # Add CC and BCC
            if request.cc:
                message["Cc"] = ", ".join(request.cc)

            # Add text and HTML parts
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")

            message.attach(text_part)
            message.attach(html_part)

            # Add attachments
            if request.attachments:
                for attachment in request.attachments:
                    await self._add_attachment(message, attachment)

            # Send email
            await self._send_smtp_email(smtp_config, message, recipient_email, request)

        except Exception as e:
            logger.error("single_email_send_failed", error=str(e))
            raise

    async def _add_attachment(self, message: MIMEMultipart, attachment: Dict[str, Any]) -> None:
        """Add attachment to email message."""
        try:
            if "content" in attachment and "filename" in attachment:
                # Base64 encoded content
                content = base64.b64decode(attachment["content"])

                part = MIMEBase("application", "octet-stream")
                part.set_payload(content)
                encoders.encode_base64(part)

                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {attachment['filename']}"
                )

                message.attach(part)

            elif "file_path" in attachment:
                # File path
                file_path = Path(attachment["file_path"])
                if file_path.exists():
                    async with aiofiles.open(file_path, "rb") as file:
                        content = await file.read()

                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(content)
                    encoders.encode_base64(part)

                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {file_path.name}"
                    )

                    message.attach(part)

        except Exception as e:
            logger.error("attachment_addition_failed", error=str(e))

    async def _send_smtp_email(
        self,
        smtp_config: EmailProviderConfig,
        message: MIMEMultipart,
        recipient_email: str,
        request: EmailRequest
    ) -> None:
        """Send email via SMTP."""
        try:
            # Build recipient list
            recipients = [recipient_email]
            if request.cc:
                recipients.extend(request.cc)
            if request.bcc:
                recipients.extend(request.bcc)

            # Send email using aiosmtplib
            await aiosmtplib.send(
                message,
                hostname=smtp_config.smtp_host,
                port=smtp_config.smtp_port,
                username=smtp_config.smtp_username,
                password=smtp_config.smtp_password,
                use_tls=smtp_config.settings.get("use_tls", True),
                start_tls=smtp_config.settings.get("use_tls", True)
            )

        except Exception as e:
            logger.error("smtp_email_send_failed", error=str(e))
            raise

    async def _store_delivery_status(
        self,
        notification_id: str,
        request: EmailRequest,
        delivery_details: List[Dict[str, Any]],
        delivered_count: int,
        failed_count: int
    ) -> None:
        """Store delivery status in cache."""
        try:
            status_data = {
                "notification_id": notification_id,
                "type": "email",
                "sent_at": datetime.utcnow().isoformat(),
                "delivered_count": delivered_count,
                "failed_count": failed_count,
                "total_count": len(request.recipients),
                "delivery_details": delivery_details,
                "request_data": request.dict()
            }

            # Store in Redis with TTL (30 days)
            await self.redis_client.set(
                f"email_delivery:{notification_id}",
                json.dumps(status_data),
                ex=86400 * 30
            )

        except Exception as e:
            logger.error("delivery_status_storage_failed", error=str(e))

    async def create_template(self, template: NotificationTemplate) -> str:
        """Create email template."""
        try:
            template_id = template.id or str(uuid.uuid4())

            # Store template in cache
            template_data = {
                "id": template_id,
                "name": template.name,
                "subject": template.subject,
                "html_body": template.html_body,
                "text_body": template.body,
                "variables": template.variables or [],
                "metadata": template.metadata or {},
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat(),
                "version": template.version,
                "is_active": template.is_active
            }

            await self.redis_client.set(
                f"email_template:{template_id}",
                json.dumps(template_data),
                ex=86400 * 365  # 1 year
            )

            # Update local cache
            self.template_cache[template_id] = template_data

            logger.info("email_template_created", template_id=template_id)

            return template_id

        except Exception as e:
            logger.error("email_template_creation_failed", error=str(e))
            raise

    async def _get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get email template by ID."""
        try:
            # Check local cache first
            if template_id in self.template_cache:
                return self.template_cache[template_id]

            # Get from Redis
            template_data = await self.redis_client.get(f"email_template:{template_id}")
            if template_data:
                template = json.loads(template_data)
                self.template_cache[template_id] = template
                return template

            return None

        except Exception as e:
            logger.error("email_template_retrieval_failed", error=str(e))
            return None

    async def get_template_list(self) -> List[Dict[str, Any]]:
        """Get list of all email templates."""
        try:
            template_keys = await self.redis_client.keys("email_template:*")
            templates = []

            for template_key in template_keys:
                template_data = await self.redis_client.get(template_key)
                if template_data:
                    template = json.loads(template_data)
                    templates.append({
                        "id": template["id"],
                        "name": template["name"],
                        "created_at": template["created_at"],
                        "updated_at": template["updated_at"],
                        "version": template["version"],
                        "is_active": template["is_active"]
                    })

            return templates

        except Exception as e:
            logger.error("email_template_list_retrieval_failed", error=str(e))
            return []

    async def update_template(self, template_id: str, template: NotificationTemplate) -> None:
        """Update email template."""
        try:
            # Get existing template
            existing_template = await self._get_template(template_id)
            if not existing_template:
                raise Exception(f"Template {template_id} not found")

            # Update template data
            template_data = {
                "id": template_id,
                "name": template.name,
                "subject": template.subject,
                "html_body": template.html_body,
                "text_body": template.body,
                "variables": template.variables or [],
                "metadata": template.metadata or {},
                "created_at": existing_template["created_at"],
                "updated_at": datetime.utcnow().isoformat(),
                "version": existing_template["version"] + 1,
                "is_active": template.is_active
            }

            await self.redis_client.set(
                f"email_template:{template_id}",
                json.dumps(template_data),
                ex=86400 * 365  # 1 year
            )

            # Update local cache
            self.template_cache[template_id] = template_data

            logger.info("email_template_updated", template_id=template_id)

        except Exception as e:
            logger.error("email_template_update_failed", error=str(e))
            raise

    async def delete_template(self, template_id: str) -> None:
        """Delete email template."""
        try:
            # Remove from Redis
            await self.redis_client.delete(f"email_template:{template_id}")

            # Remove from local cache
            if template_id in self.template_cache:
                del self.template_cache[template_id]

            logger.info("email_template_deleted", template_id=template_id)

        except Exception as e:
            logger.error("email_template_deletion_failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check email service health."""
        try:
            # Check Redis connection
            await self.redis_client.ping()

            # Check SMTP configuration
            if not self.smtp_configs:
                return False

            return True

        except Exception as e:
            logger.error("email_service_health_check_failed", error=str(e))
            return False

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()

            logger.info("email_service_cleanup_completed")

        except Exception as e:
            logger.error("email_service_cleanup_failed", error=str(e))
