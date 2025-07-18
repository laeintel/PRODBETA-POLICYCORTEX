# PolicyCortex Notification Service

A comprehensive notification service for the PolicyCortex platform that handles email, SMS, push notifications, webhooks, alerts, and subscription management.

## Features

### Core Notification Types
- **Email Notifications**: SMTP, SendGrid, Mailgun, Azure Communication Services
- **SMS Notifications**: Twilio, AWS SNS, MessageBird, Nexmo/Vonage
- **Push Notifications**: Firebase Cloud Messaging, Apple Push Notification Service, OneSignal
- **Webhook Notifications**: HTTP/HTTPS webhooks with retry logic and signature validation
- **Alert Management**: Escalation rules, auto-resolution, and monitoring integration

### Advanced Features
- **Template Engine**: Jinja2-based email and SMS templates
- **Subscription Management**: User preferences and notification subscriptions
- **Notification Scheduler**: Scheduled and recurring notifications with cron support
- **Analytics & Tracking**: Delivery tracking, performance metrics, and reporting
- **Rate Limiting**: Per-user and per-service rate limits
- **Circuit Breaker**: Automatic failover and provider health monitoring
- **Authentication**: JWT-based authentication with Azure AD integration
- **Azure Integration**: Native Azure Communication Services support

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│ Notification    │───▶│   Providers     │
│                 │    │    Service      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │   Analytics     │
                       │   (Cache/Queue) │    │   & Tracking    │
                       └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.11+
- Redis
- Azure subscription (optional, for Azure Communication Services)

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the service:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8005 --reload
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t policycortex/notification-service .
   ```

2. Run the container:
   ```bash
   docker run -p 8005:8005 --env-file .env policycortex/notification-service
   ```

## API Documentation

### Health Endpoints

- `GET /health` - Basic health check
- `GET /ready` - Readiness check with dependency validation
- `GET /metrics` - Prometheus metrics

### Notification Endpoints

#### Send Notification
```http
POST /api/v1/notifications/send
Content-Type: application/json
Authorization: Bearer <token>

{
  "type": "email",
  "priority": "high",
  "recipients": [
    {
      "email": "user@example.com",
      "name": "John Doe"
    }
  ],
  "content": {
    "subject": "Important Alert",
    "body": "This is an important notification.",
    "html_body": "<p>This is an <strong>important</strong> notification.</p>"
  }
}
```

#### Send Email
```http
POST /api/v1/notifications/email/send
Content-Type: application/json
Authorization: Bearer <token>

{
  "recipients": [{"email": "user@example.com"}],
  "content": {
    "subject": "Hello",
    "body": "Hello World!",
    "template_id": "welcome-email"
  },
  "from_email": "noreply@policycortex.com",
  "track_opens": true,
  "track_clicks": true
}
```

#### Send SMS
```http
POST /api/v1/notifications/sms/send
Content-Type: application/json
Authorization: Bearer <token>

{
  "recipients": [{"phone": "+1234567890"}],
  "content": {
    "body": "Your verification code is: 123456"
  },
  "provider": "twilio"
}
```

#### Send Push Notification
```http
POST /api/v1/notifications/push/send
Content-Type: application/json
Authorization: Bearer <token>

{
  "recipients": [{"device_token": "device_token_here"}],
  "content": {
    "title": "New Message",
    "body": "You have a new message"
  },
  "badge": 1,
  "sound": "default"
}
```

#### Send Webhook
```http
POST /api/v1/notifications/webhook/send
Content-Type: application/json
Authorization: Bearer <token>

{
  "url": "https://example.com/webhook",
  "method": "POST",
  "content": {
    "body": "Webhook payload"
  },
  "headers": {
    "X-Custom-Header": "value"
  },
  "auth_token": "webhook_auth_token"
}
```

### Alert Management

#### Create Alert
```http
POST /api/v1/notifications/alerts/create
Content-Type: application/json
Authorization: Bearer <token>

{
  "title": "High CPU Usage",
  "description": "Server CPU usage exceeded 90%",
  "severity": "critical",
  "source": "monitoring",
  "initial_recipients": [
    {"email": "admin@example.com"}
  ],
  "escalation_rules": [
    {
      "level": 1,
      "delay_minutes": 15,
      "recipients": [{"email": "manager@example.com"}],
      "notification_types": ["email", "sms"]
    }
  ]
}
```

### Subscription Management

#### Create Subscription
```http
POST /api/v1/notifications/subscriptions
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_id": "user123",
  "channel": "email",
  "topic": "security_alerts",
  "preferences": {
    "email_enabled": true,
    "sms_enabled": false,
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "08:00"
  }
}
```

#### Update Preferences
```http
PUT /api/v1/notifications/preferences/{user_id}
Content-Type: application/json
Authorization: Bearer <token>

{
  "email_enabled": true,
  "sms_enabled": true,
  "push_enabled": true,
  "quiet_hours_start": "23:00",
  "quiet_hours_end": "07:00",
  "timezone": "UTC"
}
```

### Scheduled Notifications

#### Schedule Notification
```http
POST /api/v1/notifications/schedule
Content-Type: application/json
Authorization: Bearer <token>

{
  "type": "email",
  "recipients": [{"email": "user@example.com"}],
  "content": {
    "subject": "Daily Report",
    "body": "Your daily report is ready"
  },
  "scheduled_time": "2024-01-01T09:00:00Z",
  "recurrence": "0 9 * * *",
  "end_date": "2024-12-31T23:59:59Z"
}
```

### Analytics

#### Get Statistics
```http
GET /api/v1/notifications/analytics/stats?start_date=2024-01-01&end_date=2024-01-31
Authorization: Bearer <token>
```

#### Get Delivery Status
```http
GET /api/v1/notifications/{notification_id}/status
Authorization: Bearer <token>
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVICE_NAME` | Service name | `notification-service` |
| `SERVICE_PORT` | Service port | `8005` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `JWT_SECRET_KEY` | JWT secret key | Required |
| `AZURE_COMMUNICATION_CONNECTION_STRING` | Azure Communication Services connection string | Optional |
| `TWILIO_ACCOUNT_SID` | Twilio account SID | Optional |
| `FCM_SERVER_KEY` | Firebase Cloud Messaging server key | Optional |
| `SMTP_HOST` | SMTP server host | `localhost` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Provider Configuration

The service supports multiple providers for each notification type:

#### Email Providers
- SMTP (default)
- SendGrid
- Mailgun
- Azure Communication Services

#### SMS Providers
- Twilio
- AWS SNS
- MessageBird
- Nexmo/Vonage

#### Push Notification Providers
- Firebase Cloud Messaging
- Apple Push Notification Service
- OneSignal

## Templates

### Email Templates

Email templates use Jinja2 syntax:

```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>Hello {{ user.name }}!</h1>
    <p>{{ message }}</p>
    <p>Best regards,<br>{{ company.name }}</p>
</body>
</html>
```

### SMS Templates

SMS templates are plain text with variable substitution:

```
Hello {{ user.name }}! Your verification code is: {{ code }}
```

## Error Handling

The service implements comprehensive error handling:

- **Retry Logic**: Automatic retries with exponential backoff
- **Circuit Breaker**: Automatic failover when providers are unhealthy
- **Fallback Providers**: Automatic switching to backup providers
- **Error Tracking**: Detailed error logging and metrics

## Monitoring

### Metrics

The service exposes Prometheus metrics:

- `notification_requests_total` - Total notification requests
- `notification_delivery_seconds` - Notification delivery time
- `notifications_sent_total` - Total notifications sent by type and status
- `notification_provider_health` - Provider health status

### Health Checks

- **Liveness**: `/health` endpoint
- **Readiness**: `/ready` endpoint with dependency checks
- **Metrics**: `/metrics` endpoint for Prometheus

## Security

### Authentication
- JWT-based authentication
- Azure AD integration
- API key support

### Authorization
- Role-based access control
- Permission-based operations
- User-specific rate limiting

### Data Protection
- Webhook signature validation
- TLS encryption
- Sensitive data redaction in logs

## Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
pytest tests/load/
```

## Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: notification-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: notification-service
  template:
    metadata:
      labels:
        app: notification-service
    spec:
      containers:
      - name: notification-service
        image: policycortex/notification-service:latest
        ports:
        - containerPort: 8005
        env:
        - name: REDIS_URL
          value: "redis://redis:6379/0"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: notification-secrets
              key: jwt-secret-key
```

### Docker Compose

```yaml
version: '3.8'
services:
  notification-service:
    build: .
    ports:
      - "8005:8005"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact:
- Email: dev@policycortex.com
- Documentation: https://docs.policycortex.com
- Issues: https://github.com/policycortex/notification-service/issues