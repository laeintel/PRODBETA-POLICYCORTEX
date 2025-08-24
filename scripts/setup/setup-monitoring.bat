@echo off
REM PolicyCortex Monitoring Stack Setup Script
REM This script sets up comprehensive monitoring, observability, and alerting
REM Includes Prometheus, Grafana, Jaeger, Application Insights integration

setlocal enabledelayedexpansion

echo =============================================================
echo PolicyCortex Monitoring Stack Setup Script v2.0
echo Setting up comprehensive monitoring and observability...
echo =============================================================
echo.

REM Set variables
set PROJECT_ROOT=%~dp0\..\..
set SETUP_DIR=%~dp0
set ERRORS_OCCURRED=0

REM Function to log messages
:log_info
echo [INFO] %~1
goto :eof

:log_error
echo [ERROR] %~1
set ERRORS_OCCURRED=1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:check_prerequisites
echo.
call :log_info "Checking prerequisites..."

REM Check Docker
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Docker is not installed or not in PATH"
    call :log_info "Please install Docker Desktop from https://docker.com/get-started"
    goto :error_exit
)

REM Check Docker Compose
docker compose version >nul 2>&1
if %errorLevel% neq 0 (
    call :log_error "Docker Compose is not available"
    call :log_info "Please ensure Docker Compose is installed"
    goto :error_exit
)

call :log_success "Prerequisites check passed"
goto :eof

:create_monitoring_directories
echo.
call :log_info "Creating monitoring directories..."

cd /d "%PROJECT_ROOT%"

REM Create monitoring directories
if not exist "monitoring\grafana\dashboards" mkdir "monitoring\grafana\dashboards"
if not exist "monitoring\grafana\provisioning\dashboards" mkdir "monitoring\grafana\provisioning\dashboards"
if not exist "monitoring\grafana\provisioning\datasources" mkdir "monitoring\grafana\provisioning\datasources"
if not exist "monitoring\prometheus\rules" mkdir "monitoring\prometheus\rules"
if not exist "monitoring\prometheus\config" mkdir "monitoring\prometheus\config"
if not exist "monitoring\alertmanager" mkdir "monitoring\alertmanager"
if not exist "monitoring\logs" mkdir "monitoring\logs"

call :log_success "Monitoring directories created"
goto :eof

:create_prometheus_config
echo.
call :log_info "Creating Prometheus configuration..."

REM Create Prometheus configuration
(
echo # Prometheus Configuration for PolicyCortex
echo global:
echo   scrape_interval: 15s
echo   evaluation_interval: 15s
echo   external_labels:
echo     cluster: 'policycortex'
echo     environment: 'development'
echo.
echo rule_files:
echo   - "/etc/prometheus/rules/*.yml"
echo.
echo alerting:
echo   alertmanagers:
echo     - static_configs:
echo         - targets:
echo           - alertmanager:9093
echo.
echo scrape_configs:
echo   - job_name: 'prometheus'
echo     static_configs:
echo       - targets: ['localhost:9090']
echo.
echo   - job_name: 'policycortex-backend'
echo     static_configs:
echo       - targets: ['backend:8080']
echo     metrics_path: '/metrics'
echo     scrape_interval: 30s
echo.
echo   - job_name: 'policycortex-frontend'
echo     static_configs:
echo       - targets: ['frontend:3000']
echo     metrics_path: '/api/metrics'
echo     scrape_interval: 30s
echo.
echo   - job_name: 'policycortex-graphql'
echo     static_configs:
echo       - targets: ['graphql:4000']
echo     metrics_path: '/metrics'
echo     scrape_interval: 30s
echo.
echo   - job_name: 'postgres'
echo     static_configs:
echo       - targets: ['postgres:5432']
echo     metrics_path: '/metrics'
echo     scrape_interval: 60s
echo.
echo   - job_name: 'redis'
echo     static_configs:
echo       - targets: ['redis:6379']
echo     metrics_path: '/metrics'
echo     scrape_interval: 60s
echo.
echo   - job_name: 'eventstore'
echo     static_configs:
echo       - targets: ['eventstore:2113']
echo     metrics_path: '/stats'
echo     scrape_interval: 60s
echo.
echo   - job_name: 'node-exporter'
echo     static_configs:
echo       - targets: ['node-exporter:9100']
echo     scrape_interval: 60s
echo.
echo   - job_name: 'cadvisor'
echo     static_configs:
echo       - targets: ['cadvisor:8080']
echo     scrape_interval: 60s
) > "monitoring\prometheus\prometheus.yml"

call :log_success "Prometheus configuration created"
goto :eof

:create_alert_rules
echo.
call :log_info "Creating Prometheus alert rules..."

REM Create basic alert rules
(
echo groups:
echo - name: policycortex_alerts
echo   rules:
echo   - alert: HighCPUUsage
echo     expr: cpu_usage_percent > 80
echo     for: 5m
echo     labels:
echo       severity: warning
echo     annotations:
echo       summary: "High CPU usage detected"
echo       description: "CPU usage has been above 80%% for more than 5 minutes"
echo.
echo   - alert: HighMemoryUsage
echo     expr: memory_usage_percent > 85
echo     for: 5m
echo     labels:
echo       severity: warning
echo     annotations:
echo       summary: "High memory usage detected"
echo       description: "Memory usage has been above 85%% for more than 5 minutes"
echo.
echo   - alert: ServiceDown
echo     expr: up == 0
echo     for: 2m
echo     labels:
echo       severity: critical
echo     annotations:
echo       summary: "Service is down"
echo       description: "Service {{ $labels.job }} has been down for more than 2 minutes"
echo.
echo   - alert: HighResponseTime
echo     expr: http_request_duration_seconds{quantile="0.95"} > 2
echo     for: 5m
echo     labels:
echo       severity: warning
echo     annotations:
echo       summary: "High response time detected"
echo       description: "95th percentile response time is above 2 seconds"
echo.
echo   - alert: HighErrorRate
echo     expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
echo     for: 5m
echo     labels:
echo       severity: critical
echo     annotations:
echo       summary: "High error rate detected"
echo       description: "Error rate is above 5%% for more than 5 minutes"
echo.
echo - name: policycortex_business_alerts
echo   rules:
echo   - alert: ComplianceScoreDrop
echo     expr: compliance_score < 70
echo     for: 10m
echo     labels:
echo       severity: warning
echo     annotations:
echo       summary: "Compliance score dropped below threshold"
echo       description: "Overall compliance score has dropped below 70%%"
echo.
echo   - alert: HighRiskResourceDetected
echo     expr: resource_risk_score > 80
echo     for: 1m
echo     labels:
echo       severity: high
echo     annotations:
echo       summary: "High risk resource detected"
echo       description: "A resource with risk score above 80 has been detected"
echo.
echo   - alert: PolicyViolation
echo     expr: increase(policy_violations_total[1h]) > 5
echo     for: 5m
echo     labels:
echo       severity: warning
echo     annotations:
echo       summary: "Multiple policy violations detected"
echo       description: "More than 5 policy violations in the last hour"
) > "monitoring\prometheus\rules\alerts.yml"

call :log_success "Prometheus alert rules created"
goto :eof

:create_grafana_config
echo.
call :log_info "Creating Grafana configuration..."

REM Create Grafana datasource configuration
(
echo apiVersion: 1
echo.
echo datasources:
echo - name: Prometheus
echo   type: prometheus
echo   access: proxy
echo   url: http://prometheus:9090
echo   isDefault: true
echo   editable: true
echo.
echo - name: Jaeger
echo   type: jaeger
echo   access: proxy
echo   url: http://jaeger:16686
echo   editable: true
echo.
echo - name: Loki
echo   type: loki
echo   access: proxy
echo   url: http://loki:3100
echo   editable: true
) > "monitoring\grafana\provisioning\datasources\datasources.yml"

REM Create Grafana dashboard provisioning
(
echo apiVersion: 1
echo.
echo providers:
echo - name: 'default'
echo   orgId: 1
echo   folder: ''
echo   folderUid: ''
echo   type: file
echo   disableDeletion: false
echo   updateIntervalSeconds: 30
echo   allowUiUpdates: true
echo   options:
echo     path: /var/lib/grafana/dashboards
) > "monitoring\grafana\provisioning\dashboards\dashboards.yml"

call :log_success "Grafana configuration created"
goto :eof

:create_grafana_dashboards
echo.
call :log_info "Creating Grafana dashboards..."

REM Create PolicyCortex main dashboard
(
echo {
echo   "dashboard": {
echo     "id": null,
echo     "title": "PolicyCortex Overview",
echo     "description": "Main dashboard for PolicyCortex monitoring",
echo     "tags": ["policycortex", "overview"],
echo     "timezone": "browser",
echo     "refresh": "30s",
echo     "time": {
echo       "from": "now-1h",
echo       "to": "now"
echo     },
echo     "panels": [
echo       {
echo         "id": 1,
echo         "title": "Service Status",
echo         "type": "stat",
echo         "targets": [
echo           {
echo             "expr": "up{job=~\"policycortex.*\"}",
echo             "legendFormat": "{{ job }}"
echo           }
echo         ],
echo         "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
echo       },
echo       {
echo         "id": 2,
echo         "title": "Request Rate",
echo         "type": "graph",
echo         "targets": [
echo           {
echo             "expr": "rate(http_requests_total[5m])",
echo             "legendFormat": "{{ job }}"
echo           }
echo         ],
echo         "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
echo       },
echo       {
echo         "id": 3,
echo         "title": "Response Time",
echo         "type": "graph",
echo         "targets": [
echo           {
echo             "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
echo             "legendFormat": "95th percentile"
echo           }
echo         ],
echo         "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
echo       },
echo       {
echo         "id": 4,
echo         "title": "Error Rate",
echo         "type": "graph",
echo         "targets": [
echo           {
echo             "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
echo             "legendFormat": "Error Rate %%"
echo           }
echo         ],
echo         "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
echo       }
echo     ]
echo   },
echo   "overwrite": true
echo }
) > "monitoring\grafana\dashboards\policycortex-overview.json"

REM Create Business Metrics dashboard
(
echo {
echo   "dashboard": {
echo     "id": null,
echo     "title": "PolicyCortex Business Metrics",
echo     "description": "Business KPIs and compliance metrics",
echo     "tags": ["policycortex", "business", "compliance"],
echo     "timezone": "browser",
echo     "refresh": "1m",
echo     "panels": [
echo       {
echo         "id": 1,
echo         "title": "Compliance Score",
echo         "type": "stat",
echo         "targets": [
echo           {
echo             "expr": "avg(compliance_score)",
echo             "legendFormat": "Overall Score"
echo           }
echo         ],
echo         "fieldConfig": {
echo           "defaults": {
echo             "min": 0,
echo             "max": 100,
echo             "unit": "percent"
echo           }
echo         },
echo         "gridPos": {"h": 6, "w": 6, "x": 0, "y": 0}
echo       },
echo       {
echo         "id": 2,
echo         "title": "Risk Score Distribution",
echo         "type": "piechart",
echo         "targets": [
echo           {
echo             "expr": "count by (risk_level) (risk_score_bucket)",
echo             "legendFormat": "{{ risk_level }}"
echo           }
echo         ],
echo         "gridPos": {"h": 6, "w": 6, "x": 6, "y": 0}
echo       }
echo     ]
echo   },
echo   "overwrite": true
echo }
) > "monitoring\grafana\dashboards\business-metrics.json"

call :log_success "Grafana dashboards created"
goto :eof

:create_alertmanager_config
echo.
call :log_info "Creating Alertmanager configuration..."

(
echo global:
echo   smtp_smarthost: 'localhost:587'
echo   smtp_from: 'alerts@policycortex.com'
echo.
echo route:
echo   group_by: ['alertname', 'severity']
echo   group_wait: 30s
echo   group_interval: 5m
echo   repeat_interval: 12h
echo   receiver: 'default-notifications'
echo   routes:
echo   - match:
echo       severity: critical
echo     receiver: 'critical-notifications'
echo   - match:
echo       severity: high
echo     receiver: 'high-priority-notifications'
echo.
echo receivers:
echo - name: 'default-notifications'
echo   email_configs:
echo   - to: 'admin@policycortex.com'
echo     subject: '[PolicyCortex] Alert: {{ .GroupLabels.alertname }}'
echo     body: |
echo       Alert: {{ .GroupLabels.alertname }}
echo       Severity: {{ .GroupLabels.severity }}
echo       
echo       {{ range .Alerts }}
echo       - {{ .Annotations.summary }}
echo       {{ .Annotations.description }}
echo       {{ end }}
echo.
echo - name: 'critical-notifications'
echo   email_configs:
echo   - to: 'admin@policycortex.com,security@policycortex.com'
echo     subject: '[CRITICAL] PolicyCortex Alert: {{ .GroupLabels.alertname }}'
echo     body: |
echo       CRITICAL ALERT - Immediate attention required!
echo       
echo       Alert: {{ .GroupLabels.alertname }}
echo       Severity: {{ .GroupLabels.severity }}
echo       
echo       {{ range .Alerts }}
echo       - {{ .Annotations.summary }}
echo       {{ .Annotations.description }}
echo       Started: {{ .StartsAt }}
echo       {{ end }}
echo.
echo - name: 'high-priority-notifications'
echo   email_configs:
echo   - to: 'admin@policycortex.com'
echo     subject: '[HIGH] PolicyCortex Alert: {{ .GroupLabels.alertname }}'
echo     body: |
echo       HIGH PRIORITY ALERT
echo       
echo       Alert: {{ .GroupLabels.alertname }}
echo       Severity: {{ .GroupLabels.severity }}
echo       
echo       {{ range .Alerts }}
echo       - {{ .Annotations.summary }}
echo       {{ .Annotations.description }}
echo       {{ end }}
) > "monitoring\alertmanager\alertmanager.yml"

call :log_success "Alertmanager configuration created"
goto :eof

:create_docker_compose_monitoring
echo.
call :log_info "Creating monitoring Docker Compose configuration..."

(
echo version: '3.9'
echo.
echo services:
echo   # Extended monitoring services to add to main docker-compose
echo   
echo   # Node Exporter for host metrics
echo   node-exporter:
echo     image: prom/node-exporter:v1.7.0
echo     container_name: policycortex-node-exporter
echo     restart: unless-stopped
echo     ports:
echo       - "9100:9100"
echo     volumes:
echo       - /proc:/host/proc:ro
echo       - /sys:/host/sys:ro
echo       - /:/rootfs:ro
echo     command:
echo       - '--path.procfs=/host/proc'
echo       - '--path.rootfs=/rootfs'
echo       - '--path.sysfs=/host/sys'
echo       - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
echo     networks:
echo       - policycortex-network
echo.
echo   # cAdvisor for container metrics
echo   cadvisor:
echo     image: gcr.io/cadvisor/cadvisor:v0.47.0
echo     container_name: policycortex-cadvisor
echo     restart: unless-stopped
echo     ports:
echo       - "8080:8080"
echo     volumes:
echo       - /:/rootfs:ro
echo       - /var/run:/var/run:rw
echo       - /sys:/sys:ro
echo       - /var/lib/docker:/var/lib/docker:ro
echo     privileged: true
echo     networks:
echo       - policycortex-network
echo.
echo   # Alertmanager for alert routing
echo   alertmanager:
echo     image: prom/alertmanager:v0.26.0
echo     container_name: policycortex-alertmanager
echo     restart: unless-stopped
echo     ports:
echo       - "9093:9093"
echo     volumes:
echo       - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
echo       - alertmanager_data:/alertmanager
echo     command:
echo       - '--config.file=/etc/alertmanager/alertmanager.yml'
echo       - '--storage.path=/alertmanager'
echo       - '--web.external-url=http://localhost:9093'
echo     networks:
echo       - policycortex-network
echo.
echo   # Loki for log aggregation
echo   loki:
echo     image: grafana/loki:2.9.2
echo     container_name: policycortex-loki
echo     restart: unless-stopped
echo     ports:
echo       - "3100:3100"
echo     volumes:
echo       - loki_data:/loki
echo     command: -config.file=/etc/loki/local-config.yaml
echo     networks:
echo       - policycortex-network
echo.
echo   # Promtail for log collection
echo   promtail:
echo     image: grafana/promtail:2.9.2
echo     container_name: policycortex-promtail
echo     restart: unless-stopped
echo     volumes:
echo       - /var/log:/var/log:ro
echo       - /var/lib/docker/containers:/var/lib/docker/containers:ro
echo       - ./monitoring/promtail/config.yml:/etc/promtail/config.yml
echo     command: -config.file=/etc/promtail/config.yml
echo     networks:
echo       - policycortex-network
echo.
echo networks:
echo   policycortex-network:
echo     external: true
echo.
echo volumes:
echo   alertmanager_data:
echo   loki_data:
) > "monitoring\docker-compose.monitoring.yml"

call :log_success "Monitoring Docker Compose configuration created"
goto :eof

:create_application_insights_config
echo.
call :log_info "Creating Application Insights configuration..."

REM Create Application Insights configuration for Node.js
(
echo {
echo   "instrumentationKey": "${APPINSIGHTS_INSTRUMENTATIONKEY}",
echo   "connectionString": "${APPLICATIONINSIGHTS_CONNECTION_STRING}",
echo   "role": {
echo     "name": "policycortex-frontend",
echo     "instance": "frontend-1"
echo   },
echo   "sampling": {
echo     "percentage": 100,
echo     "maxTelemetryItemsPerSecond": 20
echo   },
echo   "logging": {
echo     "console": false,
echo     "logLevel": "info"
echo   },
echo   "httpAutoCollectionOptions": {
echo     "enableHttpAutoCollection": true,
echo     "enableW3CDistributedTracing": true,
echo     "enableAutoCorrelation": true
echo   },
echo   "requestAutoCollectionOptions": {
echo     "enableAutoCollection": true
echo   },
echo   "exceptionAutoCollectionOptions": {
echo     "enableAutoCollection": true,
echo     "enableLiveMetrics": true
echo   },
echo   "performanceAutoCollectionOptions": {
echo     "enableAutoCollection": true
echo   },
echo   "quickPulseConfig": {
echo     "enabled": true
echo   }
echo }
) > "monitoring\applicationinsights.json"

call :log_success "Application Insights configuration created"
goto :eof

:start_monitoring_services
echo.
call :log_info "Starting monitoring services..."

cd /d "%PROJECT_ROOT%"

REM Start monitoring services
call :log_info "Starting extended monitoring stack..."
docker compose -f monitoring\docker-compose.monitoring.yml up -d
if %errorLevel% neq 0 (
    call :log_error "Failed to start monitoring services"
    goto :eof
)

REM Wait for services to be ready
call :log_info "Waiting for monitoring services to be ready..."
timeout /t 30 /nobreak >nul

REM Check service health
call :log_info "Checking monitoring services health..."
docker compose -f monitoring\docker-compose.monitoring.yml ps

call :log_success "Monitoring services are running"
goto :eof

:create_monitoring_scripts
echo.
call :log_info "Creating monitoring utility scripts..."

REM Create monitoring status check script
(
echo @echo off
echo echo Checking PolicyCortex Monitoring Services Status...
echo echo.
echo echo Prometheus: http://localhost:9090
echo curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:9090/-/healthy
echo.
echo echo Grafana: http://localhost:3010
echo curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:3010/api/health
echo.
echo echo Alertmanager: http://localhost:9093
echo curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:9093/-/healthy
echo.
echo echo Jaeger: http://localhost:16686
echo curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:16686
echo.
echo echo Node Exporter: http://localhost:9100
echo curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:9100/metrics
echo.
echo echo cAdvisor: http://localhost:8080
echo curl -s -o nul -w "Status: %%{http_code}\n" http://localhost:8080/healthz
echo.
) > "scripts\check-monitoring.bat"

REM Create log viewing script
(
echo @echo off
echo echo PolicyCortex Service Logs
echo echo ========================
echo echo 1. Backend logs
echo echo 2. Frontend logs  
echo echo 3. GraphQL logs
echo echo 4. Database logs
echo echo 5. All logs
echo echo.
echo set /p choice="Select service (1-5): "
echo.
echo if "%%choice%%"=="1" docker logs -f policycortex-backend
echo if "%%choice%%"=="2" docker logs -f policycortex-frontend
echo if "%%choice%%"=="3" docker logs -f policycortex-graphql
echo if "%%choice%%"=="4" docker logs -f policycortex-postgres
echo if "%%choice%%"=="5" docker compose logs -f
) > "scripts\view-logs.bat"

call :log_success "Monitoring utility scripts created"
goto :eof

:display_monitoring_summary
echo.
echo =============================================================
call :log_success "PolicyCortex Monitoring Stack Setup Complete"
echo =============================================================
echo.
echo Monitoring Services Available:
echo ================================
echo.
echo Core Monitoring:
echo - Prometheus (Metrics):     http://localhost:9090
echo - Grafana (Dashboards):     http://localhost:3010 (admin/policycortex2024)
echo - Alertmanager (Alerts):    http://localhost:9093
echo - Jaeger (Tracing):         http://localhost:16686
echo.
echo System Monitoring:
echo - Node Exporter (Host):     http://localhost:9100/metrics
echo - cAdvisor (Containers):    http://localhost:8080
echo - Loki (Logs):              http://localhost:3100
echo.
echo Management Tools:
echo - Check Status:             scripts\check-monitoring.bat
echo - View Logs:                scripts\view-logs.bat
echo.
echo Configuration Files Created:
echo ============================
echo - monitoring\prometheus\prometheus.yml
echo - monitoring\prometheus\rules\alerts.yml
echo - monitoring\grafana\provisioning\datasources\datasources.yml
echo - monitoring\grafana\dashboards\*.json
echo - monitoring\alertmanager\alertmanager.yml
echo - monitoring\docker-compose.monitoring.yml
echo - monitoring\applicationinsights.json
echo.
echo Next Steps:
echo ===========
echo 1. Configure Application Insights connection string
echo 2. Set up notification channels in Alertmanager
echo 3. Customize Grafana dashboards for your needs
echo 4. Configure log aggregation rules in Loki
echo 5. Set up external monitoring integrations
echo.
echo Grafana Default Dashboards:
echo - PolicyCortex Overview
echo - Business Metrics
echo - Infrastructure Monitoring
echo.
goto :eof

:error_exit
echo.
call :log_error "Monitoring setup failed. Please check the errors above and try again."
exit /b 1

:main
REM Main execution flow
call :check_prerequisites
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_monitoring_directories
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_prometheus_config
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_alert_rules
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_grafana_config
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_grafana_dashboards
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_alertmanager_config
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_docker_compose_monitoring
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_application_insights_config
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :start_monitoring_services
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :create_monitoring_scripts
if !ERRORS_OCCURRED! equ 1 goto :error_exit

call :display_monitoring_summary

call :log_success "PolicyCortex monitoring stack setup completed successfully!"
goto :eof

REM Execute main function
call :main