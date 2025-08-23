# Cloud ITSM Solution Guide

## Overview

PolicyCortex now includes a comprehensive Cloud IT Service Management (ITSM) solution that provides complete visibility and control over all cloud resources across Azure, AWS, and GCP.

## 🎯 Key Features

### 1. **Multi-Cloud Resource Management**
- **Real-time Discovery**: Automatically discover and track resources across all connected clouds
- **Unified View**: Single pane of glass for Azure, AWS, and GCP resources
- **Resource States**: Track 8 different resource states:
  - ✅ **Running**: Actively consuming resources
  - ⏸️ **Stopped**: Provisioned but not running
  - 💤 **Idle**: Running but inactive (< 5% CPU for > 7 days)
  - 👻 **Orphaned**: No owner or associated application
  - ⚠️ **Degraded**: Running with performance/health issues
  - 📅 **Scheduled**: Automated start/stop schedule configured
  - 🔧 **Maintenance**: Currently under maintenance
  - 🗑️ **Decommissioned**: Marked for deletion

### 2. **Application Lifecycle Management**
- Track application health and performance
- Monitor resource consumption and costs
- Detect orphaned and idle resources automatically
- Map application dependencies
- SLA compliance tracking

### 3. **Service Management**
- Service catalog with health dashboard
- SLA tracking and reporting
- Service dependency mapping
- Availability and performance metrics
- Incident correlation

### 4. **ITSM Processes**
- **Incident Management**: Track and resolve incidents
- **Change Management**: Manage change requests with approval workflows
- **Problem Management**: Root cause analysis and permanent fixes
- **Asset Management**: Complete IT asset lifecycle tracking
- **CMDB**: Configuration management with relationship mapping

## 📍 Navigation

Access Cloud ITSM from the sidebar menu:

```
Cloud ITSM
├── Resource Inventory      (/itsm/inventory)
├── Applications            (/itsm/applications)
├── Service Catalog         (/itsm/services)
├── Incident Management     (/itsm/incidents)
├── Change Management       (/itsm/changes)
├── Problem Management      (/itsm/problems)
├── Asset Management        (/itsm/assets)
└── CMDB                    (/itsm/cmdb)
```

## 🚀 Quick Start

### 1. **View All Resources**
Navigate to **Cloud ITSM > Resource Inventory** to see all resources across your clouds:
- Filter by cloud provider (Azure, AWS, GCP)
- Filter by resource state (Running, Stopped, Idle, etc.)
- Search by name, type, or tags
- Export to CSV for reporting

### 2. **Manage Applications**
Go to **Cloud ITSM > Applications** to:
- View application health scores
- Monitor resource consumption
- Identify orphaned resources
- Track costs per application
- View performance metrics

### 3. **Track Services**
Visit **Cloud ITSM > Service Catalog** to:
- Monitor service health
- Track SLA compliance
- View service dependencies
- Analyze availability metrics

## 💡 Key Capabilities

### Resource Inventory Features
- **Bulk Operations**: Select multiple resources for start/stop/delete
- **Quick Actions**: One-click actions for common tasks
- **Export**: Download resource data as CSV or JSON
- **Advanced Filtering**: Filter by multiple criteria simultaneously
- **Cost Tracking**: See cost per resource with optimization recommendations

### Application Management
- **Health Scoring**: Automatic health score calculation based on:
  - Resource utilization
  - Error rates
  - Response times
  - SLA compliance
- **Orphan Detection**: Automatically identify resources with:
  - No tags
  - No recent activity
  - No dependencies
  - No assigned owner

### Service Monitoring
- **Real-time Health**: Live service health monitoring
- **SLA Tracking**: Monitor and report on SLA compliance
- **Dependency Mapping**: Visualize service dependencies
- **Performance Metrics**: Track key performance indicators

## 🔍 Advanced Queries

### Find Idle Resources
1. Go to **Resource Inventory**
2. Filter by Status: "Idle"
3. View resources with < 5% CPU usage for > 7 days
4. Take action to optimize costs

### Identify Orphaned Resources
1. Navigate to **Applications**
2. Click "Orphaned Resources" tab
3. Review resources without owners
4. Assign owners or schedule for deletion

### Track Changes
1. Open **Change Management**
2. View scheduled, in-progress, and completed changes
3. Assess risk and impact
4. Track approval status

## 📊 Dashboards

### ITSM Dashboard
The main ITSM dashboard provides:
- **Infrastructure Health Score**: Overall health of your cloud infrastructure
- **Resource Distribution**: Resources by cloud provider and type
- **Cost Analysis**: Current spend and optimization opportunities
- **Active Issues**: Incidents, changes, and problems
- **Service Health**: Status of critical services

### Resource States Widget
Visual representation of resource states:
- Green: Healthy running resources
- Yellow: Resources needing attention (idle, scheduled)
- Red: Critical issues (degraded, orphaned)

## 🛠️ Automation

### Automated Actions
- **Auto-tagging**: Automatically tag resources based on patterns
- **Scheduled Start/Stop**: Configure resources to start/stop on schedule
- **Orphan Cleanup**: Automatically flag or delete orphaned resources
- **Cost Optimization**: Receive recommendations for rightsizing

### Bulk Operations
1. Select multiple resources using checkboxes
2. Choose bulk action (Start, Stop, Tag, Delete)
3. Confirm action
4. Monitor progress in real-time

## 📈 Reporting

### Available Reports
- **Resource Utilization Report**: CPU, memory, storage usage
- **Cost Analysis Report**: Spending by service, department, project
- **Compliance Report**: Resources meeting/violating policies
- **SLA Compliance Report**: Service level achievement
- **Incident Analysis Report**: Incident trends and patterns

### Export Options
- **CSV**: For spreadsheet analysis
- **JSON**: For programmatic processing
- **PDF**: For management reporting (coming soon)

## 🔐 Security & Compliance

### Security Features
- **Access Control**: Role-based access to ITSM functions
- **Audit Trail**: Complete history of all actions
- **Compliance Checking**: Verify resources against policies
- **Security Scanning**: Identify security vulnerabilities

### Compliance Tracking
- Monitor resource compliance with organizational policies
- Track security standards adherence
- Generate compliance reports
- Receive alerts for violations

## 🔧 Configuration

### Setting Resource States
Resources are automatically classified based on:
- **Running**: provisioningState = "Succeeded" and powerState = "Running"
- **Stopped**: provisioningState = "Succeeded" and powerState = "Deallocated"
- **Idle**: CPU < 5% for > 7 days
- **Orphaned**: No tags["owner"] and no recent activity
- **Degraded**: Health status != "Healthy" or active alerts
- **Scheduled**: Has tags["auto-shutdown-schedule"]
- **Maintenance**: Has tags["maintenance-window"]
- **Decommissioned**: Has tags["decommission-date"]

### Customizing Thresholds
Edit thresholds in Settings:
- Idle threshold (default: 5% CPU for 7 days)
- Orphan detection rules
- Cost optimization thresholds
- Alert sensitivity

## 📝 Best Practices

1. **Regular Reviews**: Review orphaned and idle resources weekly
2. **Tagging Strategy**: Implement consistent tagging for better tracking
3. **Cost Optimization**: Act on optimization recommendations monthly
4. **SLA Monitoring**: Set up alerts for SLA breaches
5. **Change Planning**: Use change management for all production changes
6. **Asset Tracking**: Keep asset information up-to-date
7. **CMDB Accuracy**: Regularly validate CMDB relationships

## 🚨 Troubleshooting

### No Resources Showing
- Verify cloud credentials are configured
- Check network connectivity to cloud APIs
- Ensure proper permissions for resource discovery

### Incorrect Resource States
- Refresh resource cache
- Verify tagging is properly configured
- Check resource activity logs

### Performance Issues
- Reduce number of resources displayed per page
- Use filters to narrow results
- Clear browser cache

## 📖 API Reference

### Key Endpoints
- `GET /api/v1/itsm/inventory` - List all resources
- `GET /api/v1/itsm/applications` - List applications
- `GET /api/v1/itsm/services` - Service catalog
- `POST /api/v1/itsm/inventory/bulk` - Bulk operations
- `GET /api/v1/itsm/inventory/export` - Export data

## 🎉 Getting Started

1. Navigate to **Cloud ITSM** in the sidebar
2. Start with **Resource Inventory** to see all your resources
3. Check **Applications** for orphaned resources
4. Review **Service Catalog** for service health
5. Set up **Incidents** for issue tracking
6. Configure **CMDB** for relationship mapping

The Cloud ITSM solution provides everything you need to manage your cloud infrastructure efficiently, reduce costs, and maintain high service levels.