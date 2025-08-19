# PolicyCortex Demo Guide

## Demo Overview

This guide consolidates all demo scenarios and scripts for PolicyCortex v2, showcasing the four patent technologies and production-ready capabilities.

## Executive Demo Script (15 minutes)

### Opening (2 minutes)
**"PolicyCortex v2 represents the next generation of cloud governance - moving from reactive to predictive, siloed to unified, and manual to automated."**

#### Key Value Propositions
1. **Unified Governance** - Single pane of glass for security, compliance, and cost
2. **Predictive Intelligence** - Proactive drift detection and prevention
3. **Conversational Interface** - Natural language governance interaction
4. **Automated Remediation** - One-click policy violation fixes

### Patent Technology Demonstration (10 minutes)

#### 🎯 Patent 1: Cross-Domain Governance Correlation Engine (2.5 min)
**Demo Flow:**
1. Navigate to unified dashboard (`/dashboard`)
2. Show real-time correlation of security, cost, and compliance metrics
3. Highlight cross-domain risk insights
4. Demonstrate API endpoint: `GET /api/v1/metrics`

**Key Talking Points:**
- "Traditional tools operate in silos - PolicyCortex correlates across all governance domains"
- "See how security violations correlate with cost spikes and compliance drift"
- "ML-powered pattern detection identifies risks before they become incidents"

#### 🎯 Patent 2: Conversational Governance Intelligence (2.5 min)
**Demo Flow:**
1. Open conversational interface (`/chat`)
2. Ask natural language questions:
   - "Show me all non-compliant resources in production"
   - "What's my security posture in East US region?"
   - "Predict next month's Azure costs"
3. Demonstrate contextual follow-up questions

**Key Talking Points:**
- "Governance teams can now query policies in plain English"
- "No need to learn complex query languages or navigate multiple dashboards"
- "Context-aware conversations with governance data"

#### 🎯 Patent 3: Unified AI-Driven Platform (2.5 min)
**Demo Flow:**
1. Show integrated dashboard with all governance domains
2. Navigate through unified resource views (`/resources`)
3. Demonstrate single workflow for security, compliance, and cost actions
4. Show AI-driven recommendations panel

**Key Talking Points:**
- "One platform replaces multiple point solutions"
- "AI orchestrates governance actions across all domains"
- "Unified experience reduces tool sprawl and training overhead"

#### 🎯 Patent 4: Predictive Policy Compliance (2.5 min)
**Demo Flow:**
1. Navigate to predictions dashboard (`/monitoring`)
2. Show compliance drift forecasting
3. Demonstrate early warning alerts
4. Show predictive API: `GET /api/v1/predictions`

**Key Talking Points:**
- "Move from reactive to proactive governance"
- "Predict policy violations before they occur"
- "85%+ accuracy in compliance drift detection"

### Automated Remediation Demo (2 minutes)

#### One-Click Remediation Flow
1. Identify policy violation in dashboard
2. Click "Auto-Remediate" button
3. Show approval workflow (if enabled)
4. Demonstrate real-time progress tracking
5. Show rollback capabilities

**Key Talking Points:**
- "Reduce manual remediation time from hours to minutes"
- "Built-in approval workflows with break-glass access"
- "Complete audit trail and rollback capabilities"

### Closing (1 minute)
**"PolicyCortex v2 delivers the future of cloud governance today - unified, intelligent, and automated."**

## Technical Deep-Dive Demo (30 minutes)

### Architecture Overview (5 minutes)
- Show system architecture diagram
- Explain Rust performance advantages
- Demonstrate sub-millisecond API responses
- Highlight post-quantum security features

### Azure Integration Demo (10 minutes)
1. **Real-time Resource Discovery**
   - Show Azure Resource Graph integration
   - Demonstrate cross-subscription queries
   - Display resource relationship mapping

2. **Policy Synchronization**
   - Show Azure Policy integration
   - Demonstrate policy assignment tracking
   - Display compliance status monitoring

3. **Security Center Integration**
   - Show security recommendation ingestion
   - Demonstrate threat correlation
   - Display unified security posture

### Multi-Tenant Capabilities (5 minutes)
1. Switch between tenant contexts
2. Show isolated data views
3. Demonstrate tenant-specific policies
4. Show RBAC enforcement

### Performance Demonstration (5 minutes)
1. Show API response time metrics
2. Demonstrate concurrent user handling
3. Show caching effectiveness
4. Display system scaling capabilities

### Advanced Analytics (5 minutes)
1. Show correlation analysis in action
2. Demonstrate ML model predictions
3. Display trend analysis and forecasting
4. Show custom analytics creation

## Demo Scenarios by Use Case

### Scenario 1: Security Operations Center (SOC)
**Persona**: SOC Analyst
**Goal**: Investigate security incidents with governance context

**Demo Flow:**
1. Security alert appears in dashboard
2. Use conversational interface: "Show me all resources affected by CVE-2023-XXXX"
3. Display cross-domain impact analysis
4. Show automated remediation options
5. Execute remediation with approval workflow

**Value Demonstrated:**
- Unified security and governance view
- Natural language incident investigation
- Automated response capabilities

### Scenario 2: Cloud Financial Management
**Persona**: FinOps Manager
**Goal**: Optimize cloud costs with governance controls

**Demo Flow:**
1. Show cost spike prediction alert
2. Correlate cost increase with compliance violations
3. Use AI recommendations for cost optimization
4. Execute cost controls with governance approval
5. Show projected savings and compliance impact

**Value Demonstrated:**
- Predictive cost management
- Cost-compliance correlation
- Automated cost optimization

### Scenario 3: Compliance Team
**Persona**: Compliance Officer
**Goal**: Ensure continuous compliance across all environments

**Demo Flow:**
1. Show compliance dashboard with drift predictions
2. Ask conversational AI: "Are we ready for our SOC 2 audit?"
3. Display compliance gaps and remediation plan
4. Execute bulk remediation for compliance fixes
5. Generate compliance report

**Value Demonstrated:**
- Proactive compliance management
- Natural language compliance queries
- Automated compliance remediation

### Scenario 4: DevOps Team
**Persona**: DevOps Engineer
**Goal**: Deploy infrastructure with built-in governance

**Demo Flow:**
1. Show infrastructure deployment request
2. Display governance policy checks
3. Show automated policy enforcement
4. Demonstrate approval workflow for policy exceptions
5. Track deployment with continuous governance monitoring

**Value Demonstrated:**
- Governance-as-code integration
- Automated policy enforcement
- Continuous monitoring

## Demo Environment Setup

### Prerequisites
- Azure subscription with sample resources
- PolicyCortex v2 deployed and configured
- Demo data seeded
- User accounts with appropriate roles

### Quick Demo Setup
```bash
# Start demo environment
scripts/demo-ready.bat

# (Optional) Seed demo data — not required for the default simulated demo
# scripts/seed-demo-data.bat

# Verify UI
curl http://localhost:3000/health || echo "Open http://localhost:3000 in your browser"
```

### Demo Data Requirements
Not required for the base demo; the app provides simulated/mocked endpoints.
For live data, set `USE_REAL_DATA=true` and configure Azure credentials.

## Demo Readiness Checklist

### Technical Verification
- [ ] All services running and responsive
- [ ] Azure integration functional
- [ ] Demo data loaded and current
- [ ] User accounts configured
- [ ] Network connectivity verified

### Demo Content
- [ ] Slides prepared and tested
- [ ] Demo scripts rehearsed
- [ ] Backup scenarios prepared
- [ ] Questions and objections anticipated
- [ ] Follow-up materials ready

### Audience Preparation
- [ ] Audience background understood
- [ ] Use cases aligned with audience needs
- [ ] Technical depth appropriate
- [ ] Business value clearly articulated
- [ ] Next steps defined

## Troubleshooting Common Demo Issues

### Service Connectivity
- **Issue**: Backend services not responding
- **Solution**: Restart services with `.\restart-services.bat`

### Authentication Problems
- **Issue**: Azure AD login failures
- **Solution**: Enable demo mode with `NEXT_PUBLIC_DEMO_MODE=true`

### Data Display Issues
- **Issue**: Empty dashboards or missing data
- **Solution**: Re-seed demo data with `.\scripts\seed-demo-data.bat`

### Performance Problems
- **Issue**: Slow response times
- **Solution**: Restart cache services and check Azure connectivity

## Post-Demo Follow-up

### Technical Questions
- Provide architecture documentation
- Share API documentation
- Offer technical deep-dive sessions
- Discuss integration requirements

### Business Discussions
- ROI analysis and business case
- Implementation timeline and requirements
- Pricing and licensing discussions
- Proof of concept planning

### Next Steps
- Schedule follow-up meetings
- Provide trial access
- Begin technical evaluation
- Plan pilot implementation

PolicyCortex v2 demonstrations showcase the platform's unique capabilities and competitive advantages, positioning it as the future of intelligent cloud governance.