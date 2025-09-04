# PolicyCortex PAYBACK Pillar - ROI Engine

## Overview

The PolicyCortex ROI Engine is an enterprise-grade financial impact tracking and simulation system designed to demonstrate 8-15% cloud cost savings within 90 days. It provides CFO-ready metrics, what-if simulations, and comprehensive P&L statements for governance initiatives.

## Key Features

### 1. Cost Calculator
- **Per-policy financial impact**: Calculates precise financial impact of each governance policy
- **Cloud waste elimination**: Identifies and quantifies unused resources, oversized instances
- **Incident prevention value**: Calculates savings from prevented security incidents
- **Azure Cost Management Integration**: Real-time cost data from Azure

### 2. What-If Simulator
- **Monte Carlo simulations**: 10,000 iterations for high confidence projections
- **30/60/90-day projections**: Financial forecasts with confidence intervals
- **Scenario planning**: Model policy changes, resource scaling, automation deployment
- **Sensitivity analysis**: Identify critical parameters affecting ROI

### 3. P&L Dashboard
- **Governance P&L statements**: CFO-ready financial statements
- **ROI metrics**: Percentage returns, payback periods, NPV calculations
- **KPI tracking**: Savings rate, prevention value, risk reduction
- **Executive reporting**: HTML/JSON reports with visualizations

## API Endpoints

### GET /api/v1/roi/dashboard
Returns P&L dashboard metrics with comprehensive financial analysis.

**Query Parameters:**
- `period`: daily|weekly|monthly|quarterly|yearly (default: monthly)
- `detailed`: boolean (default: true)

**Response:**
```json
{
  "pl_statement": {
    "revenue": {
      "direct_cost_savings": 15000,
      "incident_prevention": 8000,
      "productivity_gains": 5000,
      "total": 28000
    },
    "costs": {
      "platform": 10000,
      "implementation": 5000,
      "total": 15000
    },
    "net_results": {
      "net_benefit": 13000,
      "roi_percentage": 86.7,
      "payback_months": 3.5
    }
  },
  "kpis": {
    "savings_rate": 12.5,
    "automation_hours": 120
  }
}
```

### POST /api/v1/roi/simulate
Run what-if financial simulations with Monte Carlo analysis.

**Request Body:**
```json
{
  "scenario_type": "COST_OPTIMIZATION",
  "parameters": [
    {
      "name": "policy_adoption",
      "current_value": 0.75,
      "proposed_value": 0.90,
      "unit": "percentage",
      "impact_factor": 0.3
    }
  ],
  "current_monthly_cost": 100000,
  "time_horizon_days": 90
}
```

**Response:**
```json
{
  "simulation": {
    "expected_savings": 45000,
    "roi_percentage": 250,
    "payback_period_days": 45,
    "confidence_level": 0.87,
    "confidence_intervals": {
      "30": [40000, 50000],
      "60": [80000, 100000],
      "90": [120000, 150000]
    }
  },
  "recommendations": [
    "PRIORITY: Quick win with <30 day payback - implement immediately"
  ]
}
```

### GET /api/v1/roi/savings
Get detailed cost savings breakdown.

**Query Parameters:**
- `days`: 1-365 (default: 30)
- `category`: optional filter

**Response:**
```json
{
  "total_savings": 45000,
  "categories": {
    "compute": 18000,
    "storage": 9000,
    "network": 6750
  },
  "top_policies": [
    {
      "policy_name": "VM Rightsizing Policy",
      "savings": 15750,
      "violations_prevented": 23
    }
  ],
  "optimization_opportunities": [
    {
      "resource_id": "vm-prod-001",
      "potential_savings": 1500,
      "actions": ["Rightsize: CPU at 15%"],
      "payback_days": 7
    }
  ]
}
```

### GET /api/v1/roi/report
Generate executive ROI report.

**Query Parameters:**
- `period_days`: 30-365 (default: 90)
- `format`: json|html|pdf (default: json)
- `include_projections`: boolean (default: true)

**Response:**
```json
{
  "executive_summary": "PolicyCortex delivered $54,000 in net benefits...",
  "financial_impact": { /* P&L data */ },
  "kpis": { /* KPI metrics */ },
  "recommendations": [ /* Action items */ ],
  "projections": {
    "next_quarter": {
      "estimated_savings": 62100,
      "projected_roi": 287.5
    }
  },
  "achievements": {
    "cost_savings_achieved": "15.0%",
    "key_wins": [
      "Prevented 67 policy violations",
      "Saved 280 hours through automation"
    ]
  }
}
```

## Usage Examples

### Python Integration
```python
from core.roi import CostCalculator, WhatIfSimulator, PLDashboardMetrics

# Initialize services
calculator = CostCalculator(subscription_id="your-sub-id")
simulator = WhatIfSimulator(seed=42)
metrics = PLDashboardMetrics()

# Calculate policy impact
impact = await calculator.calculate_policy_impact(
    policy_id="pol-001",
    policy_metrics={
        "violations_prevented": 23,
        "automation_hours_saved": 45
    },
    time_range_days=30
)

print(f"Total savings: ${impact.total_impact}")
print(f"ROI: {impact.roi_percentage}%")
```

### What-If Simulation
```python
from core.roi import ScenarioType, ScenarioParameter

# Define scenario parameters
params = [
    ScenarioParameter(
        name="automation_scope",
        current_value=100,
        proposed_value=200,
        unit="processes",
        impact_factor=0.25,
        confidence_interval=(180, 220)
    )
]

# Run simulation
result = await simulator.simulate_scenario(
    scenario_type=ScenarioType.AUTOMATION_DEPLOYMENT,
    parameters=params,
    current_monthly_cost=Decimal("100000"),
    time_horizon_days=90
)

print(f"Expected savings: ${result.expected_savings}")
print(f"Confidence: {result.confidence_level:.1%}")
```

## Key Metrics Tracked

### Financial Metrics
- **Direct Cost Savings**: Infrastructure cost reductions
- **Incident Prevention Value**: Avoided security breach costs
- **Productivity Gains**: Automation-driven efficiency
- **Compliance Penalty Avoidance**: Risk-adjusted regulatory savings

### Operational Metrics
- **Policies Enforced**: Number of active governance policies
- **Violations Prevented**: Security/compliance violations avoided
- **Resources Optimized**: Cloud resources rightsized/optimized
- **Automation Hours Saved**: Manual work eliminated

### KPIs
- **Savings Rate**: Percentage of cloud spend saved
- **ROI Percentage**: Return on PolicyCortex investment
- **Payback Period**: Time to recover investment
- **Risk Reduction**: Quantified risk mitigation value

## Industry Benchmarks

The ROI engine uses industry-standard benchmarks:

- **Incident Costs**:
  - Minor: $500
  - Moderate: $5,000
  - Major: $50,000
  - Critical: $500,000

- **Compliance Penalties**:
  - GDPR: Up to $20M or 4% revenue
  - HIPAA: $50,000 per violation
  - PCI-DSS: $100,000 per month
  - SOX: $1M+ criminal penalties

- **Optimization Targets**:
  - Compute: 30% typical savings
  - Storage: 25% typical savings
  - Network: 20% typical savings
  - Licensing: 35% typical savings

## Performance Targets

- **8-15% cost savings** within 90 days
- **<100ms** response time for dashboard queries
- **<5s** for what-if simulations (10,000 iterations)
- **95% confidence** in financial projections
- **Real-time** Azure cost data integration

## Installation

```bash
# Install dependencies
pip install -r core/roi/requirements.txt

# Set Azure credentials
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"

# Run API server (if standalone)
cd backend/services/api_gateway
uvicorn main:app --reload --port 8090
```

## Architecture

```
core/roi/
├── cost_calculator/       # Cost impact calculations
│   └── calculator.py      # Main calculator engine
├── what_if_engine/        # Monte Carlo simulations
│   └── simulator.py       # What-if simulator
├── pl_dashboard/          # P&L metrics generation
│   └── metrics.py         # Dashboard metrics service
└── impact_analyzer/       # Impact analysis (optional)
    └── analyzer.py        # Deep impact analytics
```

## Contributing

The ROI engine is a critical component of PolicyCortex's value proposition. When contributing:

1. Maintain high confidence levels (>85%) in financial calculations
2. Use industry-standard benchmarks and methodologies
3. Ensure all monetary values use `Decimal` for precision
4. Include comprehensive error handling and logging
5. Add unit tests for new calculation methods

## License

Proprietary - PolicyCortex. Patent pending technologies implemented.