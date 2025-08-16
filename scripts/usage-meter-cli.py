#!/usr/bin/env python3
"""
PolicyCortex Usage Meter CLI
Command-line tool for managing usage quotas, tiers, and billing
"""

import asyncio
import json
import sys
import argparse
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import httpx

console = Console()

@dataclass
class TierUsage:
    tier: str
    tenant_count: int
    total_api_calls: int
    total_revenue: float
    avg_quota_usage: float

@dataclass
class TenantUsage:
    tenant_id: str
    tier: str
    total_api_calls: int
    quota_usage_percentage: float
    monthly_cost: float
    last_activity: datetime

class UsageMeterCLI:
    """Main CLI interface for usage metering"""
    
    def __init__(self, api_base_url: str = "http://localhost:8083"):
        self.api_base_url = api_base_url
        
    async def initialize(self):
        """Initialize connections"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/health")
                if response.status_code != 200:
                    console.print("[red]❌ Cannot connect to Usage Metering Service[/red]")
                    sys.exit(1)
            
            console.print("[green]✅ Connected to Usage Metering Service[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
            sys.exit(1)

@click.group()
@click.option('--api-url', default='http://localhost:8083', help='Usage Metering Service URL')
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
@click.pass_context
def cli(ctx, api_url, format):
    """PolicyCortex Usage Meter CLI"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url
    ctx.obj['format'] = format
    ctx.obj['cli'] = UsageMeterCLI(api_url)

@cli.command()
@click.option('--tenant-id', help='Filter by tenant ID')
@click.option('--usage-type', type=click.Choice(['api_call', 'prediction', 'analysis', 'storage', 'compute']), help='Filter by usage type')
@click.option('--days', default=7, help='Number of days to analyze')
@click.pass_context
async def usage(ctx, tenant_id, usage_type, days):
    """Show usage statistics"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if tenant_id:
        usage_data = await get_tenant_usage(cli_instance.api_base_url, tenant_id, usage_type)
        display_tenant_usage(usage_data, ctx.obj['format'])
    else:
        usage_data = await get_overall_usage(cli_instance.api_base_url, days)
        display_overall_usage(usage_data, ctx.obj['format'])

@cli.command()
@click.option('--tenant-id', help='Filter by tenant ID')
@click.pass_context
async def quotas(ctx, tenant_id):
    """Show quota information"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if tenant_id:
        quota_data = await get_tenant_quotas(cli_instance.api_base_url, tenant_id)
        display_tenant_quotas(tenant_id, quota_data, ctx.obj['format'])
    else:
        all_quotas = await get_all_quotas(cli_instance.api_base_url)
        display_all_quotas(all_quotas, ctx.obj['format'])

@cli.command()
@click.argument('tenant_id')
@click.argument('tier', type=click.Choice(['free', 'pro', 'enterprise']))
@click.pass_context
async def set_tier(ctx, tenant_id, tier):
    """Set tenant tier"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    await update_tenant_tier(cli_instance.api_base_url, tenant_id, tier)
    console.print(f"[green]✅ Tenant {tenant_id} tier updated to {tier}[/green]")

@cli.command()
@click.pass_context
async def tiers(ctx):
    """Show tier configurations"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    tier_configs = await get_tier_configs(cli_instance.api_base_url)
    display_tier_configs(tier_configs, ctx.obj['format'])

@cli.command()
@click.option('--days', default=30, help='Number of days for billing period')
@click.option('--tenant-id', help='Filter by tenant ID')
@click.pass_context
async def billing(ctx, days, tenant_id):
    """Show billing information"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    billing_data = await get_billing_data(cli_instance.api_base_url, days, tenant_id)
    display_billing_data(billing_data, ctx.obj['format'])

@cli.command()
@click.option('--threshold', default=80, help='Warning threshold percentage')
@click.pass_context
async def alerts(ctx, threshold):
    """Show quota alerts"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    alerts_data = await get_quota_alerts(cli_instance.api_base_url, threshold)
    display_quota_alerts(alerts_data, ctx.obj['format'])

@cli.command()
@click.argument('tenant_id')
@click.argument('usage_type', type=click.Choice(['api_call', 'prediction', 'analysis', 'storage', 'compute']))
@click.argument('quantity', type=float)
@click.pass_context
async def record(ctx, tenant_id, usage_type, quantity):
    """Manually record usage"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    usage_data = {
        "tenant_id": tenant_id,
        "api_endpoint": "/manual",
        "usage_type": usage_type,
        "quantity": quantity,
        "unit": "manual",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await record_usage_manual(cli_instance.api_base_url, usage_data)
    console.print(f"[green]✅ Usage recorded: {quantity} {usage_type} for {tenant_id}[/green]")

@cli.command()
@click.option('--output-file', type=click.Path(), help='Export to file')
@click.option('--tenant-id', help='Filter by tenant ID')
@click.option('--days', default=30, help='Number of days to export')
@click.pass_context
async def export(ctx, output_file, tenant_id, days):
    """Export usage data"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    export_data = await export_usage_data(cli_instance.api_base_url, tenant_id, days)
    
    if output_file:
        await save_export_to_file(export_data, output_file)
        console.print(f"[green]Usage data exported to {output_file}[/green]")
    else:
        if ctx.obj['format'] == 'json':
            console.print(json.dumps(export_data, indent=2, default=str))
        elif ctx.obj['format'] == 'yaml':
            console.print(yaml.dump(export_data, default_flow_style=False))
        else:
            display_export_data(export_data)

# Helper functions

async def get_tenant_usage(api_url: str, tenant_id: str, usage_type: Optional[str] = None) -> List[Dict]:
    """Get usage data for a specific tenant"""
    async with httpx.AsyncClient() as client:
        params = {}
        if usage_type:
            params['usage_type'] = usage_type
        
        response = await client.get(f"{api_url}/usage/{tenant_id}", params=params)
        response.raise_for_status()
        return response.json()

async def get_overall_usage(api_url: str, days: int) -> Dict:
    """Get overall usage statistics"""
    # This would call a hypothetical aggregate endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/usage/aggregate", params={'days': days})
        response.raise_for_status()
        return response.json()

async def get_tenant_quotas(api_url: str, tenant_id: str) -> List[Dict]:
    """Get quota data for a specific tenant"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/quotas/{tenant_id}")
        response.raise_for_status()
        return response.json()

async def get_all_quotas(api_url: str) -> List[Dict]:
    """Get all quota data"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/quotas")
        response.raise_for_status()
        return response.json()

async def update_tenant_tier(api_url: str, tenant_id: str, tier: str):
    """Update tenant tier"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api_url}/tiers/{tenant_id}", params={'tier': tier})
        response.raise_for_status()

async def get_tier_configs(api_url: str) -> Dict:
    """Get tier configurations"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/tiers")
        response.raise_for_status()
        return response.json()

async def get_billing_data(api_url: str, days: int, tenant_id: Optional[str] = None) -> Dict:
    """Get billing data"""
    # Mock implementation
    return {
        "total_revenue": 1234.56,
        "total_calls": 50000,
        "period_days": days,
        "tenant_breakdown": [
            {"tenant_id": "tenant-1", "tier": "pro", "revenue": 99.0, "calls": 45000},
            {"tenant_id": "tenant-2", "tier": "free", "revenue": 0.0, "calls": 5000}
        ]
    }

async def get_quota_alerts(api_url: str, threshold: int) -> List[Dict]:
    """Get quota alerts"""
    # Mock implementation
    return [
        {
            "tenant_id": "tenant-1",
            "usage_type": "api_call",
            "usage_percentage": 85.0,
            "current_usage": 42500,
            "monthly_limit": 50000
        }
    ]

async def record_usage_manual(api_url: str, usage_data: Dict):
    """Manually record usage"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api_url}/usage/record", json=usage_data)
        response.raise_for_status()

async def export_usage_data(api_url: str, tenant_id: Optional[str], days: int) -> Dict:
    """Export usage data"""
    # Mock implementation
    return {
        "export_date": datetime.utcnow().isoformat(),
        "period_days": days,
        "tenant_filter": tenant_id,
        "usage_events": [
            {
                "timestamp": "2024-01-01T00:00:00",
                "tenant_id": "tenant-1",
                "usage_type": "api_call",
                "quantity": 1,
                "cost": 0.001
            }
        ]
    }

async def save_export_to_file(data: Dict, file_path: str):
    """Save export data to file"""
    with open(file_path, 'w') as f:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            yaml.dump(data, f, default_flow_style=False)
        else:
            json.dump(data, f, indent=2, default=str)

def display_tenant_usage(usage_data: List[Dict], format: str):
    """Display tenant usage data"""
    if format == 'json':
        console.print(json.dumps(usage_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(usage_data, default_flow_style=False))
    else:
        table = Table(title="Tenant Usage")
        table.add_column("Usage Type", style="cyan")
        table.add_column("Total Quantity", style="green")
        table.add_column("Total Cost", style="yellow")
        
        for usage in usage_data:
            table.add_row(
                usage['usage_type'],
                str(usage['total_quantity']),
                f"${usage['total_cost']:.4f}"
            )
        
        console.print(table)

def display_overall_usage(usage_data: Dict, format: str):
    """Display overall usage data"""
    if format == 'json':
        console.print(json.dumps(usage_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(usage_data, default_flow_style=False))
    else:
        console.print(Panel.fit(f"📊 Overall Usage ({usage_data.get('period_days', 'N/A')} days)", title="PolicyCortex"))
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total API Calls", f"{usage_data.get('total_calls', 0):,}")
        table.add_row("Total Revenue", f"${usage_data.get('total_revenue', 0):.2f}")
        table.add_row("Active Tenants", str(usage_data.get('active_tenants', 0)))
        table.add_row("Avg Calls/Tenant", f"{usage_data.get('avg_calls_per_tenant', 0):.1f}")
        
        console.print(table)

def display_tenant_quotas(tenant_id: str, quota_data: List[Dict], format: str):
    """Display tenant quota data"""
    if format == 'json':
        console.print(json.dumps(quota_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(quota_data, default_flow_style=False))
    else:
        table = Table(title=f"Quotas for {tenant_id}")
        table.add_column("Usage Type", style="cyan")
        table.add_column("Monthly Limit", style="blue")
        table.add_column("Current Usage", style="green")
        table.add_column("Usage %", style="yellow")
        
        for quota in quota_data:
            usage_pct = quota.get('usage_percentage', 0)
            color = "red" if usage_pct >= 90 else "yellow" if usage_pct >= 80 else "green"
            
            table.add_row(
                quota['usage_type'],
                f"{quota['monthly_limit']:,}",
                f"{quota['current_usage']:,}",
                f"[{color}]{usage_pct:.1f}%[/{color}]"
            )
        
        console.print(table)

def display_all_quotas(all_quotas: List[Dict], format: str):
    """Display all quota data"""
    if format == 'json':
        console.print(json.dumps(all_quotas, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(all_quotas, default_flow_style=False))
    else:
        table = Table(title="All Tenant Quotas")
        table.add_column("Tenant ID", style="cyan")
        table.add_column("Usage Type", style="blue")
        table.add_column("Usage %", style="yellow")
        table.add_column("Status", style="green")
        
        for quota in all_quotas:
            usage_pct = quota.get('usage_percentage', 0)
            if usage_pct >= 90:
                status = "[red]Critical[/red]"
            elif usage_pct >= 80:
                status = "[yellow]Warning[/yellow]"
            else:
                status = "[green]OK[/green]"
            
            table.add_row(
                quota['tenant_id'],
                quota['usage_type'],
                f"{usage_pct:.1f}%",
                status
            )
        
        console.print(table)

def display_tier_configs(tier_configs: Dict, format: str):
    """Display tier configurations"""
    if format == 'json':
        console.print(json.dumps(tier_configs, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(tier_configs, default_flow_style=False))
    else:
        for tier_name, config in tier_configs.items():
            console.print(Panel.fit(f"💎 {config['name']} - ${config['monthly_cost']}/month", title=tier_name.title()))
            
            table = Table()
            table.add_column("Quota Type", style="cyan")
            table.add_column("Monthly Limit", style="green")
            table.add_column("Overage Rate", style="yellow")
            
            for quota_type, limit in config['quotas'].items():
                overage_rate = config['overage_rates'].get(quota_type, 0)
                table.add_row(
                    quota_type.replace('_', ' ').title(),
                    f"{limit:,}",
                    f"${overage_rate:.4f}"
                )
            
            console.print(table)
            console.print()

def display_billing_data(billing_data: Dict, format: str):
    """Display billing data"""
    if format == 'json':
        console.print(json.dumps(billing_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(billing_data, default_flow_style=False))
    else:
        console.print(Panel.fit(f"💰 Billing Summary ({billing_data['period_days']} days)", title="Revenue"))
        
        summary_table = Table()
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Revenue", f"${billing_data['total_revenue']:.2f}")
        summary_table.add_row("Total API Calls", f"{billing_data['total_calls']:,}")
        summary_table.add_row("Revenue per Call", f"${billing_data['total_revenue'] / billing_data['total_calls']:.6f}")
        
        console.print(summary_table)
        console.print()
        
        if billing_data.get('tenant_breakdown'):
            tenant_table = Table(title="Tenant Breakdown")
            tenant_table.add_column("Tenant ID", style="cyan")
            tenant_table.add_column("Tier", style="blue")
            tenant_table.add_column("Revenue", style="green")
            tenant_table.add_column("API Calls", style="yellow")
            
            for tenant in billing_data['tenant_breakdown']:
                tenant_table.add_row(
                    tenant['tenant_id'],
                    tenant['tier'].title(),
                    f"${tenant['revenue']:.2f}",
                    f"{tenant['calls']:,}"
                )
            
            console.print(tenant_table)

def display_quota_alerts(alerts_data: List[Dict], format: str):
    """Display quota alerts"""
    if format == 'json':
        console.print(json.dumps(alerts_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(alerts_data, default_flow_style=False))
    else:
        if not alerts_data:
            console.print("[green]✅ No quota alerts[/green]")
            return
        
        table = Table(title="⚠️  Quota Alerts")
        table.add_column("Tenant ID", style="cyan")
        table.add_column("Usage Type", style="blue")
        table.add_column("Usage %", style="yellow")
        table.add_column("Current/Limit", style="green")
        
        for alert in alerts_data:
            usage_pct = alert['usage_percentage']
            color = "red" if usage_pct >= 95 else "yellow"
            
            table.add_row(
                alert['tenant_id'],
                alert['usage_type'],
                f"[{color}]{usage_pct:.1f}%[/{color}]",
                f"{alert['current_usage']:,} / {alert['monthly_limit']:,}"
            )
        
        console.print(table)

def display_export_data(export_data: Dict):
    """Display export data summary"""
    console.print(Panel.fit(f"📤 Usage Export - {export_data['period_days']} days", title="Export Summary"))
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Export Date", export_data['export_date'])
    table.add_row("Period Days", str(export_data['period_days']))
    table.add_row("Tenant Filter", export_data.get('tenant_filter', 'All'))
    table.add_row("Total Events", f"{len(export_data['usage_events']):,}")
    
    console.print(table)

if __name__ == '__main__':
    # Handle async CLI
    import sys
    import inspect
    
    # Patch click to handle async commands
    original_invoke = click.Context.invoke
    
    def patched_invoke(self, callback, *args, **kwargs):
        if inspect.iscoroutinefunction(callback):
            return asyncio.run(callback(*args, **kwargs))
        return original_invoke(self, callback, *args, **kwargs)
    
    click.Context.invoke = patched_invoke
    
    cli()