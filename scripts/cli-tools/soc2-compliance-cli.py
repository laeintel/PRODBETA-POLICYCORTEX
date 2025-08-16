#!/usr/bin/env python3
"""
PolicyCortex SOC-2 Compliance CLI
Command-line tool for managing SOC-2 compliance and evidence collection
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live
import httpx

console = Console()

@dataclass
class ControlStatus:
    control_id: str
    control_name: str
    implementation_status: str
    last_tested: datetime
    test_results: str
    evidence_count: int
    deficiencies_count: int

@dataclass
class EvidenceItem:
    evidence_id: str
    control_id: str
    evidence_type: str
    evidence_source: str
    collection_date: datetime
    description: str

class SOC2ComplianceCLI:
    """Main CLI interface for SOC-2 compliance management"""
    
    def __init__(self, api_base_url: str = "http://localhost:8086"):
        self.api_base_url = api_base_url
        
    async def initialize(self):
        """Initialize connections"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/health")
                if response.status_code != 200:
                    console.print("[red]❌ Cannot connect to Defender Integration Service[/red]")
                    sys.exit(1)
            
            console.print("[green]✅ Connected to Defender Integration Service[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
            sys.exit(1)

@click.group()
@click.option('--api-url', default='http://localhost:8086', help='Defender Integration Service URL')
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
@click.pass_context
def cli(ctx, api_url, format):
    """PolicyCortex SOC-2 Compliance CLI"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url
    ctx.obj['format'] = format
    ctx.obj['cli'] = SOC2ComplianceCLI(api_url)

@cli.group()
def controls():
    """SOC-2 controls management commands"""
    pass

@cli.group()
def evidence():
    """Evidence collection and management commands"""
    pass

@cli.group()
def reports():
    """SOC-2 report generation commands"""
    pass

@cli.group()
def defender():
    """Microsoft Defender integration commands"""
    pass

@controls.command()
@click.option('--category', help='Filter by control category (CC1, CC2, A1, etc.)')
@click.option('--status', type=click.Choice(['implemented', 'partially_implemented', 'not_implemented']), help='Filter by implementation status')
@click.pass_context
async def list(ctx, category, status):
    """List SOC-2 controls"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    controls_data = await get_soc2_controls(cli_instance.api_base_url, category, status)
    display_controls(controls_data, ctx.obj['format'])

@controls.command()
@click.argument('control_id')
@click.pass_context
async def show(ctx, control_id):
    """Show detailed control information"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    control_data = await get_control_details(cli_instance.api_base_url, control_id)
    evidence_data = await get_evidence_by_control(cli_instance.api_base_url, control_id)
    
    display_control_details(control_data, evidence_data, ctx.obj['format'])

@controls.command()
@click.pass_context
async def dashboard(ctx):
    """Show SOC-2 compliance dashboard"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    await show_compliance_dashboard(cli_instance.api_base_url)

@evidence.command()
@click.option('--control-id', help='Collect evidence for specific control')
@click.option('--category', help='Collect evidence for all controls in category')
@click.pass_context
async def collect(ctx, control_id, category):
    """Collect SOC-2 compliance evidence"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting evidence...", total=None)
        
        try:
            control_ids = None
            if control_id:
                control_ids = [control_id]
            elif category:
                # Get all controls for category
                all_controls = await get_soc2_controls(cli_instance.api_base_url)
                control_ids = [c['control_id'] for c in all_controls if c['control_category'] == category]
            
            result = await collect_soc2_evidence(cli_instance.api_base_url, control_ids)
            progress.update(task, description="✅ Evidence collection complete")
            
            display_evidence_collection_result(result, ctx.obj['format'])
            
        except Exception as e:
            progress.update(task, description="❌ Evidence collection failed")
            console.print(f"[red]Evidence collection failed: {e}[/red]")

@evidence.command()
@click.argument('control_id')
@click.option('--type', help='Filter by evidence type')
@click.option('--days', default=30, help='Show evidence from last N days')
@click.pass_context
async def show(ctx, control_id, type, days):
    """Show evidence for a control"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    evidence_data = await get_evidence_by_control(cli_instance.api_base_url, control_id)
    
    # Filter by type if specified
    if type:
        evidence_data = [e for e in evidence_data if e['evidence_type'] == type]
    
    # Filter by date range
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    evidence_data = [e for e in evidence_data if datetime.fromisoformat(e['collection_date'].replace('Z', '+00:00')) > cutoff_date]
    
    display_evidence_list(evidence_data, ctx.obj['format'])

@reports.command()
@click.option('--start-date', required=True, help='Report period start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='Report period end date (YYYY-MM-DD)')
@click.option('--controls', help='Comma-separated list of control IDs to include')
@click.option('--format-type', default='pdf', type=click.Choice(['pdf', 'json', 'csv']), help='Report format')
@click.option('--include-evidence/--no-evidence', default=True, help='Include evidence in report')
@click.option('--output', help='Output file path')
@click.pass_context
async def generate(ctx, start_date, end_date, controls, format_type, include_evidence, output):
    """Generate SOC-2 compliance report"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        return
    
    # Parse controls list
    control_list = None
    if controls:
        control_list = [c.strip() for c in controls.split(',')]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating SOC-2 report...", total=None)
        
        try:
            report_request = {
                "report_period_start": start_dt.isoformat(),
                "report_period_end": end_dt.isoformat(),
                "controls": control_list,
                "include_evidence": include_evidence,
                "output_format": format_type
            }
            
            result = await generate_soc2_report(cli_instance.api_base_url, report_request)
            progress.update(task, description="✅ Report generation complete")
            
            console.print(f"[green]✅ Report generated successfully[/green]")
            console.print(f"[blue]Report ID: {result['report_id']}[/blue]")
            console.print(f"[blue]File Path: {result['file_path']}[/blue]")
            console.print(f"[blue]Controls Tested: {result['controls_tested']}[/blue]")
            console.print(f"[blue]Evidence Items: {result['evidence_items']}[/blue]")
            
            if output and result['file_path']:
                # Copy file to specified output location
                import shutil
                shutil.copy2(result['file_path'], output)
                console.print(f"[green]Report copied to: {output}[/green]")
            
        except Exception as e:
            progress.update(task, description="❌ Report generation failed")
            console.print(f"[red]Report generation failed: {e}[/red]")

@reports.command()
@click.pass_context
async def list(ctx):
    """List generated SOC-2 reports"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    reports_data = await get_soc2_reports(cli_instance.api_base_url)
    display_reports_list(reports_data, ctx.obj['format'])

@defender.command()
@click.option('--start-time', help='Start time for alert ingestion (ISO format)')
@click.option('--end-time', help='End time for alert ingestion (ISO format)')
@click.option('--severity', multiple=True, help='Filter by severity (can be used multiple times)')
@click.option('--include-resolved', is_flag=True, help='Include resolved alerts')
@click.pass_context
async def ingest(ctx, start_time, end_time, severity, include_resolved):
    """Ingest Microsoft Defender alerts"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting Defender alerts...", total=None)
        
        try:
            ingest_request = {
                "include_resolved": include_resolved
            }
            
            if start_time:
                ingest_request["start_time"] = start_time
            if end_time:
                ingest_request["end_time"] = end_time
            if severity:
                ingest_request["severity_filter"] = list(severity)
            
            result = await ingest_defender_alerts(cli_instance.api_base_url, ingest_request)
            progress.update(task, description="✅ Alert ingestion complete")
            
            console.print(f"[green]✅ Alert ingestion started[/green]")
            
        except Exception as e:
            progress.update(task, description="❌ Alert ingestion failed")
            console.print(f"[red]Alert ingestion failed: {e}[/red]")

@defender.command()
@click.option('--limit', default=50, help='Limit number of alerts to show')
@click.option('--severity', help='Filter by severity')
@click.option('--status', help='Filter by status')
@click.pass_context
async def alerts(ctx, limit, severity, status):
    """Show Defender alerts"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    alerts_data = await get_defender_alerts(cli_instance.api_base_url, limit, severity, status)
    display_defender_alerts(alerts_data, ctx.obj['format'])

@cli.command()
@click.option('--watch', is_flag=True, help='Watch for changes')
@click.option('--interval', default=30, help='Watch interval in seconds')
@click.pass_context
async def status(ctx, watch, interval):
    """Show overall compliance status"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if watch:
        while True:
            console.clear()
            await show_overall_status(cli_instance.api_base_url)
            await asyncio.sleep(interval)
    else:
        await show_overall_status(cli_instance.api_base_url)

# Helper functions

async def get_soc2_controls(api_url: str, category: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
    """Get SOC-2 controls"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/soc2/controls")
        response.raise_for_status()
        controls = response.json()
        
        # Apply filters
        if category:
            controls = [c for c in controls if c['control_category'] == category]
        if status:
            controls = [c for c in controls if c['implementation_status'] == status]
        
        return controls

async def get_control_details(api_url: str, control_id: str) -> Dict:
    """Get detailed control information"""
    controls = await get_soc2_controls(api_url)
    for control in controls:
        if control['control_id'] == control_id:
            return control
    raise ValueError(f"Control {control_id} not found")

async def get_evidence_by_control(api_url: str, control_id: str) -> List[Dict]:
    """Get evidence for specific control"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/soc2/evidence/{control_id}")
        response.raise_for_status()
        return response.json()

async def collect_soc2_evidence(api_url: str, control_ids: Optional[List[str]] = None) -> Dict:
    """Collect SOC-2 evidence"""
    async with httpx.AsyncClient() as client:
        params = {}
        if control_ids:
            params['control_ids'] = control_ids
        
        response = await client.post(f"{api_url}/soc2/evidence/collect", params=params)
        response.raise_for_status()
        return response.json()

async def generate_soc2_report(api_url: str, report_request: Dict) -> Dict:
    """Generate SOC-2 report"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{api_url}/soc2/report/generate", json=report_request)
        response.raise_for_status()
        return response.json()

async def get_soc2_reports(api_url: str) -> List[Dict]:
    """Get list of SOC-2 reports"""
    # This would be implemented with an actual API endpoint
    return []

async def ingest_defender_alerts(api_url: str, ingest_request: Dict) -> Dict:
    """Ingest Defender alerts"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api_url}/defender/ingest", json=ingest_request)
        response.raise_for_status()
        return response.json()

async def get_defender_alerts(api_url: str, limit: int, severity: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
    """Get Defender alerts"""
    params = {'limit': limit}
    if severity:
        params['severity'] = severity
    if status:
        params['status'] = status
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/defender/alerts", params=params)
        response.raise_for_status()
        return response.json()

def display_controls(controls_data: List[Dict], format: str):
    """Display SOC-2 controls"""
    if format == 'json':
        console.print(json.dumps(controls_data, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(controls_data, default_flow_style=False))
    else:
        table = Table(title="SOC-2 Controls")
        table.add_column("Control ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Category", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Last Tested", style="gray50")
        
        for control in controls_data:
            status_color = {
                'implemented': 'green',
                'partially_implemented': 'yellow',
                'not_implemented': 'red'
            }.get(control['implementation_status'], 'white')
            
            table.add_row(
                control['control_id'],
                control['control_name'][:50] + "..." if len(control['control_name']) > 50 else control['control_name'],
                control['control_category'],
                f"[{status_color}]{control['implementation_status']}[/{status_color}]",
                control.get('last_tested', 'Never')
            )
        
        console.print(table)

def display_control_details(control_data: Dict, evidence_data: List[Dict], format: str):
    """Display detailed control information"""
    if format == 'json':
        combined_data = {
            'control': control_data,
            'evidence': evidence_data
        }
        console.print(json.dumps(combined_data, indent=2, default=str))
    elif format == 'yaml':
        combined_data = {
            'control': control_data,
            'evidence': evidence_data
        }
        console.print(yaml.dump(combined_data, default_flow_style=False))
    else:
        console.print(Panel.fit(f"🛡️ {control_data['control_name']}", title="Control Details"))
        
        details_table = Table()
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="green")
        
        details_table.add_row("Control ID", control_data['control_id'])
        details_table.add_row("Category", control_data['control_category'])
        details_table.add_row("Status", control_data['implementation_status'])
        details_table.add_row("Last Tested", str(control_data.get('last_tested', 'Never')))
        details_table.add_row("Test Results", control_data.get('test_results', 'N/A'))
        details_table.add_row("Evidence Items", str(len(evidence_data)))
        
        console.print(details_table)
        
        if evidence_data:
            console.print("\n[bold]Evidence Items:[/bold]")
            evidence_table = Table()
            evidence_table.add_column("Type", style="cyan")
            evidence_table.add_column("Source", style="blue")
            evidence_table.add_column("Collection Date", style="gray50")
            evidence_table.add_column("Description", style="green")
            
            for evidence in evidence_data[:10]:  # Show first 10 items
                evidence_table.add_row(
                    evidence['evidence_type'],
                    evidence['evidence_source'],
                    evidence['collection_date'][:10],  # Just the date part
                    evidence['evidence_description'][:50] + "..." if len(evidence['evidence_description']) > 50 else evidence['evidence_description']
                )
            
            console.print(evidence_table)
            
            if len(evidence_data) > 10:
                console.print(f"\n[gray50]... and {len(evidence_data) - 10} more evidence items[/gray50]")

def display_evidence_collection_result(result: Dict, format: str):
    """Display evidence collection results"""
    if format == 'json':
        console.print(json.dumps(result, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(result, default_flow_style=False))
    else:
        console.print("[green]✅ Evidence collection completed[/green]")
        
        total_evidence = sum(len(evidence_list) for evidence_list in result.values())
        console.print(f"[blue]Total evidence items collected: {total_evidence}[/blue]")
        
        for control_id, evidence_list in result.items():
            console.print(f"[cyan]{control_id}[/cyan]: {len(evidence_list)} items")

def display_evidence_list(evidence_data: List[Dict], format: str):
    """Display evidence list"""
    if format == 'json':
        console.print(json.dumps(evidence_data, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(evidence_data, default_flow_style=False))
    else:
        if not evidence_data:
            console.print("[yellow]No evidence found matching criteria[/yellow]")
            return
        
        table = Table(title="Compliance Evidence")
        table.add_column("Type", style="cyan")
        table.add_column("Source", style="blue")
        table.add_column("Date", style="gray50")
        table.add_column("Description", style="green")
        
        for evidence in evidence_data:
            table.add_row(
                evidence['evidence_type'],
                evidence['evidence_source'],
                evidence['collection_date'][:10],
                evidence['evidence_description'][:60] + "..." if len(evidence['evidence_description']) > 60 else evidence['evidence_description']
            )
        
        console.print(table)

def display_reports_list(reports_data: List[Dict], format: str):
    """Display reports list"""
    if format == 'json':
        console.print(json.dumps(reports_data, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(reports_data, default_flow_style=False))
    else:
        if not reports_data:
            console.print("[yellow]No reports found[/yellow]")
            return
        
        table = Table(title="SOC-2 Reports")
        table.add_column("Report ID", style="cyan")
        table.add_column("Period", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Generated", style="gray50")
        
        for report in reports_data:
            period = f"{report['report_period_start'][:10]} to {report['report_period_end'][:10]}"
            table.add_row(
                report['report_id'],
                period,
                report['status'],
                report['generated_at'][:10]
            )
        
        console.print(table)

def display_defender_alerts(alerts_data: List[Dict], format: str):
    """Display Defender alerts"""
    if format == 'json':
        console.print(json.dumps(alerts_data, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(alerts_data, default_flow_style=False))
    else:
        if not alerts_data:
            console.print("[yellow]No alerts found[/yellow]")
            return
        
        table = Table(title="Microsoft Defender Alerts")
        table.add_column("Title", style="cyan")
        table.add_column("Severity", style="red")
        table.add_column("Status", style="yellow")
        table.add_column("Product", style="blue")
        table.add_column("Time", style="gray50")
        
        for alert in alerts_data:
            severity_color = {
                'Critical': 'red',
                'High': 'orange',
                'Medium': 'yellow',
                'Low': 'green'
            }.get(alert['severity'], 'white')
            
            table.add_row(
                alert['title'][:40] + "..." if len(alert['title']) > 40 else alert['title'],
                f"[{severity_color}]{alert['severity']}[/{severity_color}]",
                alert['status'],
                alert['product'],
                alert['first_activity_time'][:16]
            )
        
        console.print(table)

async def show_compliance_dashboard(api_url: str):
    """Show SOC-2 compliance dashboard"""
    console.print(Panel.fit("🛡️ SOC-2 Compliance Dashboard", title="PolicyCortex"))
    
    try:
        controls = await get_soc2_controls(api_url)
        
        # Create status summary
        status_counts = {
            'implemented': 0,
            'partially_implemented': 0,
            'not_implemented': 0
        }
        
        category_counts = {}
        
        for control in controls:
            status_counts[control['implementation_status']] += 1
            category = control['control_category']
            if category not in category_counts:
                category_counts[category] = {'total': 0, 'implemented': 0}
            category_counts[category]['total'] += 1
            if control['implementation_status'] == 'implemented':
                category_counts[category]['implemented'] += 1
        
        # Status overview
        status_tree = Tree("📊 Implementation Status")
        status_tree.add(f"[green]Implemented: {status_counts['implemented']}[/green]")
        status_tree.add(f"[yellow]Partially Implemented: {status_counts['partially_implemented']}[/yellow]")
        status_tree.add(f"[red]Not Implemented: {status_counts['not_implemented']}[/red]")
        
        console.print(status_tree)
        console.print()
        
        # Category breakdown
        category_tree = Tree("📋 By Category")
        for category, counts in category_counts.items():
            percentage = (counts['implemented'] / counts['total']) * 100
            color = "green" if percentage >= 90 else "yellow" if percentage >= 70 else "red"
            category_tree.add(f"[{color}]{category}: {counts['implemented']}/{counts['total']} ({percentage:.1f}%)[/{color}]")
        
        console.print(category_tree)
        
    except Exception as e:
        console.print(f"[red]Failed to load dashboard data: {e}[/red]")

async def show_overall_status(api_url: str):
    """Show overall compliance and security status"""
    console.print(Panel.fit("🔒 PolicyCortex Security & Compliance Status", title="Overview"))
    
    try:
        # Get controls status
        controls = await get_soc2_controls(api_url)
        implemented_count = len([c for c in controls if c['implementation_status'] == 'implemented'])
        total_count = len(controls)
        compliance_percentage = (implemented_count / total_count) * 100 if total_count > 0 else 0
        
        # Get recent alerts
        alerts = await get_defender_alerts(api_url, 10)
        critical_alerts = len([a for a in alerts if a['severity'] == 'Critical'])
        high_alerts = len([a for a in alerts if a['severity'] == 'High'])
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="metrics", size=8),
            Layout(name="alerts", size=10)
        )
        
        # Metrics section
        metrics_table = Table(title="Key Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Status", style="yellow")
        
        compliance_color = "green" if compliance_percentage >= 90 else "yellow" if compliance_percentage >= 70 else "red"
        alert_color = "red" if critical_alerts > 0 else "yellow" if high_alerts > 0 else "green"
        
        metrics_table.add_row(
            "SOC-2 Compliance",
            f"{compliance_percentage:.1f}%",
            f"[{compliance_color}]{'✅ Good' if compliance_percentage >= 90 else '⚠️ Needs Attention' if compliance_percentage >= 70 else '❌ Critical'}[/{compliance_color}]"
        )
        
        metrics_table.add_row(
            "Security Alerts",
            f"{len(alerts)} recent",
            f"[{alert_color}]{'✅ Good' if critical_alerts == 0 and high_alerts == 0 else '⚠️ Monitor' if critical_alerts == 0 else '❌ Action Required'}[/{alert_color}]"
        )
        
        layout["metrics"].update(Panel(metrics_table, title="Status Overview"))
        
        # Recent alerts section
        if alerts:
            recent_alerts_table = Table(title="Recent Security Alerts")
            recent_alerts_table.add_column("Severity", style="red")
            recent_alerts_table.add_column("Title", style="cyan")
            recent_alerts_table.add_column("Time", style="gray50")
            
            for alert in alerts[:5]:  # Show top 5
                severity_color = {
                    'Critical': 'red',
                    'High': 'orange',
                    'Medium': 'yellow',
                    'Low': 'green'
                }.get(alert['severity'], 'white')
                
                recent_alerts_table.add_row(
                    f"[{severity_color}]{alert['severity']}[/{severity_color}]",
                    alert['title'][:50] + "..." if len(alert['title']) > 50 else alert['title'],
                    alert['first_activity_time'][:16]
                )
            
            layout["alerts"].update(Panel(recent_alerts_table, title="Security Alerts"))
        else:
            layout["alerts"].update(Panel("[green]No recent security alerts[/green]", title="Security Alerts"))
        
        console.print(layout)
        
    except Exception as e:
        console.print(f"[red]Failed to load status data: {e}[/red]")

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